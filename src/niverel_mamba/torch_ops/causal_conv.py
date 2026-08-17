"""Depthwise causal conv1d with document-boundary masking.

Upstream computes this as ``conv1d(x)[..., :-(d_conv - 1)]`` with
``padding=d_conv - 1``, and delegates the ``seq_idx`` variant to the
``causal_conv1d`` CUDA kernel, which zeroes any tap reading a position that
belongs to a different document.

We express both as an explicit sum of ``d_conv`` shifted, masked products.
That form is used **unconditionally**, including when ``seq_idx is None``, and
that choice is deliberate:

The strict-reset test compares ``model(x, seq_idx=s)`` against
``cat([model(doc) for doc in docs])``, and the per-document calls pass
``seq_idx=None``. If those took an ``F.conv1d`` fast path while the segmented
call took the masked path, the two sides would disagree by ~1e-7 purely from
float32 reassociation -- noise that has nothing to do with the invariant under
test and would force the tolerance to be loosened for the wrong reason. One
code path makes the two sides bit-identical.

``conv_impl="native"`` is available as an opt-in performance flag and is
documented as *not* numerically identical.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..errors import InvalidSeqIdxError

__all__ = ["SEQ_IDX_SENTINEL", "causal_conv1d", "validate_seq_idx"]

#: Left-pad value for ``seq_idx``. Never a valid document id, so taps reaching
#: before position 0 are masked out. The data there is zero anyway; masking it
#: keeps the length-1-document case exact rather than accidentally exact.
SEQ_IDX_SENTINEL = -1


def validate_seq_idx(seq_idx: torch.Tensor, batch: int, length: int) -> torch.Tensor:
    """Check the invariant every reset mask depends on.

    ``seq_idx`` must be non-decreasing along the sequence axis. The whole
    chunked derivation rests on "``seq_idx[i] == seq_idx[j]`` for ``i <= j``
    iff no boundary lies in ``(i, j]``", which is false the moment an id
    repeats non-contiguously (``0, 1, 0``).
    """
    if seq_idx.dim() != 2:
        raise InvalidSeqIdxError(f"seq_idx must be (batch, seqlen), got shape {tuple(seq_idx.shape)}")
    if seq_idx.shape != (batch, length):
        raise InvalidSeqIdxError(
            f"seq_idx shape {tuple(seq_idx.shape)} does not match input (batch={batch}, seqlen={length})"
        )
    if seq_idx.numel() and length > 1:
        deltas = seq_idx[:, 1:].to(torch.int64) - seq_idx[:, :-1].to(torch.int64)
        if bool((deltas < 0).any()):
            row = int(torch.nonzero(deltas < 0)[0][0])
            raise InvalidSeqIdxError(
                f"seq_idx must be non-decreasing along the sequence axis; row {row} decreases. "
                "Document ids have to be contiguous and ordered for strict reset to be well defined."
            )
    return seq_idx


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    seq_idx: torch.Tensor | None = None,
    activation: str | None = "silu",
    impl: str = "masked",
) -> torch.Tensor:
    """Causal depthwise convolution over ``(batch, channels, seqlen)``.

    Parameters
    ----------
    x
        ``(batch, channels, seqlen)``.
    weight
        ``(channels, 1, d_conv)`` or ``(channels, d_conv)``. Index
        ``d_conv - 1`` is the **newest** tap, i.e. position ``t`` itself.
    seq_idx
        ``(batch, seqlen)`` document ids. Taps crossing a boundary are zeroed.
    activation
        ``"silu"`` / ``"swish"`` or ``None``.
    impl
        ``"masked"`` (default, exact and boundary-aware) or ``"native"``
        (``F.conv1d``, faster but numerically non-identical and unable to
        honour ``seq_idx``).
    """
    if weight.dim() == 3:
        if weight.shape[1] != 1:
            raise ValueError(f"expected depthwise weight (C, 1, W), got {tuple(weight.shape)}")
        weight2d = weight.squeeze(1)
    elif weight.dim() == 2:
        weight2d = weight
    else:
        raise ValueError(f"conv weight must be 2-D or 3-D, got {tuple(weight.shape)}")

    batch, channels, length = x.shape
    width = weight2d.shape[-1]
    if weight2d.shape[0] != channels:
        raise ValueError(
            f"conv weight has {weight2d.shape[0]} channels but input has {channels}"
        )

    if impl == "native":
        if seq_idx is not None:
            raise ValueError(
                "impl='native' cannot honour seq_idx; use the default 'masked' implementation"
            )
        out = F.conv1d(x, weight2d.unsqueeze(1), bias, padding=width - 1, groups=channels)
        out = out[..., :length]
    elif impl == "masked":
        if seq_idx is not None:
            validate_seq_idx(seq_idx, batch, length)

        padded = F.pad(x, (width - 1, 0))
        if seq_idx is not None:
            padded_idx = F.pad(seq_idx.to(torch.int64), (width - 1, 0), value=SEQ_IDX_SENTINEL)

        out = torch.zeros_like(x)
        for k in range(width):
            # k = 0 is the oldest tap, k = width - 1 is position t itself.
            tap = padded[:, :, k : k + length]
            contribution = weight2d[None, :, k, None] * tap
            if seq_idx is not None:
                same_doc = (padded_idx[:, k : k + length] == seq_idx.to(torch.int64)).to(x.dtype)
                contribution = contribution * same_doc.unsqueeze(1)
            out = out + contribution
        if bias is not None:
            out = out + bias[None, :, None]
    else:
        raise ValueError(f"unknown conv impl {impl!r}; expected 'masked' or 'native'")

    if activation is None:
        return out
    if activation in ("silu", "swish"):
        return F.silu(out)
    raise ValueError(f"unsupported activation {activation!r}")
