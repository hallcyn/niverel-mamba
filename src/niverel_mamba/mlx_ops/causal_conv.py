"""Depthwise causal conv1d in MLX, as a sum of masked shifts.

``mx.conv1d`` expects NLC layout and its depthwise support (``groups ==
in_channels``) has varied across MLX versions. The sum-of-shifts form needs
only multiply, add and pad, so it is version-proof -- and it is the same
formulation the torch backend uses, which makes a line-by-line parity review
possible.
"""

from __future__ import annotations

import mlx.core as mx

from ._common import silu

__all__ = ["SEQ_IDX_SENTINEL", "causal_conv1d"]

SEQ_IDX_SENTINEL = -1


def causal_conv1d(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    seq_idx: mx.array | None = None,
    activation: str | None = "silu",
) -> mx.array:
    """Causal depthwise convolution over ``(batch, channels, seqlen)``.

    ``weight`` is ``(channels, 1, d_conv)`` or ``(channels, d_conv)``; index
    ``d_conv - 1`` is the newest tap, i.e. position ``t`` itself.
    """
    if weight.ndim == 3:
        weight2d = weight.reshape(weight.shape[0], weight.shape[2])
    elif weight.ndim == 2:
        weight2d = weight
    else:
        raise ValueError(f"conv weight must be 2-D or 3-D, got {weight.shape}")

    batch, channels, length = x.shape
    width = weight2d.shape[-1]

    padded = mx.pad(x, [(0, 0), (0, 0), (width - 1, 0)])
    if seq_idx is not None:
        seq_i = seq_idx.astype(mx.int32)
        padded_idx = mx.pad(seq_i, [(0, 0), (width - 1, 0)], constant_values=SEQ_IDX_SENTINEL)

    out = mx.zeros((batch, channels, length), dtype=x.dtype)
    for k in range(width):
        tap = padded[:, :, k : k + length]
        contribution = weight2d[None, :, k, None] * tap
        if seq_idx is not None:
            same_doc = (padded_idx[:, k : k + length] == seq_i).astype(x.dtype)
            contribution = contribution * mx.expand_dims(same_doc, 1)
        out = out + contribution

    if bias is not None:
        out = out + bias[None, :, None]

    if activation is None:
        return out
    if activation in ("silu", "swish"):
        return silu(out)
    raise ValueError(f"unsupported activation {activation!r}")
