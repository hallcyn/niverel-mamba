"""Gated RMSNorm, ported from upstream's ``rms_norm_ref``.

Two details cause almost every bug here, so they are stated up front:

* ``norm_before_gate=False`` -- upstream's default and Niverel's -- means
  ``norm(x * silu(z))``. The other order, ``norm(x) * silu(z)``, is what
  ``rms_norm_ref`` does by *its* default. Getting them backwards produces
  plausible numbers that are simply wrong.
* ``eps`` is ``1e-5``. ``Mamba2.__init__`` hard-codes that when constructing
  ``RMSNormGated``; ``rms_norm_ref``'s own signature default is ``1e-6``.

Everything is computed in float32 (or float64) and cast back to the input
dtype at the very end, exactly as upstream does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gated_rmsnorm"]


def gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    bias: torch.Tensor | None = None,
    upcast: bool = True,
) -> torch.Tensor:
    """RMS-normalise ``x``, gated by ``z``.

    Parameters
    ----------
    x
        ``(..., N)``. Normalised over the last axis, in groups of ``group_size``.
    weight
        ``(N,)`` gain, applied after normalisation.
    z
        Optional gate of the same shape as ``x``.
    group_size
        Elements per normalisation group. ``None`` means one group of ``N``,
        which is what ``ngroups=1`` reduces to.
    norm_before_gate
        ``False`` (default): ``norm(x * silu(z))``.
        ``True``: ``norm(x) * silu(z)``.
    """
    out_dtype = x.dtype
    if upcast:
        work_dtype = x.dtype if x.dtype == torch.float64 else torch.float32
        x = x.to(work_dtype)
        z = z.to(work_dtype) if z is not None else None
        weight = weight.to(work_dtype)
        bias = bias.to(work_dtype) if bias is not None else None

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    n = x.shape[-1]
    if group_size is None or group_size == n:
        variance = x.square().mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
    else:
        if n % group_size != 0:
            raise ValueError(f"last dim {n} is not divisible by group_size {group_size}")
        grouped = x.reshape(*x.shape[:-1], n // group_size, group_size)
        variance = grouped.square().mean(dim=-1, keepdim=True)
        out = (grouped * torch.rsqrt(variance + eps)).reshape(x.shape)

    out = out * weight
    if bias is not None:
        out = out + bias

    if z is not None and norm_before_gate:
        out = out * F.silu(z)

    return out.to(out_dtype)
