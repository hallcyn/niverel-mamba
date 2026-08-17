"""Gated RMSNorm in MLX -- the same semantics as the torch port.

``norm_before_gate=False`` means ``norm(x * silu(z))``, and ``eps`` is
``1e-5`` because that is what ``Mamba2.__init__`` hard-codes.
"""

from __future__ import annotations

import mlx.core as mx

from ._common import silu

__all__ = ["gated_rmsnorm"]


def gated_rmsnorm(
    x: mx.array,
    weight: mx.array,
    z: mx.array | None = None,
    *,
    eps: float = 1e-5,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    bias: mx.array | None = None,
) -> mx.array:
    out_dtype = x.dtype
    work = mx.float32

    x = x.astype(work)
    z = z.astype(work) if z is not None else None
    weight = weight.astype(work)
    bias = bias.astype(work) if bias is not None else None

    if z is not None and not norm_before_gate:
        x = x * silu(z)

    n = x.shape[-1]
    if group_size is None or group_size == n:
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        out = x * mx.rsqrt(variance + eps)
    else:
        if n % group_size != 0:
            raise ValueError(f"last dim {n} is not divisible by group_size {group_size}")
        grouped = x.reshape(*x.shape[:-1], n // group_size, group_size)
        variance = mx.mean(mx.square(grouped), axis=-1, keepdims=True)
        out = (grouped * mx.rsqrt(variance + eps)).reshape(x.shape)

    out = out * weight
    if bias is not None:
        out = out + bias
    if z is not None and norm_before_gate:
        out = out * silu(z)
    return out.astype(out_dtype)
