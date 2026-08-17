"""Shared MLX helpers.

Three MLX facts shape everything in this package:

* **No in-place mutation.** Which is why the ``step`` API is functional on
  *both* backends -- one contract beats matching upstream's plumbing.
* **Lazy evaluation.** Graphs are built, not run, until something forces
  them. During an autoregressive decode that means the state's graph grows
  without bound unless it is evaluated each step. This is the single most
  common MLX performance trap, so :func:`eval_state` exists and is called.
* **Uneven coverage of ``einsum`` and depthwise ``conv1d``** across versions.
  We therefore express every contraction as ``reshape -> matmul -> reshape``
  and the convolution as a sum of shifts -- version-proof, and structurally
  line-for-line comparable with the torch implementation.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

__all__ = [
    "NEG_INF_SURROGATE",
    "broadcast_groups",
    "causal_mask",
    "eval_state",
    "prepare_dt",
    "segsum_exponent",
    "silu",
    "softplus",
]

#: Same rationale as the torch side: ``exp(-1e30) == 0`` exactly, and no
#: ``inf`` ever enters the graph.
NEG_INF_SURROGATE = -1e30


def softplus(x: mx.array) -> mx.array:
    """``log(1 + exp(x))``, numerically stable.

    ``logaddexp(x, 0)`` matches torch's ``F.softplus`` including its
    threshold-20 shortcut: for ``x > 20`` both return ``x`` to within float32
    epsilon.
    """
    return mx.logaddexp(x, mx.zeros_like(x))


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def causal_mask(size: int, dtype: Any = mx.bool_) -> mx.array:
    """Lower-triangular mask built from ``arange`` rather than ``tril``.

    Identical on both backends and free of ``tril``'s dtype quirks.
    """
    idx = mx.arange(size)
    mask = idx[:, None] >= idx[None, :]
    return mask if dtype is mx.bool_ else mask.astype(dtype)


def prepare_dt(
    dt_raw: mx.array,
    dt_bias: mx.array | None,
    *,
    dtype: Any = mx.float32,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
) -> mx.array:
    """Upstream's exact ``dt`` pipeline: upcast, bias, softplus, clamp."""
    dt = dt_raw.astype(dtype)
    if dt_bias is not None:
        dt = dt + dt_bias.astype(dtype)
    if dt_softplus:
        dt = softplus(dt)
    if dt_limit != (0.0, math.inf):
        dt = mx.clip(dt, dt_limit[0], dt_limit[1])
    return dt


def broadcast_groups(tensor: mx.array, nheads: int, group_axis: int = 2) -> mx.array:
    """Expand ``(..., ngroups, d_state)`` to ``(..., nheads, d_state)``."""
    ngroups = tensor.shape[group_axis]
    if ngroups == nheads:
        return tensor
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    return mx.repeat(tensor, repeats=nheads // ngroups, axis=group_axis)


def segsum_exponent(x: mx.array) -> mx.array:
    """Causal segment-sum exponent, masked before any exponentiation."""
    length = x.shape[-1]
    cumulative = mx.cumsum(x, axis=-1)
    exponent = mx.expand_dims(cumulative, -1) - mx.expand_dims(cumulative, -2)
    mask = causal_mask(length)
    return mx.where(mask, exponent, mx.full(exponent.shape, NEG_INF_SURROGATE, exponent.dtype))


def eval_state(*arrays: Any) -> None:
    """Force evaluation of the given arrays.

    Called at the end of every ``step`` so an autoregressive loop does not
    accumulate an unbounded lazy graph.
    """
    flat: list[mx.array] = []
    for item in arrays:
        if item is None:
            continue
        if isinstance(item, mx.array):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            flat.extend(a for a in item if isinstance(a, mx.array))
    if flat:
        mx.eval(*flat)
