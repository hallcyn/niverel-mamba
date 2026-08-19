"""Shared helpers for the SSD implementations.

Both the sequential oracle and the chunked path must preprocess ``dt`` and
broadcast ``B``/``C`` across heads in *exactly* the same way, otherwise the
two disagree for reasons that have nothing to do with the algorithms. Doing
it once here is the cheapest way to guarantee that.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..errors import UnsupportedDtypeError

__all__ = [
    "NEG_INF_SURROGATE",
    "broadcast_groups",
    "prepare_dt",
    "resolve_work_dtype",
    "segsum_exponent",
]

#: Stand-in for ``-inf`` when masking a decay exponent.
#:
#: ``exp(-1e30)`` is exactly ``0.0`` in float32 and float64, so this is
#: value-identical to ``-inf`` -- but no ``inf`` ever enters the graph, which
#: matters twice over: ``inf * 0 = NaN`` would poison the forward pass, and
#: the backward of ``exp`` at ``-inf`` is a ``0 * inf`` form on some paths.
#: It also sidesteps the differing ``inf`` handling of MPS and MLX.
NEG_INF_SURROGATE = -1e30


def resolve_work_dtype(
    tensor: torch.Tensor,
    requested: torch.dtype | None,
    *,
    allow_downcast: bool = False,
) -> torch.dtype:
    """Pick the dtype the SSD core should compute in.

    The rule, from the brief's fidelity requirements: weights keep whatever
    dtype they were loaded in, but the scan itself always runs in float32,
    with float64 reserved for the CPU oracle.

    MPS has no float64. Rather than silently downcast -- which would let a
    "float64 certification" quietly become a float32 one -- we raise.
    """
    device_type = tensor.device.type
    if requested is None:
        return torch.float64 if tensor.dtype == torch.float64 else torch.float32
    if requested == torch.float64 and device_type == "mps":
        if allow_downcast:
            return torch.float32
        raise UnsupportedDtypeError(
            "float64 is not available on MPS. Run the float64 oracle on CPU, or pass "
            "allow_downcast=True to accept float32 explicitly -- this package will not "
            "downgrade precision behind your back."
        )
    return requested


def prepare_dt(
    dt_raw: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    work_dtype: torch.dtype,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
) -> torch.Tensor:
    """Apply upstream's exact ``dt`` pipeline.

    The order is load-bearing and matches ``_chunk_cumsum_fwd``:
    upcast, then add the bias, then softplus, then clamp to ``dt_limit``.
    Any other order changes the result.
    """
    dt = dt_raw.to(work_dtype)
    if dt_bias is not None:
        dt = dt + dt_bias.to(work_dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit != (0.0, math.inf):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    return dt


def broadcast_groups(tensor: torch.Tensor, nheads: int, group_axis: int = 2) -> torch.Tensor:
    """Expand ``(..., ngroups, d_state)`` to ``(..., nheads, d_state)``.

    Head ``h`` reads group ``h // (nheads // ngroups)``. For the common
    ``ngroups == 1`` case this is a free broadcast rather than a copy.
    """
    ngroups = tensor.shape[group_axis]
    if ngroups == nheads:
        return tensor
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")
    return tensor.repeat_interleave(nheads // ngroups, dim=group_axis)


def segsum_exponent(x: torch.Tensor) -> torch.Tensor:
    """Causal segment-sum exponent: ``out[..., l, s] = sum_{j=s+1}^{l} x[..., j]``.

    Entries above the diagonal are set to :data:`NEG_INF_SURROGATE` *before*
    any exponentiation, so ``exp`` of the result is lower-triangular with an
    exact zero above the diagonal. Masking after ``exp`` would be wrong: for
    ``s > l`` the raw exponent is positive and can overflow to ``inf``.
    """
    length = x.shape[-1]
    cumulative = torch.cumsum(x, dim=-1)
    exponent = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
    causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    return torch.where(causal, exponent, torch.full_like(exponent, NEG_INF_SURROGATE))
