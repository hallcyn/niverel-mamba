"""The sequential SSD oracle -- slow, readable, and the source of truth.

This is a literal transcription of the Mamba2 recurrence, one timestep at a
time::

    dt_t = clamp(softplus(dt_raw_t + dt_bias), *dt_limit)
    h_t  = exp(dt_t * A) * h_{t-1} + dt_t * (x_t (x) B_t)
    y_t  = C_t . h_t + D * x_t

Memory is O(state), not O(L^2): the loop never materialises a decay matrix.
It is far too slow for production -- and that is fine. Its job is to be
obviously correct, so that the chunked implementation has something to be
checked against. Per the brief, it is never to be deleted, even once the
chunked path is faster.

Note what ``D`` multiplies: the *raw* post-conv ``x``, never ``dt * x``. And
note that ``dt`` scales ``B``, not ``A`` a second time -- upstream calls its
own minimal implementation as ``ssd_minimal_discrete(x * dt, A * dt, B, C)``,
so inside that function the symbol ``A`` already means ``dA``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ._common import broadcast_groups, prepare_dt, resolve_work_dtype

__all__ = ["ssd_sequential"]


def ssd_sequential(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    work_dtype: torch.dtype | None = None,
    allow_downcast: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the SSD recurrence explicitly.

    Parameters
    ----------
    x
        ``(batch, seqlen, nheads, headdim)``.
    dt_raw
        ``(batch, seqlen, nheads)``, pre-softplus and pre-bias.
    A
        ``(nheads,)``, already ``-exp(A_log)`` and non-positive.
    B, C
        ``(batch, seqlen, ngroups, d_state)``.
    D
        ``(nheads,)`` or ``(nheads, headdim)``. Applied to raw ``x``.
    z
        Optional gate, only used when the caller has ``rmsnorm=False``.
    seq_idx
        ``(batch, seqlen)``. At every change of id the state resets to zero.
    initial_states
        ``(batch, nheads, headdim, d_state)`` state entering position 0.

    Returns
    -------
    ``(y, final_state)`` with ``y`` in the input dtype and ``final_state`` in
    the working dtype.
    """
    batch, length, nheads, headdim = x.shape
    out_dtype = x.dtype
    work = resolve_work_dtype(x, work_dtype, allow_downcast=allow_downcast)

    A_w = A.to(work)
    dt = prepare_dt(dt_raw, dt_bias, work_dtype=work, dt_softplus=dt_softplus, dt_limit=dt_limit)
    dA = dt * A_w  # (batch, seqlen, nheads), non-positive

    B_h = broadcast_groups(B.to(work), nheads)  # (batch, seqlen, nheads, d_state)
    C_h = broadcast_groups(C.to(work), nheads)
    x_w = x.to(work)
    d_state = B_h.shape[-1]

    if initial_states is None:
        state = torch.zeros(batch, nheads, headdim, d_state, dtype=work, device=x.device)
    else:
        state = initial_states.to(work)

    if seq_idx is not None:
        seq_idx = seq_idx.to(torch.int64)

    outputs = []
    for t in range(length):
        if seq_idx is not None and t > 0:
            # Reset *before* the update, so that the first token of a new
            # document still contributes its own injection. That is exactly
            # what starting a fresh forward pass at h = 0 would do.
            keep = (seq_idx[:, t] == seq_idx[:, t - 1]).to(work).view(batch, 1, 1, 1)
            state = state * keep

        decay = torch.exp(dA[:, t]).view(batch, nheads, 1, 1)
        injection = (
            dt[:, t].view(batch, nheads, 1, 1)
            * x_w[:, t].unsqueeze(-1)
            * B_h[:, t].unsqueeze(-2)
        )
        state = decay * state + injection
        y_t = (state * C_h[:, t].unsqueeze(-2)).sum(dim=-1)  # (batch, nheads, headdim)
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1)  # (batch, seqlen, nheads, headdim)

    if D is not None:
        D_w = D.to(work)
        y = y + (D_w.view(nheads, 1) if D_w.dim() == 1 else D_w) * x_w

    if z is not None:
        y = y * F.silu(z.to(work))

    return y.to(out_dtype), state
