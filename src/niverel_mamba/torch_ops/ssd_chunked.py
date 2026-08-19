"""Chunked SSD with strict ``seq_idx`` reset.

The algorithm is the paper's chunked decomposition (upstream's
``ssd_minimal_discrete``), hardened for production: internal padding to a
multiple of ``chunk_size``, ``initial_states``, a returned final state, and
document-boundary resets that do not have to align with chunk boundaries.

Cost is ``(L / Q) * Q^2`` rather than ``L^2`` -- a global ``L x L`` decay
matrix is never materialised, which is the entire point.

Padding
-------
``dt`` is padded with **exact zeros after** softplus and clamping, never
before. Padding ``dt_raw`` with zeros would be wrong, because
``softplus(0 + dt_bias) > 0`` -- the pad positions would carry real decay and
``final_state`` would come out silently attenuated. With ``dt_pad = 0`` we get
``exp(dA_pad) = 1`` (identity carry) and a zero injection, so
``h[Lp-1] == h[L-1]`` exactly.

``seq_idx`` is padded by **replicating the last id**, never with a sentinel.
A sentinel would make ``v[Cn-1]`` bogus, the state mask would then reject
every real position of the final chunk, and both that chunk's state and
``final_state`` would collapse to zero.

Strict reset
------------
With ``seq_idx`` non-decreasing (enforced), "same id" is equivalent to "no
boundary in between", and four masks suffice. Writing ``S`` for the chunked
``seq_idx`` and ``v = [S[0,0], S[:,Q-1]]`` for the id owning the state
*entering* each chunk:

===============  ===================  =================================
mask             shape                applied to
===============  ===================  =================================
``segmask``      ``(b, Cn, Q, Q)``    ``CB`` (per group, so cheap)
``statemask``    ``(b, Cn, Q)``       ``decay_states``
``carrymask``    ``(b, Cn+1, Cn+1)``  ``decay_chunk``
``outmask``      ``(b, Cn, Q)``       ``Y_off``
===============  ===================  =================================

The one non-obvious step is ``carrymask``. Inter-chunk passing needs the
product of indicators ``prod_{j=c}^{z-1} g_j`` where ``g_j`` marks "chunk j
contains no boundary". Because ids are monotone, that whole product collapses
to the single equality ``v[c] == v[z]`` -- no cumulative product required.

None of the four masks depends on the head index, which is why they cost
essentially nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ._common import (
    NEG_INF_SURROGATE,
    broadcast_groups,
    prepare_dt,
    resolve_work_dtype,
    segsum_exponent,
)
from .causal_conv import validate_seq_idx

__all__ = ["ssd_chunked"]

#: Cap on the intra-chunk decay tensor before the chunk loop kicks in.
#: ``Lmat`` is ``(b, H, Cn, Q, Q)``, i.e. ``b * H * Lp * Q`` elements, so at
#: L=8192 with 24 heads and Q=256 it is ~200 MB in float32. Splitting the
#: chunk axis is numerically identical -- terms 1 and 2 are independent per
#: chunk -- so this is purely a memory ceiling, never a change of result.
_DECAY_ELEMENT_BUDGET = 64 * 1024 * 1024


def _auto_chunk_block(batch: int, nheads: int, n_chunks: int, chunk_size: int) -> int:
    per_chunk = batch * nheads * chunk_size * chunk_size
    if per_chunk == 0:
        return n_chunks
    block = max(1, _DECAY_ELEMENT_BUDGET // per_chunk)
    return min(n_chunks, block)


def ssd_chunked(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int = 256,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    work_dtype: torch.dtype | None = None,
    allow_downcast: bool = False,
    chunk_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked SSD. Same signature and semantics as :func:`ssd_sequential`.

    Returns ``(y, final_state)``.
    """
    batch, length, nheads, headdim = x.shape
    out_dtype = x.dtype
    work = resolve_work_dtype(x, work_dtype, allow_downcast=allow_downcast)
    device = x.device

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if seq_idx is not None:
        validate_seq_idx(seq_idx, batch, length)
        seq_idx = seq_idx.to(torch.int64)

    # ---------------------------------------------------------------
    # dt pipeline, then padding. Order matters: see the module docstring.
    # ---------------------------------------------------------------
    A_w = A.to(work)
    dt = prepare_dt(dt_raw, dt_bias, work_dtype=work, dt_softplus=dt_softplus, dt_limit=dt_limit)

    n_pad = (-length) % chunk_size
    padded_length = length + n_pad
    n_chunks = padded_length // chunk_size

    x_w = x.to(work)
    B_w = broadcast_groups(B.to(work), nheads)  # (b, L, H, N)
    C_w = broadcast_groups(C.to(work), nheads)
    d_state = B_w.shape[-1]

    if n_pad:
        x_p = F.pad(x_w, (0, 0, 0, 0, 0, n_pad))
        B_p = F.pad(B_w, (0, 0, 0, 0, 0, n_pad))
        C_p = F.pad(C_w, (0, 0, 0, 0, 0, n_pad))
        dt_p = F.pad(dt, (0, 0, 0, n_pad))  # exact zeros, post-softplus
        seq_idx_p: torch.Tensor | None = (
            torch.cat([seq_idx, seq_idx[:, -1:].expand(batch, n_pad)], dim=1)
            if seq_idx is not None
            else None
        )
    else:
        x_p, B_p, C_p, dt_p = x_w, B_w, C_w, dt
        seq_idx_p = seq_idx

    # ---------------------------------------------------------------
    # Chunked quantities
    # ---------------------------------------------------------------
    xc = x_p.reshape(batch, n_chunks, chunk_size, nheads, headdim)
    Bc = B_p.reshape(batch, n_chunks, chunk_size, nheads, d_state)
    Cc = C_p.reshape(batch, n_chunks, chunk_size, nheads, d_state)
    dtc = dt_p.reshape(batch, n_chunks, chunk_size, nheads)

    dA = (dtc * A_w).permute(0, 3, 1, 2)  # (b, H, Cn, Q), non-positive
    cs = torch.cumsum(dA, dim=-1)
    csl = cs[..., -1]  # (b, H, Cn)

    Xdt = xc * dtc.unsqueeze(-1)  # (b, Cn, Q, H, P)

    # ---------------------------------------------------------------
    # Reset masks
    # ---------------------------------------------------------------
    segmask: torch.Tensor | None = None
    statemask: torch.Tensor | None = None
    carrymask: torch.Tensor | None = None
    outmask: torch.Tensor | None = None
    if seq_idx_p is not None:
        S = seq_idx_p.reshape(batch, n_chunks, chunk_size)
        v = torch.cat([S[:, 0, 0:1], S[:, :, -1]], dim=1)  # (b, Cn+1)
        segmask = (S.unsqueeze(-1) == S.unsqueeze(-2)).to(work)  # (b, Cn, Q, Q)
        statemask = (v[:, 1:].unsqueeze(-1) == S).to(work)  # (b, Cn, Q)
        carrymask = (v.unsqueeze(-1) == v.unsqueeze(-2)).to(work)  # (b, Cn+1, Cn+1)
        outmask = (v[:, :-1].unsqueeze(-1) == S).to(work)  # (b, Cn, Q)

    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=device).tril()

    # ---------------------------------------------------------------
    # Term 2: per-chunk states (cheap, computed for every chunk at once)
    # ---------------------------------------------------------------
    decay_states = torch.exp(csl.unsqueeze(-1) - cs)  # (b, H, Cn, Q), all <= 1
    if statemask is not None:
        decay_states = decay_states * statemask.unsqueeze(1)
    states = torch.einsum("bclhp,bhcl,bclhn->bchpn", Xdt, decay_states, Bc)

    # ---------------------------------------------------------------
    # Term 3: inter-chunk state passing
    # ---------------------------------------------------------------
    if initial_states is None:
        first = torch.zeros(batch, 1, nheads, headdim, d_state, dtype=work, device=device)
    else:
        first = initial_states.to(work).unsqueeze(1)
    stacked = torch.cat([first, states], dim=1)  # (b, Cn+1, H, P, N)

    decay_chunk = torch.exp(segsum_exponent(F.pad(csl, (1, 0))))  # (b, H, Cn+1, Cn+1)
    if carrymask is not None:
        decay_chunk = decay_chunk * carrymask.unsqueeze(1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, stacked)
    entering, final_state = new_states[:, :-1], new_states[:, -1]

    # ---------------------------------------------------------------
    # Terms 1 and 4: intra-chunk and inherited output.
    # Split across the chunk axis to bound peak memory. Each chunk is
    # independent here, so this is a pure loop split, not a reassociation.
    # ---------------------------------------------------------------
    block = chunk_block if chunk_block is not None else _auto_chunk_block(
        batch, nheads, n_chunks, chunk_size
    )
    block = max(1, min(block, n_chunks))

    pieces = []
    for start in range(0, n_chunks, block):
        stop = min(start + block, n_chunks)
        sl = slice(start, stop)

        cs_b = cs[:, :, sl]  # (b, H, cb, Q)
        exponent = cs_b.unsqueeze(-1) - cs_b.unsqueeze(-2)  # [l, s] = cs_l - cs_s
        exponent = torch.where(
            causal, exponent, torch.full_like(exponent, NEG_INF_SURROGATE)
        )
        decay_intra = torch.exp(exponent)  # (b, H, cb, Q, Q)

        CB = torch.einsum("bclhn,bcshn->bchls", Cc[:, sl], Bc[:, sl])  # (b, cb, H, Q, Q)
        if segmask is not None:
            CB = CB * segmask[:, sl].unsqueeze(2)
        M = CB * decay_intra.permute(0, 2, 1, 3, 4)
        y_diag = torch.einsum("bchls,bcshp->bclhp", M, Xdt[:, sl])

        y_off = torch.einsum("bclhn,bchpn->bclhp", Cc[:, sl], entering[:, sl])
        y_off = y_off * torch.exp(cs_b).permute(0, 2, 3, 1).unsqueeze(-1)
        if outmask is not None:
            y_off = y_off * outmask[:, sl].unsqueeze(-1).unsqueeze(-1)

        pieces.append(y_diag + y_off)

    y = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
    y = y.reshape(batch, padded_length, nheads, headdim)[:, :length]

    if D is not None:
        D_w = D.to(work)
        y = y + (D_w.view(nheads, 1) if D_w.dim() == 1 else D_w) * x_w

    if z is not None:
        y = y * F.silu(z.to(work))

    return y.to(out_dtype), final_state
