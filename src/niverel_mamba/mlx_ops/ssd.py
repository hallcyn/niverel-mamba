"""Chunked SSD in MLX.

A direct translation of :mod:`niverel_mamba.torch_ops.ssd_chunked`, including
the same four reset masks and the same padding rules. Read that module for
the derivation; this file only differs in how contractions are spelled.

Every ``einsum`` becomes ``reshape -> matmul -> reshape``. The five needed
are, in torch notation:

===================================  ============================
contraction                          what it computes
===================================  ============================
``bclhn,bcshn->bchls``               ``CB``
``bchls,bcshp->bclhp``               intra-chunk output
``bclhp,bhcl,bclhn->bchpn``          per-chunk states
``bclhn,bchpn->bclhp``               inherited output
``bhzc,bchpn->bzhpn``                inter-chunk state passing
===================================  ============================
"""

from __future__ import annotations

import math

import mlx.core as mx

from ._common import (
    NEG_INF_SURROGATE,
    broadcast_groups,
    causal_mask,
    prepare_dt,
    segsum_exponent,
    silu,
)

__all__ = ["ssd_chunked", "ssd_sequential"]


def ssd_chunked(
    x: mx.array,
    dt_raw: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    *,
    chunk_size: int = 256,
    D: mx.array | None = None,
    z: mx.array | None = None,
    dt_bias: mx.array | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
    seq_idx: mx.array | None = None,
    initial_states: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Chunked SSD. Returns ``(y, final_state)``.

    Shapes match the torch implementation: ``x`` is ``(b, L, H, P)``,
    ``dt_raw`` is ``(b, L, H)``, ``B``/``C`` are ``(b, L, G, N)``.
    """
    batch, length, nheads, headdim = x.shape
    out_dtype = x.dtype
    work = mx.float32

    A_w = A.astype(work)
    dt = prepare_dt(dt_raw, dt_bias, dtype=work, dt_softplus=dt_softplus, dt_limit=dt_limit)

    n_pad = (-length) % chunk_size
    padded_length = length + n_pad
    n_chunks = padded_length // chunk_size

    x_w = x.astype(work)
    B_w = broadcast_groups(B.astype(work), nheads)
    C_w = broadcast_groups(C.astype(work), nheads)
    d_state = B_w.shape[-1]

    if n_pad:
        x_p = mx.pad(x_w, [(0, 0), (0, n_pad), (0, 0), (0, 0)])
        B_p = mx.pad(B_w, [(0, 0), (0, n_pad), (0, 0), (0, 0)])
        C_p = mx.pad(C_w, [(0, 0), (0, n_pad), (0, 0), (0, 0)])
        # Zeros AFTER softplus/clamp. See the torch module for why padding
        # dt_raw instead would silently attenuate final_state.
        dt_p = mx.pad(dt, [(0, 0), (0, n_pad), (0, 0)])
        if seq_idx is not None:
            tail = mx.broadcast_to(seq_idx[:, -1:], (batch, n_pad))
            seq_p = mx.concatenate([seq_idx, tail], axis=1)
        else:
            seq_p = None
    else:
        x_p, B_p, C_p, dt_p, seq_p = x_w, B_w, C_w, dt, seq_idx

    xc = x_p.reshape(batch, n_chunks, chunk_size, nheads, headdim)
    Bc = B_p.reshape(batch, n_chunks, chunk_size, nheads, d_state)
    Cc = C_p.reshape(batch, n_chunks, chunk_size, nheads, d_state)
    dtc = dt_p.reshape(batch, n_chunks, chunk_size, nheads)

    dA = mx.transpose(dtc * A_w, (0, 3, 1, 2))  # (b, H, Cn, Q)
    cs = mx.cumsum(dA, axis=-1)
    csl = cs[..., -1]  # (b, H, Cn)

    Xdt = xc * mx.expand_dims(dtc, -1)  # (b, Cn, Q, H, P)

    # ---------------- reset masks ----------------
    if seq_p is not None:
        S = seq_p.reshape(batch, n_chunks, chunk_size)
        v = mx.concatenate([S[:, 0, 0:1], S[:, :, -1]], axis=1)  # (b, Cn+1)
        segmask = (mx.expand_dims(S, -1) == mx.expand_dims(S, -2)).astype(work)
        statemask = (mx.expand_dims(v[:, 1:], -1) == S).astype(work)
        carrymask = (mx.expand_dims(v, -1) == mx.expand_dims(v, -2)).astype(work)
        outmask = (mx.expand_dims(v[:, :-1], -1) == S).astype(work)
    else:
        segmask = statemask = carrymask = outmask = None

    # ---------------- term 2: per-chunk states ----------------
    decay_states = mx.exp(mx.expand_dims(csl, -1) - cs)  # (b, H, Cn, Q)
    if statemask is not None:
        decay_states = decay_states * mx.expand_dims(statemask, 1)

    # bclhp,bhcl,bclhn->bchpn
    w_lhs = mx.transpose(decay_states, (0, 2, 3, 1))  # (b, Cn, Q, H)
    Xdt_w = Xdt * mx.expand_dims(w_lhs, -1)  # (b, Cn, Q, H, P)
    lhs = mx.transpose(Xdt_w, (0, 1, 3, 4, 2))  # (b, Cn, H, P, Q)
    rhs = mx.transpose(Bc, (0, 1, 3, 2, 4))  # (b, Cn, H, Q, N)
    states = mx.matmul(lhs, rhs)  # (b, Cn, H, P, N)

    # ---------------- term 3: inter-chunk state passing ----------------
    if initial_states is None:
        first = mx.zeros((batch, 1, nheads, headdim, d_state), dtype=work)
    else:
        first = mx.expand_dims(initial_states.astype(work), 1)
    stacked = mx.concatenate([first, states], axis=1)  # (b, Cn+1, H, P, N)

    decay_chunk = mx.exp(segsum_exponent(mx.pad(csl, [(0, 0), (0, 0), (1, 0)])))
    if carrymask is not None:
        decay_chunk = decay_chunk * mx.expand_dims(carrymask, 1)

    # bhzc,bchpn->bzhpn
    dc = decay_chunk.reshape(batch * nheads, n_chunks + 1, n_chunks + 1)
    st = mx.transpose(stacked, (0, 2, 1, 3, 4)).reshape(
        batch * nheads, n_chunks + 1, headdim * d_state
    )
    new_states = mx.matmul(dc, st).reshape(batch, nheads, n_chunks + 1, headdim, d_state)
    new_states = mx.transpose(new_states, (0, 2, 1, 3, 4))  # (b, Cn+1, H, P, N)
    entering, final_state = new_states[:, :-1], new_states[:, -1]

    # ---------------- terms 1 and 4 ----------------
    exponent = mx.expand_dims(cs, -1) - mx.expand_dims(cs, -2)  # (b, H, Cn, Q, Q)
    mask = causal_mask(chunk_size)
    exponent = mx.where(mask, exponent, mx.full(exponent.shape, NEG_INF_SURROGATE, work))
    decay_intra = mx.exp(exponent)

    # bclhn,bcshn->bchls
    Cq = mx.transpose(Cc, (0, 1, 3, 2, 4)).reshape(batch * n_chunks * nheads, chunk_size, d_state)
    Bq = mx.transpose(Bc, (0, 1, 3, 2, 4)).reshape(batch * n_chunks * nheads, chunk_size, d_state)
    CB = mx.matmul(Cq, mx.swapaxes(Bq, -1, -2)).reshape(
        batch, n_chunks, nheads, chunk_size, chunk_size
    )
    if segmask is not None:
        CB = CB * mx.expand_dims(segmask, 2)
    M = CB * mx.transpose(decay_intra, (0, 2, 1, 3, 4))

    # bchls,bcshp->bclhp
    M_flat = M.reshape(batch * n_chunks * nheads, chunk_size, chunk_size)
    X_flat = mx.transpose(Xdt, (0, 1, 3, 2, 4)).reshape(
        batch * n_chunks * nheads, chunk_size, headdim
    )
    y_diag = mx.matmul(M_flat, X_flat).reshape(
        batch, n_chunks, nheads, chunk_size, headdim
    )
    y_diag = mx.transpose(y_diag, (0, 1, 3, 2, 4))  # (b, Cn, Q, H, P)

    # bclhn,bchpn->bclhp
    C_flat = mx.transpose(Cc, (0, 1, 3, 2, 4)).reshape(
        batch * n_chunks * nheads, chunk_size, d_state
    )
    S_flat = mx.swapaxes(entering, -1, -2).reshape(
        batch * n_chunks * nheads, d_state, headdim
    )
    y_off = mx.matmul(C_flat, S_flat).reshape(batch, n_chunks, nheads, chunk_size, headdim)
    y_off = mx.transpose(y_off, (0, 1, 3, 2, 4))
    y_off = y_off * mx.expand_dims(mx.transpose(mx.exp(cs), (0, 2, 3, 1)), -1)
    if outmask is not None:
        y_off = y_off * mx.expand_dims(mx.expand_dims(outmask, -1), -1)

    y = (y_diag + y_off).reshape(batch, padded_length, nheads, headdim)[:, :length]

    if D is not None:
        D_w = D.astype(work)
        y = y + (D_w.reshape(nheads, 1) if D_w.ndim == 1 else D_w) * x_w

    if z is not None:
        y = y * silu(z.astype(work))

    return y.astype(out_dtype), final_state


def ssd_sequential(
    x: mx.array,
    dt_raw: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    *,
    D: mx.array | None = None,
    z: mx.array | None = None,
    dt_bias: mx.array | None = None,
    dt_softplus: bool = True,
    dt_limit: tuple[float, float] = (0.0, math.inf),
    seq_idx: mx.array | None = None,
    initial_states: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Explicit MLX recurrence. Slow; used only to cross-check the chunked path."""
    batch, length, nheads, headdim = x.shape
    out_dtype = x.dtype
    work = mx.float32

    A_w = A.astype(work)
    dt = prepare_dt(dt_raw, dt_bias, dtype=work, dt_softplus=dt_softplus, dt_limit=dt_limit)
    dA = dt * A_w
    B_h = broadcast_groups(B.astype(work), nheads)
    C_h = broadcast_groups(C.astype(work), nheads)
    x_w = x.astype(work)
    d_state = B_h.shape[-1]

    state = (
        mx.zeros((batch, nheads, headdim, d_state), dtype=work)
        if initial_states is None
        else initial_states.astype(work)
    )

    outputs = []
    for t in range(length):
        if seq_idx is not None and t > 0:
            keep = (seq_idx[:, t] == seq_idx[:, t - 1]).astype(work).reshape(batch, 1, 1, 1)
            state = state * keep
        decay = mx.exp(dA[:, t]).reshape(batch, nheads, 1, 1)
        injection = (
            dt[:, t].reshape(batch, nheads, 1, 1)
            * mx.expand_dims(x_w[:, t], -1)
            * mx.expand_dims(B_h[:, t], -2)
        )
        state = decay * state + injection
        outputs.append(mx.sum(state * mx.expand_dims(C_h[:, t], -2), axis=-1))

    y = mx.stack(outputs, axis=1)
    if D is not None:
        D_w = D.astype(work)
        y = y + (D_w.reshape(nheads, 1) if D_w.ndim == 1 else D_w) * x_w
    if z is not None:
        y = y * silu(z.astype(work))
    return y.astype(out_dtype), state
