"""The MLX Mamba2 block.

This deliberately does **not** imitate
``torch.nn.Module``. It is a plain object that takes canonical weights via
:meth:`Mamba2.load_canonical_weights` and is called directly.

The weight conversion is reversible by construction: canonical tensors are
stored under their canonical names, with no reshaping or renaming, so
``canonical -> MLX -> canonical`` returns identical bytes. The round-trip
test proves it rather than assuming it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import numpy as np

from ..config import Mamba2Config
from ..errors import InvalidSeqIdxError
from ..schema import WeightContract
from ..weights import validate_state_dict
from ._common import eval_state, silu, softplus
from .causal_conv import causal_conv1d
from .gated_rmsnorm import gated_rmsnorm
from .ssd import ssd_chunked, ssd_sequential
from .state import Mamba2State

__all__ = ["Mamba2", "from_mlx_weights", "to_mlx_weights"]


def _to_mx(value: Any) -> mx.array:
    """Convert a torch / numpy / MLX tensor to an ``mx.array`` losslessly."""
    if isinstance(value, mx.array):
        return value
    if hasattr(value, "detach"):  # torch
        value = value.detach().cpu().numpy()
    return mx.array(np.ascontiguousarray(np.asarray(value)))


def to_mlx_weights(
    state_dict: Mapping[str, Any],
    config: Mamba2Config,
    *,
    contract: WeightContract | None = None,
) -> dict[str, mx.array]:
    """Canonical weights -> MLX arrays. Names and shapes are unchanged."""
    checked = validate_state_dict(state_dict, config, contract=contract)
    return {key: _to_mx(value) for key, value in checked.items()}


def from_mlx_weights(
    weights: Mapping[str, mx.array],
    config: Mamba2Config,
    *,
    contract: WeightContract | None = None,
) -> dict[str, np.ndarray]:
    """MLX arrays -> canonical numpy tensors. The exact inverse of the above."""
    plain = {key: np.array(value) for key, value in weights.items()}
    return validate_state_dict(plain, config, contract=contract)


class Mamba2:
    """A portable Mamba2 mixer implemented in pure MLX."""

    def __init__(self, config: Mamba2Config, *, ssd_impl: str = "chunked") -> None:
        if ssd_impl not in ("chunked", "sequential"):
            raise ValueError(f"unknown ssd_impl {ssd_impl!r}")
        self.config = config
        self.ssd_impl = ssd_impl
        self.weights: dict[str, mx.array] = {}

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def load_canonical_weights(
        self,
        state_dict: Mapping[str, Any],
        *,
        contract: WeightContract | None = None,
    ) -> None:
        """Load weights, validated strictly against the contract."""
        self.weights = to_mlx_weights(state_dict, self.config, contract=contract)
        eval_state(list(self.weights.values()))

    def canonical_weights(self, *, contract: WeightContract | None = None) -> dict[str, np.ndarray]:
        """Return the weights in canonical form, for a round-trip back to torch."""
        return from_mlx_weights(self.weights, self.config, contract=contract)

    def _w(self, name: str) -> mx.array:
        try:
            return self.weights[name]
        except KeyError as exc:
            raise KeyError(
                f"{name!r} has not been loaded; call load_canonical_weights() first"
            ) from exc

    def _A(self) -> mx.array:
        """``A = -exp(A_log)``, always in float32."""
        return -mx.exp(self._w("A_log").astype(mx.float32))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        u: mx.array,
        seq_idx: mx.array | None = None,
        *,
        initial_states: mx.array | None = None,
        return_final_state: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        config = self.config
        batch, length, _ = u.shape

        if seq_idx is not None:
            _validate_seq_idx(seq_idx, batch, length)

        zxbcdt = mx.matmul(u, mx.swapaxes(self._w("in_proj.weight"), 0, 1))
        if config.bias:
            zxbcdt = zxbcdt + self._w("in_proj.bias")

        splits = _split_offsets(config.in_proj_split)
        z0, x0, z, xBC, dt_raw = (zxbcdt[..., a:b] for a, b in splits)

        xBC = causal_conv1d(
            mx.swapaxes(xBC, 1, 2),
            self._w("conv1d.weight"),
            self._w("conv1d.bias") if config.conv_bias else None,
            seq_idx=seq_idx,
            activation="silu",
        )
        xBC = mx.swapaxes(xBC, 1, 2)

        d_ssm = config.effective_d_ssm
        gn = config.ngroups * config.d_state
        x = xBC[..., :d_ssm]
        B = xBC[..., d_ssm : d_ssm + gn]
        C = xBC[..., d_ssm + gn : d_ssm + 2 * gn]

        x_heads = x.reshape(batch, length, config.nheads, config.headdim)
        B_g = B.reshape(batch, length, config.ngroups, config.d_state)
        C_g = C.reshape(batch, length, config.ngroups, config.d_state)

        D_raw = self._w("D")
        D_arg = D_raw.reshape(config.nheads, config.headdim) if config.D_has_hdim else D_raw
        z_arg = None if config.rmsnorm else z.reshape(batch, length, config.nheads, config.headdim)

        scan = ssd_chunked if self.ssd_impl == "chunked" else ssd_sequential
        kwargs: dict[str, Any] = {
            "D": D_arg,
            "z": z_arg,
            "dt_bias": self._w("dt_bias"),
            "dt_softplus": True,
            "dt_limit": config.dt_limit,
            "seq_idx": seq_idx,
            "initial_states": initial_states,
        }
        if self.ssd_impl == "chunked":
            kwargs["chunk_size"] = config.chunk_size

        y, final_state = scan(x_heads, dt_raw, self._A(), B_g, C_g, **kwargs)
        y = y.reshape(batch, length, d_ssm)

        if config.rmsnorm:
            y = gated_rmsnorm(
                y,
                self._w("norm.weight"),
                z,
                eps=config.norm_epsilon,
                group_size=config.norm_group_size,
                norm_before_gate=config.norm_before_gate,
            )

        if config.d_mlp > 0:
            y = mx.concatenate([silu(z0) * x0, y], axis=-1)

        out = mx.matmul(y, mx.swapaxes(self._w("out_proj.weight"), 0, 1))
        if config.bias:
            out = out + self._w("out_proj.bias")

        if return_final_state:
            return out, final_state
        return out

    # ------------------------------------------------------------------
    # Stateful API
    # ------------------------------------------------------------------

    def allocate_inference_state(self, batch_size: int = 1, *, seq_idx: int = 0) -> Mamba2State:
        config = self.config
        return Mamba2State.allocate(
            batch_size,
            conv_dim=config.conv_dim,
            d_conv=config.d_conv,
            nheads=config.nheads,
            headdim=config.headdim,
            d_state=config.d_state,
            seq_idx=seq_idx,
        )

    def step(
        self,
        x_t: mx.array,
        state: Mamba2State,
        seq_idx_t: mx.array | int | None = None,
    ) -> tuple[mx.array, Mamba2State]:
        """Advance one timestep. Returns ``(y_t, new_state)``; nothing mutates."""
        config = self.config
        if x_t.ndim == 3:
            x_t = x_t[:, 0]
        batch = x_t.shape[0]

        if seq_idx_t is not None:
            current = (
                mx.full((batch,), seq_idx_t, dtype=mx.int32)
                if isinstance(seq_idx_t, int)
                else seq_idx_t.reshape(batch).astype(mx.int32)
            )
            state = state.reset_where(current != state.seq_idx)
            state = Mamba2State(state.conv_state, state.ssm_state, current, state.pos)

        zxbcdt = mx.matmul(x_t, mx.swapaxes(self._w("in_proj.weight"), 0, 1))
        if config.bias:
            zxbcdt = zxbcdt + self._w("in_proj.bias")
        splits = _split_offsets(config.in_proj_split)
        z0, x0, z, xBC, dt_raw = (zxbcdt[..., a:b] for a, b in splits)

        conv_state = mx.concatenate(
            [state.conv_state[..., 1:], mx.expand_dims(xBC, -1)], axis=-1
        )
        weight = self._w("conv1d.weight")
        weight2d = weight.reshape(weight.shape[0], weight.shape[-1])
        xBC_out = mx.sum(conv_state * weight2d, axis=-1)
        if config.conv_bias:
            xBC_out = xBC_out + self._w("conv1d.bias")
        xBC_out = silu(xBC_out)

        d_ssm = config.effective_d_ssm
        gn = config.ngroups * config.d_state
        x = xBC_out[..., :d_ssm]
        B = xBC_out[..., d_ssm : d_ssm + gn]
        C = xBC_out[..., d_ssm + gn : d_ssm + 2 * gn]

        work = mx.float32
        A = self._A()
        dt = softplus(dt_raw.astype(work) + self._w("dt_bias").astype(work))
        if config.has_dt_limit:
            dt = mx.clip(dt, config.dt_limit[0], config.dt_limit[1])

        heads, headdim, d_state = config.nheads, config.headdim, config.d_state
        x_h = x.astype(work).reshape(batch, heads, headdim)
        B_g = B.astype(work).reshape(batch, config.ngroups, d_state)
        C_g = C.astype(work).reshape(batch, config.ngroups, d_state)
        repeats = heads // config.ngroups
        B_h = mx.repeat(B_g, repeats=repeats, axis=1) if repeats > 1 else B_g
        C_h = mx.repeat(C_g, repeats=repeats, axis=1) if repeats > 1 else C_g

        decay = mx.exp(dt * A).reshape(batch, heads, 1, 1)
        injection = (
            dt.reshape(batch, heads, 1, 1)
            * mx.expand_dims(x_h, -1)
            * mx.expand_dims(B_h, -2)
        )
        ssm_state = state.ssm_state.astype(work) * decay + injection

        y = mx.sum(ssm_state * mx.expand_dims(C_h, -2), axis=-1)
        D_w = self._w("D").astype(work)
        y = y + (D_w.reshape(heads, 1) if not config.D_has_hdim else D_w.reshape(heads, headdim)) * x_h
        y = y.reshape(batch, d_ssm)

        if config.rmsnorm:
            y = gated_rmsnorm(
                y,
                self._w("norm.weight"),
                z,
                eps=config.norm_epsilon,
                group_size=config.norm_group_size,
                norm_before_gate=config.norm_before_gate,
            )
        else:
            y = y * silu(z)

        if config.d_mlp > 0:
            y = mx.concatenate([silu(z0) * x0, y], axis=-1)

        out = mx.matmul(y, mx.swapaxes(self._w("out_proj.weight"), 0, 1))
        if config.bias:
            out = out + self._w("out_proj.bias")

        new_state = Mamba2State(
            conv_state=conv_state,
            ssm_state=ssm_state,
            seq_idx=state.seq_idx,
            pos=state.pos + 1,
        )
        # Without this the decode loop accumulates an unbounded lazy graph.
        eval_state(out, new_state.conv_state, new_state.ssm_state)
        return out, new_state


def _split_offsets(sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    offsets = []
    start = 0
    for size in sizes:
        offsets.append((start, start + size))
        start += size
    return offsets


def _validate_seq_idx(seq_idx: mx.array, batch: int, length: int) -> None:
    if tuple(seq_idx.shape) != (batch, length):
        raise InvalidSeqIdxError(
            f"seq_idx shape {tuple(seq_idx.shape)} does not match input (batch={batch}, seqlen={length})"
        )
    if length > 1:
        deltas = seq_idx[:, 1:].astype(mx.int32) - seq_idx[:, :-1].astype(mx.int32)
        if bool(mx.any(deltas < 0).item()):
            raise InvalidSeqIdxError(
                "seq_idx must be non-decreasing along the sequence axis; document ids have to be "
                "contiguous and ordered for strict reset to be well defined."
            )
