"""Inference state for the MLX backend.

Mirrors :class:`niverel_mamba.torch_ops.state.Mamba2State`. Purely functional:
nothing here mutates, because MLX has no in-place mutation to mutate with.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from ._common import eval_state

__all__ = ["Mamba2State"]


@dataclass
class Mamba2State:
    """Recurrent state of a single MLX Mamba2 block."""

    conv_state: mx.array  # (batch, conv_dim, d_conv); index -1 is newest
    ssm_state: mx.array  # (batch, nheads, headdim, d_state)
    seq_idx: mx.array  # (batch,)
    pos: mx.array  # (batch,)

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        *,
        conv_dim: int,
        d_conv: int,
        nheads: int,
        headdim: int,
        d_state: int,
        dtype: Any = mx.float32,
        seq_idx: int = 0,
    ) -> Mamba2State:
        return cls(
            conv_state=mx.zeros((batch_size, conv_dim, d_conv), dtype=dtype),
            ssm_state=mx.zeros((batch_size, nheads, headdim, d_state), dtype=mx.float32),
            seq_idx=mx.full((batch_size,), seq_idx, dtype=mx.int32),
            pos=mx.zeros((batch_size,), dtype=mx.int32),
        )

    def reset_where(self, new_document: mx.array) -> Mamba2State:
        """Zero conv and SSM state for rows that just started a document."""
        keep = mx.logical_not(new_document)
        return replace(
            self,
            conv_state=self.conv_state * keep.astype(self.conv_state.dtype).reshape(-1, 1, 1),
            ssm_state=self.ssm_state * keep.astype(self.ssm_state.dtype).reshape(-1, 1, 1, 1),
            pos=mx.where(new_document, mx.zeros_like(self.pos), self.pos),
        )

    def eval(self) -> Mamba2State:
        """Force evaluation, so a decode loop cannot grow an unbounded graph."""
        eval_state(self.conv_state, self.ssm_state, self.seq_idx, self.pos)
        return self

    @property
    def batch_size(self) -> int:
        return int(self.conv_state.shape[0])
