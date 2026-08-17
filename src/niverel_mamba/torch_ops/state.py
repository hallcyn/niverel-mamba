"""Inference state for the stateful ``step`` API.

The brief requires the state to carry ``conv_state``, ``ssm_state`` and the
position / segment identity, which is exactly what :class:`Mamba2State` holds.

The API is **functional**: ``step`` returns a new state rather than mutating
the one it was given. Upstream mutates in place because its ``InferenceParams``
cache requires it, but MLX has no in-place mutation, and carrying one contract
across both backends is worth more than matching upstream's plumbing. A
mutating ``step_`` is offered separately for hot decode loops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

__all__ = ["Mamba2State"]


@dataclass
class Mamba2State:
    """Recurrent state of a single Mamba2 block.

    Attributes
    ----------
    conv_state
        ``(batch, conv_dim, d_conv)``. Index ``-1`` is the most recent input,
        matching the tap ordering of ``conv1d.weight``.
    ssm_state
        ``(batch, nheads, headdim, d_state)``. Kept in float32 regardless of
        the weight dtype -- a documented, strictly-better divergence from
        upstream, and consistent with the chunked path which also passes
        states in float32.
    seq_idx
        ``(batch,)``. The document each row is currently inside.
    pos
        ``(batch,)``. Position within the current document.
    """

    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    seq_idx: torch.Tensor
    pos: torch.Tensor

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
        device: torch.device | str | None = None,
        conv_dtype: torch.dtype = torch.float32,
        ssm_dtype: torch.dtype = torch.float32,
        seq_idx: int = 0,
    ) -> Mamba2State:
        """Allocate a zeroed state for ``batch_size`` independent streams."""
        return cls(
            conv_state=torch.zeros(batch_size, conv_dim, d_conv, dtype=conv_dtype, device=device),
            ssm_state=torch.zeros(
                batch_size, nheads, headdim, d_state, dtype=ssm_dtype, device=device
            ),
            seq_idx=torch.full((batch_size,), seq_idx, dtype=torch.int64, device=device),
            pos=torch.zeros(batch_size, dtype=torch.int64, device=device),
        )

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Mamba2State:
        """Move / cast the state. Integer bookkeeping is never cast to float."""
        conv = self.conv_state.to(device=device, dtype=dtype) if dtype else self.conv_state.to(device)
        ssm = self.ssm_state.to(device=device, dtype=dtype) if dtype else self.ssm_state.to(device)
        return Mamba2State(
            conv_state=conv,
            ssm_state=ssm,
            seq_idx=self.seq_idx.to(device),
            pos=self.pos.to(device),
        )

    def detach(self) -> Mamba2State:
        return Mamba2State(
            conv_state=self.conv_state.detach(),
            ssm_state=self.ssm_state.detach(),
            seq_idx=self.seq_idx,
            pos=self.pos,
        )

    def clone(self) -> Mamba2State:
        return Mamba2State(
            conv_state=self.conv_state.clone(),
            ssm_state=self.ssm_state.clone(),
            seq_idx=self.seq_idx.clone(),
            pos=self.pos.clone(),
        )

    def reset_where(self, new_document: torch.Tensor) -> Mamba2State:
        """Zero conv and SSM state for the rows that just started a document.

        Masked rather than branched, so that a batch containing a mix of
        continuing and restarting rows stays a single vectorised call.
        """
        keep_conv = (~new_document).to(self.conv_state.dtype).view(-1, 1, 1)
        keep_ssm = (~new_document).to(self.ssm_state.dtype).view(-1, 1, 1, 1)
        return replace(
            self,
            conv_state=self.conv_state * keep_conv,
            ssm_state=self.ssm_state * keep_ssm,
            pos=torch.where(new_document, torch.zeros_like(self.pos), self.pos),
        )

    @property
    def batch_size(self) -> int:
        return int(self.conv_state.shape[0])
