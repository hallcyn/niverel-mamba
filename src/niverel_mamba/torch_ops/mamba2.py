"""The portable Mamba2 block as a real ``torch.nn.Module``.

Parameter names, shapes and ``state_dict`` ordering are those of the weight
contract, which was extracted from ``mamba_ssm.Mamba2`` 2.3.2.post1. That is
what lets this module be dropped into an existing H-Net in place of the
upstream one without touching a checkpoint.

The forward keeps upstream's order exactly::

    u -> in_proj -> split [z0, x0, z, xBC, dt]
      -> depthwise causal conv1d over [x, B, C] -> SiLU
      -> split [x, B, C]
      -> dt = softplus(dt + dt_bias);  A = -exp(A_log)
      -> SSD  -> D skip
      -> gated RMSNorm
      -> gated-MLP branch when d_mlp > 0
      -> out_proj
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Mamba2Config
from ..errors import UnsupportedConfigError
from .causal_conv import causal_conv1d, validate_seq_idx
from .gated_rmsnorm import gated_rmsnorm
from .ssd_chunked import ssd_chunked
from .ssd_sequential import ssd_sequential
from .state import Mamba2State

__all__ = ["SSD_IMPLEMENTATIONS", "Mamba2"]

SSD_IMPLEMENTATIONS = ("chunked", "sequential", "per_segment")

SSDImpl = Literal["chunked", "sequential", "per_segment"]


class Mamba2(nn.Module):
    """A portable Mamba2 mixer.

    Parameters
    ----------
    config
        The canonical configuration.
    ssd_impl
        ``"chunked"`` (default, production), ``"sequential"`` (the float64
        oracle), or ``"per_segment"`` (an independent second oracle that
        literally splits the batch at document boundaries and runs each
        document separately -- useful precisely because it shares no code
        with the masking path it is used to check).
    work_dtype
        dtype of the SSD core. ``None`` means float32, or float64 when the
        input is float64.
    """

    def __init__(
        self,
        config: Mamba2Config,
        *,
        ssd_impl: SSDImpl = "chunked",
        work_dtype: torch.dtype | None = None,
        allow_downcast: bool = False,
        conv_impl: str = "masked",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if ssd_impl not in SSD_IMPLEMENTATIONS:
            raise UnsupportedConfigError(
                f"unknown ssd_impl {ssd_impl!r}; expected one of {SSD_IMPLEMENTATIONS}"
            )
        self.config = config
        self.ssd_impl = ssd_impl
        self.work_dtype = work_dtype
        self.allow_downcast = allow_downcast
        self.conv_impl = conv_impl

        factory: dict[str, Any] = {"device": device, "dtype": dtype}

        self.in_proj = nn.Linear(config.d_model, config.d_in_proj, bias=config.bias, **factory)
        self.conv1d = nn.Conv1d(
            in_channels=config.conv_dim,
            out_channels=config.conv_dim,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.conv_dim,
            padding=config.d_conv - 1,
            **factory,
        )
        self.dt_bias = nn.Parameter(torch.empty(config.nheads, **factory))
        self.A_log = nn.Parameter(torch.empty(config.nheads, **factory))
        self.D = nn.Parameter(torch.empty(config.d_D, **factory))

        if config.rmsnorm:
            self.norm = _GatedRMSNorm(
                config.effective_d_ssm,
                eps=config.norm_epsilon,
                group_size=config.norm_group_size,
                norm_before_gate=config.norm_before_gate,
                **factory,
            )
        else:
            self.norm = None  # type: ignore[assignment]

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias, **factory)

        # Upstream marks these as weight-decay exempt; carry the flags so an
        # optimiser configured for upstream behaves identically here.
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation, mirroring upstream so a freshly built module is
    # distributionally identical to a freshly built upstream one.
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        config = self.config
        with torch.no_grad():
            if config.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -config.conv_init, config.conv_init)

            dt = torch.exp(
                torch.rand(config.nheads, device=self.dt_bias.device)
                * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            ).clamp(min=config.dt_init_floor)
            # Inverse of softplus.
            self.dt_bias.copy_((dt + torch.log(-torch.expm1(-dt))).to(self.dt_bias.dtype))

            lo, hi = config.A_init_range
            a = torch.empty(config.nheads, device=self.A_log.device).uniform_(lo, hi)
            self.A_log.copy_(torch.log(a).to(self.A_log.dtype))

            self.D.fill_(1.0)

    # ------------------------------------------------------------------
    # Derived tensors
    # ------------------------------------------------------------------

    def _A(self) -> torch.Tensor:
        """``A = -exp(A_log)``, always computed in float32.

        Upstream is emphatic about the ``.float()``: in float16 a stored
        ``A_log`` can exponentiate to ``-inf``.
        """
        return -torch.exp(self.A_log.float())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        u: torch.Tensor,
        seq_idx: torch.Tensor | None = None,
        *,
        initial_states: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the block over a whole sequence.

        Parameters
        ----------
        u
            ``(batch, seqlen, d_model)``.
        seq_idx
            ``(batch, seqlen)`` document ids. At every change, both the
            convolution state and the SSM state reset to zero.
        """
        config = self.config
        batch, length, _ = u.shape
        if seq_idx is not None:
            validate_seq_idx(seq_idx, batch, length)

        zxbcdt = self.in_proj(u)
        z0, x0, z, xBC, dt_raw = torch.split(zxbcdt, list(config.in_proj_split), dim=-1)

        xBC = causal_conv1d(
            xBC.transpose(1, 2),
            self.conv1d.weight,
            self.conv1d.bias,
            seq_idx=seq_idx,
            activation="silu",
            impl=self.conv_impl,
        ).transpose(1, 2)

        x, B, C = torch.split(
            xBC,
            [config.effective_d_ssm, config.ngroups * config.d_state, config.ngroups * config.d_state],
            dim=-1,
        )

        x_heads = x.reshape(batch, length, config.nheads, config.headdim)
        B_g = B.reshape(batch, length, config.ngroups, config.d_state)
        C_g = C.reshape(batch, length, config.ngroups, config.d_state)
        D_arg = self.D.reshape(config.nheads, config.headdim) if config.D_has_hdim else self.D
        # ``z`` gates inside the scan only when there is no gated RMSNorm to
        # do it afterwards.
        z_arg = None if config.rmsnorm else z.reshape_as(x_heads)

        y, final_state = self._run_ssd(
            x_heads,
            dt_raw,
            B_g,
            C_g,
            D=D_arg,
            z=z_arg,
            seq_idx=seq_idx,
            initial_states=initial_states,
        )
        y = y.reshape(batch, length, config.effective_d_ssm)

        if config.rmsnorm:
            y = self.norm(y, z)

        if config.d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)
        if return_final_state:
            return out, final_state
        return out

    def _run_ssd(
        self,
        x_heads: torch.Tensor,
        dt_raw: torch.Tensor,
        B_g: torch.Tensor,
        C_g: torch.Tensor,
        *,
        D: torch.Tensor | None,
        z: torch.Tensor | None,
        seq_idx: torch.Tensor | None,
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        common: dict[str, Any] = {
            "D": D,
            "z": z,
            "dt_bias": self.dt_bias,
            "dt_softplus": True,
            "dt_limit": config.dt_limit,
            "work_dtype": self.work_dtype,
            "allow_downcast": self.allow_downcast,
        }
        A = self._A()

        if self.ssd_impl == "sequential":
            return ssd_sequential(
                x_heads, dt_raw, A, B_g, C_g, seq_idx=seq_idx, initial_states=initial_states, **common
            )
        if self.ssd_impl == "chunked":
            return ssd_chunked(
                x_heads,
                dt_raw,
                A,
                B_g,
                C_g,
                chunk_size=config.chunk_size,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **common,
            )
        return _ssd_per_segment(
            x_heads,
            dt_raw,
            A,
            B_g,
            C_g,
            chunk_size=config.chunk_size,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **common,
        )

    # ------------------------------------------------------------------
    # Stateful API
    # ------------------------------------------------------------------

    def allocate_inference_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
        seq_idx: int = 0,
    ) -> Mamba2State:
        """Allocate a zeroed :class:`Mamba2State` for autoregressive decoding."""
        config = self.config
        return Mamba2State.allocate(
            batch_size,
            conv_dim=config.conv_dim,
            d_conv=config.d_conv,
            nheads=config.nheads,
            headdim=config.headdim,
            d_state=config.d_state,
            device=device if device is not None else self.out_proj.weight.device,
            conv_dtype=self.conv1d.weight.dtype,
            ssm_dtype=torch.float32,
            seq_idx=seq_idx,
        )

    def step(
        self,
        x_t: torch.Tensor,
        state: Mamba2State,
        seq_idx_t: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, Mamba2State]:
        """Advance one timestep. Returns ``(y_t, new_state)``; does not mutate.

        ``x_t`` is ``(batch, d_model)`` or ``(batch, 1, d_model)``.
        """
        config = self.config
        squeezed = False
        if x_t.dim() == 3:
            if x_t.shape[1] != 1:
                raise ValueError(f"step consumes one token at a time, got seqlen {x_t.shape[1]}")
            x_t = x_t[:, 0]
            squeezed = True
        batch = x_t.shape[0]

        # Document switch first, so a restarting row sees a zeroed state.
        if seq_idx_t is not None:
            current = (
                torch.full((batch,), seq_idx_t, dtype=torch.int64, device=x_t.device)
                if isinstance(seq_idx_t, int)
                else seq_idx_t.reshape(batch).to(torch.int64)
            )
            new_document = current != state.seq_idx
            state = state.reset_where(new_document)
            state = Mamba2State(state.conv_state, state.ssm_state, current, state.pos)

        zxbcdt = self.in_proj(x_t)
        z0, x0, z, xBC, dt_raw = torch.split(zxbcdt, list(config.in_proj_split), dim=-1)

        # Conv step. ``cat`` rather than upstream's ``roll`` + index-assign:
        # numerically identical, allocation-free of aliasing hazards, and it
        # keeps MPS and autograd happy.
        conv_state = torch.cat(
            [state.conv_state[..., 1:], xBC.to(state.conv_state.dtype).unsqueeze(-1)], dim=-1
        )
        weight = self.conv1d.weight.squeeze(1)  # (conv_dim, d_conv)
        xBC_out = (conv_state * weight).sum(dim=-1)
        if self.conv1d.bias is not None:
            xBC_out = xBC_out + self.conv1d.bias
        xBC_out = F.silu(xBC_out).to(x_t.dtype)

        x, B, C = torch.split(
            xBC_out,
            [config.effective_d_ssm, config.ngroups * config.d_state, config.ngroups * config.d_state],
            dim=-1,
        )

        work = torch.float64 if x_t.dtype == torch.float64 else torch.float32
        A = self._A().to(work)

        # Aligned with the forward path: float32, bias then softplus then the
        # dt_limit clamp. Upstream's non-kernel step does neither, which makes
        # it inconsistent with its own forward for bf16 or limited-dt configs.
        dt = dt_raw.to(work) + self.dt_bias.to(work)
        dt = F.softplus(dt)
        if config.has_dt_limit:
            dt = dt.clamp(min=config.dt_limit[0], max=config.dt_limit[1])

        heads, headdim, d_state = config.nheads, config.headdim, config.d_state
        x_h = x.to(work).reshape(batch, heads, headdim)
        B_g = B.to(work).reshape(batch, config.ngroups, d_state)
        C_g = C.to(work).reshape(batch, config.ngroups, d_state)
        repeats = heads // config.ngroups
        # Generalised beyond upstream, which asserts ngroups == 1 on this path.
        B_h = B_g.repeat_interleave(repeats, dim=1) if repeats > 1 else B_g
        C_h = C_g.repeat_interleave(repeats, dim=1) if repeats > 1 else C_g

        decay = torch.exp(dt * A).reshape(batch, heads, 1, 1)
        injection = dt.reshape(batch, heads, 1, 1) * x_h.unsqueeze(-1) * B_h.unsqueeze(-2)
        ssm_state = state.ssm_state.to(work) * decay + injection

        y = (ssm_state * C_h.unsqueeze(-2)).sum(dim=-1)  # (batch, heads, headdim)
        D_w = self.D.to(work)
        y = y + (D_w.reshape(heads, 1) if not config.D_has_hdim else D_w.reshape(heads, headdim)) * x_h
        y = y.reshape(batch, config.effective_d_ssm).to(x_t.dtype)

        y = self.norm(y, z) if config.rmsnorm else y * F.silu(z)

        if config.d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)
        new_state = Mamba2State(
            conv_state=conv_state,
            ssm_state=ssm_state.to(state.ssm_state.dtype),
            seq_idx=state.seq_idx,
            pos=state.pos + 1,
        )
        if squeezed:
            out = out.unsqueeze(1)
        return out, new_state


class _GatedRMSNorm(nn.Module):
    """``RMSNormGated`` with the same parameter name upstream uses (``weight``)."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        return gated_rmsnorm(
            x,
            self.weight,
            z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
        )


def _ssd_per_segment(
    x: torch.Tensor,
    dt_raw: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    *,
    chunk_size: int,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A second, independent oracle: literally split, run, concatenate.

    It shares no masking code with :func:`ssd_chunked`, which is exactly what
    makes it valuable as a cross-check -- a bug in the mask algebra cannot
    hide in both.
    """
    if seq_idx is None:
        return ssd_chunked(
            x, dt_raw, A, B, C, chunk_size=chunk_size, initial_states=initial_states, **kwargs
        )

    batch, length = x.shape[0], x.shape[1]
    rows_y = []
    final_states = []
    for b in range(batch):
        ids = seq_idx[b]
        boundaries = [0]
        for t in range(1, length):
            if int(ids[t]) != int(ids[t - 1]):
                boundaries.append(t)
        boundaries.append(length)

        pieces = []
        # Only the first document inherits ``initial_states``; every later one
        # starts from zero, which is what strict reset means.
        incoming = initial_states[b : b + 1] if initial_states is not None else None
        segment_final = None
        for start, stop in pairwise(boundaries):
            y_seg, segment_final = ssd_chunked(
                x[b : b + 1, start:stop],
                dt_raw[b : b + 1, start:stop],
                A,
                B[b : b + 1, start:stop],
                C[b : b + 1, start:stop],
                chunk_size=chunk_size,
                initial_states=incoming,
                **kwargs,
            )
            pieces.append(y_seg)
            incoming = None
        rows_y.append(torch.cat(pieces, dim=1))
        assert segment_final is not None  # length >= 1 guarantees one segment
        final_states.append(segment_final)
    return torch.cat(rows_y, dim=0), torch.cat(final_states, dim=0)
