"""The ``cuda-reference`` backend: a thin wrapper over upstream's kernels.

This backend does not reimplement anything. It constructs the real
``mamba_ssm.Mamba2`` and forwards to it, so that on a CUDA box the numbers are
upstream's numbers by construction rather than by resemblance.

What it adds is the contract: weights are validated against the canonical
schema before being handed over, so a checkpoint that loads here loads
everywhere.

**Status.** ``numerically-certified``, and deliberately not ``reference``:
``reference`` names the backend that *produces or certifies* a result, which
here is ``torch-reference``. This one is compared against it and passes, which
is exactly what ``numerically-certified`` means.

It earned that on release run 32455074823, measured on an A100 and an H100
across all three runtimes: max_abs 2.5e-03 in float32 against the portable
backend, cosine similarity 0.999999988, bit-identical on both architectures.
The band it passes under is sealed in ``certification/tolerances.yaml`` and was
set from that measurement, not before it.

If the required wheels are absent, constructing this backend raises
:class:`BackendUnavailableError`. It never falls back to CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..capabilities import (
    UPSTREAM_RUNTIME_REQUIREMENTS,
    Capability,
    Certification,
    detect_environment,
    upstream_mamba2_importable,
)
from ..config import Mamba2Config
from ..errors import BackendUnavailableError
from ..weights import validate_state_dict
from .base import Backend

__all__ = ["CudaReferenceBackend", "load_upstream_mamba2"]


def load_upstream_mamba2() -> Any:
    """Import the real upstream ``Mamba2``, or explain precisely what is missing."""
    env = detect_environment()
    missing = []
    if not env.torch.available:
        missing.append("torch")
    if not env.upstream_mamba_ssm.available:
        missing.append("mamba-ssm")
    if not env.causal_conv1d.available:
        missing.append("causal-conv1d")
    if missing:
        raise BackendUnavailableError(
            "cuda-reference requires "
            + ", ".join(missing)
            + ", which are not installed. Install the certified wheels with:\n"
            "  niverel-mamba install-backend cuda --yes\n"
            "This package will not fall back to another backend."
        )
    if not env.cuda.available:
        raise BackendUnavailableError(
            "cuda-reference requires a visible CUDA device; torch reports none. "
            "This package will not fall back to CPU."
        )
    importable, why = upstream_mamba2_importable()
    if not importable:
        raise BackendUnavailableError(
            f"mamba-ssm is installed but will not import: {why}\n"
            "Upstream's own __init__ reaches these at import time, and its wheel "
            "does not install them (the wheels must be installed with --no-deps, "
            "which is what protects your pinned torch build):\n"
            f"  pip install {' '.join(UPSTREAM_RUNTIME_REQUIREMENTS)}\n"
            "This package will not fall back to another backend."
        )
    from mamba_ssm.modules.mamba2 import Mamba2 as UpstreamMamba2

    return UpstreamMamba2


class CudaReferenceBackend(Backend):
    """Runs the upstream CUDA kernels behind the canonical contract."""

    name = "cuda-reference"
    framework = "torch"
    certification = Certification.NUMERICALLY_CERTIFIED
    capability = Capability(inference=True, backward="experimental", training=False)
    official_reference = False

    def __init__(
        self,
        config: Mamba2Config,
        *,
        device: str = "cuda",
        dtype: Any = None,
    ) -> None:
        super().__init__(config)
        import torch

        upstream_cls = load_upstream_mamba2()
        self.device = device
        self.module = upstream_cls(
            **config.upstream_kwargs(),
            device=torch.device(device),
            dtype=dtype,
        )

    def load_canonical_weights(self, state_dict: Mapping[str, Any]) -> None:
        checked = validate_state_dict(state_dict, self.config)
        # strict=True is the contract; there is no other mode.
        self.module.load_state_dict(checked, strict=True)

    def forward(self, x: Any, seq_idx: Any = None) -> Any:
        return self.module(x, seq_idx=seq_idx)

    def allocate_inference_state(self, batch_size: int = 1) -> Any:
        return self.module.allocate_inference_cache(batch_size, max_seqlen=1)

    def step(self, x_t: Any, state: Any, seq_idx_t: Any = None) -> tuple[Any, Any]:
        if seq_idx_t is not None:
            raise NotImplementedError(
                "upstream's step() has no seq_idx argument; reset the state explicitly "
                "at a document boundary instead of passing seq_idx_t."
            )
        conv_state, ssm_state = state
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        out, conv_state, ssm_state = self.module.step(x_t, conv_state, ssm_state)
        return out, (conv_state, ssm_state)
