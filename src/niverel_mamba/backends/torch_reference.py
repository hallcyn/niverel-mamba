"""The ``torch-reference`` backend: portable pure PyTorch.

Runs on Linux CPU, Linux CUDA without Mamba kernels, macOS CPU and macOS MPS.
Fidelity comes before speed here -- this is the implementation everything else
is measured against.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..capabilities import Capability, Certification
from ..config import Mamba2Config
from ..errors import UnsupportedConfigError
from ..weights import validate_state_dict
from .base import Backend

__all__ = ["TorchReferenceBackend"]


class TorchReferenceBackend(Backend):
    """Wraps :class:`niverel_mamba.torch_ops.mamba2.Mamba2` behind the contract."""

    name = "torch-reference"
    framework = "torch"
    #: Certified against the float64 sequential oracle under sealed
    #: tolerances. Not yet certified against CUDA -- every report names the
    #: reference it was actually measured against.
    certification = Certification.NUMERICALLY_CERTIFIED
    capability = Capability(inference=True, backward="experimental", training=False)
    official_reference = False

    def __init__(
        self,
        config: Mamba2Config,
        *,
        device: str = "cpu",
        dtype: Any = None,
        ssd_impl: str = "chunked",
    ) -> None:
        super().__init__(config)
        import torch

        from ..torch_ops.mamba2 import SSD_IMPLEMENTATIONS, Mamba2

        if ssd_impl not in SSD_IMPLEMENTATIONS:
            raise UnsupportedConfigError(
                f"unknown ssd_impl {ssd_impl!r}; expected one of {SSD_IMPLEMENTATIONS}"
            )
        self.device = device
        self.module = Mamba2(
            config,
            ssd_impl=cast("Any", ssd_impl),
            device=torch.device(device),
            dtype=dtype,
        )

    def load_canonical_weights(self, state_dict: Mapping[str, Any]) -> None:
        import torch

        checked = validate_state_dict(state_dict, self.config)
        target = self.module.in_proj.weight
        moved = {
            key: torch.as_tensor(value).to(device=target.device, dtype=target.dtype)
            for key, value in checked.items()
        }
        self.module.load_state_dict(moved, strict=True)

    def forward(self, x: Any, seq_idx: Any = None) -> Any:
        return self.module(x, seq_idx=seq_idx)

    def allocate_inference_state(self, batch_size: int = 1) -> Any:
        return self.module.allocate_inference_state(batch_size, device=self.device)

    def step(self, x_t: Any, state: Any, seq_idx_t: Any = None) -> tuple[Any, Any]:
        return self.module.step(x_t, state, seq_idx_t=seq_idx_t)
