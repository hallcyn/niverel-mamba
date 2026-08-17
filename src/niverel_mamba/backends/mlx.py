"""The ``mlx`` backend for Apple Silicon.

Published as ``experimental`` until its parity report against
``torch-reference`` CPU is measured and sealed. The brief is explicit that MLX
must not be announced before parity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..capabilities import Capability, Certification
from ..config import Mamba2Config
from ..errors import BackendUnavailableError
from .base import Backend

__all__ = ["MlxBackend"]


class MlxBackend(Backend):
    """Wraps :class:`niverel_mamba.mlx_ops.mamba2.Mamba2` behind the contract."""

    name = "mlx"
    framework = "mlx"
    certification = Certification.EXPERIMENTAL
    capability = Capability(inference=True, backward=False, training=False)
    official_reference = False

    def __init__(self, config: Mamba2Config, *, ssd_impl: str = "chunked") -> None:
        super().__init__(config)
        try:
            from ..mlx_ops.mamba2 import Mamba2
        except ImportError as exc:
            raise BackendUnavailableError(
                "the mlx backend requires MLX on Apple Silicon: "
                "pip install 'niverel-mamba[mlx]'"
            ) from exc
        self.module = Mamba2(config, ssd_impl=ssd_impl)

    def load_canonical_weights(self, state_dict: Mapping[str, Any]) -> None:
        self.module.load_canonical_weights(state_dict)

    def canonical_weights(self) -> dict[str, Any]:
        """Convert back to canonical form. Byte-identical to what was loaded."""
        return self.module.canonical_weights()

    def forward(self, x: Any, seq_idx: Any = None) -> Any:
        return self.module(x, seq_idx)

    def allocate_inference_state(self, batch_size: int = 1) -> Any:
        return self.module.allocate_inference_state(batch_size)

    def step(self, x_t: Any, state: Any, seq_idx_t: Any = None) -> tuple[Any, Any]:
        return self.module.step(x_t, state, seq_idx_t=seq_idx_t)
