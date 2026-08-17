"""The interface every backend presents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ..capabilities import Capability, Certification
from ..config import Mamba2Config

__all__ = ["Backend"]


class Backend(ABC):
    """A concrete way of running a Mamba2 block.

    Backends differ in framework and device, never in weights: they all load
    the same canonical state dict, strictly.
    """

    name: str
    framework: str
    certification: Certification
    capability: Capability
    official_reference: bool = False

    def __init__(self, config: Mamba2Config) -> None:
        self.config = config

    @abstractmethod
    def load_canonical_weights(self, state_dict: Mapping[str, Any]) -> None:
        """Load weights, refusing anything that violates the contract."""

    @abstractmethod
    def forward(self, x: Any, seq_idx: Any = None) -> Any:
        """Run a full sequence."""

    @abstractmethod
    def allocate_inference_state(self, batch_size: int = 1) -> Any:
        """Allocate a zeroed recurrent state."""

    @abstractmethod
    def step(self, x_t: Any, state: Any, seq_idx_t: Any = None) -> tuple[Any, Any]:
        """Advance one timestep, returning ``(y_t, new_state)``."""

    def identity(self, device: str | None = None) -> dict[str, Any]:
        """Self-description, in the shape brief section 3.3 specifies."""
        return {
            "backend": self.name,
            "framework": self.framework,
            "device": device,
            "certification": self.certification.value,
            "official_reference": self.official_reference,
            "capability": self.capability.to_dict(),
        }
