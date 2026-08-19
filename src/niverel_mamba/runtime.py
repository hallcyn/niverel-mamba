"""High-level runtime entry point.

``load_mamba2`` is the one call that ties everything together: resolve a
backend honestly, build it, load weights strictly, and hand back both the
model and a full description of what is actually running.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .backends import Backend, build_backend
from .config import Mamba2Config
from .registry import Resolution, resolve

__all__ = ["Runtime", "describe", "load_mamba2"]


@dataclass
class Runtime:
    """A ready-to-use backend plus the identity of what was selected."""

    backend: Backend
    resolution: Resolution
    config: Mamba2Config

    def forward(self, x: Any, seq_idx: Any = None) -> Any:
        return self.backend.forward(x, seq_idx)

    __call__ = forward

    def allocate_inference_state(self, batch_size: int = 1) -> Any:
        return self.backend.allocate_inference_state(batch_size)

    def step(self, x_t: Any, state: Any, seq_idx_t: Any = None) -> tuple[Any, Any]:
        return self.backend.step(x_t, state, seq_idx_t=seq_idx_t)

    def identity(self) -> dict[str, Any]:
        """What is actually running, for a caller that needs to display it."""
        return self.resolution.to_dict()


def load_mamba2(
    config: Mamba2Config,
    weights: Mapping[str, Any] | None = None,
    *,
    backend: str = "auto",
    device: str | None = None,
    dtype: Any = None,
    **backend_kwargs: Any,
) -> Runtime:
    """Resolve a backend, build it, and optionally load weights strictly.

    ``backend="auto"`` selects the most trusted available backend and records
    that it did so. Any explicit name is honoured exactly or raises
    :class:`~niverel_mamba.errors.BackendUnavailableError` -- there is never a
    silent switch to something else.
    """
    resolution = resolve(backend, device)

    kwargs: dict[str, Any] = dict(backend_kwargs)
    if resolution.framework == "torch":
        kwargs.setdefault("device", resolution.device)
        if dtype is not None:
            kwargs.setdefault("dtype", dtype)

    instance = build_backend(resolution.backend, config, **kwargs)
    if weights is not None:
        instance.load_canonical_weights(weights)
    return Runtime(backend=instance, resolution=resolution, config=config)


def describe(backend: str = "auto", device: str | None = None) -> dict[str, Any]:
    """Resolve a backend request without building anything."""
    return resolve(backend, device).to_dict()
