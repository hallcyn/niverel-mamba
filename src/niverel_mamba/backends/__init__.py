"""Backend implementations.

Imported lazily: pulling in this package must not import torch or MLX, so that
a torch-only user never pays for MLX and vice versa.
"""

from __future__ import annotations

from typing import Any

from .base import Backend

__all__ = ["Backend", "build_backend"]


def build_backend(name: str, config: Any, **kwargs: Any) -> Backend:
    """Instantiate a backend by canonical name."""
    from ..registry import get_backend

    spec = get_backend(name)
    if spec.name == "torch-reference":
        from .torch_reference import TorchReferenceBackend

        return TorchReferenceBackend(config, **kwargs)
    if spec.name == "cuda-reference":
        from .cuda_reference import CudaReferenceBackend

        return CudaReferenceBackend(config, **kwargs)
    if spec.name == "mlx":
        from .mlx import MlxBackend

        return MlxBackend(config, **kwargs)
    raise AssertionError(f"unhandled backend {spec.name!r}")  # pragma: no cover
