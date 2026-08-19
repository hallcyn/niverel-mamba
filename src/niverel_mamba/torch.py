"""``niverel_mamba.torch`` -- the PyTorch-facing public API.

This is the import Niverel uses to replace ``from mamba_ssm import Mamba2``::

    from niverel_mamba.torch import Mamba2

``Mamba2`` here is a real ``torch.nn.Module`` with upstream's parameter names,
so it drops into an existing H-Net without touching a checkpoint.
"""

from __future__ import annotations

from typing import Any

from .config import Mamba2Config
from .errors import UnknownBackendError
from .torch_ops.mamba2 import SSD_IMPLEMENTATIONS, Mamba2
from .torch_ops.state import Mamba2State

__all__ = ["SSD_IMPLEMENTATIONS", "Mamba2", "Mamba2Config", "Mamba2State", "build_mamba2"]


def build_mamba2(config: Mamba2Config, backend: str = "reference", **kwargs: Any) -> Any:
    """Build a torch Mamba2 for the named backend.

    ``"reference"`` / ``"torch-reference"`` gives the portable module;
    ``"cuda-reference"`` gives the upstream CUDA one, or raises if its wheels
    are absent. There is no fallback between them.
    """
    if backend in ("reference", "torch-reference", "torch", "portable", "niverel-torch"):
        return Mamba2(config, **kwargs)
    if backend in ("cuda-reference", "cuda", "upstream", "upstream-cuda"):
        from .backends.cuda_reference import load_upstream_mamba2

        upstream_cls = load_upstream_mamba2()
        return upstream_cls(**config.upstream_kwargs(), **kwargs)
    raise UnknownBackendError(
        f"unknown torch backend {backend!r}; expected 'reference' or 'cuda-reference'"
    )
