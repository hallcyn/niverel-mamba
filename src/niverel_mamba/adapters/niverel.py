"""Integration helpers for Niverel's H-Net.

Niverel currently does::

    from mamba_ssm import Mamba2
    ...
    self.mixer = Mamba2(d_model=..., d_state=..., d_conv=..., expand=...)

and passes ``seq_idx`` for varlen document isolation (``boundary_policy:
strict_reset``). The goal here is to make the swap conditional and total:
the ``state_dict`` must stay byte-identical, so Foundation V3 checkpoints are
never rewritten.

    from niverel_mamba.adapters.niverel import build_mamba2

    mixer = build_mamba2(config, backend=os.environ.get("NIVEREL_MAMBA_BACKEND", "auto"))
"""

from __future__ import annotations

from typing import Any

from ..config import Mamba2Config
from ..errors import UnknownBackendError
from ..registry import resolve

__all__ = ["NIVEREL_BACKENDS", "build_mamba2", "mamba_kwargs_to_config", "niverel_v3_config"]

#: The backend names Niverel itself uses, mapped to ours.
NIVEREL_BACKENDS = {
    "upstream-cuda": "cuda-reference",
    "niverel-torch": "torch-reference",
    "niverel-mlx": "mlx",
    "auto": "auto",
}


def niverel_v3_config(**overrides: Any) -> Mamba2Config:
    """The Foundation V3 Mamba2 configuration.

    ``headdim``, ``ngroups`` and ``chunk_size`` are never passed by Niverel's
    ``mamba_lm.py``, so upstream's defaults (64 / 1 / 256) apply. Recording
    them explicitly is the point: a default that is never written down is a
    default that silently changes.
    """
    base: dict[str, Any] = {
        "d_model": 768,
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "chunk_size": 256,
        "bias": False,
        "conv_bias": True,
    }
    base.update(overrides)
    return Mamba2Config(**base)


def mamba_kwargs_to_config(**kwargs: Any) -> Mamba2Config:
    """Turn an upstream ``Mamba2(...)`` call's kwargs into a canonical config.

    Lets a caller keep writing the upstream constructor signature while
    getting a validated, serialisable config out of it.
    """
    ignored = {"layer_idx", "process_group", "sequence_parallel", "use_mem_eff_path", "device", "dtype"}
    return Mamba2Config(**{k: v for k, v in kwargs.items() if k not in ignored})


def build_mamba2(config: Mamba2Config, backend: str = "auto", **kwargs: Any) -> Any:
    """The factory brief section 16 asks for.

    Returns a module whose ``state_dict`` is identical either way, so the
    choice of backend never leaks into a checkpoint.
    """
    canonical = NIVEREL_BACKENDS.get(backend, backend)
    if canonical == "auto":
        canonical = resolve("auto").backend

    if canonical == "torch-reference":
        from ..torch_ops.mamba2 import Mamba2

        return Mamba2(config, **kwargs)
    if canonical == "cuda-reference":
        from ..backends.cuda_reference import load_upstream_mamba2

        upstream_cls = load_upstream_mamba2()
        return upstream_cls(**config.upstream_kwargs(), **kwargs)
    if canonical == "mlx":
        from ..mlx_ops.mamba2 import Mamba2 as MlxMamba2

        return MlxMamba2(config, **kwargs)
    raise UnknownBackendError(
        f"unknown backend {backend!r}; expected one of {sorted(NIVEREL_BACKENDS)} "
        "or a canonical backend name"
    )
