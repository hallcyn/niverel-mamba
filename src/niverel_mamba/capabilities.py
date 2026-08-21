"""Runtime capability detection.

Everything here is pure inspection. Importing this module -- or the package --
downloads nothing, spawns no subprocess and compiles nothing, which is a hard
requirement of this package.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Capability",
    "Certification",
    "Environment",
    "FrameworkInfo",
    "detect_environment",
    "mlx_available",
    "torch_available",
]


class Certification(str, Enum):
    """How much a backend's numbers have actually been proven.

    These are the four statuses this project publishes, and they are reported
    as-is. A backend is never advertised at a level it has not earned.
    """

    #: The exact backend used to produce or certify a result.
    REFERENCE = "reference"
    #: Compared against a reference under sealed tolerances, and passed.
    NUMERICALLY_CERTIFIED = "numerically-certified"
    #: Functional, but not yet certified against anything.
    EXPERIMENTAL = "experimental"
    #: This combination is explicitly refused.
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Capability:
    """What a backend can honestly claim to do."""

    inference: bool = True
    #: ``"experimental"`` until gradients have been compared against CUDA.
    backward: bool | str = "experimental"
    #: Only ever ``True`` once the gradient comparison exists.
    training: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameworkInfo:
    available: bool
    version: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Environment:
    """A snapshot of what this machine can run."""

    python_version: str
    platform_system: str
    platform_machine: str
    platform_release: str
    torch: FrameworkInfo
    mlx: FrameworkInfo
    cuda: FrameworkInfo
    mps: FrameworkInfo
    upstream_mamba_ssm: FrameworkInfo
    causal_conv1d: FrameworkInfo
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _distribution_installed(name: str) -> bool:
    """Whether a distribution of this name is installed, per its metadata."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import distribution as _distribution

    try:
        _distribution(name)
    except PackageNotFoundError:
        return False
    except Exception:  # pragma: no cover - broken metadata on disk
        return False
    return True


def _module_available(name: str, distribution_name: str | None = None) -> bool:
    """Whether ``name`` is a real, importable framework.

    A bare ``find_spec(...) is not None`` is not enough: any directory of that
    name on ``sys.path`` becomes an implicit namespace package and answers
    yes. That is not hypothetical -- this repository has a ``tests/mlx``
    directory, and under pytest it made this function report MLX as installed
    on a machine that had none.

    Nor is "reject namespace packages" enough on its own, because **MLX really
    is one**: ``site-packages/mlx`` has no top-level ``__init__.py``, so its
    spec has no loader and no origin either. The two cases are indistinguishable
    by spec alone.

    Installed *metadata* is what separates them: a real distribution has a
    ``.dist-info``, an incidental directory never does. The spec check is kept
    only as a fallback for the unusual case of an importable module with a real
    loader but no metadata (a vendored single-file module, say).
    """
    if _distribution_installed(distribution_name or name):
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    return spec.origin is not None or spec.loader is not None


#: Upstream's own import-time requirements that its wheels do not carry when
#: installed with ``--no-deps`` -- which is how they must be installed, since
#: that is what protects the pinned torch build from being replaced.
#:
#: Derived from upstream's import graph rather than guessed: walking the hard
#: imports from ``mamba_ssm/__init__`` reaches thirty-one modules and six
#: external names. Three need no action -- ``packaging`` is a core dependency
#: here, ``triton`` ships with the CUDA torch build, and ``selective_scan_cuda``
#: is the compiled extension inside the wheel itself. These three are what is
#: left.
#:
#: ``huggingface_hub`` is listed even though ``transformers`` depends on it,
#: because ``mamba_ssm/modules/mamba2.py`` imports it directly. Inheriting a
#: direct import through someone else's dependency is a bet on their packaging.
#:
#: ``einops`` was missing until `install-backend` was finally run in a clean
#: environment: every environment that had exercised it before also carried
#: einops for other reasons, so `import mamba_ssm.modules.mamba2` succeeded and
#: the gap stayed invisible.
UPSTREAM_RUNTIME_REQUIREMENTS = ("einops", "huggingface_hub", "transformers")


def upstream_mamba2_importable() -> tuple[bool, str | None]:
    """Whether upstream's ``Mamba2`` can be *imported*, not merely installed.

    Distribution metadata proves a wheel is present. It does not prove the
    package imports, and for mamba-ssm 2.3.2.post1 the two came apart:
    ``mamba_ssm/__init__.py`` pulls in ``models.mixer_seq_simple``, which
    reaches ``transformers`` -- absent, because the certified wheels are
    installed with ``--no-deps``.

    The distinction is not academic, and the way it fails is worse than a plain
    error. ``__init__`` imports ``modules.mamba2`` *before* the line that
    fails, so when the package import blows up Python has already cached
    ``mamba_ssm.modules.mamba2`` in ``sys.modules``. A later
    ``from mamba_ssm.modules.mamba2 import Mamba2`` in the **same process**
    then succeeds from that cache, without re-running ``__init__``. A GPU
    certification run passed its parity tests exactly that way, against a
    backend that a fresh process could not load at all.

    So availability is answered by importing, in the only way that means
    anything: the same import the backend itself performs.
    """
    global _IMPORTABLE
    if _IMPORTABLE is not None:
        return _IMPORTABLE
    try:
        from mamba_ssm.modules.mamba2 import Mamba2  # noqa: F401
    except Exception as exc:
        missing = getattr(exc, "name", None)
        hint = (
            f"; upstream requires {missing!r}, which its wheel does not install"
            if missing in UPSTREAM_RUNTIME_REQUIREMENTS
            else ""
        )
        # Deliberately not cached: a failure can be repaired inside the same
        # process by installing what is missing, and a stale "no" would then
        # outlive the problem.
        return False, f"{type(exc).__name__}: {exc}{hint}"
    _IMPORTABLE = (True, None)
    return _IMPORTABLE


#: Only a success is memoised; see above.
_IMPORTABLE: tuple[bool, str | None] | None = None


def torch_available() -> bool:
    return _module_available("torch")


def mlx_available() -> bool:
    return _module_available("mlx")


def _probe_torch() -> tuple[FrameworkInfo, FrameworkInfo, FrameworkInfo]:
    """Return ``(torch, cuda, mps)`` information without importing gratuitously."""
    if not torch_available():
        absent = FrameworkInfo(False, detail="torch is not installed")
        return (
            FrameworkInfo(False, detail="install with: pip install 'niverel-mamba[torch]'"),
            absent,
            absent,
        )
    import torch

    torch_info = FrameworkInfo(True, version=torch.__version__)

    cuda_version = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            detail = f"sm_{major}{minor} ({torch.cuda.get_device_name(0)})"
        except Exception:  # pragma: no cover - driver quirks
            detail = "available"
        cuda_info = FrameworkInfo(True, version=cuda_version, detail=detail)
    else:
        cuda_info = FrameworkInfo(
            False,
            version=cuda_version,
            detail="no CUDA device visible to torch",
        )

    mps_info = _probe_mps(torch)
    return torch_info, cuda_info, mps_info


def _probe_mps(torch: Any) -> FrameworkInfo:
    """Whether MPS is not merely advertised but actually usable.

    ``torch.backends.mps.is_available()`` is not sufficient. GitHub's
    virtualised macOS runners answer *true* and then fail every allocation:

        RuntimeError: MPS backend out of memory (MPS allocated: 0 bytes,
        max allowed: 7.93 GiB). Tried to allocate 21.00 KiB on shared pool.

    Believing the flag on such a machine means ``doctor`` advertises MPS and
    ``auto`` selects it, only for the first real tensor to blow up. Reporting a
    device we cannot use would be its own kind of silent lie, so we spend one
    tiny allocation to find out.
    """
    if not torch.backends.mps.is_built():
        return FrameworkInfo(False, detail="not built")
    if not torch.backends.mps.is_available():
        return FrameworkInfo(False, detail="built but unavailable")
    try:
        probe = torch.zeros(8, device="mps")
        float((probe + 1).sum().cpu())
    except Exception as exc:
        return FrameworkInfo(
            False,
            detail=f"reports available but cannot allocate ({type(exc).__name__})",
        )
    return FrameworkInfo(True)


def _probe_mlx() -> FrameworkInfo:
    if not mlx_available():
        detail = "install with: pip install 'niverel-mamba[mlx]'"
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            detail = "MLX requires macOS on Apple Silicon"
        return FrameworkInfo(False, detail=detail)
    try:
        import mlx

        version = getattr(mlx, "__version__", None)
        if version is None:
            from importlib.metadata import version as _version

            version = _version("mlx")
        return FrameworkInfo(True, version=version)
    except Exception as exc:  # pragma: no cover - defensive
        return FrameworkInfo(False, detail=f"MLX present but unusable: {exc}")


def _probe_distribution(name: str, module: str) -> FrameworkInfo:
    if not _module_available(module, distribution_name=name):
        return FrameworkInfo(False, detail=f"{name} is not installed")
    try:
        from importlib.metadata import version as _version

        return FrameworkInfo(True, version=_version(name))
    except Exception:
        return FrameworkInfo(True, version=None)


def detect_environment() -> Environment:
    """Inspect the current process. Cheap, side-effect free, and never cached
    across processes so that a freshly installed backend is picked up."""
    torch_info, cuda_info, mps_info = _probe_torch()
    return Environment(
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        platform_release=platform.release(),
        torch=torch_info,
        mlx=_probe_mlx(),
        cuda=cuda_info,
        mps=mps_info,
        upstream_mamba_ssm=_probe_distribution("mamba-ssm", "mamba_ssm"),
        causal_conv1d=_probe_distribution("causal-conv1d", "causal_conv1d"),
        extras={"executable": sys.executable},
    )
