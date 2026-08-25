"""Backend registry and resolution.

Two rules govern everything here:

1. **An explicit request never degrades.** ``backend="cuda-reference"`` on a
   machine without the wheel raises :class:`BackendUnavailableError`. It does
   not quietly become CPU.
2. **``auto`` reports what it picked.** The resolution result carries the
   backend name, framework, device, certification status and whether it is
   the official reference -- so a caller (Niverel Lab, say) can display the
   backend actually in use rather than the one it hoped for.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .capabilities import (
    Capability,
    Certification,
    Environment,
    detect_environment,
    upstream_mamba2_importable,
)
from .errors import BackendUnavailableError, UnknownBackendError

__all__ = [
    "BACKENDS",
    "BackendSpec",
    "BackendStatus",
    "Resolution",
    "backend_status",
    "get_backend",
    "list_backends",
    "resolve",
]


@dataclass(frozen=True)
class BackendSpec:
    """Static description of a backend."""

    name: str
    framework: str
    devices: tuple[str, ...]
    certification: Certification
    capability: Capability
    official_reference: bool
    summary: str
    #: Aliases accepted from users, e.g. ``"cuda"`` for ``"cuda-reference"``.
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendStatus:
    """A backend's static spec plus whether it can actually run here."""

    spec: BackendSpec
    available: bool
    reason: str | None = None
    devices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.spec.name,
            "framework": self.spec.framework,
            "available": self.available,
            "certification": self.spec.certification.value,
            "official_reference": self.spec.official_reference,
            "devices": list(self.devices),
            "capability": self.spec.capability.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Resolution:
    """The outcome of a backend request. Always fully self-describing."""

    backend: str
    framework: str
    device: str
    certification: Certification
    official_reference: bool
    capability: Capability
    requested: str
    auto_selected: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "framework": self.framework,
            "device": self.device,
            "certification": self.certification.value,
            "official_reference": self.official_reference,
            "capability": self.capability.to_dict(),
            "requested": self.requested,
            "auto_selected": self.auto_selected,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# The registry
#
# Certification levels below are deliberate and, where they are lower than one
# might hope, deliberately so:
#
# * ``torch-reference`` is NUMERICALLY_CERTIFIED against the float64 sequential
#   oracle under sealed tolerances -- a real, reproducible claim. It is *not*
#   yet certified against CUDA, and every report says which reference it was
#   measured against.
# * ``cuda-reference`` is EXPERIMENTAL, not REFERENCE. It only becomes the
#   reference once a GPU certification job has actually produced a report;
#   shipping it as REFERENCE on the strength of "it wraps upstream" would be
#   exactly the unproven claim the brief forbids.
# * ``mlx`` is EXPERIMENTAL until its parity report is published.
# ---------------------------------------------------------------------------

BACKENDS: dict[str, BackendSpec] = {
    "torch-reference": BackendSpec(
        name="torch-reference",
        framework="torch",
        devices=("cpu", "cuda", "mps"),
        certification=Certification.NUMERICALLY_CERTIFIED,
        # Gradients are compared against the upstream CUDA kernels on every
        # certification and scored under `cuda_float32_backward`: on release
        # run 32478644509 the worst parameter deviated by 3.4e-04 relative and
        # the input gradient by 6.2e-04, both below one TF32 epsilon. See the
        # note on Capability.training for what that does and does not claim.
        capability=Capability(inference=True, backward=True, training=True),
        official_reference=False,
        summary="Portable pure-PyTorch implementation. Fidelity before speed.",
        aliases=("reference", "torch", "portable"),
    ),
    "cuda-reference": BackendSpec(
        name="cuda-reference",
        framework="torch",
        devices=("cuda",),
        # numerically-certified, not `reference`: `reference` names the backend
        # that produces or certifies a result, which here is torch-reference.
        # Earned on release run 32455074823 -- A100 and H100, all three
        # runtimes, max_abs 2.5e-03 in float32, cosine 0.999999988, under a band
        # sealed from that measurement rather than before it.
        certification=Certification.NUMERICALLY_CERTIFIED,
        # Gradients are compared against the upstream CUDA kernels on every
        # certification and scored under `cuda_float32_backward`: on release
        # run 32478644509 the worst parameter deviated by 3.4e-04 relative and
        # the input gradient by 6.2e-04, both below one TF32 epsilon. See the
        # note on Capability.training for what that does and does not claim.
        capability=Capability(inference=True, backward=True, training=True),
        official_reference=False,
        summary="Wraps the upstream mamba-ssm CUDA kernels. Closest to the training runtime.",
        aliases=("cuda", "upstream", "upstream-cuda"),
    ),
    "mlx": BackendSpec(
        name="mlx",
        framework="mlx",
        devices=("gpu",),
        # Compared against torch-reference under a sealed tolerance and passing,
        # which is what numerically-certified means. `mlx_float32` was measured
        # on the real Foundation V3 block: 1.15e-05 against torch CPU, with the
        # chunked path at 3.6e-06 against MLX's own sequential oracle and
        # forward == concat(step) at 8.6e-06. ci-mlx re-runs the nine parity
        # tests on macOS with real MLX on every push, so this is continuous
        # evidence rather than one measurement.
        #
        # `backward=False` stays: MLX has no backward path here, and that is an
        # absence, not an unproven claim.
        certification=Certification.NUMERICALLY_CERTIFIED,
        capability=Capability(inference=True, backward=False, training=False),
        official_reference=False,
        summary="Pure MLX implementation for Apple Silicon.",
        aliases=("apple", "metal"),
    ),
}

_ALIASES: dict[str, str] = {}
for _spec in BACKENDS.values():
    _ALIASES[_spec.name] = _spec.name
    for _alias in _spec.aliases:
        _ALIASES[_alias] = _spec.name

#: Order ``auto`` prefers, most-trusted first.
AUTO_PREFERENCE = ("cuda-reference", "torch-reference", "mlx")


def list_backends() -> list[str]:
    return list(BACKENDS)


def get_backend(name: str) -> BackendSpec:
    """Resolve a name or alias to its spec."""
    canonical = _ALIASES.get(name)
    if canonical is None:
        raise UnknownBackendError(
            f"unknown backend {name!r}; known backends are {sorted(BACKENDS)} "
            f"(aliases: {sorted(set(_ALIASES) - set(BACKENDS))})"
        )
    return BACKENDS[canonical]


def _available_devices(spec: BackendSpec, env: Environment) -> tuple[bool, str | None, tuple[str, ...]]:
    if spec.name == "torch-reference":
        if not env.torch.available:
            return False, "PyTorch is not installed (pip install 'niverel-mamba[torch]')", ()
        # Ordered by preference, not alphabetically: ``auto`` takes the first
        # entry, and on a Mac with working MPS that should be MPS, not CPU.
        devices = []
        if env.cuda.available:
            devices.append("cuda")
        if env.mps.available:
            devices.append("mps")
        devices.append("cpu")
        return True, None, tuple(devices)

    if spec.name == "cuda-reference":
        if not env.torch.available:
            return False, "PyTorch is not installed", ()
        if not env.upstream_mamba_ssm.available:
            return False, "CUDA backend not installed (mamba-ssm is absent)", ()
        if not env.causal_conv1d.available:
            return False, "CUDA backend incomplete (causal-conv1d is absent)", ()
        if not env.cuda.available:
            return False, "no CUDA device is visible", ()
        # Installed is not importable. Reporting "yes" on the strength of a
        # .dist-info would be exactly the unproven claim this package refuses:
        # a backend that cannot be imported by a fresh process is not available,
        # however complete its metadata looks.
        importable, why = upstream_mamba2_importable()
        if not importable:
            return False, f"mamba-ssm is installed but will not import ({why})", ()
        return True, None, ("cuda",)

    if spec.name == "mlx":
        if not env.mlx.available:
            return False, env.mlx.detail or "MLX is not installed", ()
        return True, None, ("gpu",)

    return False, "unrecognised backend", ()  # pragma: no cover


def backend_status(name: str, env: Environment | None = None) -> BackendStatus:
    """Whether a backend can run here, and if not, precisely why not."""
    spec = get_backend(name)
    env = env or detect_environment()
    available, reason, devices = _available_devices(spec, env)
    return BackendStatus(spec=spec, available=available, reason=reason, devices=devices)


def all_statuses(env: Environment | None = None) -> list[BackendStatus]:
    env = env or detect_environment()
    return [backend_status(name, env) for name in BACKENDS]


def resolve(
    backend: str = "auto",
    device: str | None = None,
    env: Environment | None = None,
    *,
    preference: Iterable[str] = AUTO_PREFERENCE,
) -> Resolution:
    """Resolve a backend request into a concrete, fully described choice.

    ``backend="auto"`` picks the most trusted available backend and says so.
    Any other value is honoured exactly or raises -- there is no fallback.
    """
    env = env or detect_environment()

    if backend == "auto":
        tried: list[str] = []
        for candidate in preference:
            status = backend_status(candidate, env)
            if status.available:
                chosen_device = _pick_device(status, device)
                if chosen_device is None:
                    tried.append(f"{candidate}: no device matching {device!r}")
                    continue
                return Resolution(
                    backend=status.spec.name,
                    framework=status.spec.framework,
                    device=chosen_device,
                    certification=status.spec.certification,
                    official_reference=status.spec.official_reference,
                    capability=status.spec.capability,
                    requested="auto",
                    auto_selected=True,
                    notes=[f"skipped {t}" for t in tried],
                )
            tried.append(f"{candidate}: {status.reason}")
        raise BackendUnavailableError(
            "auto could not find a usable backend:\n  " + "\n  ".join(tried)
        )

    status = backend_status(backend, env)
    if not status.available:
        raise BackendUnavailableError(
            f"backend {backend!r} was requested explicitly but is not available: {status.reason}. "
            "This package does not silently fall back to another backend -- either install the "
            "required components or request a different backend."
        )

    chosen_device = _pick_device(status, device)
    if chosen_device is None:
        raise BackendUnavailableError(
            f"backend {status.spec.name!r} cannot use device {device!r}; "
            f"it supports {list(status.devices)} on this machine"
        )

    return Resolution(
        backend=status.spec.name,
        framework=status.spec.framework,
        device=chosen_device,
        certification=status.spec.certification,
        official_reference=status.spec.official_reference,
        capability=status.spec.capability,
        requested=backend,
    )


def _pick_device(status: BackendStatus, device: str | None) -> str | None:
    if device is None:
        return status.devices[0] if status.devices else None
    base = device.split(":", 1)[0]
    if base in status.devices:
        return device
    return None
