"""Backend resolution: no silent fallback, and always self-describing."""

from __future__ import annotations

import pytest

from niverel_mamba import backend_status, list_backends, resolve
from niverel_mamba.capabilities import Certification, Environment, FrameworkInfo
from niverel_mamba.errors import BackendUnavailableError, UnknownBackendError


def _env(**overrides):
    absent = FrameworkInfo(False, detail="not installed")
    base = {
        "python_version": "3.12.0",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "platform_release": "26.0.0",
        "torch": FrameworkInfo(True, "2.13.0"),
        "mlx": FrameworkInfo(True, "0.32.0"),
        "cuda": absent,
        "mps": FrameworkInfo(True),
        "upstream_mamba_ssm": absent,
        "causal_conv1d": absent,
    }
    base.update(overrides)
    return Environment(**base)


def test_explicit_cuda_request_raises_rather_than_falling_back():
    """The single most important behaviour in the package."""
    with pytest.raises(BackendUnavailableError) as excinfo:
        resolve("cuda-reference", env=_env())
    message = str(excinfo.value)
    assert "not available" in message
    assert "does not silently fall back" in message


def test_auto_reports_what_it_picked():
    resolution = resolve("auto", env=_env())
    assert resolution.auto_selected is True
    assert resolution.requested == "auto"
    payload = resolution.to_dict()
    for field in ("backend", "framework", "device", "certification", "official_reference"):
        assert field in payload


def test_auto_prefers_mps_over_cpu_on_apple_silicon():
    assert resolve("auto", env=_env()).device == "mps"


def test_auto_falls_through_to_mlx_when_torch_is_absent():
    env = _env(torch=FrameworkInfo(False, detail="not installed"), mps=FrameworkInfo(False))
    resolution = resolve("auto", env=env)
    assert resolution.backend == "mlx"
    assert any("torch-reference" in note for note in resolution.notes)


def test_auto_raises_when_nothing_is_usable():
    env = _env(
        torch=FrameworkInfo(False, detail="absent"),
        mlx=FrameworkInfo(False, detail="absent"),
        mps=FrameworkInfo(False),
    )
    with pytest.raises(BackendUnavailableError, match="could not find a usable backend"):
        resolve("auto", env=env)


def test_unknown_backend_name():
    with pytest.raises(UnknownBackendError, match="unknown backend"):
        resolve("tensorflow")


@pytest.mark.parametrize("alias,canonical", [
    ("cuda", "cuda-reference"),
    ("reference", "torch-reference"),
    ("torch", "torch-reference"),
    ("metal", "mlx"),
])
def test_aliases(alias, canonical):
    assert backend_status(alias, _env()).spec.name == canonical


def test_unavailable_backend_explains_why():
    status = backend_status("cuda-reference", _env())
    assert status.available is False
    assert "mamba-ssm" in (status.reason or "")


def test_cuda_reference_is_certified_but_never_the_reference():
    """Wrapping upstream is not certifying, and being certified is not being *the* reference.

    `REFERENCE` names the backend that produces or certifies a result, which
    here is torch-reference. cuda-reference is compared against it and passes,
    which is what NUMERICALLY_CERTIFIED means and all it may ever claim.

    It reached that status on release run 32455074823, measured on an A100 and
    an H100 across all three runtimes. The earlier form of this test asserted
    EXPERIMENTAL and is what kept the claim honest until then; this form keeps
    the remaining one honest.
    """
    from niverel_mamba.registry import BACKENDS

    spec = BACKENDS["cuda-reference"]
    assert spec.official_reference is False
    assert spec.certification is Certification.NUMERICALLY_CERTIFIED
    assert spec.certification is not Certification.REFERENCE


def test_a_certified_backend_has_evidence_behind_its_class():
    """A published status must be backed by an observed tolerance, not a guess.

    Three certification runs were lost to bands sized for arithmetic the
    hardware does not perform. A backend may not advertise certification while
    the class it was scored against has never been measured.
    """
    from niverel_mamba.certification.tolerances import load_tolerances

    table = load_tolerances()
    assert table["cuda_float32"].observed, (
        "cuda-reference is published as certified, so its gate must carry the "
        "measurement it was certified under"
    )


def test_mlx_is_certified_for_what_it_actually_does():
    """Parity was published, so the status follows -- and only that far.

    `mlx_float32` is measured on the real Foundation V3 block and ci-mlx re-runs
    the nine parity tests on macOS with real MLX on every push. That earns
    NUMERICALLY_CERTIFIED for inference.

    It earns nothing for backward, which MLX does not implement here. That is an
    absence rather than an unproven claim, and the capability must keep saying
    so: a certified backend is not thereby a complete one.
    """
    from niverel_mamba.certification.tolerances import load_tolerances
    from niverel_mamba.registry import BACKENDS

    spec = BACKENDS["mlx"]
    assert spec.certification is Certification.NUMERICALLY_CERTIFIED
    assert spec.certification is not Certification.REFERENCE
    assert spec.capability.backward is False, "MLX has no backward path here"
    assert spec.capability.training is False
    assert load_tolerances()["mlx_float32"].observed, (
        "a published certification must carry the measurement behind it"
    )


def test_no_backend_claims_training():
    """training=True requires a gradient comparison against CUDA that has not
    happened yet. Every backend must say so."""
    from niverel_mamba.registry import BACKENDS

    for spec in BACKENDS.values():
        assert spec.capability.training is False
        assert spec.capability.inference is True


def test_all_backends_are_listed():
    assert set(list_backends()) == {"torch-reference", "cuda-reference", "mlx"}
