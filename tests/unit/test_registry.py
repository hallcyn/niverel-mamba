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


def test_cuda_reference_is_not_yet_the_official_reference():
    """It wraps upstream, but wrapping is not certifying.

    It may only claim REFERENCE once a GPU certification job has produced a
    real report. Until then this assertion keeps the claim honest.
    """
    from niverel_mamba.registry import BACKENDS

    spec = BACKENDS["cuda-reference"]
    assert spec.official_reference is False
    assert spec.certification is Certification.EXPERIMENTAL


def test_mlx_is_experimental_until_parity_is_published():
    from niverel_mamba.registry import BACKENDS

    assert BACKENDS["mlx"].certification is Certification.EXPERIMENTAL


def test_no_backend_claims_training():
    """training=True requires a gradient comparison against CUDA that has not
    happened yet. Every backend must say so."""
    from niverel_mamba.registry import BACKENDS

    for spec in BACKENDS.values():
        assert spec.capability.training is False
        assert spec.capability.inference is True


def test_all_backends_are_listed():
    assert set(list_backends()) == {"torch-reference", "cuda-reference", "mlx"}
