"""Shared fixtures and skip logic for the test suite.

Skips are always *explicit and named*: a suite that cannot run on this machine
says which component is missing. Nothing is masked with ``continue-on-error``
or a bare try/except.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _have(module: str) -> bool:
    """Real installed framework, not a same-named directory on sys.path.

    Delegates to the package's own probe so the test suite and the product
    agree on what "installed" means. See
    `niverel_mamba.capabilities._module_available`: `tests/mlx` used to shadow
    the real MLX package here, and MLX itself is a namespace package, so
    installed metadata is the only reliable discriminator.
    """
    from niverel_mamba.capabilities import _module_available

    return _module_available(module)


HAVE_TORCH = _have("torch")
HAVE_MLX = _have("mlx")

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="PyTorch is not installed")
requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="MLX is not installed")


def _mps_status() -> tuple[bool, str]:
    """Usable MPS, not merely advertised MPS.

    Delegates to the package probe: hosted macOS runners report MPS as
    available and then fail every allocation, which would run the MPS suite
    against a device that cannot hold a tensor.
    """
    if not HAVE_TORCH:
        return False, "PyTorch is not installed"
    from niverel_mamba.capabilities import detect_environment

    info = detect_environment().mps
    return info.available, info.detail or "no MPS device available"


def _cuda_available() -> bool:
    if not HAVE_TORCH:
        return False
    import torch

    return bool(torch.cuda.is_available())


_MPS_USABLE, _MPS_REASON = _mps_status()
requires_mps = pytest.mark.skipif(not _MPS_USABLE, reason=f"MPS unusable: {_MPS_REASON}")
requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="no CUDA device available")


def requires_measured_tolerance(name: str) -> Any:
    """Skip a test whose tolerance class has never been observed.

    A tolerance that has not been measured is a guess, and asserting a guess
    turns a correct implementation into a red build. That is not hypothetical:
    the CUDA parity tests failed twice on rented GPUs against bands sized for
    arithmetic the hardware does not perform -- first bfloat16 against an
    unrounded float32 reference, then float32 against kernels that multiply in
    TF32, which carries ten mantissa bits rather than twenty-four.

    So these tests wait for evidence. They activate the moment
    `tolerances.yaml` carries an `observed` block for their class, which is
    exactly what a measurement run produces.
    """
    try:
        from niverel_mamba.certification.tolerances import load_tolerances

        observed = load_tolerances()[name].observed
    except Exception as exc:  # pragma: no cover - defensive
        return pytest.mark.skipif(True, reason=f"cannot read tolerance {name!r}: {exc}")
    return pytest.mark.skipif(
        observed is None,
        reason=(
            f"tolerance class {name!r} has no observed data yet; "
            f"run `niverel-mamba verify --certify cuda-reference --measure` on a GPU "
            f"and seal the result before asserting a bound"
        ),
    )


def _upstream_wheel() -> Path | None:
    try:
        from _upstream_env import find_wheel

        return find_wheel()
    except Exception:
        return None


HAVE_UPSTREAM_WHEEL = _upstream_wheel() is not None
requires_upstream_wheel = pytest.mark.skipif(
    not HAVE_UPSTREAM_WHEEL,
    reason="the pinned mamba_ssm wheel was not found; see scripts/_upstream_env.py",
)


def _fixture_present(name: str, filename: str) -> bool:
    return (REPO_ROOT / "fixtures" / name / filename).is_file()


HAVE_NIVEREL_FIXTURE = _fixture_present("niverel", "block.safetensors")
requires_niverel_fixture = pytest.mark.skipif(
    not HAVE_NIVEREL_FIXTURE,
    reason="run: python scripts/make_golden_fixture.py --niverel  (needs HF_TOKEN)",
)

HAVE_SYNTH_FIXTURES = _fixture_present("tiny", "weights.safetensors")
requires_synth_fixtures = pytest.mark.skipif(
    not HAVE_SYNTH_FIXTURES,
    reason="run: python scripts/make_golden_fixture.py",
)


# ---------------------------------------------------------------------------
# Reusable configurations and data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_config() -> Any:
    from niverel_mamba import Mamba2Config

    return Mamba2Config(
        d_model=16, d_state=4, d_conv=4, expand=2, headdim=8, ngroups=1, chunk_size=4
    )


@pytest.fixture(scope="session")
def grouped_config() -> Any:
    """ngroups > 1, so group broadcasting is genuinely exercised."""
    from niverel_mamba import Mamba2Config

    return Mamba2Config(
        d_model=32, d_state=8, d_conv=4, expand=2, headdim=8, ngroups=2, chunk_size=16
    )


@pytest.fixture(scope="session")
def v3_config() -> Any:
    from niverel_mamba.adapters.niverel import niverel_v3_config

    return niverel_v3_config()


@pytest.fixture
def make_model():
    """Build a float64 model with deterministic weights."""

    def _make(config: Any, seed: int = 0, **kwargs: Any) -> Any:
        import torch

        from niverel_mamba.torch_ops.mamba2 import Mamba2

        torch.manual_seed(seed)
        model = Mamba2(config, **kwargs).double()
        model.eval()
        return model

    return _make


@pytest.fixture
def scan_inputs():
    """Random SSD inputs of a requested shape, in float64."""

    def _make(
        batch: int = 2,
        length: int = 64,
        nheads: int = 6,
        headdim: int = 8,
        ngroups: int = 3,
        d_state: int = 16,
        seed: int = 0,
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(seed)
        return {
            "x": torch.randn(batch, length, nheads, headdim, dtype=torch.float64),
            "dt_raw": torch.randn(batch, length, nheads, dtype=torch.float64) - 2,
            "A": -torch.exp(torch.rand(nheads, dtype=torch.float64) * 2),
            "B": torch.randn(batch, length, ngroups, d_state, dtype=torch.float64),
            "C": torch.randn(batch, length, ngroups, d_state, dtype=torch.float64),
            "D": torch.randn(nheads, dtype=torch.float64),
            "dt_bias": torch.randn(nheads, dtype=torch.float64) * 0.1,
        }

    return _make
