"""The sealed tolerance table."""

from __future__ import annotations

import numpy as np
import pytest

from niverel_mamba.certification import compare, get_tolerance, load_tolerances
from niverel_mamba.errors import CertificationError
from tests.conftest import requires_torch


def test_all_brief_classes_are_sealed():
    table = load_tolerances()
    assert {"cpu_float64", "cpu_float32", "cuda_bfloat16", "mps_float32", "mlx_float32"} <= set(
        table.classes
    )


def test_starting_values_were_not_loosened():
    """The sealed starting values. They may be tightened by
    observation, never widened to make a test pass."""
    starting = {
        "cpu_float64": (1.0e-10, 1.0e-9),
        "cpu_float32": (2.0e-5, 2.0e-4),
        "cuda_bfloat16": (2.0e-2, 2.0e-2),
        "mps_float32": (1.0e-4, 1.0e-3),
        "mlx_float32": (1.0e-4, 1.0e-3),
    }
    for name, (atol, rtol) in starting.items():
        tol = get_tolerance(name)
        assert tol.atol <= atol, f"{name} atol was widened beyond the brief's value"
        assert tol.rtol <= rtol, f"{name} rtol was widened beyond the brief's value"


def test_the_cuda_gate_admits_what_was_actually_measured():
    """A band must never be tightened below the evidence it was sealed from.

    `cuda_float32` was set from release run 32455074823: max_abs 2.4972e-03 on
    an A100 and an H100, bit-identical across both. Tightening it under that
    would fail a backend that has not changed, which is the same mistake as
    setting it from a guess -- three certification runs were lost that way.
    """
    tolerance = load_tolerances()["cuda_float32"]
    observed = tolerance.observed
    assert observed, "the gate must carry its measurement"

    worst = max(
        float(value)
        for key, value in observed.items()
        if key.endswith("max_abs")
    )
    # The reference values are of order one, so atol carries the comparison.
    assert tolerance.atol > worst, (
        f"the band ({tolerance.atol}) is below the measurement ({worst}); "
        "it would fail an unchanged backend"
    )
    assert tolerance.atol < 10 * worst, (
        f"the band ({tolerance.atol}) is more than ten times the measurement "
        f"({worst}); that is no longer a criterion"
    )


def test_bfloat16_is_recorded_as_a_measurement_and_not_as_a_gate():
    """It measures the number format, not the kernels, and must never gate again."""
    table = load_tolerances()
    observed = table["cuda_bfloat16"].observed
    assert observed, "what bfloat16 costs must stay recorded"
    assert "NOT A GATE" in table["cuda_bfloat16"].description


def test_locally_measured_classes_are_verified():
    table = load_tolerances()
    for name in ("cpu_float64", "cpu_float32", "mps_float32", "mlx_float32"):
        assert name not in table.unverified


def test_unknown_class_is_an_error():
    with pytest.raises(CertificationError, match="unknown tolerance class"):
        get_tolerance("wishful_float2")


def test_comparison_is_elementwise_not_average():
    """One bad element must fail, even if the mean looks excellent."""
    reference = np.ones(10_000)
    candidate = reference.copy()
    candidate[0] += 1.0
    result = compare(candidate, reference, name="t", tolerance="cpu_float64")
    assert result.passed is False
    assert result.violations == 1
    assert result.mean_abs_error < 1e-3  # the average hides it; the criterion does not


def test_nan_never_passes():
    reference = np.ones(8)
    candidate = reference.copy()
    candidate[3] = np.nan
    result = compare(candidate, reference, name="t", tolerance="cpu_float64")
    assert result.passed is False
    assert "NaN" in (result.detail or "")


def test_shape_mismatch_is_a_failure_not_an_exception():
    result = compare(np.ones(4), np.ones(5), name="t", tolerance="cpu_float64")
    assert result.passed is False
    assert "shape mismatch" in (result.detail or "")


@requires_torch
def test_float64_precision_is_not_lost_in_conversion():
    """to_numpy must not route float64 through float32.

    Regression: an earlier version did, and every float64 report came back as
    exactly 0.0 error -- agreement it had not demonstrated.
    """
    import torch

    from niverel_mamba.certification.compare import to_numpy

    tensor = torch.tensor([1.0 + 1e-12], dtype=torch.float64)
    assert to_numpy(tensor)[0] != 1.0
