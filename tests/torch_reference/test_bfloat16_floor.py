"""Why the CUDA campaign compares at equal data, measured without a GPU.

A certification run on an A100 failed the `cuda_bfloat16` band, and the obvious
reading -- the CUDA backend is wrong -- was wrong. The band was never reachable.

This reproduces the finding with the portable implementation alone: no CUDA, no
upstream kernels, merely rounding inputs and weights to bfloat16 already puts
elements outside the band. What that comparison measures is the cost of
bfloat16, which is a property of the number format and not something a backend
can be certified out of.

It runs on CPU in a second, and it is the evidence that justified changing the
comparison rather than the tolerance.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.fixture
def segmented():
    import torch

    from niverel_mamba.certification.golden import load_fixture
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    fixture = load_fixture("segmented")
    weights = fixture.torch_weights(dtype=torch.float32)
    inputs = fixture.torch_inputs(dtype=torch.float32)

    model = Mamba2(fixture.config).float()
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, weights, inputs


def _band(reference):
    """The sealed cuda_bfloat16 band, in the allclose form the report uses."""
    from niverel_mamba.certification.tolerances import load_tolerances

    tolerance = load_tolerances()["cuda_bfloat16"]
    return tolerance.atol + tolerance.rtol * reference.abs()


def test_bfloat16_rounding_alone_leaves_the_band(segmented):
    """No CUDA, no upstream kernels, and elements are already outside."""
    import torch

    model, weights, inputs = segmented
    x, seq_idx = inputs["x"], inputs.get("seq_idx")

    with torch.no_grad():
        exact = model(x, seq_idx=seq_idx)

    rounded = model.__class__(model.config).float()
    rounded.load_state_dict({k: v.bfloat16().float() for k, v in weights.items()}, strict=True)
    rounded.eval()
    with torch.no_grad():
        lossy = rounded(x.bfloat16().float(), seq_idx=seq_idx)

    deviation = (lossy - exact).abs()
    outside = int((deviation > _band(exact)).sum())

    assert outside > 0, (
        "if bfloat16 rounding alone now fits the band, the comparison this test "
        "justifies can be simplified -- re-measure before assuming it still cannot"
    )
    # The reference output has RMS ~1 and bfloat16 carries ~3.9e-3 of relative
    # precision, so a 2e-2 absolute band over outputs formed by cancellation is
    # simply not reachable.
    assert deviation.max() > 2e-2


def test_at_equal_data_the_portable_backend_is_exact(segmented):
    """The other half of the argument: equal data leaves nothing to explain.

    Feeding the *same* bfloat16-rounded values to the same implementation
    reproduces it bit for bit, so whatever a candidate deviates by at equal data
    is the candidate's own arithmetic -- which is what certification should
    measure.
    """
    import torch

    model, weights, inputs = segmented
    x, seq_idx = inputs["x"], inputs.get("seq_idx")

    rounded_weights = {k: v.bfloat16().float() for k, v in weights.items()}
    x_equal = x.bfloat16().float()

    outputs = []
    for _ in range(2):
        candidate = model.__class__(model.config).float()
        candidate.load_state_dict(rounded_weights, strict=True)
        candidate.eval()
        with torch.no_grad():
            outputs.append(candidate(x_equal, seq_idx=seq_idx))

    assert torch.equal(outputs[0], outputs[1])
