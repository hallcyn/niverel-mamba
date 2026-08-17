"""torch-reference on MPS, measured against the same code on CPU."""

from __future__ import annotations

import pytest

from tests.conftest import requires_mps, requires_torch

pytestmark = [requires_torch, requires_mps, pytest.mark.mps]


def _pair(config, seed=0):
    import torch

    from niverel_mamba.torch_ops.mamba2 import Mamba2

    torch.manual_seed(seed)
    cpu = Mamba2(config).float().eval()
    mps = Mamba2(config).float().to("mps").eval()
    mps.load_state_dict(cpu.state_dict(), strict=True)
    return cpu, mps


def _assert_within(candidate, reference, name):
    from niverel_mamba.certification import compare

    result = compare(candidate, reference, name=name, tolerance="mps_float32")
    assert result.passed, result.detail or f"{name}: max_abs={result.max_abs_error:.3e}"


def test_forward_matches_cpu(grouped_config):
    import torch

    cpu, mps = _pair(grouped_config)
    x = torch.randn(2, 70, grouped_config.d_model)
    with torch.no_grad():
        _assert_within(mps(x.to("mps")).cpu(), cpu(x), "mps_forward")


def test_seq_idx_reset_matches_cpu(grouped_config):
    import torch

    cpu, mps = _pair(grouped_config)
    length = 70
    x = torch.randn(2, length, grouped_config.d_model)
    seq_idx = torch.zeros(2, length, dtype=torch.int32)
    seq_idx[:, 17:] = 1
    seq_idx[:, 18:] = 2
    seq_idx[:, 45:] = 3
    with torch.no_grad():
        _assert_within(
            mps(x.to("mps"), seq_idx=seq_idx.to("mps")).cpu(),
            cpu(x, seq_idx=seq_idx),
            "mps_seq_idx",
        )


def test_strict_reset_holds_on_mps(grouped_config):
    """The invariant must hold *on the device*, not only via CPU comparison."""
    import torch

    _, mps = _pair(grouped_config)
    length = 60
    x = torch.randn(2, length, grouped_config.d_model, device="mps")
    seq_idx = torch.zeros(2, length, dtype=torch.int32, device="mps")
    seq_idx[:, 25:] = 1

    with torch.no_grad():
        segmented = mps(x, seq_idx=seq_idx)
        separate = torch.cat([mps(x[:, :25]), mps(x[:, 25:])], dim=1)
    _assert_within(segmented.cpu(), separate.cpu(), "mps_strict_reset")


def test_step_matches_forward_on_mps(grouped_config):
    import torch

    _, mps = _pair(grouped_config)
    x = torch.randn(1, 24, grouped_config.d_model, device="mps")
    with torch.no_grad():
        forward = mps(x)
        state = mps.allocate_inference_state(1)
        outs = []
        for t in range(x.shape[1]):
            y_t, state = mps.step(x[:, t], state)
            outs.append(y_t)
    _assert_within(torch.stack(outs, 1).cpu(), forward.cpu(), "mps_step")


def test_float64_is_refused_rather_than_downcast(grouped_config):
    """MPS has no float64. A certification run must fail loudly instead of
    quietly becoming a float32 one."""
    import torch

    from niverel_mamba.errors import UnsupportedDtypeError

    _, mps = _pair(grouped_config)
    mps.work_dtype = torch.float64
    x = torch.randn(1, 16, grouped_config.d_model, device="mps")
    with pytest.raises(UnsupportedDtypeError, match="float64 is not available on MPS"):
        mps(x)


def test_explicit_downcast_is_available_when_asked_for(grouped_config):
    import torch

    _, mps = _pair(grouped_config)
    mps.work_dtype = torch.float64
    mps.allow_downcast = True
    x = torch.randn(1, 16, grouped_config.d_model, device="mps")
    with torch.no_grad():
        assert torch.isfinite(mps(x)).all()


@pytest.mark.slow
def test_l8192_on_mps(grouped_config):
    import torch

    cpu, mps = _pair(grouped_config)
    length = 8192
    x = torch.randn(1, length, grouped_config.d_model)
    seq_idx = torch.zeros(1, length, dtype=torch.int32)
    for i, start in enumerate(range(700, length, 700)):
        seq_idx[0, start:] = i + 1
    with torch.no_grad():
        _assert_within(
            mps(x.to("mps"), seq_idx=seq_idx.to("mps")).cpu(),
            cpu(x, seq_idx=seq_idx),
            "mps_l8192",
        )
