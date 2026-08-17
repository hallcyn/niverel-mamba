"""dtype handling and the refusal to downcast silently."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_forward_preserves_input_dtype(make_model, tiny_config, dtype_name):
    import torch

    dtype = getattr(torch, dtype_name)
    model = make_model(tiny_config).to(dtype)
    x = torch.randn(1, 12, tiny_config.d_model, dtype=dtype)
    with torch.no_grad():
        assert model(x).dtype == dtype


def test_bfloat16_weights_load_and_run(make_model, tiny_config):
    """A bf16 checkpoint must load; the scan still runs in float32 internally."""
    import torch

    from niverel_mamba.torch_ops.mamba2 import Mamba2

    model = Mamba2(tiny_config).to(torch.bfloat16)
    x = torch.randn(1, 12, tiny_config.d_model, dtype=torch.bfloat16)
    with torch.no_grad():
        y = model(x)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


def test_A_is_always_computed_in_float32(tiny_config):
    """Upstream is emphatic: a stored A_log in float16 can exp() to -inf."""
    import torch

    from niverel_mamba.torch_ops.mamba2 import Mamba2

    model = Mamba2(tiny_config).to(torch.float16)
    assert model._A().dtype == torch.float32
    assert torch.isfinite(model._A()).all()


def test_float64_request_on_cpu_is_honoured(scan_inputs):
    import torch

    from niverel_mamba.torch_ops._common import resolve_work_dtype

    x = torch.zeros(1, dtype=torch.float32)
    assert resolve_work_dtype(x, torch.float64) == torch.float64


def test_float32_is_the_default_work_dtype():
    import torch

    from niverel_mamba.torch_ops._common import resolve_work_dtype

    assert resolve_work_dtype(torch.zeros(1, dtype=torch.float32), None) == torch.float32
    assert resolve_work_dtype(torch.zeros(1, dtype=torch.bfloat16), None) == torch.float32
    assert resolve_work_dtype(torch.zeros(1, dtype=torch.float64), None) == torch.float64
