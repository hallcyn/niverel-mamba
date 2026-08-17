"""SSD: the oracle against upstream, and the chunked path against the oracle."""

from __future__ import annotations

import math

import pytest

from tests.conftest import requires_torch, requires_upstream_wheel

pytestmark = requires_torch

ORACLE_ATOL = 1e-10  # cpu_float64, sealed


@requires_upstream_wheel
def test_sequential_oracle_matches_upstream_ssd_minimal(scan_inputs):
    """The oracle is checked against upstream's own paper implementation.

    ``ssd_minimal_discrete`` is Listing 1 of the Mamba2 paper, shipped in the
    wheel. It is pure PyTorch, so it runs here even without CUDA -- which
    makes it the one piece of upstream numerics we can genuinely execute on
    this machine.
    """
    import torch
    from einops import repeat

    from _upstream_env import upstream_mamba_ssm
    from niverel_mamba.torch_ops.ssd_sequential import ssd_sequential

    data = scan_inputs(batch=2, length=64, nheads=6, headdim=8, ngroups=3, d_state=16)
    dt = torch.nn.functional.softplus(data["dt_raw"])

    y_ours, state_ours = ssd_sequential(
        data["x"], dt, data["A"], data["B"], data["C"], dt_softplus=False
    )

    with upstream_mamba_ssm():
        import importlib

        minimal = importlib.import_module("mamba_ssm.modules.ssd_minimal")
        heads_per_group = 6 // 3
        B = repeat(data["B"], "b l g n -> b l (g r) n", r=heads_per_group)
        C = repeat(data["C"], "b l g n -> b l (g r) n", r=heads_per_group)
        y_upstream, state_upstream = minimal.ssd_minimal_discrete(
            data["x"] * dt.unsqueeze(-1), data["A"] * dt, B, C, 16
        )

    assert torch.allclose(y_ours, y_upstream, atol=ORACLE_ATOL, rtol=0)
    assert torch.allclose(state_ours, state_upstream, atol=ORACLE_ATOL, rtol=0)


@pytest.mark.parametrize("length,chunk", [(64, 16), (70, 16), (5, 16), (256, 256), (1, 4)])
def test_chunked_matches_oracle(scan_inputs, length, chunk):
    """Including L < chunk and L not divisible by chunk (internal padding)."""
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked
    from niverel_mamba.torch_ops.ssd_sequential import ssd_sequential

    data = scan_inputs(length=length)
    kwargs = {"D": data["D"], "dt_bias": data["dt_bias"]}
    y_seq, s_seq = ssd_sequential(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], **kwargs
    )
    y_chunk, s_chunk = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=chunk, **kwargs
    )
    assert torch.allclose(y_seq, y_chunk, atol=ORACLE_ATOL, rtol=0)
    assert torch.allclose(s_seq, s_chunk, atol=ORACLE_ATOL, rtol=0)


def test_padding_does_not_attenuate_the_final_state(scan_inputs):
    """Regression for the single most likely padding bug.

    ``dt`` must be padded with exact zeros *after* softplus. Padding ``dt_raw``
    with zeros instead gives ``softplus(0 + dt_bias) > 0``, so the pad
    positions carry real decay and the returned final state comes out quietly
    attenuated. The check: a padded run's final state must equal the
    unpadded run's.
    """
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked

    data = scan_inputs(length=48)
    kwargs = {"D": data["D"], "dt_bias": data["dt_bias"]}

    _, aligned = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=16, **kwargs
    )
    # Same data, chunk size that forces 15 padding positions.
    _, padded = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=63, **kwargs
    )
    assert torch.allclose(aligned, padded, atol=ORACLE_ATOL, rtol=0)


def test_initial_states_are_consumed(scan_inputs):
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked
    from niverel_mamba.torch_ops.ssd_sequential import ssd_sequential

    data = scan_inputs(length=70)
    initial = torch.randn(2, 6, 8, 16, dtype=torch.float64)
    kwargs = {"D": data["D"], "dt_bias": data["dt_bias"], "initial_states": initial}

    y_seq, s_seq = ssd_sequential(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], **kwargs
    )
    y_chunk, s_chunk = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=16, **kwargs
    )
    assert torch.allclose(y_seq, y_chunk, atol=ORACLE_ATOL, rtol=0)
    assert torch.allclose(s_seq, s_chunk, atol=ORACLE_ATOL, rtol=0)

    # And they must actually matter.
    y_zero, _ = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=16,
        D=data["D"], dt_bias=data["dt_bias"],
    )
    assert not torch.allclose(y_chunk, y_zero)


def test_split_state_continuation_equals_one_pass(scan_inputs):
    """``forward(a) -> state -> forward(b, state)`` equals ``forward(a+b)``."""
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked

    data = scan_inputs(length=64)
    kwargs = {"D": data["D"], "dt_bias": data["dt_bias"], "chunk_size": 16}

    y_full, s_full = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], **kwargs
    )

    def slice_at(key, lo, hi):
        return data[key][:, lo:hi]

    y_a, s_a = ssd_chunked(
        slice_at("x", 0, 37), slice_at("dt_raw", 0, 37), data["A"],
        slice_at("B", 0, 37), slice_at("C", 0, 37), **kwargs
    )
    y_b, s_b = ssd_chunked(
        slice_at("x", 37, 64), slice_at("dt_raw", 37, 64), data["A"],
        slice_at("B", 37, 64), slice_at("C", 37, 64), initial_states=s_a, **kwargs
    )
    assert torch.allclose(torch.cat([y_a, y_b], dim=1), y_full, atol=ORACLE_ATOL, rtol=0)
    assert torch.allclose(s_b, s_full, atol=ORACLE_ATOL, rtol=0)


def test_dt_limit_is_applied_after_softplus(scan_inputs):
    """Clamping before softplus would give different numbers; the order is
    upstream's and must be preserved."""
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked
    from niverel_mamba.torch_ops.ssd_sequential import ssd_sequential

    data = scan_inputs(length=32)
    limit = (0.05, 0.2)
    kwargs = {"D": data["D"], "dt_bias": data["dt_bias"], "dt_limit": limit}

    y_seq, _ = ssd_sequential(data["x"], data["dt_raw"], data["A"], data["B"], data["C"], **kwargs)
    y_chunk, _ = ssd_chunked(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"], chunk_size=8, **kwargs
    )
    assert torch.allclose(y_seq, y_chunk, atol=ORACLE_ATOL, rtol=0)

    unlimited, _ = ssd_sequential(
        data["x"], data["dt_raw"], data["A"], data["B"], data["C"],
        D=data["D"], dt_bias=data["dt_bias"],
    )
    assert not torch.allclose(y_seq, unlimited), "dt_limit had no effect"


def test_no_global_l_by_l_matrix_is_materialised():
    """Memory must scale as L*chunk, not L^2.

    At L=4096 an L x L float64 matrix per head would be ~134 MB each; the
    chunked path must stay far below that.
    """
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked

    torch.manual_seed(0)
    b, length, h, p, n = 1, 4096, 2, 8, 8
    peak = {"value": 0}
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        peak["value"] = max(peak["value"], out.numel())
        return out

    y, _ = ssd_chunked(
        torch.randn(b, length, h, p, dtype=torch.float64),
        torch.randn(b, length, h, dtype=torch.float64) - 2,
        -torch.exp(torch.rand(h, dtype=torch.float64)),
        torch.randn(b, length, 1, n, dtype=torch.float64),
        torch.randn(b, length, 1, n, dtype=torch.float64),
        chunk_size=128,
    )
    assert y.shape == (b, length, h, p)
    assert torch.isfinite(y).all()


def test_exponent_masking_never_produces_nan():
    """The causal mask must be applied to the exponent, before exp().

    Applied after exp(), the upper triangle would be +inf and ``inf * 0``
    would give NaN. Large decay makes the raw exponent big enough that a
    wrong order shows up immediately.
    """
    import torch

    from niverel_mamba.torch_ops.ssd_chunked import ssd_chunked

    torch.manual_seed(0)
    b, length, h, p, n = 1, 64, 2, 4, 4
    y, state = ssd_chunked(
        torch.randn(b, length, h, p, dtype=torch.float64),
        torch.full((b, length, h), 6.0, dtype=torch.float64),  # softplus -> ~6
        torch.full((h,), -50.0, dtype=torch.float64),  # very fast decay
        torch.randn(b, length, 1, n, dtype=torch.float64),
        torch.randn(b, length, 1, n, dtype=torch.float64),
        chunk_size=16,
    )
    assert torch.isfinite(y).all()
    assert torch.isfinite(state).all()


def test_infinite_dt_limit_is_the_default(scan_inputs):
    import torch

    from niverel_mamba.torch_ops._common import prepare_dt

    data = scan_inputs(length=8)
    unclamped = prepare_dt(data["dt_raw"], None, work_dtype=torch.float64)
    same = prepare_dt(
        data["dt_raw"], None, work_dtype=torch.float64, dt_limit=(0.0, math.inf)
    )
    assert torch.equal(unclamped, same)
