"""The stateful API: forward == concat(step), and state bookkeeping."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch

ATOL = 1e-10


def _decode(model, x, seq_idx=None, dtype=None):
    import torch

    state = model.allocate_inference_state(x.shape[0])
    if dtype is not None:
        state = state.to(dtype=dtype)
    outs = []
    for t in range(x.shape[1]):
        idx = seq_idx[:, t] if seq_idx is not None else None
        y_t, state = model.step(x[:, t], state, seq_idx_t=idx)
        outs.append(y_t)
    return torch.stack(outs, dim=1), state


def test_forward_equals_concat_step(make_model, grouped_config):
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(0)
    x = torch.randn(2, 70, grouped_config.d_model, dtype=torch.float64)
    with torch.no_grad():
        forward = model(x)
        stepped, _ = _decode(model, x, dtype=torch.float64)
    assert torch.allclose(forward, stepped, atol=ATOL, rtol=0)


def test_forward_equals_concat_step_with_seq_idx(make_model, grouped_config):
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(0)
    length = 70
    x = torch.randn(2, length, grouped_config.d_model, dtype=torch.float64)
    seq_idx = torch.zeros(2, length, dtype=torch.int32)
    seq_idx[:, 17:] = 1
    seq_idx[:, 18:] = 2
    seq_idx[:, 45:] = 3
    with torch.no_grad():
        forward = model(x, seq_idx=seq_idx)
        stepped, _ = _decode(model, x, seq_idx, dtype=torch.float64)
    assert torch.allclose(forward, stepped, atol=ATOL, rtol=0)


def test_state_carries_the_required_fields(make_model, tiny_config):
    """conv_state, ssm_state, and position / segment identity (brief 7)."""
    model = make_model(tiny_config)
    state = model.allocate_inference_state(3)
    assert state.conv_state.shape == (3, tiny_config.conv_dim, tiny_config.d_conv)
    assert state.ssm_state.shape == (
        3, tiny_config.nheads, tiny_config.headdim, tiny_config.d_state
    )
    assert state.seq_idx.shape == (3,)
    assert state.pos.shape == (3,)


def test_step_does_not_mutate_the_state_it_was_given(make_model, tiny_config):
    """The API is functional, so MLX and torch can share one contract."""
    import torch

    model = make_model(tiny_config)
    x = torch.randn(1, tiny_config.d_model, dtype=torch.float64)
    state = model.allocate_inference_state(1).to(dtype=torch.float64)
    before_conv = state.conv_state.clone()
    before_ssm = state.ssm_state.clone()

    with torch.no_grad():
        _, new_state = model.step(x, state)

    assert torch.equal(state.conv_state, before_conv)
    assert torch.equal(state.ssm_state, before_ssm)
    assert not torch.equal(new_state.conv_state, before_conv)


def test_position_advances_and_resets(make_model, tiny_config):
    import torch

    model = make_model(tiny_config)
    x = torch.randn(2, tiny_config.d_model, dtype=torch.float64)
    state = model.allocate_inference_state(2).to(dtype=torch.float64)
    with torch.no_grad():
        for _ in range(4):
            _, state = model.step(x, state, seq_idx_t=torch.zeros(2, dtype=torch.int64))
        assert state.pos.tolist() == [4, 4]
        _, state = model.step(x, state, seq_idx_t=torch.tensor([0, 1]))
    assert state.pos.tolist() == [5, 1], "only the row that changed document resets"


def test_document_switch_zeroes_only_the_switching_row(make_model, tiny_config):
    """Batched decode: one row restarting must not disturb the others."""
    import torch

    model = make_model(tiny_config)
    torch.manual_seed(0)
    x = torch.randn(2, 6, tiny_config.d_model, dtype=torch.float64)

    seq_idx = torch.zeros(2, 6, dtype=torch.int32)
    seq_idx[0, 3:] = 1  # row 0 starts a new document at t=3; row 1 never does

    with torch.no_grad():
        batched, _ = _decode(model, x, seq_idx, dtype=torch.float64)
        row0, _ = _decode(model, x[0:1], seq_idx[0:1], dtype=torch.float64)
        row1, _ = _decode(model, x[1:2], seq_idx[1:2], dtype=torch.float64)

    assert torch.allclose(batched[0:1], row0, atol=ATOL, rtol=0)
    assert torch.allclose(batched[1:2], row1, atol=ATOL, rtol=0)


def test_step_state_matches_forward_final_state(make_model, grouped_config):
    """Decoding k tokens must leave the same SSM state as forward over k."""
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(0)
    x = torch.randn(1, 33, grouped_config.d_model, dtype=torch.float64)
    with torch.no_grad():
        _, forward_state = model(x, return_final_state=True)
        _, decoded = _decode(model, x, dtype=torch.float64)
    assert torch.allclose(decoded.ssm_state, forward_state, atol=ATOL, rtol=0)


def test_step_accepts_both_input_ranks(make_model, tiny_config):
    import torch

    model = make_model(tiny_config)
    x = torch.randn(1, tiny_config.d_model, dtype=torch.float64)
    state = model.allocate_inference_state(1).to(dtype=torch.float64)
    with torch.no_grad():
        flat, _ = model.step(x, state)
        seq, _ = model.step(x.unsqueeze(1), state)
    assert flat.shape == (1, tiny_config.d_model)
    assert seq.shape == (1, 1, tiny_config.d_model)
    assert torch.allclose(flat, seq[:, 0], atol=1e-14, rtol=0)


def test_step_rejects_multiple_tokens(make_model, tiny_config):
    import torch

    model = make_model(tiny_config)
    state = model.allocate_inference_state(1).to(dtype=torch.float64)
    with pytest.raises(ValueError, match="one token at a time"):
        model.step(torch.randn(1, 3, tiny_config.d_model, dtype=torch.float64), state)


def test_step_generalises_beyond_ngroups_one(make_model):
    """Upstream's non-kernel step asserts ngroups == 1. We generalise, and the
    forward/step agreement is the proof it is done correctly."""
    import torch

    from niverel_mamba import Mamba2Config

    config = Mamba2Config(
        d_model=48, d_state=8, d_conv=4, expand=2, headdim=8, ngroups=3, chunk_size=8
    )
    model = make_model(config, ssd_impl="chunked")
    torch.manual_seed(0)
    x = torch.randn(2, 20, config.d_model, dtype=torch.float64)
    with torch.no_grad():
        forward = model(x)
        stepped, _ = _decode(model, x, dtype=torch.float64)
    assert torch.allclose(forward, stepped, atol=ATOL, rtol=0)
