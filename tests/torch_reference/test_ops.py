"""Causal convolution and gated RMSNorm against their upstream definitions."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch

ATOL = 1e-12


def test_masked_conv_matches_native_conv_without_seq_idx():
    """Must reproduce upstream's ``conv1d(padding=W-1)[..., :-(W-1)]`` exactly."""
    import torch
    import torch.nn.functional as F

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    torch.manual_seed(0)
    channels, width, length = 12, 4, 37
    x = torch.randn(2, channels, length, dtype=torch.float64)
    weight = torch.randn(channels, 1, width, dtype=torch.float64)
    bias = torch.randn(channels, dtype=torch.float64)

    ours = causal_conv1d(x, weight, bias, activation=None)
    upstream = F.conv1d(x, weight, bias, padding=width - 1, groups=channels)[..., :length]
    assert torch.allclose(ours, upstream, atol=ATOL, rtol=0)


def test_convolution_is_causal():
    """Changing a future input must never alter a past output."""
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    torch.manual_seed(0)
    x = torch.randn(1, 6, 20, dtype=torch.float64)
    weight = torch.randn(6, 1, 4, dtype=torch.float64)

    base = causal_conv1d(x, weight, None, activation=None)
    perturbed_input = x.clone()
    perturbed_input[:, :, 12:] += 100.0
    perturbed = causal_conv1d(perturbed_input, weight, None, activation=None)
    assert torch.allclose(base[:, :, :12], perturbed[:, :, :12], atol=ATOL, rtol=0)


def test_newest_tap_is_the_last_index():
    """``weight[:, d_conv-1]`` multiplies position t itself.

    With a one-hot kernel at the last index the convolution is the identity;
    getting the orientation backwards would shift the signal by W-1.
    """
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    channels, width, length = 3, 4, 10
    x = torch.arange(channels * length, dtype=torch.float64).reshape(1, channels, length)
    weight = torch.zeros(channels, 1, width, dtype=torch.float64)
    weight[:, 0, width - 1] = 1.0
    assert torch.equal(causal_conv1d(x, weight, None, activation=None), x)


def test_conv_resets_at_document_boundaries():
    """Taps crossing a boundary must be zeroed."""
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    torch.manual_seed(0)
    channels, length = 5, 24
    x = torch.randn(1, channels, length, dtype=torch.float64)
    weight = torch.randn(channels, 1, 4, dtype=torch.float64)
    bias = torch.randn(channels, dtype=torch.float64)

    seq_idx = torch.zeros(1, length, dtype=torch.int32)
    seq_idx[0, 10:] = 1
    masked = causal_conv1d(x, weight, bias, seq_idx=seq_idx, activation=None)

    first = causal_conv1d(x[:, :, :10], weight, bias, activation=None)
    second = causal_conv1d(x[:, :, 10:], weight, bias, activation=None)
    assert torch.allclose(masked, torch.cat([first, second], dim=-1), atol=ATOL, rtol=0)


def test_length_one_document_uses_only_its_own_tap():
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    torch.manual_seed(0)
    channels, length = 3, 8
    x = torch.randn(1, channels, length, dtype=torch.float64)
    weight = torch.randn(channels, 1, 4, dtype=torch.float64)
    bias = torch.randn(channels, dtype=torch.float64)

    seq_idx = torch.tensor([[0, 0, 0, 1, 2, 2, 2, 2]], dtype=torch.int32)
    out = causal_conv1d(x, weight, bias, seq_idx=seq_idx, activation=None)
    expected = weight[:, 0, -1] * x[0, :, 3] + bias
    assert torch.allclose(out[0, :, 3], expected, atol=ATOL, rtol=0)


def test_native_impl_refuses_seq_idx():
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    x = torch.randn(1, 3, 8, dtype=torch.float64)
    weight = torch.randn(3, 1, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="cannot honour seq_idx"):
        causal_conv1d(
            x, weight, None, seq_idx=torch.zeros(1, 8, dtype=torch.int32), impl="native"
        )


# ---------------------------------------------------------------------------
# Gated RMSNorm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("group_size", [None, 8])
def test_gated_rmsnorm_matches_upstream_reference(norm_before_gate, group_size):
    """Ported line for line from upstream's ``rms_norm_ref``."""
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    from niverel_mamba.torch_ops.gated_rmsnorm import gated_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    z = torch.randn(2, 5, 16, dtype=torch.float64)
    weight = torch.randn(16, dtype=torch.float64)
    eps = 1e-5

    def upstream_ref(x, weight, z, eps, group_size, norm_before_gate):
        x = x.float()
        z = z.float()
        weight = weight.float()
        if z is not None and not norm_before_gate:
            x = x * F.silu(z)
        if group_size is None:
            rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
            out = x * rstd * weight
        else:
            grouped = rearrange(x, "... (g d) -> ... g d", d=group_size)
            rstd = 1 / torch.sqrt((grouped.square()).mean(dim=-1, keepdim=True) + eps)
            out = rearrange(grouped * rstd, "... g d -> ... (g d)") * weight
        if z is not None and norm_before_gate:
            out = out * F.silu(z)
        return out

    ours = gated_rmsnorm(
        x, weight, z, eps=eps, group_size=group_size, norm_before_gate=norm_before_gate
    )
    theirs = upstream_ref(x, weight, z, eps, group_size, norm_before_gate)
    assert torch.allclose(ours.float(), theirs, atol=1e-6, rtol=1e-5)


def test_gate_order_actually_differs():
    """norm(x*silu(z)) and norm(x)*silu(z) must not coincide.

    If they did, the norm_before_gate tests above would be vacuous.
    """
    import torch

    from niverel_mamba.torch_ops.gated_rmsnorm import gated_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(2, 16, dtype=torch.float64)
    z = torch.randn(2, 16, dtype=torch.float64)
    weight = torch.ones(16, dtype=torch.float64)
    after = gated_rmsnorm(x, weight, z, norm_before_gate=False)
    before = gated_rmsnorm(x, weight, z, norm_before_gate=True)
    assert not torch.allclose(after, before)


def test_rmsnorm_without_gate_is_plain_rms():
    import torch

    from niverel_mamba.torch_ops.gated_rmsnorm import gated_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float64)
    weight = torch.ones(16, dtype=torch.float64)
    out = gated_rmsnorm(x, weight, None, eps=0.0)
    expected = x / x.square().mean(-1, keepdim=True).sqrt()
    assert torch.allclose(out, expected, atol=1e-12, rtol=0)
