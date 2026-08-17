"""Mamba2Config: derived dimensions, validation and round-tripping."""

from __future__ import annotations

import dataclasses
import math

import pytest

from niverel_mamba import Mamba2Config
from niverel_mamba.errors import UnsupportedConfigError


def test_niverel_v3_derived_dimensions():
    """The numbers the whole project is anchored to.

    These come from the real upstream module, not from arithmetic done here:
    scripts/extract_weight_contract.py verifies them against an instantiated
    mamba_ssm.Mamba2 on every run.
    """
    config = Mamba2Config(d_model=768, d_state=128, d_conv=4, expand=2)
    assert config.d_inner == 1536
    assert config.effective_d_ssm == 1536
    assert config.nheads == 24
    assert config.conv_dim == 1792
    assert config.d_in_proj == 3352
    assert config.d_mlp == 0
    assert config.d_D == 24
    assert config.norm_group_size == 1536
    assert config.in_proj_split == (0, 0, 1536, 1792, 24)
    assert sum(config.in_proj_split) == config.d_in_proj


def test_upstream_defaults_are_pinned():
    """headdim/ngroups/chunk_size are never passed by Niverel, so the defaults
    are load-bearing. Pin them so a silent upstream change is caught."""
    config = Mamba2Config(d_model=768)
    assert (config.headdim, config.ngroups, config.chunk_size) == (64, 1, 256)
    assert config.norm_epsilon == 1e-5  # hard-coded in Mamba2.__init__, not 1e-6
    assert config.dt_limit == (0.0, math.inf)
    assert config.bias is False
    assert config.conv_bias is True
    assert config.rmsnorm is True
    assert config.norm_before_gate is False
    assert config.D_has_hdim is False


def test_gated_mlp_branch_dimensions():
    """``d_mlp`` is ``d_inner - d_ssm``, not half of it.

    Upstream computes it as
    ``(d_in_proj - 2*d_ssm - 2*ngroups*d_state - nheads) // 2``; because
    ``d_in_proj`` is built from ``d_inner`` rather than ``d_ssm``, the halving
    cancels. Verified directly against the real module in
    tests/contract/test_weight_contract.py.
    """
    config = Mamba2Config(d_model=128, d_state=32, expand=2, headdim=16, d_ssm=128)
    assert config.d_inner == 256
    assert config.effective_d_ssm == 128
    assert config.d_mlp == 128
    assert sum(config.in_proj_split) == config.d_in_proj


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 768, "d_state": 128, "expand": 2},
        {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "d_ssm": 128},
        {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "d_ssm": 64},
        {"d_model": 96, "d_state": 8, "expand": 4, "headdim": 12, "ngroups": 2, "d_ssm": 192},
    ],
)
def test_in_proj_splits_always_sum_to_the_projection_width(kwargs):
    """The five splits must tile ``in_proj``'s output exactly.

    If they do not, ``torch.split`` in the forward pass silently produces the
    wrong widths, which is how a gated-MLP configuration would break.
    """
    config = Mamba2Config(**kwargs)
    assert sum(config.in_proj_split) == config.d_in_proj
    # The post-MLP concatenation must end up exactly d_inner wide for out_proj.
    assert config.d_mlp + config.effective_d_ssm == config.d_inner


def test_ngroups_widens_conv_and_projection():
    narrow = Mamba2Config(d_model=128, d_state=32, expand=2, headdim=16, ngroups=1)
    wide = Mamba2Config(d_model=128, d_state=32, expand=2, headdim=16, ngroups=4)
    assert wide.conv_dim > narrow.conv_dim
    assert wide.d_in_proj > narrow.d_in_proj
    assert wide.conv_dim == wide.effective_d_ssm + 2 * 4 * 32


def test_D_has_hdim_widens_D():
    config = Mamba2Config(d_model=64, d_state=16, expand=2, headdim=16, D_has_hdim=True)
    assert config.d_D == config.effective_d_ssm
    assert config.d_D != config.nheads


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"d_model": 0}, "positive"),
        ({"d_model": 64, "headdim": 7}, "divisible by headdim"),
        ({"d_model": 64, "headdim": 16, "ngroups": 100}, "cannot exceed nheads"),
        ({"d_model": 64, "headdim": 16, "ngroups": 3}, "divisible by ngroups"),
        ({"d_model": 64, "d_ssm": 999}, "must lie in"),
        ({"d_model": 64, "dt_limit": (1.0, 0.5)}, "lo < hi"),
        ({"d_model": 64, "dt_limit": (-1.0, 5.0)}, "lower bound"),
        ({"d_model": 64, "norm_epsilon": 0.0}, "norm_epsilon"),
        ({"d_model": 64, "rmsnorm": False, "norm_before_gate": True}, "requires rmsnorm"),
    ],
)
def test_invalid_configurations_are_refused(kwargs, fragment):
    with pytest.raises(UnsupportedConfigError, match=fragment):
        Mamba2Config(**kwargs)


def test_to_dict_from_dict_round_trip():
    config = Mamba2Config(d_model=768, d_state=128, dt_limit=(0.0, 0.5), conv_init=0.5)
    assert Mamba2Config.from_dict(config.to_dict()) == config


def test_infinite_dt_limit_survives_json():
    """float('inf') is not JSON-representable; the encoding must round-trip."""
    import json

    config = Mamba2Config(d_model=64)
    payload = json.loads(json.dumps(config.to_dict()))
    assert payload["dt_limit"] == ["inf", "inf"] or payload["dt_limit"][1] == "inf"
    assert Mamba2Config.from_dict(payload) == config


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(UnsupportedConfigError, match="unknown configuration keys"):
        Mamba2Config.from_dict({"d_model": 64, "nonsense": 1})


def test_upstream_kwargs_omit_runtime_only_options():
    kwargs = Mamba2Config(d_model=64).upstream_kwargs()
    for runtime_only in ("layer_idx", "process_group", "use_mem_eff_path", "device", "dtype"):
        assert runtime_only not in kwargs


def test_config_is_hashable_and_frozen():
    config = Mamba2Config(d_model=64)
    assert hash(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.d_model = 128  # type: ignore[misc]
