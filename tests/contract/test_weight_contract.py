"""The contract must describe the *real* upstream module, not a belief about it.

These tests import the genuine ``mamba_ssm.modules.mamba2`` out of the pinned
wheel (see scripts/_upstream_env.py) and compare it against the packaged
contract. They are the Phase 0 gate: if they fail, everything downstream is
built on a wrong premise.
"""

from __future__ import annotations

import pytest

from niverel_mamba import Mamba2Config, default_contract
from tests.conftest import requires_torch, requires_upstream_wheel

pytestmark = [requires_torch, requires_upstream_wheel]

SWEEP = [
    {"d_model": 768, "d_state": 128, "d_conv": 4, "expand": 2},
    {"d_model": 16, "d_state": 4, "d_conv": 4, "expand": 2, "headdim": 8},
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "bias": True},
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "conv_bias": False},
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "rmsnorm": False},
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "D_has_hdim": True},
    {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "ngroups": 4},
    {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "d_ssm": 128},
    {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "d_ssm": 64},
]


@pytest.fixture(scope="module")
def upstream():
    from _upstream_env import import_upstream_mamba2, upstream_mamba_ssm

    with upstream_mamba_ssm() as source:
        yield import_upstream_mamba2(source), source


@pytest.mark.parametrize("kwargs", SWEEP, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()))
def test_contract_matches_upstream_state_dict(upstream, kwargs):
    """Keys, shapes and ordering must all agree with the real module."""
    upstream_cls, _ = upstream
    config = Mamba2Config(**kwargs)
    module = upstream_cls(**config.upstream_kwargs())

    observed = {name: tuple(t.shape) for name, t in module.state_dict().items()}
    predicted = default_contract().expected(config)

    assert list(observed) == list(predicted), "key set or ordering differs from upstream"
    assert observed == predicted, "shapes differ from upstream"


@pytest.mark.parametrize("kwargs", SWEEP, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()))
def test_derived_dimensions_match_upstream(upstream, kwargs):
    """Our derived properties must equal upstream's own attributes."""
    upstream_cls, _ = upstream
    config = Mamba2Config(**kwargs)
    module = upstream_cls(**config.upstream_kwargs())

    assert config.d_inner == module.d_inner
    assert config.effective_d_ssm == module.d_ssm
    assert config.nheads == module.nheads
    assert config.ngroups == module.ngroups
    assert config.headdim == module.headdim
    assert config.chunk_size == module.chunk_size
    assert config.conv_dim == module.conv1d.weight.shape[0]
    assert config.d_in_proj == module.in_proj.weight.shape[0]

    # Upstream's own d_mlp expression, evaluated verbatim.
    d_mlp_upstream = (
        module.in_proj.weight.shape[0]
        - 2 * module.d_ssm
        - 2 * module.ngroups * module.d_state
        - module.nheads
    ) // 2
    assert config.d_mlp == d_mlp_upstream
    assert sum(config.in_proj_split) == module.in_proj.weight.shape[0]


def test_gated_norm_epsilon_is_1e5_not_1e6(upstream):
    """Mamba2.__init__ hard-codes eps=1e-5 when building RMSNormGated,
    overriding rms_norm_ref's own 1e-6 default. Using 1e-6 is a silent bug."""
    upstream_cls, _ = upstream
    module = upstream_cls(**Mamba2Config(d_model=64, d_state=16, headdim=16).upstream_kwargs())
    assert module.norm.eps == 1e-5
    assert module.norm.norm_before_gate is False
    assert Mamba2Config(d_model=64).norm_epsilon == module.norm.eps


def test_upstream_version_is_the_pinned_one(upstream):
    _, source = upstream
    assert source.version == "2.3.2.post1"
    assert default_contract().upstream_version == source.version


def test_upstream_has_no_init_states(upstream):
    """The brief listed init_states as 'if enabled'. It does not exist."""
    upstream_cls, _ = upstream
    module = upstream_cls(**Mamba2Config(d_model=64, d_state=16, headdim=16).upstream_kwargs())
    assert not any("init_states" in k for k in module.state_dict())


def test_our_module_state_dict_equals_upstreams(upstream):
    """The Niverel integration requirement, on a fresh module."""
    from niverel_mamba.torch_ops.mamba2 import Mamba2 as PortableMamba2

    upstream_cls, _ = upstream
    config = Mamba2Config(d_model=768, d_state=128, d_conv=4, expand=2)
    up = upstream_cls(**config.upstream_kwargs())
    portable = PortableMamba2(config)

    portable.load_state_dict(up.state_dict(), strict=True)
    assert portable.state_dict().keys() == up.state_dict().keys()
    for key, tensor in up.state_dict().items():
        assert portable.state_dict()[key].shape == tensor.shape


def test_stub_kernels_refuse_to_run(upstream):
    """The import shims must explode if used, never silently no-op.

    A stub that quietly returned zeros would let a contract be "extracted"
    from numbers upstream never produced. Every Triton kernel in the imported
    tree is a stub, and each must raise on launch.
    """
    import mamba_ssm.ops.triton.ssd_chunk_state as chunk_state

    from _upstream_env import StubInvocationError, _Stub

    kernels = [
        (name, obj)
        for name, obj in vars(chunk_state).items()
        if name.endswith("_kernel") and isinstance(obj, _Stub)
    ]
    assert kernels, "expected the Triton kernels to have been replaced by stubs"

    for _name, kernel in kernels:
        with pytest.raises(StubInvocationError, match="Refusing to return fabricated values"):
            kernel(1, 2, 3)
        # Kernels are launched as ``kernel[grid](...)``; that path must fail too.
        with pytest.raises(StubInvocationError):
            kernel[(1, 1, 1)](1, 2, 3)


def test_stub_modules_refuse_to_run(upstream):
    """The same guarantee for the non-Triton shims (causal_conv1d, the .so)."""
    import causal_conv1d

    from _upstream_env import StubInvocationError

    with pytest.raises(StubInvocationError):
        causal_conv1d.causal_conv1d_fn(None)
