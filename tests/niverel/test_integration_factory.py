"""The drop-in swap Niverel needs."""

from __future__ import annotations

import pytest

from tests.conftest import requires_torch, requires_upstream_wheel

pytestmark = requires_torch


def test_v3_config_helper_matches_the_training_config():
    from niverel_mamba.adapters.niverel import niverel_v3_config

    config = niverel_v3_config()
    assert (config.d_model, config.d_state, config.d_conv, config.expand) == (768, 128, 4, 2)
    assert (config.headdim, config.ngroups, config.chunk_size) == (64, 1, 256)


def test_upstream_constructor_kwargs_become_a_config():
    """Niverel calls Mamba2(d_model=..., d_state=..., d_conv=..., expand=...)."""
    from niverel_mamba.adapters.niverel import mamba_kwargs_to_config

    config = mamba_kwargs_to_config(d_model=768, d_state=128, d_conv=4, expand=2, layer_idx=3)
    assert config.d_model == 768
    assert config.nheads == 24


def test_factory_builds_the_portable_backend():
    from niverel_mamba.adapters.niverel import build_mamba2, niverel_v3_config
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    assert isinstance(build_mamba2(niverel_v3_config(), "niverel-torch"), Mamba2)


def test_factory_refuses_cuda_when_absent():
    import torch

    from niverel_mamba.adapters.niverel import build_mamba2, niverel_v3_config
    from niverel_mamba.errors import BackendUnavailableError

    if torch.cuda.is_available():
        pytest.skip("a CUDA device is present, so this refusal does not apply")

    with pytest.raises(BackendUnavailableError):
        build_mamba2(niverel_v3_config(), "upstream-cuda")


@requires_upstream_wheel
def test_state_dict_is_identical_between_backends():
    """The brief's exact acceptance snippet.

    ``portable.load_state_dict(upstream.state_dict(), strict=True)`` and
    identical key sets -- so swapping backends can never require rewriting a
    checkpoint.
    """
    from _upstream_env import import_upstream_mamba2, upstream_mamba_ssm
    from niverel_mamba.adapters.niverel import niverel_v3_config
    from niverel_mamba.torch_ops.mamba2 import Mamba2 as PortableMamba2

    config = niverel_v3_config()
    with upstream_mamba_ssm() as source:
        upstream_cls = import_upstream_mamba2(source)
        upstream = upstream_cls(**config.upstream_kwargs())
        portable = PortableMamba2(config)

        portable.load_state_dict(upstream.state_dict(), strict=True)
        assert portable.state_dict().keys() == upstream.state_dict().keys()
        for key, tensor in upstream.state_dict().items():
            assert portable.state_dict()[key].shape == tensor.shape
            assert portable.state_dict()[key].dtype == tensor.dtype
