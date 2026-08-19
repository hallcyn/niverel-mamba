"""The real Foundation V3 block: strict load and cross-backend agreement.

This is the proof the whole project exists for -- the same bytes that trained
on CUDA, loading and running on a Mac.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from tests.conftest import (
    requires_mlx,
    requires_mps,
    requires_niverel_fixture,
    requires_torch,
)

pytestmark = [requires_torch, requires_niverel_fixture, pytest.mark.niverel]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def fixture():
    from niverel_mamba.certification.golden import load_fixture

    return load_fixture("niverel")


@pytest.fixture(scope="module")
def block(fixture):
    return fixture.torch_weights()


@pytest.fixture(scope="module")
def model(fixture, block):
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    instance = Mamba2(fixture.config)
    instance.load_state_dict(block, strict=True)
    instance.eval()
    return instance


def test_manifest_records_real_provenance(fixture):
    """The fixture must be traceable to a specific checkpoint revision."""
    manifest = fixture.manifest
    assert manifest["source_repository"] == "thibaud-perrin/niverel-5b-v3-hnet-jepa-seed1337"
    assert manifest["run_id"] == "foundation-5b-v3-wave2-hnet-jepa-seed1337"
    assert len(manifest["source_sha256"]) == 64
    assert manifest["source_revision"]
    assert fixture.is_real_checkpoint


def test_config_matches_the_training_runtime_config(fixture):
    """Values come from runtime_config.yaml of the actual run."""
    config = fixture.config
    assert (config.d_model, config.d_state, config.d_conv, config.expand) == (768, 128, 4, 2)
    # Never passed by Niverel, so upstream defaults apply and are load-bearing.
    assert (config.headdim, config.ngroups, config.chunk_size) == (64, 1, 256)
    assert config.d_inner == 1536
    assert config.nheads == 24


def test_strict_load_of_the_real_block(fixture, block):
    """load_state_dict(..., strict=True) -- no missing, no unexpected keys."""
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    instance = Mamba2(fixture.config)
    missing, unexpected = instance.load_state_dict(block, strict=True)
    assert list(missing) == []
    assert list(unexpected) == []
    assert set(instance.state_dict()) == set(block)


def test_block_satisfies_the_contract(fixture, block):
    from niverel_mamba import validate_state_dict

    checked = validate_state_dict(block, fixture.config)
    assert set(checked) == set(block)


def test_checkpoint_contains_the_expected_block_count(fixture):
    """4 encoder + 8 decoder + 4 JEPA EMA teacher = 16 Mamba2 blocks."""
    assert len(fixture.manifest["blocks_available"]) == 16


def test_upstream_round_trip_preserves_bytes(fixture, block):
    from niverel_mamba.adapters.upstream import round_trip
    from niverel_mamba.weights import state_dict_digest

    assert state_dict_digest(block) == state_dict_digest(round_trip(block, fixture.config))


def test_implementations_agree_on_real_weights(model, fixture):
    import torch

    from niverel_mamba.certification import compare

    inputs = fixture.torch_inputs()
    x, seq_idx = inputs["x"], inputs["seq_idx"]

    with torch.no_grad():
        model.ssd_impl = "chunked"
        chunked = model(x, seq_idx=seq_idx)
        model.ssd_impl = "sequential"
        sequential = model(x, seq_idx=seq_idx)
        model.ssd_impl = "per_segment"
        per_segment = model(x, seq_idx=seq_idx)
    model.ssd_impl = "chunked"

    for name, candidate in (("sequential", sequential), ("per_segment", per_segment)):
        result = compare(candidate, chunked, name=name, tolerance="cpu_float32")
        assert result.passed, result.detail


def test_strict_reset_on_real_weights(model, fixture):
    import torch

    from niverel_mamba.certification import compare

    inputs = fixture.torch_inputs()
    x, seq_idx = inputs["x"], inputs["seq_idx"]

    bounds = [0]
    for t in range(1, x.shape[1]):
        if int(seq_idx[0, t]) != int(seq_idx[0, t - 1]):
            bounds.append(t)
    bounds.append(x.shape[1])

    with torch.no_grad():
        segmented = model(x, seq_idx=seq_idx)
        separate = torch.cat([model(x[:, s:e]) for s, e in pairwise(bounds)], dim=1)

    result = compare(separate, segmented, name="segment_reset", tolerance="cpu_float32")
    assert result.passed, result.detail


def test_forward_equals_step_on_real_weights(model, fixture):
    import torch

    from niverel_mamba.certification import compare

    inputs = fixture.torch_inputs()
    x, seq_idx = inputs["x"], inputs["seq_idx"]

    with torch.no_grad():
        forward = model(x, seq_idx=seq_idx)
        state = model.allocate_inference_state(x.shape[0])
        outs = []
        for t in range(x.shape[1]):
            y_t, state = model.step(x[:, t], state, seq_idx_t=seq_idx[:, t])
            outs.append(y_t)

    result = compare(torch.stack(outs, 1), forward, name="step", tolerance="cpu_float32")
    assert result.passed, result.detail


@requires_mps
def test_real_block_on_mps(fixture, block, model):
    import torch

    from niverel_mamba.certification import compare
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    inputs = fixture.torch_inputs()
    x, seq_idx = inputs["x"], inputs["seq_idx"]

    on_mps = Mamba2(fixture.config).to("mps").eval()
    on_mps.load_state_dict({k: v.to("mps") for k, v in block.items()}, strict=True)

    with torch.no_grad():
        expected = model(x, seq_idx=seq_idx)
        actual = on_mps(x.to("mps"), seq_idx=seq_idx.to("mps")).cpu()

    result = compare(actual, expected, name="mps_forward", tolerance="mps_float32")
    assert result.passed, result.detail


@requires_mlx
def test_real_block_on_mlx(fixture, block, model):
    import mlx.core as mx
    import numpy as np
    import torch

    from niverel_mamba.certification import compare
    from niverel_mamba.mlx_ops.mamba2 import Mamba2 as MlxMamba2

    inputs = fixture.torch_inputs()
    x, seq_idx = inputs["x"], inputs["seq_idx"]

    on_mlx = MlxMamba2(fixture.config)
    on_mlx.load_canonical_weights(block)

    with torch.no_grad():
        expected = model(x, seq_idx=seq_idx)
    actual = np.array(on_mlx(mx.array(x.numpy()), mx.array(seq_idx.numpy())))

    result = compare(actual, expected, name="mlx_forward", tolerance="mlx_float32")
    assert result.passed, result.detail


@requires_mlx
def test_real_block_survives_the_mlx_round_trip(fixture, block):
    """The acceptance criterion: the V3 checkpoint converts to MLX and back."""
    from niverel_mamba.mlx_ops.mamba2 import Mamba2 as MlxMamba2
    from niverel_mamba.weights import state_dict_digest

    on_mlx = MlxMamba2(fixture.config)
    on_mlx.load_canonical_weights(block)
    assert state_dict_digest(block) == state_dict_digest(on_mlx.canonical_weights())
