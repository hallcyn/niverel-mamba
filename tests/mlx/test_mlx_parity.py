"""The MLX backend, against torch-reference CPU."""

from __future__ import annotations

import pytest

from tests.conftest import requires_mlx, requires_torch

pytestmark = [requires_mlx, requires_torch, pytest.mark.mlx]


def _pair(config, seed=0):
    import mlx.core as mx
    import torch

    from niverel_mamba.mlx_ops.mamba2 import Mamba2 as MlxMamba2
    from niverel_mamba.torch_ops.mamba2 import Mamba2 as TorchMamba2

    torch.manual_seed(seed)
    reference = TorchMamba2(config).float().eval()
    candidate = MlxMamba2(config)
    candidate.load_canonical_weights(reference.state_dict())
    return reference, candidate, mx


def _assert_within(candidate, reference, name):
    from niverel_mamba.certification import compare

    result = compare(candidate, reference, name=name, tolerance="mlx_float32")
    assert result.passed, result.detail or f"{name}: max_abs={result.max_abs_error:.3e}"


def test_forward_matches_torch(grouped_config):
    import numpy as np
    import torch

    reference, candidate, mx = _pair(grouped_config)
    torch.manual_seed(1)
    x = torch.randn(2, 70, grouped_config.d_model)
    with torch.no_grad():
        expected = reference(x)
    actual = np.array(candidate(mx.array(x.numpy())))
    _assert_within(actual, expected, "mlx_forward")


def test_seq_idx_reset_matches_torch(grouped_config):
    import numpy as np
    import torch

    reference, candidate, mx = _pair(grouped_config)
    torch.manual_seed(1)
    length = 70
    x = torch.randn(2, length, grouped_config.d_model)
    seq_idx = torch.zeros(2, length, dtype=torch.int32)
    seq_idx[:, 17:] = 1
    seq_idx[:, 18:] = 2
    seq_idx[:, 45:] = 3
    with torch.no_grad():
        expected = reference(x, seq_idx=seq_idx)
    actual = np.array(candidate(mx.array(x.numpy()), mx.array(seq_idx.numpy())))
    _assert_within(actual, expected, "mlx_seq_idx")


def test_strict_reset_holds_inside_mlx(grouped_config):
    """The invariant must hold within MLX itself, not only against torch."""
    import numpy as np
    import torch

    _, candidate, mx = _pair(grouped_config)
    torch.manual_seed(2)
    length, split = 60, 25
    x = mx.array(torch.randn(1, length, grouped_config.d_model).numpy())
    seq_idx = np.zeros((1, length), dtype=np.int32)
    seq_idx[0, split:] = 1

    segmented = np.array(candidate(x, mx.array(seq_idx)))
    separate = np.concatenate(
        [np.array(candidate(x[:, :split])), np.array(candidate(x[:, split:]))], axis=1
    )
    _assert_within(segmented, separate, "mlx_strict_reset")


def test_chunked_matches_mlx_sequential(grouped_config):
    import numpy as np
    import torch

    _, candidate, mx = _pair(grouped_config)
    torch.manual_seed(3)
    x = mx.array(torch.randn(1, 40, grouped_config.d_model).numpy())
    candidate.ssd_impl = "chunked"
    chunked = np.array(candidate(x))
    candidate.ssd_impl = "sequential"
    sequential = np.array(candidate(x))
    _assert_within(chunked, sequential, "mlx_chunked_vs_sequential")


def test_forward_matches_concat_step(grouped_config):
    import numpy as np
    import torch

    _, candidate, mx = _pair(grouped_config)
    torch.manual_seed(4)
    length = 24
    x = mx.array(torch.randn(1, length, grouped_config.d_model).numpy())
    forward = np.array(candidate(x))

    state = candidate.allocate_inference_state(1)
    outs = []
    for t in range(length):
        y_t, state = candidate.step(x[:, t], state)
        outs.append(y_t)
    _assert_within(np.array(mx.stack(outs, axis=1)), forward, "mlx_step")


def test_weight_round_trip_is_byte_identical(grouped_config):
    """canonical -> MLX -> canonical must reproduce identical hashes.

    Brief section 3.2 requires the MLX conversion to be reversible, and
    'reversible' means the bytes come back, not that they come back close.
    """
    from niverel_mamba.weights import state_dict_digest

    reference, candidate, _ = _pair(grouped_config)
    original = reference.state_dict()
    recovered = candidate.canonical_weights()

    assert set(original) == set(recovered)
    assert state_dict_digest(original) == state_dict_digest(recovered)


def test_full_round_trip_back_into_torch(grouped_config):
    """torch state_dict -> canonical -> MLX -> canonical -> torch state_dict."""
    import torch

    from niverel_mamba.adapters.upstream import from_upstream, to_upstream
    from niverel_mamba.torch_ops.mamba2 import Mamba2 as TorchMamba2
    from niverel_mamba.weights import state_dict_digest

    reference, candidate, _ = _pair(grouped_config)
    start = reference.state_dict()

    canonical = from_upstream(start, grouped_config)
    candidate.load_canonical_weights(canonical)
    back = to_upstream(candidate.canonical_weights(), grouped_config)

    rebuilt = TorchMamba2(grouped_config).float()
    rebuilt.load_state_dict({k: torch.as_tensor(v) for k, v in back.items()}, strict=True)

    assert state_dict_digest(start) == state_dict_digest(rebuilt.state_dict())


def test_mlx_rejects_non_monotonic_seq_idx(grouped_config):
    import mlx.core as mx
    import numpy as np

    from niverel_mamba.errors import InvalidSeqIdxError

    _, candidate, _ = _pair(grouped_config)
    x = mx.array(np.random.randn(1, 6, grouped_config.d_model).astype(np.float32))
    bad = mx.array(np.array([[0, 0, 1, 1, 0, 0]], dtype=np.int32))
    with pytest.raises(InvalidSeqIdxError, match="non-decreasing"):
        candidate(x, bad)


def test_loading_refuses_a_bad_state_dict(grouped_config):
    from niverel_mamba.errors import MissingKeysError
    from niverel_mamba.mlx_ops.mamba2 import Mamba2 as MlxMamba2

    model = MlxMamba2(grouped_config)
    with pytest.raises(MissingKeysError):
        model.load_canonical_weights({"A_log": [0.0]})
