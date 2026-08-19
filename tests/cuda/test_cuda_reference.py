"""cuda-reference.

Everything numerical here is skipped without an NVIDIA GPU -- explicitly and
by name, never with continue-on-error. The behavioural tests that do NOT need
a GPU (namely: that the backend refuses to exist rather than degrading) run
everywhere, because that refusal is the property most worth guarding.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_cuda, requires_torch

pytestmark = requires_torch


def test_backend_refuses_to_build_without_the_wheels():
    """No silent fallback -- the single most important guarantee."""

    from niverel_mamba.backends.cuda_reference import load_upstream_mamba2
    from niverel_mamba.errors import BackendUnavailableError

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("mamba-ssm is installed, so the absence path cannot be exercised")

    with pytest.raises(BackendUnavailableError) as excinfo:
        load_upstream_mamba2()
    message = str(excinfo.value)
    assert "mamba-ssm" in message
    assert "will not fall back" in message


def test_error_names_the_install_command():
    """A refusal should tell the user what to do next."""
    from niverel_mamba.backends.cuda_reference import load_upstream_mamba2
    from niverel_mamba.errors import BackendUnavailableError

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("mamba-ssm is installed")

    with pytest.raises(BackendUnavailableError, match="install-backend cuda"):
        load_upstream_mamba2()


@requires_cuda
@pytest.mark.cuda
def test_cuda_matches_torch_reference():
    """Parity against the portable backend under cuda_bfloat16 tolerances.

    Until this runs on a real sm80/sm90 GPU, the cuda_bfloat16 tolerance class
    has no observed data and cuda-reference stays published as experimental.
    """
    import torch

    from niverel_mamba.adapters.niverel import niverel_v3_config
    from niverel_mamba.backends.cuda_reference import CudaReferenceBackend
    from niverel_mamba.certification import compare
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    config = niverel_v3_config()
    portable = Mamba2(config).cuda().float().eval()
    cuda = CudaReferenceBackend(config, device="cuda", dtype=torch.bfloat16)
    cuda.load_canonical_weights(portable.state_dict())

    x = torch.randn(1, 256, config.d_model, device="cuda")
    with torch.no_grad():
        expected = portable(x)
        actual = cuda.forward(x.to(torch.bfloat16))

    result = compare(actual.float(), expected, name="forward", tolerance="cuda_bfloat16")
    assert result.passed, result.detail


@requires_cuda
@pytest.mark.cuda
def test_cuda_seq_idx_reset_matches():
    import torch

    from niverel_mamba.adapters.niverel import niverel_v3_config
    from niverel_mamba.backends.cuda_reference import CudaReferenceBackend
    from niverel_mamba.certification import compare
    from niverel_mamba.torch_ops.mamba2 import Mamba2

    config = niverel_v3_config()
    portable = Mamba2(config).cuda().float().eval()
    cuda = CudaReferenceBackend(config, device="cuda", dtype=torch.bfloat16)
    cuda.load_canonical_weights(portable.state_dict())

    length = 512
    x = torch.randn(1, length, config.d_model, device="cuda")
    seq_idx = torch.zeros(1, length, dtype=torch.int32, device="cuda")
    seq_idx[0, 137:] = 1
    seq_idx[0, 300:] = 2

    with torch.no_grad():
        expected = portable(x, seq_idx=seq_idx)
        actual = cuda.forward(x.to(torch.bfloat16), seq_idx=seq_idx)

    result = compare(actual.float(), expected, name="segment_reset", tolerance="cuda_bfloat16")
    assert result.passed, result.detail
