"""Strict ``seq_idx`` reset -- the invariant Niverel's H-Net depends on.

Foundation V3 trains with ``boundary_policy: strict_reset``, so at every
document boundary both the convolution state and the SSM state must go to
zero. The certifying statement is::

    model(x, seq_idx=segments) == concat(model(doc_1), model(doc_2), ...)

Document boundaries do not align with chunk boundaries, so each of the
structurally distinct cases gets its own test rather than relying on one
random example to have covered them all.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from tests.conftest import requires_torch

pytestmark = requires_torch

ATOL = 1e-10  # cpu_float64, sealed
LENGTH = 70
CHUNK = 16


def _boundaries(seq_idx, row: int) -> list[int]:
    ids = seq_idx[row]
    bounds = [0]
    for t in range(1, len(ids)):
        if int(ids[t]) != int(ids[t - 1]):
            bounds.append(t)
    bounds.append(len(ids))
    return bounds


def _run_documents_separately(model, x, seq_idx):
    """The ground truth: every document as its own independent forward pass."""
    import torch

    rows = []
    for b in range(x.shape[0]):
        bounds = _boundaries(seq_idx, b)
        pieces = [model(x[b : b + 1, s:e]) for s, e in pairwise(bounds)]
        rows.append(torch.cat(pieces, dim=1))
    return torch.cat(rows, dim=0)


# Each entry builds a seq_idx exercising one structurally distinct case.
CASES = {
    "boundary_at_position_zero": lambda t: [0] + [1] * (t - 1),
    "single_boundary_mid_chunk": lambda t: [0] * 20 + [1] * (t - 20),
    "boundary_on_chunk_edge": lambda t: [0] * 16 + [1] * 16 + [2] * 16 + [3] * (t - 48),
    "length_one_document": lambda t: [0] * 33 + [1] + [2] * (t - 34),
    "consecutive_boundaries": lambda t: list(range(t)),
    "document_inside_one_chunk": lambda t: [0] * 17 + [1] * 3 + [2] * (t - 20),
    "boundary_at_last_position": lambda t: [0] * (t - 1) + [1],
    "non_zero_first_id": lambda t: [7] * 30 + [9] * (t - 30),
    "many_short_documents": lambda t: [i // 3 for i in range(t)],
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_segmented_equals_separate_documents(make_model, grouped_config, case):
    """The certifying invariant, per structural case."""
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(11)
    x = torch.randn(2, LENGTH, grouped_config.d_model, dtype=torch.float64)
    ids = CASES[case](LENGTH)
    seq_idx = torch.tensor([ids, ids], dtype=torch.int32)

    with torch.no_grad():
        segmented = model(x, seq_idx=seq_idx)
        separate = _run_documents_separately(model, x, seq_idx)

    assert torch.allclose(segmented, separate, atol=ATOL, rtol=0), (
        f"{case}: max diff {(segmented - separate).abs().max():.3e}"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_sequential_oracle_agrees_on_every_case(make_model, grouped_config, case):
    """The chunked masking must match the explicit recurrence too.

    Passing the previous test alone could in principle mean both sides share a
    bug; the sequential oracle resets state with a plain per-timestep
    multiply and shares no masking code with the chunked path.
    """
    import torch

    model = make_model(grouped_config)
    torch.manual_seed(11)
    x = torch.randn(2, LENGTH, grouped_config.d_model, dtype=torch.float64)
    ids = CASES[case](LENGTH)
    seq_idx = torch.tensor([ids, ids], dtype=torch.int32)

    with torch.no_grad():
        model.ssd_impl = "chunked"
        chunked = model(x, seq_idx=seq_idx)
        model.ssd_impl = "sequential"
        sequential = model(x, seq_idx=seq_idx)
        model.ssd_impl = "per_segment"
        per_segment = model(x, seq_idx=seq_idx)

    assert torch.allclose(chunked, sequential, atol=ATOL, rtol=0)
    assert torch.allclose(chunked, per_segment, atol=ATOL, rtol=0)


def test_documents_misaligned_across_batch_rows(make_model, grouped_config):
    """Masks are built per row; rows must not leak into each other."""
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(3)
    x = torch.randn(3, LENGTH, grouped_config.d_model, dtype=torch.float64)
    seq_idx = torch.zeros(3, LENGTH, dtype=torch.int32)
    seq_idx[0, 10:] = 1
    seq_idx[0, 55:] = 2
    seq_idx[1, 3:] = 1
    seq_idx[1, 4:] = 2
    seq_idx[1, 60:] = 3
    seq_idx[2, 33:] = 1

    with torch.no_grad():
        segmented = model(x, seq_idx=seq_idx)
        separate = _run_documents_separately(model, x, seq_idx)
    assert torch.allclose(segmented, separate, atol=ATOL, rtol=0)


def test_a_row_with_no_boundary_is_unaffected(make_model, grouped_config):
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    torch.manual_seed(5)
    x = torch.randn(1, LENGTH, grouped_config.d_model, dtype=torch.float64)
    with torch.no_grad():
        plain = model(x)
        with_ids = model(x, seq_idx=torch.zeros(1, LENGTH, dtype=torch.int32))
    assert torch.equal(plain, with_ids), "an all-same seq_idx must be bit-identical to none"


def test_conv_path_is_identical_with_and_without_seq_idx(make_model, grouped_config):
    """Both sides of the invariant must take the same convolution code path.

    If ``seq_idx=None`` took an F.conv1d fast path while the segmented call
    took the masked one, the two would differ by float32 reassociation noise
    unrelated to strict reset -- and the tolerance would have to be loosened
    for the wrong reason. Bit-identical is the requirement.
    """
    import torch

    from niverel_mamba.torch_ops.causal_conv import causal_conv1d

    torch.manual_seed(0)
    x = torch.randn(2, 8, 40, dtype=torch.float64)
    weight = torch.randn(8, 1, 4, dtype=torch.float64)
    bias = torch.randn(8, dtype=torch.float64)

    without = causal_conv1d(x, weight, bias, seq_idx=None)
    single_doc = causal_conv1d(x, weight, bias, seq_idx=torch.zeros(2, 40, dtype=torch.int32))
    assert torch.equal(without, single_doc)


def test_non_monotonic_seq_idx_is_refused(make_model, tiny_config):
    """The masking derivation is only valid for non-decreasing ids.

    Ids like ``0, 1, 0`` break the "same id iff no boundary in between"
    equivalence, so they must be rejected rather than silently mis-masked.
    """
    import torch

    from niverel_mamba.errors import InvalidSeqIdxError

    model = make_model(tiny_config)
    x = torch.randn(1, 6, tiny_config.d_model, dtype=torch.float64)
    bad = torch.tensor([[0, 0, 1, 1, 0, 0]], dtype=torch.int32)
    with pytest.raises(InvalidSeqIdxError, match="non-decreasing"):
        model(x, seq_idx=bad)


def test_seq_idx_shape_is_validated(make_model, tiny_config):
    import torch

    from niverel_mamba.errors import InvalidSeqIdxError

    model = make_model(tiny_config)
    x = torch.randn(1, 6, tiny_config.d_model, dtype=torch.float64)
    with pytest.raises(InvalidSeqIdxError, match="does not match input"):
        model(x, seq_idx=torch.zeros(1, 5, dtype=torch.int32))


@pytest.mark.slow
def test_l8192_with_many_documents(make_model, grouped_config):
    """The real Niverel context length, with realistic document structure."""
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    length = 8192
    torch.manual_seed(1337)
    x = torch.randn(1, length, grouped_config.d_model, dtype=torch.float64)

    generator = torch.Generator().manual_seed(99)
    lengths = torch.randint(50, 900, (40,), generator=generator).tolist()
    ids: list[int] = []
    for doc, size in enumerate(lengths):
        ids.extend([doc] * size)
        if len(ids) >= length:
            break
    ids = (ids + [len(lengths)] * length)[:length]
    seq_idx = torch.tensor([ids], dtype=torch.int32)

    with torch.no_grad():
        segmented = model(x, seq_idx=seq_idx)
        separate = _run_documents_separately(model, x, seq_idx)

    assert segmented.shape == (1, length, grouped_config.d_model)
    assert torch.allclose(segmented, separate, atol=ATOL, rtol=0)
