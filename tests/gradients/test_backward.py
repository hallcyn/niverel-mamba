"""Autograd through the portable implementation.

The capability is published as ``backward: "experimental"`` and
``training: false``. These tests establish that gradients exist, are finite,
flow to every parameter, and agree between the chunked path and the
sequential oracle. They do *not* license a training claim: that needs a
comparison against CUDA, which has not happened.
"""

from __future__ import annotations

from tests.conftest import requires_torch

pytestmark = requires_torch

PARAMETERS = ["in_proj.weight", "conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias",
              "norm.weight", "out_proj.weight"]


def _grads(model, x, seq_idx=None):

    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    model(x, seq_idx=seq_idx).square().sum().backward()
    return x.grad, {name: p.grad for name, p in model.named_parameters()}


def test_gradients_reach_every_parameter(make_model, grouped_config):
    """Every parameter that must receive a gradient: input, in_proj, conv1d,
    A_log, D, out_proj."""
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    x = torch.randn(2, 40, grouped_config.d_model, dtype=torch.float64)
    grad_x, grads = _grads(model, x)

    assert grad_x is not None and torch.isfinite(grad_x).all()
    assert grad_x.abs().max() > 0

    for name in PARAMETERS:
        grad = grads[name]
        assert grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient in {name}"
        assert grad.abs().max() > 0, f"gradient for {name} is identically zero"


def test_chunked_and_oracle_gradients_agree(make_model, grouped_config):
    """If the two forwards agree, so must their backwards."""
    import torch

    x = torch.randn(2, 40, grouped_config.d_model, dtype=torch.float64)

    chunked = make_model(grouped_config, seed=3, ssd_impl="chunked")
    sequential = make_model(grouped_config, seed=3, ssd_impl="sequential")
    sequential.load_state_dict(chunked.state_dict(), strict=True)

    grad_x_chunked, grads_chunked = _grads(chunked, x)
    grad_x_seq, grads_seq = _grads(sequential, x)

    assert torch.allclose(grad_x_chunked, grad_x_seq, atol=1e-9, rtol=0)
    for name in PARAMETERS:
        assert torch.allclose(grads_chunked[name], grads_seq[name], atol=1e-9, rtol=0), name


def test_gradients_respect_document_boundaries(make_model, grouped_config):
    """Effectively no gradient flows backwards across a strict-reset boundary.

    "Effectively", not "exactly", and the distinction is worth being precise
    about because it applies to the forward direction too.

    ``segmask`` zeroes every cross-document entry of the decay matrix, so no
    earlier document contributes a *term*. But the intra-chunk decay is
    evaluated as ``exp(cs_l - cs_s)`` with ``cs`` a cumulative sum over the
    whole chunk. For two positions both in the later document, ``cs_l - cs_s``
    cancels the earlier document's contributions exactly in real arithmetic
    and only to rounding in floating point. A residual dependence around
    1e-16 relative therefore survives, in values and in gradients alike.

    This is inherent to the cumsum-difference formulation rather than to the
    masking, and upstream's Triton kernels compute the same segsum, so the
    CUDA path carries it as well. The honest claim is a bound at float64
    rounding, not a bitwise zero -- which is exactly why the brief refuses to
    promise bit-for-bit equality and asks for measured equivalence instead.
    """
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    length, split = 40, 18
    x = torch.randn(2, length, grouped_config.d_model, dtype=torch.float64).requires_grad_(True)
    seq_idx = torch.zeros(2, length, dtype=torch.int32)
    seq_idx[:, split:] = 1

    model(x, seq_idx=seq_idx)[:, split:].square().sum().backward()
    assert x.grad is not None

    inside = x.grad[:, split:].abs().max()
    leaked = x.grad[:, :split].abs().max()
    assert inside > 0.0
    assert leaked / inside < 1e-14, (
        f"gradient leaked across a boundary at {leaked / inside:.3e} of the in-document "
        "magnitude, far above float64 rounding -- this indicates a masking bug"
    )


def test_forward_is_insensitive_to_earlier_documents(make_model, grouped_config):
    """Perturbing document 1 by a factor of 1000 must not move document 2.

    The counterpart of the gradient test: a violently different earlier
    document may shift the later one only at float64 rounding, never
    proportionally to the perturbation. If the masking were wrong, a 1000x
    perturbation would show up immediately and enormously.
    """
    import torch

    model = make_model(grouped_config, ssd_impl="chunked")
    length, split = 40, 18
    torch.manual_seed(0)
    x = torch.randn(2, length, grouped_config.d_model, dtype=torch.float64)
    seq_idx = torch.zeros(2, length, dtype=torch.int32)
    seq_idx[:, split:] = 1

    perturbed = x.clone()
    perturbed[:, :split] = torch.randn_like(perturbed[:, :split]) * 1000.0

    with torch.no_grad():
        base = model(x, seq_idx=seq_idx)
        other = model(perturbed, seq_idx=seq_idx)

    tail_base, tail_other = base[:, split:], other[:, split:]
    drift = (tail_base - tail_other).abs().max() / tail_base.abs().max()
    assert drift < 1e-13, f"document 2 moved by {drift:.3e} relative when document 1 changed"


def test_gradcheck_on_a_tiny_configuration(make_model, tiny_config):
    """Full analytic-vs-numeric gradient check in float64."""
    import torch

    model = make_model(tiny_config, ssd_impl="chunked")
    x = torch.randn(1, 6, tiny_config.d_model, dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return model(inp)

    assert torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_capability_does_not_claim_training():
    """Gradients working is not the same as training being certified."""
    from niverel_mamba.registry import BACKENDS

    capability = BACKENDS["torch-reference"].capability
    assert capability.backward == "experimental"
    assert capability.training is False


def test_backward_stays_experimental_until_cuda_gradients_are_sealed():
    """Everything about the portable backward is proven; that is not the claim.

    Gradients reach every parameter, the chunked path agrees with the sequential
    oracle, none crosses a strict-reset boundary, and a float64 gradcheck matches
    analytic against numeric. None of it says upstream's kernels differentiate
    the same way, which is the condition `Capability.backward` states for
    itself.

    The certification campaign now measures that comparison and writes it into
    the report. When a tolerance class carries those observations, this test is
    what says the claim may move -- and not before.
    """
    from niverel_mamba.certification.tolerances import load_tolerances
    from niverel_mamba.registry import BACKENDS

    table = load_tolerances()
    measured = "cuda_float32_backward" in table.classes and (
        table["cuda_float32_backward"].observed is not None
    )
    for name in ("torch-reference", "cuda-reference"):
        capability = BACKENDS[name].capability
        assert capability.training is False, "training needs more than gradients agreeing"
        if not measured:
            assert capability.backward == "experimental", (
                f"{name} claims backward={capability.backward!r} while no tolerance "
                "class has observed a gradient comparison against CUDA"
            )
