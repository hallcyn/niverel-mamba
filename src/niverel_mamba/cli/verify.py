"""``niverel-mamba verify`` -- run the certification campaign and emit a report.

Compares, on a chosen fixture:

* the chunked implementation against the sequential float64 oracle;
* ``forward`` against ``concat(step)``;
* segmented ``seq_idx`` against separately-run documents;
* MPS against CPU, and MLX against torch CPU, when those are available.

Every comparison is scored against the sealed tolerances in
``certification/tolerances.yaml``. Tolerances are never widened here.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..capabilities import detect_environment
from ..errors import BackendUnavailableError

__all__ = ["add_parser", "run"]

# Deliberately NOT imported at module level: `cli.main.build_parser()` imports
# every subcommand so that `--help` can list them, so anything heavy here is
# paid for by `doctor` and `inspect` too. A cold `pip install niverel-mamba`
# followed by `niverel-mamba doctor` used to crash on numpy for exactly this
# reason -- and a diagnostic command that dies on a missing dependency is
# useless precisely when it is needed.


def add_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser("verify", help="run numerical certification and write a report")
    parser.add_argument("--fixture", default="tiny", help="tiny, segmented or niverel")
    parser.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--device", default=None, help="also compare against this torch device")
    parser.add_argument("--mlx", action="store_true", help="also compare the MLX backend")
    parser.add_argument(
        "--certify",
        default="torch-reference",
        choices=["torch-reference", "cuda-reference"],
        help=(
            "which backend the report certifies. One report, one certified backend: "
            "a report whose candidate is torch-reference proves nothing about the "
            "CUDA kernels, however expensive the machine that produced it was."
        ),
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "measure and report without passing judgement. The report is marked as a "
            "measurement and the command succeeds whatever the numbers, so that a run "
            "on rented hardware yields evidence instead of stopping at the first "
            "comparison scored against a tolerance nobody has observed yet."
        ),
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "verify needs PyTorch to run the reference implementation it certifies "
            "against: pip install 'niverel-mamba[torch]'"
        ) from exc

    from ..certification.compare import compare
    from ..certification.golden import load_fixture
    from ..certification.report import CertificationReport
    from ..errors import FixtureError
    from ..torch_ops.mamba2 import Mamba2

    try:
        fixture = load_fixture(args.fixture)
    except FixtureError as exc:
        print(str(exc))
        return 1

    env = detect_environment()
    config = fixture.config

    if args.certify == "cuda-reference":
        return _certify_cuda_reference(args, fixture, env)

    # The tiny and segmented fixtures are float64; the real V3 block is float32.
    double = not fixture.is_real_checkpoint
    dtype = torch.float64 if double else torch.float32
    tol_class = "cpu_float64" if double else "cpu_float32"

    weights = fixture.torch_weights(dtype=dtype)
    inputs = fixture.torch_inputs(dtype=dtype)
    x = inputs["x"]
    seq_idx = inputs.get("seq_idx")

    model = Mamba2(config).to(dtype)
    model.load_state_dict(weights, strict=True)
    model.eval()

    report = CertificationReport(
        reference_backend="torch-reference-cpu-sequential-oracle",
        candidate_backend="torch-reference-cpu-chunked",
        fixture=args.fixture,
        metadata={"torch": env.torch.version, "dtype": str(dtype), "real_checkpoint": fixture.is_real_checkpoint},
    )

    with torch.no_grad():
        model.ssd_impl = "sequential"
        y_oracle = model(x, seq_idx=seq_idx)
        model.ssd_impl = "chunked"
        y_chunked = model(x, seq_idx=seq_idx)
        report.add(compare(y_chunked, y_oracle, name="forward", tolerance=tol_class))

        model.ssd_impl = "per_segment"
        y_per_segment = model(x, seq_idx=seq_idx)
        report.add(
            compare(y_per_segment, y_oracle, name="per_segment_oracle", tolerance=tol_class)
        )

        # forward == concat(step)
        model.ssd_impl = "chunked"
        state = model.allocate_inference_state(x.shape[0])
        state = state.to(dtype=dtype)
        steps = []
        for t in range(x.shape[1]):
            idx = seq_idx[:, t] if seq_idx is not None else None
            y_t, state = model.step(x[:, t], state, seq_idx_t=idx)
            steps.append(y_t)
        report.add(compare(torch.stack(steps, dim=1), y_chunked, name="step", tolerance=tol_class))

        # segmented == separately-run documents
        if seq_idx is not None:
            y_separate = _run_documents(model, x, seq_idx)
            report.add(
                compare(y_separate, y_chunked, name="segment_reset", tolerance=tol_class)
            )

        if args.device:
            device = torch.device(args.device)
            model32 = Mamba2(config).to(device=device, dtype=torch.float32)
            model32.load_state_dict({k: v.float() for k, v in weights.items()}, strict=True)
            model32.eval()
            y_device = model32(
                x.float().to(device), seq_idx=seq_idx.to(device) if seq_idx is not None else None
            ).cpu()
            klass = "mps_float32" if device.type == "mps" else "cpu_float32"
            report.add(compare(y_device, y_chunked.float(), name=f"{device.type}_forward", tolerance=klass))

    if args.mlx:
        if not env.mlx.available:
            print("MLX requested but not available; skipping")
            return 1
        import mlx.core as mx
        import numpy as np

        from ..mlx_ops.mamba2 import Mamba2 as MlxMamba2

        mlx_model = MlxMamba2(config)
        mlx_model.load_canonical_weights({k: v.float() for k, v in weights.items()})
        y_mlx = np.array(
            mlx_model(
                mx.array(x.float().numpy()),
                mx.array(seq_idx.numpy()) if seq_idx is not None else None,
            )
        )
        report.add(
            compare(y_mlx, y_chunked.float(), name="mlx_forward", tolerance="mlx_float32")
        )

    print(report.summary())
    if args.report:
        path = report.write(args.report)
        print(f"\nwrote {path}")
    return 0 if report.passed else 1


def _certify_cuda_reference(args: argparse.Namespace, fixture: Any, env: Any) -> int:
    """Certify the upstream CUDA kernels against the portable backend.

    **The gate is float32.** In float32 a disagreement between two chunked
    implementations is algebraic -- reassociation, chunk boundaries, how a
    segment reset is applied -- and a tight band means something. Anything
    genuinely wrong, a mishandled reset above all, shows up as an O(1)
    difference and cannot hide under a tolerance.

    **bfloat16 is measured and reported, never gated**, because gating it
    measures the number format rather than the kernels. Two runs on an A100
    established that. Against an unrounded float32 reference the deviation was
    max_abs 3.03e-02, of which the portable implementation alone -- no CUDA at
    all -- already accounted for 3.01e-02. Feeding both sides the same bfloat16
    data removed the input rounding and brought it to 1.85e-02 for `forward`,
    but left 2.58e-02 on `segment_reset`: mean_abs 2.88e-03 on outputs of RMS
    one, which is 0.29% against bfloat16's own 0.39% of relative precision. The
    residue is the arithmetic inside the kernel, and no correct implementation
    can be certified out of it.

    So the report carries both: a verdict on the algorithm, and a measurement of
    what bfloat16 costs on this fixture for anyone choosing to run it.

    Kept separate from the default campaign, and emitting its own report,
    because a report certifies exactly one backend.
    """
    import torch

    from ..certification.compare import compare
    from ..certification.report import CertificationReport
    from ..torch_ops.mamba2 import Mamba2

    if not torch.cuda.is_available():
        print("--certify cuda-reference needs a visible CUDA device; torch reports none")
        return 1

    from ..backends.cuda_reference import CudaReferenceBackend

    config = fixture.config
    weights = fixture.torch_weights(dtype=torch.float32)
    inputs = fixture.torch_inputs(dtype=torch.float32)
    seq_idx = inputs.get("seq_idx")
    seq_idx = seq_idx.cuda() if seq_idx is not None else None
    x = inputs["x"].cuda()

    capability = torch.cuda.get_device_capability(0)
    report = CertificationReport(
        reference_backend="torch-reference-cuda-float32",
        candidate_backend="cuda-reference",
        fixture=args.fixture,
        metadata={
            "torch": env.torch.version,
            "gated_dtype": "torch.float32",
            "device": torch.cuda.get_device_name(0),
            "compute_capability": f"sm_{capability[0]}{capability[1]}",
            "real_checkpoint": fixture.is_real_checkpoint,
        },
    )

    exact = Mamba2(config).cuda().float()
    exact.load_state_dict({k: v.float().cuda() for k, v in weights.items()}, strict=True)
    exact.eval()

    # ---- the gate: float32, where a difference is the algorithm's ------------
    try:
        candidate32 = CudaReferenceBackend(config, device="cuda", dtype=torch.float32)
        candidate32.load_canonical_weights(exact.state_dict())
        with torch.no_grad():
            report.add(
                compare(
                    candidate32.forward(x).float(),
                    exact(x),
                    name="forward_float32",
                    tolerance="cuda_float32",
                )
            )
            if seq_idx is not None:
                report.add(
                    compare(
                        candidate32.forward(x, seq_idx=seq_idx).float(),
                        exact(x, seq_idx=seq_idx),
                        name="segment_reset_float32",
                        tolerance="cuda_float32",
                    )
                )
    except Exception as exc:  # the gate must fail loudly, never quietly vanish
        print(f"::error::the float32 campaign could not run: {type(exc).__name__}: {exc}")
        report.metadata["float32_error"] = f"{type(exc).__name__}: {exc}"
        if args.report:
            print(f"\nwrote {report.write(args.report)}")
        return 0 if args.measure else 1

    # ---- measured, never gated: what bfloat16 costs --------------------------
    rounded = {k: v.float().bfloat16().float().cuda() for k, v in weights.items()}
    x_bf16 = inputs["x"].bfloat16().cuda()
    x_equal = x_bf16.float()

    lossless = Mamba2(config).cuda().float()
    lossless.load_state_dict(rounded, strict=True)
    lossless.eval()
    candidate16 = CudaReferenceBackend(config, device="cuda", dtype=torch.bfloat16)
    candidate16.load_canonical_weights(lossless.state_dict())

    measured: dict[str, Any] = {
        "note": (
            "bfloat16 is reported, not gated. Against an unrounded float32 "
            "reference this measures the number format rather than the kernels: "
            "the portable implementation alone, with no CUDA involved, already "
            "deviates by 3.01e-02 on the segmented fixture."
        )
    }
    with torch.no_grad():
        for name, ids in (("forward", None), ("segment_reset", seq_idx)):
            if name == "segment_reset" and seq_idx is None:
                continue
            actual = candidate16.forward(x_bf16, seq_idx=ids).float()
            at_equal_data = (actual - lossless(x_equal, seq_idx=ids)).abs()
            against_exact = (actual - exact(x, seq_idx=ids)).abs()
            measured[name] = {
                "at_equal_data": {
                    "max_abs_error": float(at_equal_data.max()),
                    "mean_abs_error": float(at_equal_data.mean()),
                },
                "against_unrounded_float32": {
                    "max_abs_error": float(against_exact.max()),
                    "mean_abs_error": float(against_exact.mean()),
                },
            }
    report.metadata["bfloat16_measured"] = measured

    # ---- the second gate: gradients ----------------------------------------
    #
    # `Capability.backward` states its own condition: experimental until
    # gradients have been compared against CUDA. This is that comparison.
    #
    # Scored per parameter and elementwise, because gradient magnitudes differ
    # by more than an order of magnitude between parameters -- max|grad| runs
    # from 3.9 on A_log to 104 on out_proj.weight on the segmented fixture -- so
    # a single absolute figure would be meaningless read across them.
    #
    # The cotangent is seeded. An unseeded one made the first measurement
    # unreproducible: the numbers were real but no second run could confirm
    # them, which is not a basis for sealing anything.
    generator = torch.Generator(device=x.device).manual_seed(20260821)
    cotangent = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    def _grads(model: Any, forward: Any) -> tuple[Any, dict[str, Any]]:
        for parameter in model.parameters():
            parameter.grad = None
        source = x.detach().clone().requires_grad_(True)
        (forward(source) * cotangent).sum().backward()
        return source.grad, {
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    reference_input, reference_params = _grads(exact, lambda t: exact(t))
    candidate_input, candidate_params = _grads(
        candidate32.module, lambda t: candidate32.forward(t)
    )

    only_one_side = sorted(set(reference_params) ^ set(candidate_params))
    if only_one_side:
        # A parameter that receives a gradient on one side and not the other is
        # a contract failure, not a tolerance question.
        print(f"::error::gradients differ in *shape*: {only_one_side} on one side only")
        report.metadata["backward_error"] = f"parameters on one side only: {only_one_side}"
        if args.report:
            print(f"\nwrote {report.write(args.report)}")
        return 0 if args.measure else 1

    def _normalised(candidate: Any, reference: Any) -> tuple[Any, Any]:
        """Divide both by the reference's peak, so the band is scale-free.

        Gradient magnitudes are set by the cotangent, and so is the deviation:
        both scale together. Measured across twelve draws on this fixture,
        max|grad| moves by a factor of 3.09 between the smallest and largest --
        so a band in absolute units calibrated on one draw is calibrated on
        nothing. The first version of this gate was: 3.03e-02 observed, 1.0e-01
        sealed, and a plausible 9.4e-02 on another draw. A margin of 1.1, found
        before it cost a rental rather than after.
        Normalised, the quantity is the relative deviation, which is a property
        of the arithmetic rather than of the draw: 3.4e-04 on the worst
        parameter, 6.2e-04 on the input gradient.
        """
        peak = reference.abs().max()
        if peak == 0:
            return candidate, reference
        return candidate / peak, reference / peak

    report.add(
        compare(
            *_normalised(candidate_input, reference_input),
            name="input_grad_float32",
            tolerance="cuda_float32_backward",
        )
    )
    for name in sorted(reference_params):
        report.add(
            compare(
                *_normalised(candidate_params[name], reference_params[name]),
                name=f"grad_{name}",
                tolerance="cuda_float32_backward",
            )
        )

    # The magnitudes the deviations should be read against, so a report means
    # something on its own. Their absence is why the first measurement -- worst
    # parameter 3.03e-02 -- could not be interpreted without going back to a
    # machine and computing the scale by hand.
    # The comparisons above are normalised, so the raw figures are recorded
    # here: a reader should be able to see both what deviated and how large the
    # thing that deviated was, without recomputing anything.
    report.metadata["backward_reference_magnitudes"] = {
        "input_grad_max_abs": float(reference_input.abs().max()),
        "input_grad_deviation_max_abs": float((candidate_input - reference_input).abs().max()),
        **{
            key: value
            for name, grad in sorted(reference_params.items())
            for key, value in (
                (f"grad_{name}_max_abs", float(grad.abs().max())),
                (
                    f"grad_{name}_deviation_max_abs",
                    float((candidate_params[name] - grad).abs().max()),
                ),
            )
        },
    }

    if args.measure:
        report.metadata["mode"] = "measurement"
        report.metadata["note"] = (
            "MEASUREMENT ONLY. This report certifies nothing: it exists to observe "
            "the tolerance classes so they can be sealed from evidence. The release "
            "gate rejects it, which is correct."
        )

    print(report.summary())
    print("\ngradient magnitudes these deviations are read against:")
    for key, value in report.metadata["backward_reference_magnitudes"].items():
        print(f"  {key}: {value:.4f}")
    print("\nbfloat16, measured and not gated:")
    for name, values in measured.items():
        if name == "note":
            continue
        equal = values["at_equal_data"]
        exact_cmp = values["against_unrounded_float32"]
        print(
            f"  {name:14s} equal data  max_abs={equal['max_abs_error']:.3e}"
            f"  mean_abs={equal['mean_abs_error']:.3e}"
        )
        print(
            f"  {'':14s} vs float32  max_abs={exact_cmp['max_abs_error']:.3e}"
            f"  mean_abs={exact_cmp['mean_abs_error']:.3e}"
        )

    if args.report:
        print(f"\nwrote {report.write(args.report)}")
    if args.measure:
        print("\nmeasurement only: no verdict was passed, and none is implied")
        return 0
    return 0 if report.passed else 1


def _run_documents(model: Any, x: Any, seq_idx: Any) -> Any:
    """Run each document separately and concatenate, for the reset invariant."""
    import torch

    rows = []
    for b in range(x.shape[0]):
        ids = seq_idx[b]
        bounds = [0]
        for t in range(1, x.shape[1]):
            if int(ids[t]) != int(ids[t - 1]):
                bounds.append(t)
        bounds.append(x.shape[1])
        pieces = [model(x[b : b + 1, s:e]) for s, e in pairwise(bounds)]
        rows.append(torch.cat(pieces, dim=1))
    return torch.cat(rows, dim=0)
