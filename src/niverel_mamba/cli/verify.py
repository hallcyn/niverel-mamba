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
