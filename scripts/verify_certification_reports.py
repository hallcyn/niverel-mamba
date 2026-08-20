#!/usr/bin/env python
"""Decide whether a set of certification reports actually certifies the release.

The check that matters is not "did something pass" but **what** passed. v0.1.0
attached six reports produced on an A100 and an H100, every one of them green,
and not one of them certified the CUDA backend: their `candidate_backend` was
`torch-reference-cpu-chunked`, the portable implementation measured against its
own float64 oracle. A real gate, and a real result -- but a statement about
`torch-reference` that was read as a statement about `cuda-reference` because
of the hardware it ran on and the name of the file it landed in.

So this refuses on four grounds, not one:

* a report that did not pass;
* an architecture with no report at all;
* an architecture with no report for a backend we require;
* no reports whatsoever, which is the failure mode that looks most like success.

    python scripts/verify_certification_reports.py --dir certification \
        --require-arch sm80,sm90 --require-backend cuda-reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument(
        "--require-arch",
        default="sm80,sm90",
        help="comma-separated architectures that must each be covered",
    )
    parser.add_argument(
        "--require-backend",
        default="cuda-reference",
        help="comma-separated backends that must be certified on every architecture",
    )
    args = parser.parse_args()

    reports = sorted(args.dir.glob("certification-*.json"))
    if not reports:
        print(f"::error::no certification report under {args.dir}; refusing to publish")
        return 1

    # architecture -> set of backends certified green on it
    covered: dict[str, set[str]] = {}
    failures = []
    for path in reports:
        data = json.loads(path.read_text(encoding="utf-8"))
        candidate = data.get("candidate_backend", "?")
        reference = data.get("reference_backend", "?")
        arch = path.name.split("-")[1]
        status = "passed" if data.get("passed") else "FAILED"
        print(
            f"{path.name}\n"
            f"    {status}: {candidate} vs {reference}, "
            f"max_abs={data.get('max_abs_error', float('nan')):.3e}"
        )
        if not data.get("passed"):
            failures.append(path.name)
            continue
        covered.setdefault(arch, set()).add(candidate)

    if failures:
        print(f"::error::these reports did not pass: {', '.join(failures)}")
        return 1

    required_backends = {b for b in args.require_backend.split(",") if b}
    problems = []
    for arch in (a for a in args.require_arch.split(",") if a):
        certified = covered.get(arch)
        if certified is None:
            problems.append(f"{arch}: no report at all")
            continue
        missing = required_backends - certified
        if missing:
            problems.append(
                f"{arch}: nothing certifies {sorted(missing)} "
                f"(only {sorted(certified)} certified here)"
            )

    if problems:
        for problem in problems:
            print(f"::error::{problem}")
        print("::error::refusing to publish a release whose CUDA backend is uncertified")
        return 1

    print(
        f"\n{len(reports)} passing reports; "
        + "; ".join(f"{arch}: {sorted(backends)}" for arch, backends in sorted(covered.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
