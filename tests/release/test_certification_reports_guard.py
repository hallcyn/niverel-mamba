"""What the reports certify, not merely that they are green.

v0.1.0 attached six certification reports produced on an A100 and an H100.
Every one passed. Not one of them certified the CUDA backend: their
`candidate_backend` was `torch-reference-cpu-chunked` -- the portable
implementation measured against its own float64 oracle. A real gate, and a real
result, but a statement about `torch-reference` that read as a statement about
`cuda-reference` because of the hardware it ran on and the name of its file.

These tests run the actual verifier, including against the shape of the reports
that were attached to v0.1.0, which it must refuse.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_certification_reports.py"

TARGETS = ("torch211-cu128", "torch212-cu130", "torch213-cu130")


def _report(candidate: str, passed: bool = True) -> dict[str, object]:
    return {
        "passed": passed,
        "candidate_backend": candidate,
        "reference_backend": "torch-reference-cuda-float32",
        "max_abs_error": 1.2e-3,
    }


def _write(directory: Path, reports: dict[str, dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in reports.items():
        (directory / name).write_text(json.dumps(payload))


def _run(tmp_path: Path, reports: dict[str, dict[str, object]]) -> subprocess.CompletedProcess[str]:
    _write(tmp_path / "certification", reports)
    return subprocess.run(
        [
            sys.executable, str(SCRIPT), "--dir", str(tmp_path / "certification"),
            "--require-arch", "sm80,sm90",
            "--require-backend", "cuda-reference,torch-reference-cpu-chunked",
        ],
        capture_output=True, text=True,
    )


def _full_set(**overrides: object) -> dict[str, dict[str, object]]:
    reports = {}
    for arch in ("sm80", "sm90"):
        for target in TARGETS:
            for backend in ("torch-reference-cpu-chunked", "cuda-reference"):
                reports[f"certification-{arch}-cuda-wheels-{target}-{backend}.json"] = _report(
                    backend
                )
    reports.update(overrides)  # type: ignore[arg-type]
    return reports


def test_a_fully_certified_release_is_accepted(tmp_path):
    result = _run(tmp_path, _full_set())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cuda-reference" in result.stdout


def test_the_reports_v0_1_0_actually_shipped_are_refused(tmp_path):
    """Green, produced on real GPUs, and not certifying the CUDA backend."""
    shipped = {
        f"certification-{arch}-cuda-wheels-{target}.json": _report("torch-reference-cpu-chunked")
        for arch in ("sm80", "sm90")
        for target in TARGETS
    }
    result = _run(tmp_path, shipped)
    assert result.returncode != 0
    assert "nothing certifies ['cuda-reference']" in result.stdout


def test_no_reports_at_all_is_refused(tmp_path):
    """The failure mode that looks most like success."""
    (tmp_path / "certification").mkdir()
    result = _run(tmp_path, {})
    assert result.returncode != 0
    assert "no certification report" in result.stdout


def test_one_failed_report_refuses_everything(tmp_path):
    reports = _full_set()
    key = "certification-sm90-cuda-wheels-torch213-cu130-cuda-reference.json"
    reports[key] = _report("cuda-reference", passed=False)
    result = _run(tmp_path, reports)
    assert result.returncode != 0
    assert "did not pass" in result.stdout


def test_cuda_certified_on_sm80_only_is_refused(tmp_path):
    """sm80 passing says nothing about sm90, and both are shipped."""
    reports = {
        name: payload
        for name, payload in _full_set().items()
        if not (name.startswith("certification-sm90") and "cuda-reference" in name)
    }
    result = _run(tmp_path, reports)
    assert result.returncode != 0
    assert "sm90" in result.stdout
