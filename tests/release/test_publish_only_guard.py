"""`publish_only` must re-check the evidence, not skip the check.

The mode exists for one reason: a release failed *after* two GPU certifications
had passed, and re-running the whole workflow would have rented both again. It
would be a poor trade if the escape hatch also became a way to push something
uncertified to PyPI, so it verifies the certification reports attached to the
release before anything is uploaded.

The script it uses is embedded in a composite action, so this test extracts it
and runs it -- against reports that pass, reports that fail, and reports that
cover only one architecture. A test that merely grepped the YAML would pass
against a script that verified nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is in the dev extra")

ACTION = (
    Path(__file__).resolve().parent.parent.parent
    / ".github/actions/collect-distributions/action.yml"
)


def _verification_script() -> str:
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    step = next(
        s for s in action["runs"]["steps"] if "certification reports" in str(s.get("name", ""))
    )
    body = str(step["run"])
    start = body.index("<<'PY'") + len("<<'PY'")
    return body[start : body.index("\nPY")]


def _report(passed: bool) -> dict[str, object]:
    return {
        "passed": passed,
        "candidate_backend": "cuda-reference",
        "reference_backend": "torch-reference",
        "max_abs_error": 1.2e-5,
    }


def _run(tmp_path: Path, reports: dict[str, bool]) -> subprocess.CompletedProcess[str]:
    directory = tmp_path / "certification"
    directory.mkdir()
    for name, passed in reports.items():
        (directory / name).write_text(json.dumps(_report(passed)))
    script = tmp_path / "verify.py"
    script.write_text(_verification_script())
    return subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )


ALL_PASSING = {
    f"certification-{arch}-cuda-wheels-{target}.json": True
    for arch in ("sm80", "sm90")
    for target in ("torch211-cu128", "torch212-cu130", "torch213-cu130")
}


def test_a_fully_certified_release_is_accepted(tmp_path):
    result = _run(tmp_path, ALL_PASSING)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 passing reports" in result.stdout


def test_a_release_with_no_reports_is_refused(tmp_path):
    (tmp_path / "certification").mkdir()
    script = tmp_path / "verify.py"
    script.write_text(_verification_script())
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "no certification report" in result.stdout + result.stderr


def test_a_single_failed_report_refuses_the_whole_publish(tmp_path):
    reports = dict(ALL_PASSING)
    reports["certification-sm90-cuda-wheels-torch213-cu130.json"] = False
    result = _run(tmp_path, reports)
    assert result.returncode != 0
    assert "did not pass" in result.stdout + result.stderr


def test_a_missing_architecture_refuses_the_publish(tmp_path):
    """sm80 passing says nothing about sm90; both are shipped."""
    only_sm80 = {k: v for k, v in ALL_PASSING.items() if "sm80" in k}
    result = _run(tmp_path, only_sm80)
    assert result.returncode != 0
    assert "sm90" in result.stdout + result.stderr
