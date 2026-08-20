"""The pod's driver must be new enough for every runtime we ship.

A certification pod came back advertising CUDA 12.4. `torch211-cu128` ran on it
-- CUDA minor version compatibility -- and both cu130 runtimes died inside
`torch._C._cuda_init()`, after three gigabytes of torch had been installed. The
pod before it advertised 12.8 and would have failed identically had it reached
the second runtime; earlier failures hid that.

Drivers are backward compatible and not forward. A 13.x driver runs a cu128
build; a 12.x driver cannot run CUDA 13.0, because that is a major version.
Comparing minors would reject a usable pod and throw the rental away, which the
first version of this check did.

The check itself is extracted from the workflow and executed against a fake
`nvidia-smi`, so what is tested is what will run on the pod.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is in the dev extra")

WORKFLOW = (
    Path(__file__).resolve().parent.parent.parent
    / ".github" / "workflows" / "certify-cuda.yml"
)


def _preflight_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    step = next(
        s
        for s in workflow["jobs"]["certify"]["steps"]
        if "driver must be new enough" in str(s.get("name", ""))
    )
    body = str(step["run"])
    return body[body.index("<<'PY'") + len("<<'PY'") : body.index("\nPY")]


def _run(tmp_path: Path, driver: str, runtimes: dict[str, str]) -> subprocess.CompletedProcess[str]:
    for name, cuda in runtimes.items():
        directory = tmp_path / "wheelhouse" / name
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"artifacts": [{"torch_cuda": cuda}]}))

    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho 'CUDA Version: {driver}'\n")
    smi.chmod(0o755)

    return subprocess.run(
        [sys.executable, "-c", _preflight_script()],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )


SHIPPED = {
    "cuda-wheels-torch211-cu128": "12.8",
    "cuda-wheels-torch212-cu130": "13.0",
    "cuda-wheels-torch213-cu130": "13.0",
}


@pytest.mark.parametrize("driver", ["12.4", "12.8", "12.9"])
def test_a_cuda_12_driver_is_refused_because_of_the_cu130_runtimes(tmp_path, driver):
    result = _run(tmp_path, driver, SHIPPED)
    assert result.returncode == 1, result.stdout
    assert "TOO NEW FOR THIS DRIVER" in result.stdout
    assert "cu130" in result.stdout


@pytest.mark.parametrize("driver", ["12.4", "12.8"])
def test_the_cu128_runtime_is_not_refused_on_a_cuda_12_driver(tmp_path, driver):
    """Minor version compatibility is real, and rejecting it wastes the rental.

    This is what actually happened: on the 12.4 pod, torch211-cu128 ran to
    completion. A check comparing minors would have thrown that pod away.
    """
    result = _run(tmp_path, driver, {"cuda-wheels-torch211-cu128": "12.8"})
    assert result.returncode == 0, result.stdout
    assert "TOO NEW" not in result.stdout


def test_a_cuda_13_driver_accepts_everything_we_ship(tmp_path):
    result = _run(tmp_path, "13.0", SHIPPED)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ok") >= len(SHIPPED)


def test_an_unreadable_nvidia_smi_fails_closed(tmp_path):
    """Unable to check is not the same as checked."""
    (tmp_path / "wheelhouse" / "cuda-wheels-torch211-cu128").mkdir(parents=True)
    (tmp_path / "wheelhouse" / "cuda-wheels-torch211-cu128" / "manifest.json").write_text(
        json.dumps({"artifacts": [{"torch_cuda": "12.8"}]})
    )
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'no such thing here'\n")
    smi.chmod(0o755)
    result = subprocess.run(
        [sys.executable, "-c", _preflight_script()],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "could not read the driver" in result.stdout + result.stderr
