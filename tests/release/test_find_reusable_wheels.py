"""Reusing wheels is only safe if "the same inputs" is checked, not assumed.

A release that only touches this package's own code rebuilds a byte-for-byte
equivalent artefact and spends seventy minutes per runtime doing it. Skipping
that is worth real time -- but reusing wheels built from *different* Dockerfiles
or *different* pinned versions would mean certifying something nobody built from
this source, which is the one failure this project exists to prevent.

So the resolver decides by fingerprint, and every uncertainty resolves towards
building. These tests are mostly about the refusals.
"""

from __future__ import annotations

import sys
import urllib.error
from base64 import b64encode
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

finder = pytest.importorskip("find_reusable_wheels")

TARGETS = ["torch211-cu128", "torch212-cu130", "torch213-cu130"]
WORKFLOW_TEXT = """
          build-args: |
            TORCH_CUDA_ARCH_LIST=8.0;9.0
            MAMBA_SSM_VERSION=2.3.2.post1
            CAUSAL_CONV1D_VERSION=1.6.2.post1
"""


class FakeHub:
    """Just enough of the API: file blobs per ref, runs, and their artifacts."""

    def __init__(self, blobs, runs, artifacts, workflow_text=WORKFLOW_TEXT):
        self.blobs, self.runs, self.artifacts = blobs, runs, artifacts
        self.workflow_text = workflow_text
        self.failures: set[str] = set()

    def __call__(self, path, token):
        for pattern in self.failures:
            if pattern in path:
                raise urllib.error.URLError(f"refused: {path}")
        if "/contents/" in path:
            file, ref = path.split("/contents/")[1].split("?ref=")
            if file == finder.BUILD_WORKFLOW:
                return {"content": b64encode(self.workflow_text.encode()).decode()}
            return {"sha": self.blobs[ref][file]}
        if "/artifacts" in path:
            run_id = int(path.split("/runs/")[1].split("/")[0])
            return {"artifacts": self.artifacts.get(run_id, [])}
        if "/runs?" in path:
            workflow = path.split("/workflows/")[1].split("/runs")[0]
            return {"workflow_runs": self.runs.get(workflow, [])}
        raise AssertionError(f"unexpected path {path}")


def _blobs(**per_ref):
    return {ref: {name: f"{ref}-{i}" if same else f"{ref}-{name}"
                  for i, (name, same) in enumerate(files.items())}
            for ref, files in per_ref.items()}


def _same_inputs(*refs):
    return {ref: {path: f"blob-{path}" for path in
                  (*finder.FINGERPRINTED,)} for ref in refs}


def _artifacts(targets=TARGETS, expired=False):
    return [{"name": f"cuda-wheels-{t}", "expired": expired} for t in targets]


def _hub(monkeypatch, hub):
    monkeypatch.setattr(finder, "_get", hub)
    return hub


def test_a_run_with_identical_inputs_is_reused(monkeypatch):
    _hub(monkeypatch, FakeHub(
        blobs=_same_inputs("head", "older"),
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts()},
    ))
    assert finder.find("o/r", "head", TARGETS, "tok") == 7


def test_a_run_built_from_different_dockerfiles_is_refused(monkeypatch):
    blobs = _same_inputs("head", "older")
    blobs["older"]["docker/torch212-cu130.Dockerfile"] = "something-else"
    _hub(monkeypatch, FakeHub(
        blobs=blobs,
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts()},
    ))
    assert finder.find("o/r", "head", TARGETS, "tok") is None


def test_a_run_built_from_a_different_pinned_version_is_refused(monkeypatch):
    """The versions live in the workflow, so they are extracted rather than hashed."""
    hub = FakeHub(
        blobs=_same_inputs("head", "older"),
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts()},
    )
    _hub(monkeypatch, hub)
    wanted = finder.fingerprint("o/r", "head", "tok")
    hub.workflow_text = WORKFLOW_TEXT.replace("2.3.2.post1", "2.3.3")
    assert finder.fingerprint("o/r", "older", "tok") != wanted


def test_an_expired_artifact_is_refused(monkeypatch):
    _hub(monkeypatch, FakeHub(
        blobs=_same_inputs("head", "older"),
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts(expired=True)},
    ))
    assert finder.find("o/r", "head", TARGETS, "tok") is None


def test_a_run_missing_one_target_is_refused(monkeypatch):
    """Two runtimes certified and one rebuilt would be worse than rebuilding all."""
    _hub(monkeypatch, FakeHub(
        blobs=_same_inputs("head", "older"),
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts(TARGETS[:2])},
    ))
    assert finder.find("o/r", "head", TARGETS, "tok") is None


def test_an_api_that_will_not_answer_means_build(monkeypatch):
    hub = FakeHub(
        blobs=_same_inputs("head", "older"),
        runs={"build-cuda-wheels.yml": [{"id": 7, "head_sha": "older"}], "release.yml": []},
        artifacts={7: _artifacts()},
    )
    hub.failures.add("/artifacts")
    _hub(monkeypatch, hub)
    assert finder.find("o/r", "head", TARGETS, "tok") is None


def test_an_unfingerprintable_release_means_build(monkeypatch):
    hub = FakeHub(blobs=_same_inputs("head"), runs={}, artifacts={})
    hub.failures.add("/contents/")
    _hub(monkeypatch, hub)
    assert finder.find("o/r", "head", TARGETS, "tok") is None


def test_no_token_reuses_nothing(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--repo", "o/r", "--ref", "s", "--targets", "a"])
    assert finder.main() == 0
    assert "run_id=" in capsys.readouterr().out
