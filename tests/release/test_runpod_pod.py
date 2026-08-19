"""The RunPod pod controller, tested against a fake API.

Worth testing rather than eyeballing, because the failure mode is financial:
a pod that starts and is never stopped bills by the hour. The paths that matter
most are the unhappy ones -- stop() must succeed when the pod is missing, when
it never started, and when the API is unreachable, because it runs in an
`if: always()` job precisely on those paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

runpod_pod = pytest.importorskip("runpod_pod")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the HTTP layer, recording every call."""
    state = {"pods": [], "calls": []}

    def _request(method, path, payload=None):
        state["calls"].append((method, path))
        if method == "GET" and path == "/pods":
            return list(state["pods"])
        if path.endswith("/start"):
            for pod in state["pods"]:
                if pod["id"] in path:
                    pod["desiredStatus"] = "RUNNING"
            return {}
        if path.endswith("/stop"):
            for pod in state["pods"]:
                if pod["id"] in path:
                    pod["desiredStatus"] = "EXITED"
            return {}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(runpod_pod, "_request", _request)
    return state


def _pod(name="niverel-mamba-certif-a100", status="EXITED", pod_id="abc123"):
    return {"id": pod_id, "name": name, "desiredStatus": status}


def test_missing_api_key_is_refused(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(runpod_pod.RunpodError, match="RUNPOD_API_KEY is not set"):
        runpod_pod._request("GET", "/pods")


def test_start_then_stop(fake_api):
    fake_api["pods"] = [_pod()]
    runpod_pod.start("niverel-mamba-certif-a100", wait=False, timeout=1)
    assert fake_api["pods"][0]["desiredStatus"] == "RUNNING"
    runpod_pod.stop("niverel-mamba-certif-a100")
    assert fake_api["pods"][0]["desiredStatus"] == "EXITED"


def test_starting_an_already_running_pod_is_not_an_error(fake_api):
    fake_api["pods"] = [_pod(status="RUNNING")]
    runpod_pod.start("niverel-mamba-certif-a100", wait=False, timeout=1)
    assert not any(p.endswith("/start") for _, p in fake_api["calls"])


def test_stop_is_safe_when_the_pod_does_not_exist(fake_api):
    """`stop` runs in an always() job, including when start never ran."""
    fake_api["pods"] = []
    runpod_pod.stop("niverel-mamba-certif-a100")  # must not raise


def test_stop_is_safe_when_the_api_is_unreachable(monkeypatch):
    def _boom(*_a, **_k):
        raise runpod_pod.RunpodError("network down")

    monkeypatch.setattr(runpod_pod, "_request", _boom)
    runpod_pod.stop("niverel-mamba-certif-a100")  # must not raise


def test_stop_does_nothing_for_an_already_stopped_pod(fake_api):
    fake_api["pods"] = [_pod(status="EXITED")]
    runpod_pod.stop("niverel-mamba-certif-a100")
    assert not any(p.endswith("/stop") for _, p in fake_api["calls"])


def test_an_unknown_pod_name_lists_what_exists(fake_api):
    fake_api["pods"] = [_pod(name="something-else")]
    with pytest.raises(runpod_pod.RunpodError, match="something-else"):
        runpod_pod.find_pod("niverel-mamba-certif-a100")


def test_duplicate_names_are_refused(fake_api):
    fake_api["pods"] = [_pod(pod_id="a"), _pod(pod_id="b")]
    with pytest.raises(runpod_pod.RunpodError, match="names must be unique"):
        runpod_pod.find_pod("niverel-mamba-certif-a100")


def test_the_guard_fails_while_a_pod_is_still_billing(fake_api, capsys):
    fake_api["pods"] = [_pod(status="RUNNING")]
    assert runpod_pod.assert_all_stopped(["niverel-mamba-certif-a100"]) == 1
    assert "still running and billing" in capsys.readouterr().out


def test_the_guard_passes_once_everything_is_stopped(fake_api):
    fake_api["pods"] = [_pod(status="EXITED")]
    assert runpod_pod.assert_all_stopped(["niverel-mamba-certif-a100"]) == 0


def test_the_guard_ignores_unrelated_pods(fake_api):
    """Someone else's pod running is not this workflow's problem."""
    fake_api["pods"] = [_pod(name="unrelated-training-run", status="RUNNING")]
    assert runpod_pod.assert_all_stopped(["niverel-mamba-certif-a100"]) == 0


def test_start_with_wait_times_out_loudly(fake_api, monkeypatch):
    """A pod stuck in STARTING must fail the job, not hang: it is billing."""
    monkeypatch.setattr(runpod_pod.time, "sleep", lambda _s: None)
    fake_api["pods"] = [_pod(status="EXITED")]

    def _request(method, path, payload=None):
        if method == "GET":
            return [_pod(status="STARTING")]
        return {}

    monkeypatch.setattr(runpod_pod, "_request", _request)
    with pytest.raises(runpod_pod.RunpodError, match="did not reach RUNNING"):
        runpod_pod.start("niverel-mamba-certif-a100", wait=True, timeout=0.1)


@pytest.mark.parametrize("shape", [
    {"data": [_pod()]},
    {"pods": [_pod()]},
    [_pod()],
])
def test_list_pods_accepts_the_shapes_the_api_has_used(monkeypatch, shape):
    monkeypatch.setattr(runpod_pod, "_request", lambda *_a, **_k: shape)
    assert runpod_pod.list_pods()[0]["name"] == "niverel-mamba-certif-a100"
