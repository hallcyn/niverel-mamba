"""The RunPod pod controller, against a fake API.

Worth testing rather than eyeballing, because the failure modes are financial
and I hit one of them by hand: `POST /pods` has **no required fields**, so an
empty body is accepted and RunPod rents a GPU from its own defaults. An unset
field is not a validation error, it is a machine of someone else's choosing,
billing.

So the payload builder is tested field by field, and teardown is tested on the
unhappy paths -- pod absent, never created, API unreachable -- because that is
exactly when it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

runpod_pod = pytest.importorskip("runpod_pod")

VALID = {
    "name": "niverel-mamba-certif-sm80-123",
    "arch": "sm80",
    "runner_token": "tok",
    "repo_url": "https://github.com/hallcyn/niverel-mamba",
    "image": runpod_pod.DEFAULT_IMAGE,
    "volume_gb": 20,
    "disk_gb": 40,
    "country_codes": ["US"],
}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")


@pytest.fixture
def fake_api(monkeypatch):
    state = {"pods": [], "calls": [], "next_id": "pod-1"}

    def _request(method, path, payload=None):
        state["calls"].append((method, path, payload))
        if method == "GET" and path == "/pods":
            return list(state["pods"])
        if method == "POST" and path == "/pods":
            pod = {"id": state["next_id"], "name": payload["name"],
                   "desiredStatus": "RUNNING", "costPerHr": 1.19,
                   "machine": {"gpuTypeId": "NVIDIA A100 80GB PCIe", "dataCenterId": "US-TX-1"}}
            state["pods"].append(pod)
            return pod
        if method == "DELETE":
            state["pods"] = [p for p in state["pods"] if p["id"] not in path]
            return {}
        raise AssertionError(f"unexpected {method} {path}")

    monkeypatch.setattr(runpod_pod, "_request", _request)
    return state


# --------------------------------------------------------------------------
# The payload. This is the part that cost money when it was absent.
# --------------------------------------------------------------------------


def test_the_payload_never_relies_on_api_defaults():
    """Every field that decides what gets rented must be set explicitly."""
    payload = runpod_pod._create_payload(**VALID)
    for field in ("name", "imageName", "gpuTypeIds", "gpuCount", "cloudType",
                  "containerDiskInGb", "volumeInGb", "interruptible"):
        assert field in payload, f"{field} left to the API default"
    assert payload["gpuTypeIds"], "an empty GPU list lets RunPod pick anything"


def test_the_payload_offers_every_gpu_of_the_right_capability():
    """H100 stock runs out by region; any H100 is sm_90."""
    assert len(runpod_pod._create_payload(**{**VALID, "arch": "sm90",
                                             "name": "niverel-mamba-certif-sm90-1"})["gpuTypeIds"]) >= 3
    assert all("A100" in g for g in runpod_pod._create_payload(**VALID)["gpuTypeIds"])


def test_the_pod_boots_into_an_ephemeral_runner():
    payload = runpod_pod._create_payload(**VALID)
    script = payload["dockerStartCmd"][-1]
    assert "--ephemeral" in script, "a persistent runner on a public repo is a standing risk"
    assert "--unattended" in script
    assert payload["env"]["RUNNER_LABELS"] == "self-hosted,linux,x64,cuda,sm80"


def test_spot_instances_are_refused():
    """A pod reclaimed mid-certification means paying for no report."""
    assert runpod_pod._create_payload(**VALID)["interruptible"] is False


def test_an_unknown_architecture_is_refused():
    with pytest.raises(runpod_pod.RunpodError, match="unknown architecture"):
        runpod_pod._create_payload(**{**VALID, "arch": "sm89"})


def test_a_pod_name_outside_our_namespace_is_refused():
    """The teardown guard matches on the prefix; a name outside it would be
    created and then never cleaned up."""
    with pytest.raises(runpod_pod.RunpodError, match="must start with"):
        runpod_pod._create_payload(**{**VALID, "name": "something-else"})


@pytest.mark.parametrize("missing", ["runner_token", "repo_url"])
def test_a_pod_with_no_way_to_register_is_refused(missing):
    """It would boot, bill, and never take a job."""
    with pytest.raises(runpod_pod.RunpodError, match="runner token"):
        runpod_pod._create_payload(**{**VALID, missing: ""})


def test_names_are_unique_per_run_and_architecture():
    assert runpod_pod.pod_name("sm80", "42") != runpod_pod.pod_name("sm90", "42")
    assert runpod_pod.pod_name("sm80", "42") != runpod_pod.pod_name("sm80", "43")


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_create_then_terminate(fake_api):
    runpod_pod.create(arch="sm80", run_id="123", runner_token="tok",
                      repo_url="https://github.com/hallcyn/niverel-mamba")
    assert len(fake_api["pods"]) == 1
    runpod_pod.terminate(arch="sm80", run_id="123")
    assert fake_api["pods"] == []


def test_terminate_is_safe_when_the_pod_was_never_created(fake_api):
    runpod_pod.terminate(arch="sm80", run_id="123")  # must not raise


def test_terminate_is_safe_when_the_api_is_unreachable(monkeypatch):
    monkeypatch.setattr(runpod_pod, "_request",
                        lambda *a, **k: (_ for _ in ()).throw(runpod_pod.RunpodError("down")))
    runpod_pod.terminate(arch="sm80", run_id="123")  # must not raise


def test_terminate_only_touches_this_run(fake_api):
    """Another release's pod, and anything else on the account, stay untouched."""
    fake_api["pods"] = [
        {"id": "a", "name": runpod_pod.pod_name("sm80", "111"), "desiredStatus": "RUNNING"},
        {"id": "b", "name": "niverel-5b-trainer-v1", "desiredStatus": "RUNNING"},
    ]
    runpod_pod.terminate(arch="sm80", run_id="222")
    assert {p["id"] for p in fake_api["pods"]} == {"a", "b"}


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


def test_the_guard_fails_while_one_of_our_pods_is_billing(fake_api, capsys):
    fake_api["pods"] = [{"id": "a", "name": runpod_pod.pod_name("sm90", "1"),
                         "desiredStatus": "RUNNING", "costPerHr": 2.69}]
    assert runpod_pod.assert_none_running() == 1
    assert "still running and billing" in capsys.readouterr().out


def test_the_guard_ignores_pods_that_are_not_ours(fake_api):
    """Training pods on the same account are none of this workflow's business."""
    fake_api["pods"] = [{"id": "b", "name": "niverel-5b-trainer-v1", "desiredStatus": "RUNNING"}]
    assert runpod_pod.assert_none_running() == 0


def test_the_guard_fails_closed_when_it_cannot_check(monkeypatch, capsys):
    """Unable to verify is not the same as verified clean."""
    monkeypatch.setattr(runpod_pod, "_request",
                        lambda *a, **k: (_ for _ in ()).throw(runpod_pod.RunpodError("down")))
    assert runpod_pod.assert_none_running() == 1
    assert "cannot verify" in capsys.readouterr().out


def test_the_guard_passes_once_everything_is_gone(fake_api):
    assert runpod_pod.assert_none_running() == 0


def test_missing_api_key_is_refused(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(runpod_pod.RunpodError, match="RUNPOD_API_KEY is not set"):
        runpod_pod._request("GET", "/pods")


@pytest.mark.parametrize("shape", [{"data": []}, {"pods": []}, []])
def test_list_pods_accepts_the_shapes_the_api_has_used(monkeypatch, shape):
    monkeypatch.setattr(runpod_pod, "_request", lambda *a, **k: shape)
    assert runpod_pod.list_pods() == []


def test_the_pod_must_offer_a_cuda_13_driver():
    """Two of the three runtimes are built against CUDA 13.0.

    Left unset, RunPod rents whatever is free. One pod came back advertising
    CUDA 12.4 and both cu130 runtimes died in `torch._C._cuda_init()` after the
    wheels had been installed; the pod before it advertised 12.8 and would have
    failed the same way had it got that far.
    """
    payload = runpod_pod._create_payload(**VALID)
    assert payload["allowedCudaVersions"] == ["13.0"], (
        "the pod request must constrain the driver, or the rental can be unusable"
    )
