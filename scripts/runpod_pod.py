#!/usr/bin/env python
"""Start and stop a pre-created RunPod pod, by name.

The certification pods are created once, by hand, and then only started and
stopped: `niverel-mamba-certif-a100` for sm_80 and `niverel-mamba-certif-h100`
for sm_90. Creating them from CI would mean encoding image, template, region
and disk size in a workflow and re-deriving them on every run; starting an
existing pod is one call and leaves the machine's configuration where a human
can see it.

Everything here is deliberately defensive, because the failure that matters is
not "the pod would not start" -- it is "the pod started and nothing stopped
it". A GPU left running bills by the hour whether or not anyone is watching.

    export RUNPOD_API_KEY=...
    python scripts/runpod_pod.py status niverel-mamba-certif-a100
    python scripts/runpod_pod.py start  niverel-mamba-certif-a100 --wait
    python scripts/runpod_pod.py stop   niverel-mamba-certif-a100
    python scripts/runpod_pod.py assert-all-stopped
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API_ROOT = os.environ.get("RUNPOD_API_ROOT", "https://rest.runpod.io/v1")

#: States RunPod reports for a pod that is costing GPU-time.
RUNNING_STATES = frozenset({"RUNNING", "STARTING", "CREATED", "RESTARTING"})
STOPPED_STATES = frozenset({"EXITED", "STOPPED", "TERMINATED"})


class RunpodError(RuntimeError):
    """The RunPod API refused, or answered something we cannot act on."""


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunpodError(
            "RUNPOD_API_KEY is not set. In CI it comes from the repository secret "
            "of the same name; locally, export it before running this script."
        )
    url = f"{API_ROOT.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise RunpodError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RunpodError(f"{method} {url} unreachable: {exc.reason}") from exc
    return json.loads(body) if body.strip() else {}


def list_pods() -> list[dict[str, Any]]:
    payload = _request("GET", "/pods")
    # The API has returned both a bare list and {"data": [...]} across versions;
    # accept either rather than break on a shape change.
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("pods") or []
    if not isinstance(payload, list):
        raise RunpodError(f"unexpected /pods response: {str(payload)[:200]}")
    return payload


def _state(pod: dict[str, Any]) -> str:
    return str(pod.get("desiredStatus") or pod.get("status") or "UNKNOWN").upper()


def find_pod(name: str) -> dict[str, Any]:
    pods = list_pods()
    matches = [p for p in pods if p.get("name") == name]
    if not matches:
        known = ", ".join(sorted(str(p.get("name")) for p in pods)) or "none"
        raise RunpodError(
            f"no RunPod pod named {name!r}. Pods on this account: {known}. "
            "The certification pods are created by hand and only started here."
        )
    if len(matches) > 1:
        raise RunpodError(f"{len(matches)} pods are named {name!r}; names must be unique")
    return matches[0]


def start(name: str, wait: bool, timeout: int) -> dict[str, Any]:
    pod = find_pod(name)
    pod_id, state = pod["id"], _state(pod)
    if state in RUNNING_STATES:
        print(f"{name} is already {state}")
    else:
        print(f"starting {name} ({pod_id}), currently {state}")
        _request("POST", f"/pods/{pod_id}/start")
    if not wait:
        return pod
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = _state(find_pod(name))
        if current == "RUNNING":
            print(f"{name} is RUNNING")
            return find_pod(name)
        print(f"  waiting, state={current}")
        time.sleep(10)
    raise RunpodError(
        f"{name} did not reach RUNNING within {timeout}s. It may still be starting and "
        "billing: stop it manually before retrying."
    )


def stop(name: str) -> None:
    """Stop a pod. Never raises for an already-stopped pod.

    This runs in an `if: always()` job, so it must succeed on the paths where
    something else has already gone wrong -- including the pod never having
    started.
    """
    try:
        pod = find_pod(name)
    except RunpodError as exc:
        print(f"could not look up {name}: {exc}")
        print("nothing to stop")
        return
    state = _state(pod)
    if state in STOPPED_STATES:
        print(f"{name} is already {state}")
        return
    print(f"stopping {name} ({pod['id']}), currently {state}")
    _request("POST", f"/pods/{pod['id']}/stop")


def assert_all_stopped(names: list[str]) -> int:
    """Fail loudly if any certification pod is still billing.

    The last line of defence: if teardown silently failed, this turns a slow
    financial leak into a red workflow.
    """
    running = []
    for pod in list_pods():
        name = str(pod.get("name", ""))
        if names and name not in names:
            continue
        if _state(pod) in RUNNING_STATES:
            running.append(f"{name} ({_state(pod)})")
    if running:
        print("::error::these pods are still running and billing: " + ", ".join(running))
        return 1
    print("no certification pod is running")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("start", "stop", "status"):
        p = sub.add_parser(name)
        p.add_argument("pod")
        if name == "start":
            p.add_argument("--wait", action="store_true")
            p.add_argument("--timeout", type=int, default=600)

    guard = sub.add_parser("assert-all-stopped")
    guard.add_argument("--names", default="niverel-mamba-certif-a100,niverel-mamba-certif-h100")

    args = parser.parse_args()
    try:
        if args.command == "start":
            start(args.pod, args.wait, args.timeout)
        elif args.command == "stop":
            stop(args.pod)
        elif args.command == "status":
            print(_state(find_pod(args.pod)))
        else:
            return assert_all_stopped([n for n in args.names.split(",") if n])
    except RunpodError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
