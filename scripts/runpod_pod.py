#!/usr/bin/env python
"""Create and destroy a RunPod certification pod, on demand.

A pod exists only for the fifteen minutes a certification takes. Keeping two
idle pods would cost more in standing volume storage than the compute of every
release put together.

The pod boots straight into a GitHub self-hosted runner registered
`--ephemeral`, so it takes exactly one job and retires. That matters on a
public repository: the machine exists for the length of one workflow and is
gone.

    export RUNPOD_API_KEY=...
    python scripts/runpod_pod.py create --arch sm80 --run-id 123 \
        --runner-token <token> --repo-url https://github.com/hallcyn/niverel-mamba
    python scripts/runpod_pod.py terminate --arch sm80 --run-id 123
    python scripts/runpod_pod.py assert-none-running

**A note on the API that cost real money.** `POST /pods` has *no required
fields*: an empty body is accepted and RunPod rents a GPU using its own
defaults. `create()` therefore always sends an explicit name, image and GPU
list, and `_create_payload` is unit-tested to make sure it never degenerates
into something the API would happily fill in for us.
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

#: Every pod this project creates is named with this prefix, and the teardown
#: guard only ever considers pods that carry it. Other pods on the account --
#: training runs, anything else -- are never touched.
POD_PREFIX = "niverel-mamba-certif"

#: Ubuntu 24.04, hence Python 3.12, which is what the cp312 wheels require.
#: CUDA 12.8.1 also matches the cu128 reference runtime.
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

#: Every GPU of the right compute capability, not one model. H100 capacity is
#: routinely exhausted in a given region, and any H100 is sm_90, so offering
#: the whole family lets RunPod place the pod wherever there is stock.
GPUS_BY_ARCH: dict[str, list[str]] = {
    "sm80": [
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100-SXM4-40GB",
    ],
    "sm90": [
        "NVIDIA H100 80GB HBM3",
        "NVIDIA H100 NVL",
        "NVIDIA H100 PCIe",
    ],
}

RUNNING_STATES = frozenset({"RUNNING", "STARTING", "CREATED", "RESTARTING"})

#: Boots the pod straight into an ephemeral runner. `--ephemeral` is what makes
#: the machine single-use; `--unattended` stops config.sh waiting on a prompt
#: that nobody is there to answer.
RUNNER_BOOTSTRAP = r"""
set -euo pipefail
export RUNNER_ALLOW_RUNASROOT=1
mkdir -p /runner && cd /runner
VERSION=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
  | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -1)
curl -fsSL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz"
tar xzf runner.tar.gz
./config.sh --url "${REPO_URL}" --token "${RUNNER_TOKEN}" \
  --labels "${RUNNER_LABELS}" --name "${RUNNER_NAME}" --ephemeral --unattended --replace
./run.sh
"""


class RunpodError(RuntimeError):
    """The RunPod API refused, or answered something we cannot act on."""


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunpodError(
            "RUNPOD_API_KEY is not set. In CI it comes from the repository secret of "
            "the same name; locally, export it before running this script."
        )
    url = f"{API_ROOT.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RunpodError(
            f"{method} {url} -> HTTP {exc.code}: {exc.read().decode(errors='replace')[:400]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RunpodError(f"{method} {url} unreachable: {exc.reason}") from exc
    return json.loads(body) if body.strip() else {}


def list_pods() -> list[dict[str, Any]]:
    payload = _request("GET", "/pods")
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("pods") or []
    if not isinstance(payload, list):
        raise RunpodError(f"unexpected /pods response: {str(payload)[:200]}")
    return payload


def _state(pod: dict[str, Any]) -> str:
    return str(pod.get("desiredStatus") or pod.get("status") or "UNKNOWN").upper()


def pod_name(arch: str, run_id: str) -> str:
    """One pod per architecture per workflow run, so two releases cannot collide."""
    return f"{POD_PREFIX}-{arch}-{run_id}"


def _create_payload(
    *,
    name: str,
    arch: str,
    runner_token: str,
    repo_url: str,
    image: str,
    volume_gb: int,
    disk_gb: int,
    country_codes: list[str],
) -> dict[str, Any]:
    """Build the creation body. Never returns anything the API would fill in.

    Unit-tested, because `POST /pods` accepts an empty body and rents a GPU
    from its own defaults -- an unset field here is not a validation error, it
    is a machine of RunPod's choosing, billing.
    """
    if arch not in GPUS_BY_ARCH:
        raise RunpodError(f"unknown architecture {arch!r}; known: {sorted(GPUS_BY_ARCH)}")
    if not name.startswith(POD_PREFIX):
        raise RunpodError(f"pod name {name!r} must start with {POD_PREFIX!r}")
    if not runner_token or not repo_url:
        raise RunpodError("a runner token and repository URL are required")
    return {
        "name": name,
        "imageName": image,
        "gpuTypeIds": GPUS_BY_ARCH[arch],
        "gpuCount": 1,
        "cloudType": "SECURE",
        "countryCodes": country_codes,
        "containerDiskInGb": disk_gb,
        "volumeInGb": volume_gb,
        # Not interruptible: a spot pod reclaimed mid-certification would mean
        # paying for a run that produces no report.
        "interruptible": False,
        "ports": [],
        "env": {
            "REPO_URL": repo_url,
            "RUNNER_TOKEN": runner_token,
            "RUNNER_LABELS": f"self-hosted,linux,x64,cuda,{arch}",
            "RUNNER_NAME": name,
        },
        "dockerStartCmd": ["bash", "-lc", RUNNER_BOOTSTRAP],
    }


def create(
    *,
    arch: str,
    run_id: str,
    runner_token: str,
    repo_url: str,
    image: str = DEFAULT_IMAGE,
    volume_gb: int = 20,
    disk_gb: int = 40,
    country_codes: list[str] | None = None,
) -> dict[str, Any]:
    name = pod_name(arch, run_id)
    payload = _create_payload(
        name=name,
        arch=arch,
        runner_token=runner_token,
        repo_url=repo_url,
        image=image,
        volume_gb=volume_gb,
        disk_gb=disk_gb,
        country_codes=country_codes or ["US"],
    )
    print(f"creating {name} ({arch}: {', '.join(GPUS_BY_ARCH[arch])})")
    pod = _request("POST", "/pods", payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise RunpodError(f"pod created but no id returned: {str(pod)[:200]}")
    machine = pod.get("machine") or {}
    print(
        f"created {pod_id} on {machine.get('gpuTypeId')} "
        f"in {machine.get('dataCenterId')} at ${pod.get('costPerHr')}/hr"
    )
    return pod


def terminate(*, arch: str, run_id: str) -> None:
    """Destroy the pod. Never raises.

    Runs in an `if: always()` job, so it must succeed on every path where
    something else already went wrong -- including the pod never having been
    created.
    """
    name = pod_name(arch, run_id)
    try:
        pods = [p for p in list_pods() if p.get("name") == name]
    except RunpodError as exc:
        print(f"::warning::could not list pods to terminate {name}: {exc}")
        return
    if not pods:
        print(f"no pod named {name}; nothing to terminate")
        return
    for pod in pods:
        try:
            _request("DELETE", f"/pods/{pod['id']}")
            print(f"terminated {name} ({pod['id']})")
        except RunpodError as exc:
            print(f"::error::failed to terminate {pod['id']}: {exc}")


def assert_none_running(prefix: str = POD_PREFIX) -> int:
    """Fail if any certification pod is still alive.

    The last line of defence. Only pods carrying our prefix are considered:
    someone else's training run is not this workflow's business.
    """
    try:
        pods = list_pods()
    except RunpodError as exc:
        print(f"::error::cannot verify pods were cleaned up: {exc}")
        return 1
    alive = [
        f"{p.get('name')} ({p.get('id')}, {_state(p)}, ${p.get('costPerHr')}/hr)"
        for p in pods
        if str(p.get("name", "")).startswith(prefix) and _state(p) in RUNNING_STATES
    ]
    if alive:
        print("::error::certification pods still running and billing: " + ", ".join(alive))
        return 1
    print(f"no pod matching {prefix!r} is running")
    return 0


def wait_until_running(*, arch: str, run_id: str, timeout: int = 900) -> None:
    name = pod_name(arch, run_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        matching = [p for p in list_pods() if p.get("name") == name]
        if matching and _state(matching[0]) == "RUNNING":
            print(f"{name} is RUNNING")
            return
        print(f"  waiting, state={_state(matching[0]) if matching else 'absent'}")
        time.sleep(15)
    raise RunpodError(
        f"{name} did not reach RUNNING within {timeout}s; it may be billing. "
        "The teardown job will destroy it."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arch", required=True, choices=sorted(GPUS_BY_ARCH))
    common.add_argument("--run-id", required=True)

    make = sub.add_parser("create", parents=[common])
    make.add_argument("--runner-token", required=True)
    make.add_argument("--repo-url", required=True)
    make.add_argument("--image", default=DEFAULT_IMAGE)
    make.add_argument("--volume-gb", type=int, default=20)
    make.add_argument("--disk-gb", type=int, default=40)
    make.add_argument("--countries", default="US")
    make.add_argument("--wait", action="store_true")

    sub.add_parser("terminate", parents=[common])
    guard = sub.add_parser("assert-none-running")
    guard.add_argument("--prefix", default=POD_PREFIX)

    args = parser.parse_args()
    try:
        if args.command == "create":
            create(
                arch=args.arch,
                run_id=args.run_id,
                runner_token=args.runner_token,
                repo_url=args.repo_url,
                image=args.image,
                volume_gb=args.volume_gb,
                disk_gb=args.disk_gb,
                country_codes=[c for c in args.countries.split(",") if c],
            )
            if args.wait:
                wait_until_running(arch=args.arch, run_id=args.run_id)
        elif args.command == "terminate":
            terminate(arch=args.arch, run_id=args.run_id)
        else:
            return assert_none_running(args.prefix)
    except RunpodError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
