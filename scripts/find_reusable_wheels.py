#!/usr/bin/env python
"""Find a past run whose CUDA wheels are still valid for this release.

The wheels are upstream's, at pinned versions, compiled from Dockerfiles that
change rarely. A release that only touches this package's own code rebuilds a
byte-for-byte equivalent artefact and spends seventy minutes per runtime doing
it. So: look for a run that already built them from *the same inputs*, and
reuse it.

"The same inputs" is decided by fingerprint, not by trust. What determines what
comes out of the build is:

* the Dockerfiles, which define the base image, the torch build and the
  compiler flags;
* `.github/cuda-targets.json`, which defines the matrix -- torch version, CUDA
  version, python tag and the architectures nvcc targets;
* the pinned upstream versions in the build workflow.

Each is read at the released commit and at the candidate run's commit through
the GitHub API, so no extra metadata has to have been recorded at build time
and runs from before this script existed remain reusable.

It fails safe in every direction: any doubt -- a missing artifact, an expired
one, a fingerprint that cannot be computed, an API that will not answer --
returns nothing, and nothing means build. The cost of being wrong that way is
seventy minutes; the cost of the other way is certifying wheels nobody built
from this source.

    python scripts/find_reusable_wheels.py --repo owner/name --ref <sha> \
        --targets torch211-cu128,torch212-cu130,torch213-cu130
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any

API = "https://api.github.com"

#: Files whose content decides what the wheels contain.
FINGERPRINTED = (
    "docker/torch211-cu128.Dockerfile",
    "docker/torch212-cu130.Dockerfile",
    "docker/torch213-cu130.Dockerfile",
    ".github/cuda-targets.json",
)

#: The pinned upstream versions live in the build workflow rather than in a file
#: of their own, so they are extracted rather than hashed with it -- that file
#: also changes for reasons that cannot affect a wheel, such as where its layer
#: cache is stored.
PINS = re.compile(r"^\s*(MAMBA_SSM_VERSION|CAUSAL_CONV1D_VERSION)=(\S+)\s*$", re.M)
BUILD_WORKFLOW = ".github/workflows/build-cuda-wheels.yml"


def _get(path: str, token: str) -> Any:
    request = urllib.request.Request(f"{API}{path}")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def fingerprint(repo: str, ref: str, token: str) -> str | None:
    """Blob SHAs of the build inputs, plus the pinned versions, at one commit."""
    parts = []
    try:
        for path in FINGERPRINTED:
            entry = _get(f"/repos/{repo}/contents/{path}?ref={ref}", token)
            parts.append(f"{path}={entry['sha']}")

        entry = _get(f"/repos/{repo}/contents/{BUILD_WORKFLOW}?ref={ref}", token)
        import base64

        text = base64.b64decode(entry["content"]).decode("utf-8", "replace")
        pins = sorted(f"{name}={value}" for name, value in PINS.findall(text))
        if not pins:
            print(f"  no pinned versions found in {BUILD_WORKFLOW} at {ref[:8]}", file=sys.stderr)
            return None
        parts.extend(pins)
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as exc:
        print(f"  cannot fingerprint {ref[:8]}: {exc}", file=sys.stderr)
        return None
    return "\n".join(parts)


def _usable_artifacts(repo: str, run_id: int, targets: list[str], token: str) -> bool:
    try:
        payload = _get(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100", token)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"  cannot list artifacts of {run_id}: {exc}", file=sys.stderr)
        return False
    alive = {
        artifact["name"]
        for artifact in payload.get("artifacts", [])
        if not artifact.get("expired")
    }
    missing = [f"cuda-wheels-{t}" for t in targets if f"cuda-wheels-{t}" not in alive]
    if missing:
        print(f"  run {run_id} is missing {', '.join(missing)}", file=sys.stderr)
        return False
    return True


def find(repo: str, ref: str, targets: list[str], token: str, limit: int = 40) -> int | None:
    wanted = fingerprint(repo, ref, token)
    if wanted is None:
        return None

    seen: dict[str, str | None] = {}
    for workflow in ("build-cuda-wheels.yml", "release.yml"):
        try:
            runs = _get(
                f"/repos/{repo}/actions/workflows/{workflow}/runs"
                f"?status=completed&per_page={limit}",
                token,
            )
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            print(f"  cannot list runs of {workflow}: {exc}", file=sys.stderr)
            continue

        for run in runs.get("workflow_runs", []):
            head = run.get("head_sha")
            if not head:
                continue
            if head not in seen:
                seen[head] = fingerprint(repo, head, token)
            if seen[head] != wanted:
                continue
            if not _usable_artifacts(repo, run["id"], targets, token):
                continue
            print(
                f"  reusing run {run['id']} ({workflow}, {head[:8]}): "
                f"the build inputs are unchanged",
                file=sys.stderr,
            )
            return int(run["id"])
    print("  no run has usable wheels for these inputs; they will be built", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--targets", required=True)
    args = parser.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        print("  no token; not reusing anything", file=sys.stderr)
        run_id = None
    else:
        run_id = find(args.repo, args.ref, [t for t in args.targets.split(",") if t], token)

    output = os.environ.get("GITHUB_OUTPUT")
    line = f"run_id={run_id or ''}"
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
