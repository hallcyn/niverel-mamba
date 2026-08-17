#!/usr/bin/env python
"""Verify every release asset against its manifest before it goes anywhere.

Checks, for each manifest found in a directory tree:

* the manifest's schema version is one we understand;
* every artefact it names is actually present;
* every artefact's SHA-256 matches what the manifest recorded;
* nothing is present that the manifest does not describe.

Used by build-cuda-wheels (before cold-install), by certify-cuda-* (before
installing on a GPU box) and by release (before attaching anything).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BINARY_MANIFEST_SCHEMA = "niverel-mamba-binary-manifest-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path) -> list[str]:
    problems: list[str] = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema_version") != BINARY_MANIFEST_SCHEMA:
        return [f"{manifest_path}: unknown schema {manifest.get('schema_version')!r}"]

    directory = manifest_path.parent
    described = set()

    for artifact in manifest.get("artifacts", []):
        name = artifact["filename"]
        described.add(name)
        target = directory / name
        if not target.is_file():
            problems.append(f"{manifest_path}: {name} is described but missing")
            continue
        actual = sha256_file(target)
        if actual != artifact["sha256"]:
            problems.append(
                f"{manifest_path}: {name} SHA-256 mismatch\n"
                f"    manifest {artifact['sha256']}\n"
                f"    actual   {actual}"
            )
        else:
            print(f"  ok  {name}  {actual[:16]}...")

    for wheel in directory.glob("*.whl"):
        if wheel.name not in described:
            problems.append(
                f"{manifest_path}: {wheel.name} is present but not described by any manifest"
            )

    status = manifest.get("certification", {}).get("status")
    if status == "uncertified":
        print(f"  note: {manifest_path.parent.name} is marked uncertified "
              "(it must not be published until a GPU run signs off)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="succeed when no manifest is present (a core-only release)",
    )
    args = parser.parse_args()

    manifests = sorted(args.wheelhouse.rglob("manifest.json"))
    if not manifests:
        if args.allow_empty:
            print(f"no binary manifest under {args.wheelhouse}; core-only release")
            return 0
        print(f"::error::no manifest.json found under {args.wheelhouse}")
        return 1

    problems: list[str] = []
    for manifest in manifests:
        print(f"verifying {manifest}")
        problems.extend(verify(manifest))

    if problems:
        print("\nVERIFICATION FAILED\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"\nall {len(manifests)} manifest(s) verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
