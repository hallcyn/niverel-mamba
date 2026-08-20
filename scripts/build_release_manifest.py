#!/usr/bin/env python
"""Build the index `niverel-mamba install-backend cuda` downloads.

Release assets are a flat namespace and all three runtimes ship wheels with
identical filenames, so the wheels are attached as one zip per runtime. That
makes the per-runtime build manifests insufficient on their own: their
`url` fields are null, because at build time nothing knows where the artefact
will end up, and `install-backend` had no way to find anything.

This runs at release time, when the answer is known, and writes a single index
naming each runtime's archive, its URL, its SHA-256, and the SHA-256 of every
wheel inside it -- so a client can verify the download before unpacking it and
verify each wheel before installing it.

    python scripts/build_release_manifest.py --wheelhouse cuda-wheels \
        --assets assets --tag v0.1.1 --repo hallcyn/niverel-mamba \
        --output assets/cuda-manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

CUDA_INDEX_SCHEMA = "niverel-mamba-cuda-index-v1"
BINARY_MANIFEST_SCHEMA = "niverel-mamba-binary-manifest-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(wheelhouse: Path, assets: Path, tag: str, repo: str) -> dict[str, object]:
    runtimes = []
    for manifest_path in sorted(wheelhouse.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BINARY_MANIFEST_SCHEMA:
            raise SystemExit(f"{manifest_path}: unexpected schema {manifest.get('schema_version')!r}")

        # cuda-wheels/cuda-wheels-torch211-cu128/manifest.json -> torch211-cu128
        name = manifest_path.parent.name.replace("cuda-wheels-", "")
        archive = assets / f"niverel-mamba-cuda-{name}.zip"
        if not archive.is_file():
            raise SystemExit(f"no archive for runtime {name!r}: {archive} is missing")

        artifacts = manifest.get("artifacts") or []
        if not artifacts:
            raise SystemExit(f"{manifest_path}: describes no artifact")
        first = artifacts[0]

        runtimes.append(
            {
                "name": name,
                "torch_version": first["torch_version"],
                "torch_cuda": first["torch_cuda"],
                "python_tag": first["python_tag"],
                "platform": first.get("platform"),
                "architectures": first.get("architectures", []),
                "source_repository": manifest.get("source_repository"),
                "source_commit": manifest.get("source_commit"),
                "build_workflow": manifest.get("build_workflow"),
                "archive": {
                    "filename": archive.name,
                    "url": f"https://github.com/{repo}/releases/download/{tag}/{archive.name}",
                    "sha256": sha256_file(archive),
                    "size_bytes": archive.stat().st_size,
                },
                "wheels": [
                    {
                        "package": a["package"],
                        "package_version": a["package_version"],
                        "filename": a["filename"],
                        "sha256": a["sha256"],
                    }
                    for a in artifacts
                ],
            }
        )

    if not runtimes:
        raise SystemExit(f"no runtime manifest found under {wheelhouse}")

    return {
        "schema_version": CUDA_INDEX_SCHEMA,
        "release": tag,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "runtimes": runtimes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = build(args.wheelhouse, args.assets, args.tag, args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    runtimes = index["runtimes"]
    assert isinstance(runtimes, list)
    print(f"wrote {args.output} describing {len(runtimes)} runtimes:")
    for runtime in runtimes:
        print(f"  {runtime['name']:16s} torch {runtime['torch_version']} / CUDA {runtime['torch_cuda']}"
              f"  {runtime['archive']['filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
