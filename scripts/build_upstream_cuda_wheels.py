#!/usr/bin/env python
"""Build and describe the upstream CUDA wheels.

Two subcommands:

``build``
    Drive the Docker build for one target. Requires Docker with buildx and a
    CUDA toolkit image; no GPU is needed to *compile*.

``manifest``
    Emit a ``niverel-mamba-binary-manifest-v1`` document describing a
    wheelhouse: every artefact's SHA-256, the source commit it was built from,
    and the workflow that produced it. The CLI downloads this manifest first
    and verifies the wheel's SHA against it before installing anything.

Nothing here publishes. Publication requires a GPU certification run to have
passed on these exact artefacts -- see ``certify-cuda-sm80.yml``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

BINARY_MANIFEST_SCHEMA = "niverel-mamba-binary-manifest-v1"

TARGETS: dict[str, dict[str, Any]] = {
    "torch211-cu128": {
        "dockerfile": "docker/torch211-cu128.Dockerfile",
        "torch_version": "2.11.0",
        "torch_cuda": "12.8",
        "note": "the Niverel Foundation V3 training runtime",
    },
    "torch212-cu130": {
        "dockerfile": "docker/torch212-cu130.Dockerfile",
        "torch_version": "2.12.1",
        "torch_cuda": "13.0",
    },
    "torch213-cu130": {
        "dockerfile": "docker/torch213-cu130.Dockerfile",
        "torch_version": "2.13.0",
        "torch_cuda": "13.0",
    },
}

DEFAULT_ARCHITECTURES = ("sm_80", "sm_90")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_of(filename: str) -> tuple[str, str]:
    """Split ``mamba_ssm-2.3.2.post1-cp312-...whl`` into name and version."""
    stem = filename.split("-")
    name = stem[0].replace("_", "-")
    version = stem[1] if len(stem) > 1 else "unknown"
    return name, version


def cmd_build(args: argparse.Namespace) -> int:
    targets = list(TARGETS) if args.target == "all" else [args.target]
    for name in targets:
        target = TARGETS[name]
        out = REPO_ROOT / f"wheelhouse-{name}"
        out.mkdir(parents=True, exist_ok=True)
        command = [
            "docker", "buildx", "build",
            "--file", str(REPO_ROOT / target["dockerfile"]),
            "--build-arg", f"TORCH_CUDA_ARCH_LIST={args.arch_list}",
            "--build-arg", f"MAMBA_SSM_VERSION={args.mamba_ssm_version}",
            "--build-arg", f"CAUSAL_CONV1D_VERSION={args.causal_conv1d_version}",
            "--output", f"type=local,dest={out}",
            str(REPO_ROOT),
        ]
        print("+ " + " ".join(command))
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


def build_manifest(
    wheelhouse: Path,
    *,
    torch_version: str,
    torch_cuda: str,
    python_tag: str,
    architectures: tuple[str, ...],
    platform: str,
    cxx11_abi: bool,
    source_repository: str,
    source_commit: str | None,
    build_workflow: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheelhouse}")

    artifacts = []
    for wheel in wheels:
        package, version = _package_of(wheel.name)
        artifacts.append(
            {
                "package": package,
                "package_version": version,
                "filename": wheel.name,
                "url": f"{base_url.rstrip('/')}/{wheel.name}" if base_url else None,
                "sha256": sha256_file(wheel),
                "size_bytes": wheel.stat().st_size,
                "torch_version": torch_version,
                "torch_cuda": torch_cuda,
                "python_tag": python_tag,
                "platform": platform,
                "cxx11_abi": cxx11_abi,
                "architectures": list(architectures),
            }
        )

    return {
        "schema_version": BINARY_MANIFEST_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_repository": source_repository,
        "source_commit": source_commit,
        "build_workflow": build_workflow,
        "certification": {
            "status": "uncertified",
            "note": (
                "These wheels have not yet been run on a GPU. They must not be "
                "published until certify-cuda-sm80 / sm90 has produced a passing "
                "report against them."
            ),
        },
        "artifacts": artifacts,
    }


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        args.wheelhouse,
        torch_version=args.torch_version,
        torch_cuda=args.torch_cuda,
        python_tag=args.python_tag,
        architectures=tuple(a.strip() for a in args.architectures.split(",") if a.strip()),
        platform=args.platform,
        cxx11_abi=args.cxx11_abi,
        source_repository=args.source_repository,
        source_commit=args.source_commit,
        build_workflow=args.build_workflow,
        base_url=args.base_url,
    )
    payload = json.dumps(manifest, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(payload)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build wheels in a pinned CUDA container")
    build.add_argument("--target", default="all", choices=["all", *TARGETS])
    build.add_argument("--arch-list", default="8.0;9.0")
    build.add_argument("--mamba-ssm-version", default="2.3.2.post1")
    build.add_argument("--causal-conv1d-version", default="1.6.2.post1")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=cmd_build)

    manifest = sub.add_parser("manifest", help="describe a wheelhouse")
    manifest.add_argument("--wheelhouse", type=Path, required=True)
    manifest.add_argument("--torch-version", required=True)
    manifest.add_argument("--torch-cuda", required=True)
    manifest.add_argument("--python-tag", default="cp312")
    manifest.add_argument("--architectures", default=",".join(DEFAULT_ARCHITECTURES))
    manifest.add_argument("--platform", default="manylinux_2_28_x86_64")
    manifest.add_argument("--cxx11-abi", action="store_true", default=True)
    manifest.add_argument("--source-repository", default="https://github.com/state-spaces/mamba")
    manifest.add_argument("--source-commit", default=None)
    manifest.add_argument("--build-workflow", default=None)
    manifest.add_argument("--base-url", default=None)
    manifest.add_argument("--output", type=Path, default=None)
    manifest.set_defaults(func=cmd_manifest)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
