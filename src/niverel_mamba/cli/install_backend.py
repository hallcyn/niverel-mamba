"""``niverel-mamba install-backend`` -- fetch certified CUDA wheels.

Design constraints:

* **Nothing happens at import.** No download, no subprocess, no compilation
  is triggered by importing this package. Everything here runs only when the
  user invokes this command.
* **Plan by default.** Without ``--yes`` the command prints what it *would*
  do and exits. Installation requires explicit consent.
* **Manifest first, then SHA.** The build manifest is fetched and its
  recorded SHA-256 checked against the downloaded wheel before anything is
  installed. A mismatch aborts.
* **No local compilation, ever.** If no prebuilt wheel matches this
  environment, the command says so and stops. It does not fall back to
  building from source on the user's machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from ..capabilities import UPSTREAM_RUNTIME_REQUIREMENTS, detect_environment
from ..errors import AssetVerificationError
from ..version import __version__

__all__ = ["BINARY_MANIFEST_SCHEMA", "add_parser", "run", "verify_sha256"]

BINARY_MANIFEST_SCHEMA = "niverel-mamba-binary-manifest-v1"

DEFAULT_MANIFEST_URL = (
    "https://github.com/hallcyn/niverel-mamba/releases/download/v{version}/cuda-manifest.json"
)


def add_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser(
        "install-backend", help="install a certified prebuilt backend (plan-only by default)"
    )
    parser.add_argument("backend", choices=["cuda"], help="which backend to install")
    parser.add_argument("--yes", action="store_true", help="actually install (default: plan only)")
    parser.add_argument("--manifest", default=None, help="manifest URL or local path")
    parser.add_argument("--dest", type=Path, default=None, help="where to cache downloads")
    parser.set_defaults(func=run)
    return parser


def verify_sha256(path: Path, expected: str) -> str:
    """Hash a file and refuse it unless it matches the manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise AssetVerificationError(
            f"SHA-256 mismatch for {path.name}:\n  expected {expected}\n  actual   {actual}\n"
            "Refusing to install an asset that does not match its manifest."
        )
    return actual


def _fetch(location: str) -> bytes:
    if location.startswith(("http://", "https://")):
        with urllib.request.urlopen(location) as response:
            return bytes(response.read())
    return Path(location).read_bytes()


def _environment_key(env: Any) -> dict[str, Any]:
    import torch

    capability = None
    if env.cuda.available:
        try:
            major, minor = torch.cuda.get_device_capability(0)
            capability = f"sm_{major}{minor}"
        except Exception:  # pragma: no cover
            capability = None
    return {
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_version": torch.__version__.split("+")[0],
        "torch_cuda": getattr(torch.version, "cuda", None),
        "platform": f"{env.platform_system.lower()}_{env.platform_machine}",
        "architecture": capability,
    }


def _match(manifest: dict[str, Any], key: dict[str, Any]) -> list[dict[str, Any]]:
    matches = []
    for artifact in manifest.get("artifacts", []):
        if artifact.get("python_tag") != key["python_tag"]:
            continue
        if artifact.get("torch_version") != key["torch_version"]:
            continue
        if artifact.get("torch_cuda") != key["torch_cuda"]:
            continue
        if key["architecture"] and key["architecture"] not in artifact.get("architectures", []):
            continue
        matches.append(artifact)
    return matches


def run(args: argparse.Namespace) -> int:
    env = detect_environment()
    if not env.torch.available:
        print("PyTorch is not installed. Install the CUDA build of torch first:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 1

    key = _environment_key(env)
    print("Detected:")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"  Torch {env.torch.version}")
    print(f"  Torch CUDA {key['torch_cuda'] or 'none'}")
    print(f"  {env.platform_system} {env.platform_machine}")
    print(f"  GPU capability {key['architecture'] or 'none detected'}")
    print()

    location = args.manifest or DEFAULT_MANIFEST_URL.format(version=__version__)
    try:
        manifest = json.loads(_fetch(location))
    except Exception as exc:
        print(f"Could not read the build manifest at {location}:\n  {exc}")
        return 1

    if manifest.get("schema_version") != BINARY_MANIFEST_SCHEMA:
        print(
            f"Unknown manifest schema {manifest.get('schema_version')!r}; "
            f"this build understands {BINARY_MANIFEST_SCHEMA!r}."
        )
        return 1

    matches = _match(manifest, key)
    if not matches:
        print("No certified prebuilt wheel matches this environment.")
        print()
        print("This command will not compile mamba-ssm on your machine. Either install a")
        print("supported Torch/CUDA combination, or build the wheels yourself with")
        print("scripts/build_upstream_cuda_wheels.py in a CUDA development container.")
        return 1

    print("Required artifacts:")
    for artifact in matches:
        print(f"  {artifact['package']} {artifact['package_version']}  {artifact['filename']}")
        print(f"    sha256 {artifact['sha256']}")
        print(f"    built from {artifact.get('source_repository')} @ {artifact.get('source_commit')}")
    print()

    if not args.yes:
        print("Nothing was downloaded or installed.")
        print("Run again with --yes to install.")
        return 0

    dest = args.dest or Path.home() / ".cache" / "niverel-mamba" / "wheels"
    dest.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for artifact in matches:
        target = dest / artifact["filename"]
        if not target.is_file():
            print(f"downloading {artifact['filename']} ...")
            target.write_bytes(_fetch(artifact["url"]))
        verify_sha256(target, artifact["sha256"])
        print(f"verified    {artifact['filename']}")
        downloaded.append(target)

    print()
    print("installing ...")
    # --no-deps is not an optimisation: it is what stops pip from replacing the
    # CUDA torch build these wheels were compiled against.
    command = [sys.executable, "-m", "pip", "install", "--no-deps", *[str(p) for p in downloaded]]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print("pip install failed")
        return result.returncode

    # ... which leaves upstream's own import-time requirements uninstalled, so
    # they are installed here, deliberately and by name. Without this the wheels
    # land, `niverel-mamba doctor` sees their metadata, and the backend still
    # cannot be imported by a fresh process.
    print()
    print(f"installing upstream's import-time requirements: {', '.join(UPSTREAM_RUNTIME_REQUIREMENTS)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", *UPSTREAM_RUNTIME_REQUIREMENTS],
        check=False,
    )
    if result.returncode != 0:
        print("pip install failed for upstream's requirements; the backend will not import")
        return result.returncode

    print()
    print("Done. Verify with:  niverel-mamba doctor")
    return 0
