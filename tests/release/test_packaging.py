"""Packaging, import hygiene and supply-chain guarantees (brief section 15)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# `tomllib` is stdlib from 3.11 only, and ci-torch-matrix covers 3.10.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 in CI
    import tomli as tomllib


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_import_pulls_in_neither_torch_nor_mlx():
    """A torch-only user must not pay for MLX, and vice versa.

    Run in a fresh interpreter so an already-imported framework in this
    process cannot mask a regression.
    """
    code = (
        "import sys; import niverel_mamba; "
        "print('torch' in sys.modules, 'mlx' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "False False", out.stdout


def test_import_performs_no_network_or_subprocess():
    """No download, no subprocess, no compilation at import time.

    Uses an audit hook rather than monkeypatching, because the audit events
    are raised by the interpreter itself and so cannot be bypassed by a module
    that holds its own reference to ``socket`` or ``subprocess``.
    """
    code = """
import sys

FORBIDDEN = {
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
    "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.fork",
    "urllib.Request", "ftplib.connect",
}

violations = []

def hook(event, args):
    if event in FORBIDDEN:
        violations.append(event)

sys.addaudithook(hook)

import niverel_mamba
import niverel_mamba.cli.main
niverel_mamba.cli.main.build_parser()
niverel_mamba.detect_environment()

print("VIOLATIONS:" + ",".join(sorted(set(violations))))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    line = next(row for row in out.stdout.splitlines() if row.startswith("VIOLATIONS:"))
    assert line == "VIOLATIONS:", f"import had forbidden side effects: {line}"


def test_cli_help_does_not_import_frameworks():
    code = (
        "import sys; from niverel_mamba.cli.main import build_parser; build_parser(); "
        "print('torch' in sys.modules, 'mlx' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "False False", out.stdout


def test_version_is_consistent():
    import niverel_mamba

    pyproject = _pyproject()
    assert niverel_mamba.__version__ == pyproject["project"]["version"]


def test_torch_is_not_a_required_dependency():
    """Brief section 13: an MLX user must not download PyTorch."""
    pyproject = _pyproject()
    required = " ".join(pyproject["project"]["dependencies"])
    assert "torch" not in required
    assert "mlx" not in required
    extras = pyproject["project"]["optional-dependencies"]
    assert any("torch" in item for item in extras["torch"])
    assert any("mlx" in item for item in extras["mlx"])


def test_mlx_extra_is_restricted_to_apple_silicon():
    pyproject = _pyproject()
    spec = " ".join(pyproject["project"]["optional-dependencies"]["mlx"])
    assert "platform_system == 'Darwin'" in spec
    assert "platform_machine == 'arm64'" in spec


def test_mlx_is_upper_bounded():
    """The brief forbids an unbounded MLX dependency until certification is
    automated, because a minor release could silently change numerics."""
    pyproject = _pyproject()
    spec = " ".join(pyproject["project"]["optional-dependencies"]["mlx"])
    assert "<0.33" in spec


@pytest.mark.parametrize(
    "filename",
    ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "README.md", "CHANGELOG.md",
     "SECURITY.md", "CONTRIBUTING.md"],
)
def test_required_repository_files_exist(filename):
    path = REPO_ROOT / filename
    assert path.is_file(), f"{filename} is missing"
    assert path.stat().st_size > 0


def test_env_is_gitignored():
    """.env holds the HF token and must never be committable."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore

    tracked = subprocess.run(
        ["git", "ls-files", ".env"], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert tracked.stdout.strip() == "", ".env is tracked by git"


def test_no_large_binaries_are_tracked():
    """Checkpoints and wheels must stay out of the repository."""
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=REPO_ROOT, check=True
    )
    offenders = []
    for name in out.stdout.split("\n"):
        if not name:
            continue
        path = REPO_ROOT / name
        if path.is_file() and path.stat().st_size > 2_000_000:
            offenders.append((name, path.stat().st_size))
    assert not offenders, f"large tracked files: {offenders}"


def test_packaged_contract_is_present():
    """The wheel must ship the weight contract, not rely on a source checkout."""
    from niverel_mamba.schema import default_contract

    contract = default_contract()
    assert contract.upstream_version == "2.3.2.post1"
    assert contract.tensors
