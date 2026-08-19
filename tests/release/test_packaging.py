"""Packaging, import hygiene and supply-chain guarantees."""

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


_AUDIT_TEMPLATE = """
import sys

FORBIDDEN = {forbidden}
violations = []
sys.addaudithook(lambda event, args: violations.append(event) if event in FORBIDDEN else None)

{body}

print("VIOLATIONS:" + ",".join(sorted(set(violations))))
"""

NETWORK_EVENTS = (
    '"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", '
    '"urllib.Request", "ftplib.connect"'
)
EXEC_EVENTS = '"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.fork"'


def _run_audited(body: str, forbidden: str) -> str:
    """Run `body` under an interpreter audit hook, return the violations line.

    An audit hook rather than monkeypatching, because the events are raised by
    the interpreter itself and cannot be bypassed by a module holding its own
    reference to `socket` or `subprocess`.
    """
    code = _AUDIT_TEMPLATE.format(forbidden="{" + forbidden + "}", body=body)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    return next(r for r in out.stdout.splitlines() if r.startswith("VIOLATIONS:"))


def test_import_and_cli_parser_have_no_side_effects():
    """The actual requirement: importing must download, spawn and compile nothing.

    This path deliberately does not touch torch or MLX, so it is held to the
    strict standard -- no network *and* no process execution.
    """
    line = _run_audited(
        "import niverel_mamba\n"
        "import niverel_mamba.cli.main\n"
        "niverel_mamba.cli.main.build_parser()",
        f"{NETWORK_EVENTS}, {EXEC_EVENTS}",
    )
    assert line == "VIOLATIONS:", f"importing had side effects: {line}"


def test_detect_environment_never_touches_the_network():
    """`detect_environment()` may exec, but must never phone home.

    It deliberately imports whichever frameworks are installed, and on Linux
    `torch` is the CUDA build, which pulls in Triton and probes its toolchain
    with a subprocess. That is torch's behaviour on a machine that has it, not
    something this package can or should prevent -- and `doctor` exists
    precisely to inspect the machine.

    What stays forbidden is the network. A capability probe that reached out
    would be a supply-chain concern; one that runs `ptxas --version` is not.
    """
    line = _run_audited(
        "import niverel_mamba\nniverel_mamba.detect_environment()", NETWORK_EVENTS
    )
    assert line == "VIOLATIONS:", f"detect_environment() used the network: {line}"


def test_cli_help_does_not_import_frameworks():
    """Building the parser must stay light.

    `build_parser()` imports every subcommand so `--help` can list them, so any
    heavy import in one subcommand is paid for by all of them. numpy is checked
    alongside torch and mlx because of a real regression: a cold
    `pip install niverel-mamba` followed by `niverel-mamba doctor` crashed with
    ModuleNotFoundError: numpy, pulled in transitively by `verify`. A
    diagnostic command must not die on a dependency it does not use.
    """
    code = (
        "import sys; from niverel_mamba.cli.main import build_parser; build_parser(); "
        "print('torch' in sys.modules, 'mlx' in sys.modules, 'numpy' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert out.stdout.strip() == "False False False", out.stdout


def test_doctor_and_inspect_need_only_the_core_install():
    """The two commands a user runs before installing anything else.

    Exercised here against the modules directly; publish-pypi.yml runs the same
    commands against a genuinely cold install from the index.
    """
    code = (
        "from niverel_mamba.cli.main import main; "
        "import io, contextlib; buf = io.StringIO(); "
        "contextlib.redirect_stdout(buf).__enter__(); "
        "main(['doctor']); main(['inspect']); "
        "print('OK' if 'in_proj.weight' in buf.getvalue() else 'MISSING', file=__import__('sys').stderr)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stderr, out.stderr


def test_version_is_consistent():
    import niverel_mamba

    pyproject = _pyproject()
    assert niverel_mamba.__version__ == pyproject["project"]["version"]


def test_torch_is_not_a_required_dependency():
    """An MLX user must not download PyTorch."""
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
