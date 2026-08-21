"""What upstream imports, read from upstream, not from memory.

The certified wheels are installed with `--no-deps`, because that is what stops
pip replacing the exact torch build they were compiled against. Their own
requirements therefore have to be installed by name, and the list has to be
right: a missing one makes `install-backend cuda` produce an environment whose
metadata looks complete and whose package will not import.

That happened. `transformers` was found by bisecting on a GPU runner, and the
answer was recorded as complete -- but every environment that had exercised the
command also carried `einops` for other reasons, so nothing noticed until it ran
somewhere clean.

So the list is checked against upstream's actual import graph instead.
"""

from __future__ import annotations

import ast
import sys
import zipfile

import pytest

from niverel_mamba.capabilities import UPSTREAM_RUNTIME_REQUIREMENTS
from tests.conftest import _upstream_wheel, requires_upstream_wheel

pytestmark = requires_upstream_wheel


@pytest.fixture
def upstream_wheel():
    wheel = _upstream_wheel()
    assert wheel is not None  # guarded by the marker above
    return wheel

#: Reached by upstream's imports, but not ours to install.
PROVIDED_ELSEWHERE = {
    "torch",            # the user installs it; the wheels are built against it
    "triton",           # ships with the CUDA torch build
    "packaging",        # a core dependency of this package
    "selective_scan_cuda",  # the compiled extension inside the wheel
}


def _hard_imports(source: str) -> set[str]:
    """Top-level imports only: anything inside try/except is optional by design."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _external_requirements(wheel) -> set[str]:
    archive = zipfile.ZipFile(wheel)
    members = set(archive.namelist())

    def resolve(module: str) -> str | None:
        base = module.replace(".", "/")
        for candidate in (f"{base}.py", f"{base}/__init__.py"):
            if candidate in members:
                return candidate
        return None

    pending, seen, external = ["mamba_ssm/__init__.py"], set(), set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        source = archive.read(path).decode("utf-8", "replace")
        tree = ast.parse(source)
        for name in _hard_imports(source):
            if name in sys.stdlib_module_names:
                continue
            if name != "mamba_ssm":
                external.add(name)
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("mamba_ssm")
            ):
                target = resolve(node.module)
                if target:
                    pending.append(target)
    return external


def test_every_hard_import_of_upstream_is_accounted_for(upstream_wheel):
    """Nothing upstream imports at start-up may be left to chance."""
    external = _external_requirements(upstream_wheel)
    unaccounted = external - PROVIDED_ELSEWHERE - set(UPSTREAM_RUNTIME_REQUIREMENTS)
    assert not unaccounted, (
        f"upstream imports {sorted(unaccounted)} at start-up and nothing installs them; "
        f"`install-backend cuda` would leave a package that does not import"
    )


def test_nothing_superfluous_is_installed_into_a_user_environment(upstream_wheel):
    """Installing what upstream does not import is someone else's dependency."""
    external = _external_requirements(upstream_wheel)
    superfluous = set(UPSTREAM_RUNTIME_REQUIREMENTS) - external
    assert not superfluous, f"{sorted(superfluous)} is installed but never imported"


def test_einops_is_in_the_list(upstream_wheel):
    """The one that was missing, named explicitly so the regression has a home."""
    assert "einops" in UPSTREAM_RUNTIME_REQUIREMENTS
    assert "einops" in _external_requirements(upstream_wheel)
