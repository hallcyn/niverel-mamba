"""Installed is not importable.

This distinction cost a GPU certification run. The wheels were installed, their
metadata was intact, `niverel-mamba doctor` reported cuda-reference as
available -- and no fresh process could import upstream's Mamba2, because
`mamba_ssm/__init__.py` reaches `transformers` and the wheels are installed
with `--no-deps`.

The trap is that the failure hides itself. `__init__` imports `modules.mamba2`
*before* the line that fails, so Python leaves that submodule in `sys.modules`
even after the package import blows up. Any later import of it in the same
process then succeeds from cache. The certification's parity tests passed that
way, against a backend that could not actually be loaded.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from niverel_mamba import capabilities


@pytest.fixture(autouse=True)
def _forget_the_probe():
    capabilities._IMPORTABLE = None
    yield
    capabilities._IMPORTABLE = None


def _without_upstream(monkeypatch):
    for name in [n for n in sys.modules if n == "mamba_ssm" or n.startswith("mamba_ssm.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_absent_upstream_is_reported_unimportable_with_a_reason(monkeypatch):
    _without_upstream(monkeypatch)
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda name: None)
    importable, why = capabilities.upstream_mamba2_importable()
    assert importable is False
    assert why and "mamba_ssm" in why


def test_a_failure_is_never_memoised(monkeypatch):
    """It can be repaired inside the same process, and a stale "no" would lie.

    `niverel-mamba install-backend cuda` installs the missing requirement and
    the caller may then ask again without starting a new interpreter.
    """
    _without_upstream(monkeypatch)
    assert capabilities.upstream_mamba2_importable()[0] is False
    assert capabilities._IMPORTABLE is None, "a failure must not be cached"

    module = types.ModuleType("mamba_ssm.modules.mamba2")
    module.Mamba2 = object  # type: ignore[attr-defined]
    for name in ("mamba_ssm", "mamba_ssm.modules"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mamba_ssm.modules.mamba2", module)

    assert capabilities.upstream_mamba2_importable() == (True, None)


def test_the_cuda_backend_is_not_advertised_when_it_cannot_import(monkeypatch):
    """Metadata alone must never make a backend "available"."""
    from niverel_mamba import registry

    monkeypatch.setattr(
        registry, "upstream_mamba2_importable", lambda: (False, "ModuleNotFoundError: transformers")
    )
    env = capabilities.detect_environment()
    present = capabilities.FrameworkInfo(True, version="stub")
    env = dataclasses.replace(
        env,
        torch=present,
        cuda=present,
        upstream_mamba_ssm=present,
        causal_conv1d=present,
    )
    available, reason, devices = registry._available_devices(
        registry.get_backend("cuda-reference"), env
    )
    assert available is False, "an unimportable backend must never be advertised"
    assert devices == ()
    assert reason and "will not import" in reason and "transformers" in reason
