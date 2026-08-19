"""Import the *real* upstream ``mamba_ssm`` on a machine that has no CUDA.

Why this exists
---------------
The brief forbids writing the weight contract by hand from its indicative
list; it must be extracted from a real ``mamba_ssm.Mamba2`` 2.3.2.post1. But
``mamba-ssm`` only ships Linux/CUDA wheels, so it cannot be *installed* on
macOS or on a Linux CPU box.

It can still be *read*. A wheel is a zip, and the whole of ``mamba_ssm`` is
pure Python except for a handful of compiled extensions. Crucially,
``Mamba2.__init__`` touches no kernel at all: it only builds ``nn.Linear``,
``nn.Conv1d``, ``nn.Parameter`` and ``RMSNormGated``. So we can execute the
genuine module source behind import shims and instantiate it for real.

The honesty rules this module follows
-------------------------------------
* Shims exist **only to satisfy an import**. Every stub raises
  ``StubInvocationError`` the moment it is actually called, so a kernel can
  never silently turn into a no-op and produce plausible-looking numbers.
* The ``mamba_ssm`` package object is created without executing its
  ``__init__.py``. That file eagerly imports ``selective_scan_cuda``, Mamba3's
  tilelang/cute paths and the LM head -- none of which concern the Mamba2
  weight contract. Skipping it keeps the shim surface small enough to audit.
  ``mamba_ssm/modules/mamba2.py`` itself is executed verbatim, unpatched.
* Every source file that ends up contributing to the contract is hashed, so a
  contract can always be traced back to the exact bytes it came from.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
import types
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "StubInvocationError",
    "UpstreamSource",
    "find_wheel",
    "upstream_mamba_ssm",
]

# Wheels are looked up here, in order, unless an explicit path is given.
DEFAULT_WHEEL_DIRS = (
    Path(__file__).resolve().parent.parent / "vendor",
    Path.home() / "Documents/Professionnel/Hallcyn/niverel-wheels",
)

WHEEL_GLOB = "mamba_ssm-*.whl"


class StubInvocationError(RuntimeError):
    """A shimmed kernel was actually invoked.

    Reaching this means the extraction path drifted from "instantiate only"
    into "compute", which would yield numbers that are not upstream's. It is
    always a bug, never something to work around.
    """


class _Stub:
    """A named placeholder that imports cleanly and explodes on use."""

    __slots__ = ("_name", "configs")

    def __init__(self, name: str, configs: list[Any] | None = None) -> None:
        self._name = name
        # Upstream introspects ``kernel.configs`` at import time to derive
        # ``_CHUNK_STATE_BWD_DX_MIN_BLOCK_N``, so autotuned kernels must carry
        # their real config list even though the kernel itself never runs.
        self.configs = configs if configs is not None else []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise StubInvocationError(
            f"{self._name} is a stub installed only so that the upstream module could be "
            f"imported without CUDA/Triton. It was called, which means real computation was "
            f"attempted through a shim. Refusing to return fabricated values."
        )

    def __getitem__(self, item: Any) -> _Stub:
        # Triton kernels are launched as ``kernel[grid](...)``; make that path
        # fail at the call, not at the subscript, so the message stays useful.
        return self

    def __iter__(self) -> Any:
        # Without this, Python's legacy iteration protocol would fall back to
        # ``__getitem__(0), __getitem__(1), ...`` -- which returns ``self``
        # forever and hangs. Fail loudly instead.
        raise StubInvocationError(f"{self._name} was iterated; a stub has no contents")

    def __getattr__(self, item: str) -> _Stub:
        return _Stub(f"{self._name}.{item}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<stub {self._name}>"


class _StubModule(types.ModuleType):
    """A module whose every attribute is a :class:`_Stub`."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__all__: list[str] = []

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self.__name__}.{item}")


def _make_triton_stub() -> tuple[types.ModuleType, types.ModuleType]:
    """A Triton stub that survives *decoration* at import time.

    ``@triton.jit``, ``@triton.autotune(...)``, ``@triton.heuristics(...)`` and
    ``triton.Config(...)`` are all evaluated when the module is imported, so
    they must behave like real decorators. What they produce is a stub that
    raises if the kernel is ever launched.
    """
    triton = _StubModule("triton")

    def jit(fn: Any = None, **_: Any) -> Any:
        if fn is None:  # used as ``@triton.jit(...)``
            return jit
        return _Stub(f"triton.jit:{getattr(fn, '__name__', 'kernel')}")

    def _decorator_factory(label: str) -> Any:
        def outer(*args: Any, **kwargs: Any) -> Any:
            configs = kwargs.get("configs")
            if configs is None and args and isinstance(args[0], list):
                configs = args[0]

            def inner(fn: Any) -> Any:
                return _Stub(
                    f"triton.{label}:{getattr(fn, '__name__', 'kernel')}",
                    configs=list(configs) if configs is not None else None,
                )

            return inner

        return outer

    class _Config:
        def __init__(self, kwargs: dict[str, Any] | None = None, **extra: Any) -> None:
            self.kwargs = dict(kwargs or {})
            self.__dict__.update(extra)

    triton.jit = jit  # type: ignore[attr-defined]
    triton.autotune = _decorator_factory("autotune")  # type: ignore[attr-defined]
    triton.heuristics = _decorator_factory("heuristics")  # type: ignore[attr-defined]
    triton.Config = _Config  # type: ignore[attr-defined]
    triton.cdiv = lambda a, b: -(-int(a) // int(b))  # type: ignore[attr-defined]
    triton.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()  # type: ignore[attr-defined]
    triton.__version__ = "0.0.0+niverel-stub"  # type: ignore[attr-defined]

    language = _StubModule("triton.language")
    # ``BLOCK: tl.constexpr`` annotations are evaluated at def time, so
    # ``constexpr`` has to be a real object.
    language.constexpr = type("constexpr", (), {})  # type: ignore[attr-defined]
    triton.language = language  # type: ignore[attr-defined]
    return triton, language


#: Modules that must be faked for ``mamba_ssm.modules.mamba2`` to import.
#: Anything not listed here is the genuine upstream file.
_PLAIN_STUB_MODULES = (
    "causal_conv1d",
    "causal_conv1d.causal_conv1d_varlen",
    "causal_conv1d.cpp_functions",
    "selective_scan_cuda",
    "selective_scan",
    "causal_conv1d_cuda",
)


class UpstreamSource:
    """An extracted upstream tree plus the provenance of what was read."""

    def __init__(self, root: Path, wheel: Path, version: str) -> None:
        self.root = root
        self.wheel = wheel
        self.version = version
        self._hashes: dict[str, str] = {}

    @property
    def package_dir(self) -> Path:
        return self.root / "mamba_ssm"

    def record_hash(self, relative: str) -> str:
        """Hash a source file, remembering it for the provenance block."""
        if relative not in self._hashes:
            data = (self.root / relative).read_bytes()
            self._hashes[relative] = hashlib.sha256(data).hexdigest()
        return self._hashes[relative]

    @property
    def source_hashes(self) -> dict[str, str]:
        return dict(sorted(self._hashes.items()))

    def wheel_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.wheel.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()


def find_wheel(explicit: Path | None = None) -> Path:
    """Locate the pinned ``mamba_ssm`` wheel."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"mamba_ssm wheel not found at {explicit}")
        return explicit
    for directory in DEFAULT_WHEEL_DIRS:
        if not directory.is_dir():
            continue
        matches = sorted(directory.glob(WHEEL_GLOB))
        if matches:
            return matches[-1]
    searched = "\n  ".join(str(d) for d in DEFAULT_WHEEL_DIRS)
    raise FileNotFoundError(
        "Could not find a mamba_ssm wheel. Looked in:\n  "
        + searched
        + "\nPass --wheel to point at one explicitly."
    )


def _extract(wheel: Path, destination: Path) -> str:
    """Unpack the pure-Python tree and read ``__version__`` from the source."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.namelist():
            if not entry.startswith("mamba_ssm/"):
                continue
            if entry.endswith(".so") or entry.endswith("/"):
                continue
            archive.extract(entry, destination)

    init_source = (destination / "mamba_ssm" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_source)
    if match is None:
        raise RuntimeError("could not read __version__ from the upstream __init__.py")
    return match.group(1)


@contextmanager
def upstream_mamba_ssm(
    wheel: Path | None = None,
    workdir: Path | None = None,
) -> Iterator[UpstreamSource]:
    """Make the genuine ``mamba_ssm.modules.mamba2`` importable, then clean up.

    Yields an :class:`UpstreamSource` describing what was made available. On
    exit, every module and ``sys.path`` entry this added is removed, so the
    calling process is left exactly as it was found.
    """
    import tempfile

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(prefix="niverel-mamba-upstream-")
        root = Path(temporary.name)
    else:
        root = workdir
        root.mkdir(parents=True, exist_ok=True)

    wheel_path = find_wheel(wheel)
    version = _extract(wheel_path, root)
    source = UpstreamSource(root=root, wheel=wheel_path, version=version)

    saved_modules = dict(sys.modules)
    saved_path = list(sys.path)

    try:
        triton, triton_language = _make_triton_stub()
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = triton_language
        for name in _PLAIN_STUB_MODULES:
            sys.modules[name] = _StubModule(name)

        # Create the package object without running its __init__.py. See the
        # module docstring for why.
        package = types.ModuleType("mamba_ssm")
        package.__path__ = [str(source.package_dir)]  # type: ignore[attr-defined]
        package.__version__ = version  # type: ignore[attr-defined]
        sys.modules["mamba_ssm"] = package
        for sub in ("modules", "ops", "ops.triton", "distributed", "utils"):
            dotted = f"mamba_ssm.{sub}"
            sub_module = types.ModuleType(dotted)
            sub_module.__path__ = [str(source.package_dir / sub.replace(".", "/"))]  # type: ignore[attr-defined]
            sys.modules[dotted] = sub_module

        sys.path.insert(0, str(root))
        yield source
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name not in saved_modules:
                del sys.modules[name]
        sys.modules.update(saved_modules)
        if temporary is not None:
            temporary.cleanup()


def import_upstream_mamba2(source: UpstreamSource) -> Any:
    """Import the real ``Mamba2`` class and hash the files it came from."""
    for relative in (
        "mamba_ssm/__init__.py",
        "mamba_ssm/modules/mamba2.py",
        "mamba_ssm/ops/triton/layernorm_gated.py",
        "mamba_ssm/modules/ssd_minimal.py",
    ):
        source.record_hash(relative)

    module = importlib.import_module("mamba_ssm.modules.mamba2")
    return module.Mamba2
