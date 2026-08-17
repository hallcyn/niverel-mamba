"""The canonical Mamba2 weight contract.

A contract is a JSON document (see ``schemas/mamba2-upstream-2.3.2.post1.json``)
that states, for a given upstream package version, exactly which tensors a
Mamba2 block has, what shape each one is as a function of the configuration,
and whether it is conditional on a flag such as ``bias`` or ``rmsnorm``.

Shapes are **symbolic**. Storing ``["d_in_proj", "d_model"]`` rather than
``[3352, 768]`` is what makes one contract cover every configuration instead
of only Niverel's. The symbols are resolved against a
:class:`~niverel_mamba.config.Mamba2Config` at load time.

The contract is never written by hand. ``scripts/extract_weight_contract.py``
derives it from the genuine upstream module and then *verifies* the symbolic
form reproduces the real shapes across a sweep of configurations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Mamba2Config
from .errors import ContractVersionError, WeightContractError

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "TensorSpec",
    "WeightContract",
    "default_contract",
    "load_contract",
]

SCHEMA_VERSION = "niverel-mamba2-weights-v1"

#: Contract versions this package can load. A version outside this set is an
#: error, never a "best effort" load.
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"
_PACKAGED_SCHEMA_DIR = Path(__file__).resolve().parent / "_schemas"

DEFAULT_CONTRACT_FILENAME = "mamba2-upstream-2.3.2.post1.json"

#: Symbols a contract shape may reference. Each maps to a property of
#: :class:`Mamba2Config`.
SHAPE_SYMBOLS: dict[str, str] = {
    "d_model": "d_model",
    "d_state": "d_state",
    "d_conv": "d_conv",
    "headdim": "headdim",
    "ngroups": "ngroups",
    "d_inner": "d_inner",
    "d_ssm": "effective_d_ssm",
    "nheads": "nheads",
    "conv_dim": "conv_dim",
    "d_in_proj": "d_in_proj",
    "d_D": "d_D",
}


def resolve_dim(symbol: Any, config: Mamba2Config) -> int:
    """Resolve one symbolic dimension against a configuration."""
    if isinstance(symbol, int):
        return symbol
    if not isinstance(symbol, str):
        raise WeightContractError(f"shape entries must be int or symbol, got {symbol!r}")
    attribute = SHAPE_SYMBOLS.get(symbol)
    if attribute is None:
        raise WeightContractError(
            f"unknown shape symbol {symbol!r}; known symbols: {sorted(SHAPE_SYMBOLS)}"
        )
    return int(getattr(config, attribute))


@dataclass(frozen=True)
class TensorSpec:
    """One entry of the contract."""

    name: str
    shape: tuple[Any, ...]
    dtype: str
    #: Name of the boolean config flag that must be true for this tensor to
    #: exist, or ``None`` when the tensor is unconditional.
    required_if: str | None = None
    description: str = ""

    def is_present(self, config: Mamba2Config) -> bool:
        if self.required_if is None:
            return True
        return bool(getattr(config, self.required_if))

    def resolve_shape(self, config: Mamba2Config) -> tuple[int, ...]:
        return tuple(resolve_dim(dim, config) for dim in self.shape)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"shape": list(self.shape), "dtype": self.dtype}
        if self.required_if is not None:
            payload["required_if"] = self.required_if
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> TensorSpec:
        try:
            shape = tuple(payload["shape"])
            dtype = payload["dtype"]
        except KeyError as exc:
            raise WeightContractError(f"tensor {name!r} is missing {exc.args[0]!r}") from exc
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            required_if=payload.get("required_if"),
            description=payload.get("description", ""),
        )


@dataclass(frozen=True)
class WeightContract:
    """A complete, versioned weight contract."""

    schema_version: str
    upstream_package: str
    upstream_version: str
    tensors: tuple[TensorSpec, ...]
    key_order: tuple[str, ...]
    provenance: dict[str, Any]
    reference_configuration: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractVersionError(
                f"unsupported contract schema_version {self.schema_version!r}; "
                f"this build understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

    def spec(self, name: str) -> TensorSpec:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise WeightContractError(f"{name!r} is not part of the contract")

    def expected(self, config: Mamba2Config) -> dict[str, tuple[int, ...]]:
        """The exact ``{key: shape}`` a state dict must have for this config.

        Keys come back in upstream's own ``state_dict`` order, so a diff
        against a real checkpoint reads naturally.
        """
        present = {
            tensor.name: tensor.resolve_shape(config)
            for tensor in self.tensors
            if tensor.is_present(config)
        }
        ordered = {name: present[name] for name in self.key_order if name in present}
        # Defensive: never drop a tensor just because key_order is stale.
        for name, shape in present.items():
            ordered.setdefault(name, shape)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "upstream_package": self.upstream_package,
            "upstream_version": self.upstream_version,
            "key_order": list(self.key_order),
            "reference_configuration": self.reference_configuration,
            "tensors": {tensor.name: tensor.to_dict() for tensor in self.tensors},
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WeightContract:
        version = payload.get("schema_version")
        if version is None:
            raise ContractVersionError("contract has no schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ContractVersionError(
                f"unsupported contract schema_version {version!r}; "
                f"this build understands {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        tensors_payload = payload.get("tensors")
        if not isinstance(tensors_payload, dict) or not tensors_payload:
            raise WeightContractError("contract has no tensors")
        tensors = tuple(
            TensorSpec.from_dict(name, spec) for name, spec in tensors_payload.items()
        )
        key_order = tuple(payload.get("key_order") or tensors_payload.keys())
        return cls(
            schema_version=version,
            upstream_package=payload.get("upstream_package", "mamba-ssm"),
            upstream_version=payload.get("upstream_version", "unknown"),
            tensors=tensors,
            key_order=key_order,
            provenance=payload.get("provenance", {}),
            reference_configuration=payload.get("reference_configuration", {}),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _candidate_paths(filename: str) -> list[Path]:
    return [_PACKAGED_SCHEMA_DIR / filename, _SCHEMA_DIR / filename]


def load_contract(path: Path | str | None = None) -> WeightContract:
    """Load a contract from disk, or the packaged default."""
    if path is not None:
        return WeightContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    for candidate in _candidate_paths(DEFAULT_CONTRACT_FILENAME):
        if candidate.is_file():
            return WeightContract.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
    raise WeightContractError(
        f"no packaged weight contract found; looked for {DEFAULT_CONTRACT_FILENAME} in "
        + ", ".join(str(p.parent) for p in _candidate_paths(DEFAULT_CONTRACT_FILENAME))
        + ". Run scripts/extract_weight_contract.py to generate it."
    )


_cached: WeightContract | None = None


def default_contract() -> WeightContract:
    """The packaged contract, loaded once."""
    global _cached
    if _cached is None:
        _cached = load_contract()
    return _cached
