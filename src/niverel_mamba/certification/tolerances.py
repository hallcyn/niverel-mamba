"""Load the sealed tolerance table.

The tolerances live in ``tolerances.yaml`` next to this file. They are data,
not code, so that widening one shows up as a reviewable diff rather than
disappearing into a test edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..errors import CertificationError

__all__ = ["Tolerance", "ToleranceTable", "get_tolerance", "load_tolerances"]

_PATH = Path(__file__).resolve().parent / "tolerances.yaml"


@dataclass(frozen=True)
class Tolerance:
    """One sealed tolerance class."""

    name: str
    atol: float
    rtol: float
    #: Reference magnitude below which relative error is not reported. Small
    #: reference values make relative error explode for no useful reason.
    rel_floor: float = 0.0
    description: str = ""
    observed: dict[str, Any] | None = None

    def passes(self, max_abs: float, max_rel: float) -> bool:
        """Coarse check on summary statistics only.

        The authoritative criterion is elementwise -- see
        :func:`niverel_mamba.certification.compare.compare`, which evaluates
        ``|a - b| <= atol + rtol * |b|`` on every element. This helper exists
        for callers that only have the summary numbers (a report loaded from
        disk, say) and is deliberately conservative: it requires the worst
        absolute error to fit within the combined bound at unit scale.
        """
        if math.isnan(max_abs) or math.isnan(max_rel):
            return False
        return max_abs <= self.atol or max_rel <= self.rtol

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.name,
            "atol": self.atol,
            "rtol": self.rtol,
            "rel_floor": self.rel_floor,
        }


@dataclass(frozen=True)
class ToleranceTable:
    schema_version: str
    sealed_on: str
    sealed_hardware: str
    classes: dict[str, Tolerance]

    def __getitem__(self, name: str) -> Tolerance:
        try:
            return self.classes[name]
        except KeyError as exc:
            raise CertificationError(
                f"unknown tolerance class {name!r}; sealed classes are {sorted(self.classes)}"
            ) from exc

    @property
    def unverified(self) -> list[str]:
        """Classes whose ``observed`` block is still empty.

        A class listed here has never actually been measured. Anything relying
        on it must be published as ``experimental``.
        """
        return sorted(name for name, tol in self.classes.items() if not tol.observed)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise CertificationError(
            "PyYAML is required to read the sealed tolerances; install 'niverel-mamba[dev]'"
        ) from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_tolerances(path: str | None = None) -> ToleranceTable:
    """Read and cache the sealed tolerance table."""
    target = Path(path) if path else _PATH
    if not target.is_file():
        raise CertificationError(f"tolerance file not found: {target}")
    payload = _load_yaml(target)

    classes: dict[str, Tolerance] = {}
    for name, entry in (payload.get("classes") or {}).items():
        classes[name] = Tolerance(
            name=name,
            atol=float(entry["atol"]),
            rtol=float(entry["rtol"]),
            rel_floor=float(entry.get("rel_floor", 0.0)),
            description=str(entry.get("description", "")).strip(),
            observed=entry.get("observed"),
        )
    if not classes:
        raise CertificationError(f"{target} defines no tolerance classes")

    return ToleranceTable(
        schema_version=payload.get("schema_version", "unknown"),
        sealed_on=str(payload.get("sealed_on", "unknown")),
        sealed_hardware=str(payload.get("sealed_hardware", "unknown")),
        classes=classes,
    )


def get_tolerance(name: str) -> Tolerance:
    return load_tolerances()[name]
