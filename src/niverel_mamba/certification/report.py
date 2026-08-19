"""Certification reports.

Every report names the backend it was measured *against*. That field is the
whole point: "torch-reference passed" means nothing on its own, whereas
"torch-reference-mps passed against torch-reference-cpu under mps_float32"
is a claim someone can check.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..version import __version__
from .compare import Comparison
from .tolerances import load_tolerances

__all__ = ["CertificationReport", "build_report"]

REPORT_SCHEMA_VERSION = "niverel-mamba-certification-report-v1"


@dataclass
class CertificationReport:
    """A single candidate/reference comparison campaign."""

    reference_backend: str
    candidate_backend: str
    fixture: str
    comparisons: list[Comparison] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, comparison: Comparison) -> Comparison:
        self.comparisons.append(comparison)
        return comparison

    def _named(self, name: str) -> bool | None:
        for comparison in self.comparisons:
            if comparison.name == name:
                return comparison.passed
        return None

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(c.passed for c in self.comparisons)

    @property
    def max_abs_error(self) -> float:
        return max((c.max_abs_error for c in self.comparisons), default=0.0)

    @property
    def max_rel_error(self) -> float:
        return max((c.max_rel_error for c in self.comparisons), default=0.0)

    @property
    def mean_abs_error(self) -> float:
        values = [c.mean_abs_error for c in self.comparisons]
        return sum(values) / len(values) if values else 0.0

    @property
    def cosine_similarity(self) -> float:
        return min((c.cosine_similarity for c in self.comparisons), default=1.0)

    def to_dict(self) -> dict[str, Any]:
        table = load_tolerances()
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "niverel_mamba_version": __version__,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "fixture": self.fixture,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "mean_abs_error": self.mean_abs_error,
            "cosine_similarity": self.cosine_similarity,
            "forward_passed": self._named("forward"),
            "step_passed": self._named("step"),
            "segment_reset_passed": self._named("segment_reset"),
            "passed": self.passed,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "tolerances": {
                "schema_version": table.schema_version,
                "sealed_on": table.sealed_on,
                "sealed_hardware": table.sealed_hardware,
                "unverified_classes": table.unverified,
            },
            "environment": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                **self.metadata,
            },
        }

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    def summary(self) -> str:
        lines = [
            f"{self.candidate_backend}  vs  {self.reference_backend}   [{self.fixture}]",
            "",
        ]
        width = max((len(c.name) for c in self.comparisons), default=0)
        for comparison in self.comparisons:
            mark = "pass" if comparison.passed else "FAIL"
            lines.append(
                f"  {mark:4s}  {comparison.name:<{width}}  "
                f"max_abs={comparison.max_abs_error:.3e}  "
                f"max_rel={comparison.max_rel_error:.3e}  "
                f"[{comparison.tolerance_class} atol={comparison.atol:g} rtol={comparison.rtol:g}]"
            )
            if comparison.detail:
                lines.append(f"        {comparison.detail}")
        lines.append("")
        lines.append(f"  overall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def build_report(
    reference_backend: str,
    candidate_backend: str,
    fixture: str,
    comparisons: list[Comparison],
    **metadata: Any,
) -> CertificationReport:
    return CertificationReport(
        reference_backend=reference_backend,
        candidate_backend=candidate_backend,
        fixture=fixture,
        comparisons=list(comparisons),
        metadata=metadata,
    )
