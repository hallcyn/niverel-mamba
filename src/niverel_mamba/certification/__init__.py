"""Numerical certification: fixtures, comparisons, sealed tolerances, reports."""

from __future__ import annotations

from .compare import Comparison, compare, to_numpy
from .golden import GoldenFixture, available_fixtures, load_fixture
from .report import CertificationReport, build_report
from .tolerances import Tolerance, ToleranceTable, get_tolerance, load_tolerances

__all__ = [
    "CertificationReport",
    "Comparison",
    "GoldenFixture",
    "Tolerance",
    "ToleranceTable",
    "available_fixtures",
    "build_report",
    "compare",
    "get_tolerance",
    "load_fixture",
    "load_tolerances",
    "to_numpy",
]
