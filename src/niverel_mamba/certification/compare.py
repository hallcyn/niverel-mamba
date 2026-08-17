"""Numerical comparison primitives.

Framework-agnostic: anything convertible to a numpy array works, so the same
code compares torch-vs-torch, torch-vs-MLX and (on a GPU box) torch-vs-CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .tolerances import Tolerance, get_tolerance

__all__ = ["Comparison", "compare", "to_numpy"]


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert a torch / MLX / numpy tensor to a float64 numpy array.

    The widening to float64 must never pass *through* a narrower type: doing
    ``float64 -> float32 -> float64`` would quietly cap every measurement at
    float32 resolution and make a float64 certification report agreement it
    never actually demonstrated. Only dtypes numpy cannot represent at all
    (bfloat16, float8) are converted first, and then upward.
    """
    if hasattr(tensor, "detach"):  # torch
        import torch

        tensor = tensor.detach().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            # numpy has no bfloat16; float32 is the narrowest lossless target.
            tensor = tensor.to(torch.float32)
        elif tensor.dtype.is_floating_point and tensor.dtype != torch.float64:
            tensor = tensor.to(torch.float64)
        return np.asarray(tensor.numpy(), dtype=np.float64)
    return np.asarray(np.array(tensor), dtype=np.float64)


@dataclass(frozen=True)
class Comparison:
    """The result of comparing a candidate against a reference."""

    name: str
    tolerance_class: str
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    cosine_similarity: float
    atol: float
    rtol: float
    rel_floor: float
    elements: int
    passed: bool
    #: How many elements violate ``|a - b| <= atol + rtol * |b|``.
    violations: int = 0
    #: Worst ``|a - b| - (atol + rtol * |b|)``. Negative means clear headroom.
    worst_excess: float = 0.0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(
    candidate: Any,
    reference: Any,
    *,
    name: str,
    tolerance: Tolerance | str,
) -> Comparison:
    """Compare two tensors under a sealed tolerance class.

    The pass criterion is elementwise and identical to ``numpy.allclose`` /
    ``torch.allclose``::

        |candidate - reference| <= atol + rtol * |reference|

    evaluated on every element, not on a summary statistic. A single bad
    element therefore fails the comparison, which is the point.

    ``max_rel_error`` is reported only over elements where
    ``|reference| > rel_floor``. Below that, relative error is dominated by
    values that are numerically zero -- a reference of 1e-9 makes any
    difference look enormous -- so including them would exaggerate
    disagreement rather than describe it. The floor is part of the sealed
    table, not a per-call knob, and it affects only what is *reported*: the
    pass/fail decision above still covers every element.
    """
    tol = get_tolerance(tolerance) if isinstance(tolerance, str) else tolerance

    a = to_numpy(candidate)
    b = to_numpy(reference)
    if a.shape != b.shape:
        return Comparison(
            name=name,
            tolerance_class=tol.name,
            max_abs_error=float("inf"),
            max_rel_error=float("inf"),
            mean_abs_error=float("inf"),
            cosine_similarity=0.0,
            atol=tol.atol,
            rtol=tol.rtol,
            rel_floor=tol.rel_floor,
            elements=int(a.size),
            passed=False,
            detail=f"shape mismatch: candidate {a.shape} vs reference {b.shape}",
        )

    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0

    significant = np.abs(b) > tol.rel_floor
    if significant.any():
        max_rel = float((diff[significant] / np.abs(b[significant])).max())
    else:
        max_rel = 0.0

    flat_a, flat_b = a.ravel(), b.ravel()
    norm = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    cosine = float(np.dot(flat_a, flat_b) / norm) if norm > 0 else 1.0

    # The authoritative, elementwise criterion.
    bound = tol.atol + tol.rtol * np.abs(b)
    excess = diff - bound
    violations = int((excess > 0).sum())
    worst_excess = float(excess.max()) if excess.size else 0.0

    detail = None
    if not np.isfinite(diff).all():
        detail = "candidate or reference contains NaN or Inf"
    elif violations:
        detail = (
            f"{violations} of {a.size} elements exceed atol + rtol*|reference| "
            f"(worst by {worst_excess:.3e})"
        )

    passed = bool(violations == 0 and np.isfinite(diff).all())
    return Comparison(
        name=name,
        tolerance_class=tol.name,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        mean_abs_error=mean_abs,
        cosine_similarity=cosine,
        atol=tol.atol,
        rtol=tol.rtol,
        rel_floor=tol.rel_floor,
        elements=int(a.size),
        passed=passed,
        violations=violations,
        worst_excess=worst_excess,
        detail=detail,
    )
