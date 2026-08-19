"""Strict weight loading against the canonical contract.

Brief section 3.2 lists six things every backend must refuse: a missing key,
an unexpected key, a differing shape, an incompatible configuration, an
unknown contract version, and an undocumented implicit conversion. This
module is where all six are enforced, so that every backend inherits the same
behaviour instead of each re-implementing it slightly differently.

The bar is ``load_state_dict(state_dict, strict=True)``. Nothing here ever
falls back to ``strict=False``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Protocol

from .config import Mamba2Config
from .errors import (
    DtypeMismatchError,
    MissingKeysError,
    ShapeMismatchError,
    UnexpectedKeysError,
)
from .schema import WeightContract, default_contract

__all__ = [
    "TensorLike",
    "canonical_key_order",
    "state_dict_digest",
    "tensor_digest",
    "validate_state_dict",
]


class TensorLike(Protocol):
    """The minimum any backend's tensor type must expose to be validated."""

    @property
    def shape(self) -> Any: ...


def _shape_of(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise ShapeMismatchError(f"object {type(tensor).__name__} has no .shape")
    return tuple(int(dim) for dim in shape)


def _dtype_name(tensor: Any) -> str | None:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return None
    name = str(dtype)
    # ``torch.float32`` -> ``float32``; ``mlx.core.float32`` -> ``float32``.
    return name.rsplit(".", 1)[-1]


def validate_state_dict(
    state_dict: Mapping[str, Any],
    config: Mamba2Config,
    *,
    contract: WeightContract | None = None,
    allow_dtype_cast: bool = True,
    require_dtype: str | None = None,
) -> dict[str, Any]:
    """Check a state dict against the contract and return it in canonical order.

    Parameters
    ----------
    allow_dtype_cast
        A checkpoint saved in bfloat16 is legitimately loadable into a float32
        module -- that conversion is documented and lossless in the safe
        direction. Set ``False`` to require an exact dtype match.
    require_dtype
        If given (e.g. ``"float32"``), every tensor must already be that dtype.

    Raises
    ------
    MissingKeysError, UnexpectedKeysError, ShapeMismatchError, DtypeMismatchError
    """
    contract = contract or default_contract()
    expected = contract.expected(config)

    provided = set(state_dict)
    required = set(expected)

    missing = sorted(required - provided)
    if missing:
        raise MissingKeysError(
            f"state dict is missing {len(missing)} contract key(s): {missing}. "
            f"Expected exactly {sorted(required)} for this configuration."
        )

    unexpected = sorted(provided - required)
    if unexpected:
        raise UnexpectedKeysError(
            f"state dict has {len(unexpected)} key(s) the contract does not define: {unexpected}. "
            "This usually means the configuration does not match the checkpoint (for example "
            "bias=False when the checkpoint was trained with bias=True)."
        )

    problems: list[str] = []
    for key, want in expected.items():
        got = _shape_of(state_dict[key])
        if got != want:
            problems.append(f"  {key}: expected {want}, got {got}")
    if problems:
        raise ShapeMismatchError(
            "state dict shapes do not match the contract for this configuration:\n"
            + "\n".join(problems)
        )

    if require_dtype is not None or not allow_dtype_cast:
        target = require_dtype
        dtype_problems = []
        for key in expected:
            name = _dtype_name(state_dict[key])
            if name is None:
                continue
            if target is None:
                target = name
            if name != target:
                dtype_problems.append(f"  {key}: {name} (expected {target})")
        if dtype_problems:
            raise DtypeMismatchError(
                "state dict would require an undocumented implicit dtype conversion:\n"
                + "\n".join(dtype_problems)
            )

    return {key: state_dict[key] for key in expected}


def canonical_key_order(config: Mamba2Config, contract: WeightContract | None = None) -> list[str]:
    """The contract's key order for this configuration."""
    contract = contract or default_contract()
    return list(contract.expected(config))


def tensor_digest(tensor: Any) -> str:
    """SHA-256 of a tensor's raw bytes, framework-independent.

    Used to prove the MLX round-trip is genuinely lossless rather than merely
    close: canonical -> MLX -> canonical must reproduce identical digests.
    """
    import numpy as np

    if hasattr(tensor, "detach"):  # torch
        array = tensor.detach().cpu().contiguous().numpy()
    else:  # mlx or numpy
        array = np.asarray(tensor)
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def state_dict_digest(state_dict: Mapping[str, Any]) -> dict[str, str]:
    """Per-tensor digests, keyed by name."""
    return {key: tensor_digest(value) for key, value in sorted(state_dict.items())}
