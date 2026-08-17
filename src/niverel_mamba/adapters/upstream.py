"""Bijection between an upstream ``mamba_ssm.Mamba2`` state dict and ours.

At 2.3.2.post1 the mapping is the identity: our parameter names, shapes and
``state_dict`` ordering were *derived from* upstream, so nothing needs
renaming. That is the whole point -- a checkpoint is not rewritten to be
portable, it is portable already.

This module still exists, and is still the only sanctioned path, because:

* the identity is a *fact to be tested*, not an assumption. ``round_trip``
  and the contract tests prove it every CI run, so the day upstream changes a
  name we find out here rather than in a silently wrong forward pass;
* it is where an optional prefix (``decoder.0.mixer.``) is stripped, which is
  what turns a full H-Net checkpoint into a single-block state dict;
* if a future upstream version does rename something, the translation lands
  here and every backend inherits it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from ..config import Mamba2Config
from ..errors import WeightContractError
from ..schema import WeightContract, default_contract
from ..weights import validate_state_dict

__all__ = [
    "CANONICAL_TO_UPSTREAM",
    "UPSTREAM_TO_CANONICAL",
    "find_blocks",
    "from_upstream",
    "round_trip",
    "strip_prefix",
    "to_upstream",
]

T = TypeVar("T")

#: Name translation, upstream -> canonical. Empty because the two coincide at
#: 2.3.2.post1; kept explicit so a future divergence has an obvious home.
UPSTREAM_TO_CANONICAL: dict[str, str] = {}

CANONICAL_TO_UPSTREAM: dict[str, str] = {v: k for k, v in UPSTREAM_TO_CANONICAL.items()}


def strip_prefix(state_dict: Mapping[str, T], prefix: str) -> dict[str, T]:
    """Keep only keys under ``prefix`` and drop it.

    ``strip_prefix(ckpt, "decoder.0.mixer")`` turns a full H-Net checkpoint
    into a single Mamba2 block's state dict.
    """
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    out = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not out:
        raise WeightContractError(f"no keys under prefix {prefix!r}")
    return out


def find_blocks(
    state_dict: Mapping[str, Any],
    config: Mamba2Config,
    *,
    contract: WeightContract | None = None,
) -> dict[str, dict[str, Any]]:
    """Locate every Mamba2 block in a larger checkpoint.

    A prefix qualifies only if *all* of the contract's keys are present
    beneath it, so a partial or differently-shaped block is never mistaken
    for one.
    """
    contract = contract or default_contract()
    required = list(contract.expected(config))
    anchor = "in_proj.weight"

    blocks: dict[str, dict[str, Any]] = {}
    for key in state_dict:
        if not key.endswith(anchor):
            continue
        prefix = key[: -len(anchor)]
        block = {}
        for name in required:
            full = f"{prefix}{name}"
            if full not in state_dict:
                break
            block[name] = state_dict[full]
        else:
            blocks[prefix.rstrip(".")] = block
    return blocks


def from_upstream(
    state_dict: Mapping[str, Any],
    config: Mamba2Config,
    *,
    prefix: str | None = None,
    contract: WeightContract | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Upstream state dict -> canonical, validated against the contract."""
    source: Mapping[str, Any] = strip_prefix(state_dict, prefix) if prefix else state_dict
    renamed = {UPSTREAM_TO_CANONICAL.get(k, k): v for k, v in source.items()}
    if validate:
        return validate_state_dict(renamed, config, contract=contract)
    return renamed


def to_upstream(
    state_dict: Mapping[str, Any],
    config: Mamba2Config,
    *,
    prefix: str | None = None,
    contract: WeightContract | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Canonical -> upstream state dict."""
    checked = validate_state_dict(state_dict, config, contract=contract) if validate else dict(state_dict)
    renamed = {CANONICAL_TO_UPSTREAM.get(k, k): v for k, v in checked.items()}
    if prefix:
        dotted = prefix if prefix.endswith(".") else prefix + "."
        renamed = {f"{dotted}{k}": v for k, v in renamed.items()}
    return renamed


def round_trip(state_dict: Mapping[str, Any], config: Mamba2Config) -> dict[str, Any]:
    """``upstream -> canonical -> upstream``. Used by the reversibility test."""
    canonical = from_upstream(state_dict, config)
    return to_upstream(canonical, config)
