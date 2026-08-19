#!/usr/bin/env python
"""Phase 0 -- derive the Mamba2 weight contract from the real upstream module.

The brief is explicit: the contract must not be typed in by hand from its
indicative list, it must be extracted from a genuine
``mamba_ssm.Mamba2`` 2.3.2.post1. This script does that, and then does one
thing more -- it *verifies* the symbolic form it produces against a sweep of
real configurations, so the contract is proven to describe upstream rather
than merely asserted to.

    python scripts/extract_weight_contract.py
    python scripts/extract_weight_contract.py --check     # CI: fail on drift

Running it on a machine without CUDA is fine and intended: see
``scripts/_upstream_env.py`` for how the real module is imported without
Triton or compiled extensions, and why no kernel can silently run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from _upstream_env import (  # noqa: E402
    UpstreamSource,
    import_upstream_mamba2,
    upstream_mamba_ssm,
)
from niverel_mamba.config import Mamba2Config  # noqa: E402
from niverel_mamba.schema import SCHEMA_VERSION, TensorSpec, WeightContract  # noqa: E402

# --------------------------------------------------------------------------
# The symbolic contract, expressed once and then checked against reality.
# --------------------------------------------------------------------------

#: Every tensor a Mamba2 block can own, with its shape as a function of the
#: configuration. ``required_if`` names the config flag that gates it.
#:
#: Note what is *absent*: there is no ``init_states``. The brief listed it as
#: "if enabled", but Mamba2 2.3.2.post1 has no learnable initial state at all.
#: This is exactly the sort of thing the brief demanded be extracted rather
#: than assumed, and the verification sweep below is what proves it.
CONTRACT_TENSORS: tuple[TensorSpec, ...] = (
    TensorSpec(
        name="dt_bias",
        shape=("nheads",),
        dtype="float32",
        description="Pre-softplus bias on the discretisation step, one per head.",
    ),
    TensorSpec(
        name="A_log",
        shape=("nheads",),
        dtype="float32",
        description="Log of the negated state-decay rate; A = -exp(A_log), always in float32.",
    ),
    TensorSpec(
        name="D",
        shape=("d_D",),
        dtype="float32",
        description="Skip-connection gain. Length nheads, or d_ssm when D_has_hdim.",
    ),
    TensorSpec(
        name="in_proj.weight",
        shape=("d_in_proj", "d_model"),
        dtype="float32",
        description="Fused input projection producing [z0, x0, z, xBC, dt] in that order.",
    ),
    TensorSpec(
        name="in_proj.bias",
        shape=("d_in_proj",),
        dtype="float32",
        required_if="bias",
        description="Present only when bias=True.",
    ),
    TensorSpec(
        name="conv1d.weight",
        shape=("conv_dim", 1, "d_conv"),
        dtype="float32",
        description="Depthwise causal conv over [x, B, C]; index d_conv-1 is the newest tap.",
    ),
    TensorSpec(
        name="conv1d.bias",
        shape=("conv_dim",),
        dtype="float32",
        required_if="conv_bias",
        description="Present only when conv_bias=True.",
    ),
    TensorSpec(
        name="norm.weight",
        shape=("d_ssm",),
        dtype="float32",
        required_if="rmsnorm",
        description="Gated RMSNorm gain, grouped by d_ssm // ngroups, eps=1e-5.",
    ),
    TensorSpec(
        name="out_proj.weight",
        shape=("d_model", "d_inner"),
        dtype="float32",
        description="Output projection back to d_model.",
    ),
    TensorSpec(
        name="out_proj.bias",
        shape=("d_model",),
        dtype="float32",
        required_if="bias",
        description="Present only when bias=True.",
    ),
)

#: Upstream's own ``state_dict()`` ordering. Preserved so that a diff against
#: a real checkpoint reads in the order a human expects.
KEY_ORDER: tuple[str, ...] = (
    "dt_bias",
    "A_log",
    "D",
    "in_proj.weight",
    "in_proj.bias",
    "conv1d.weight",
    "conv1d.bias",
    "norm.weight",
    "out_proj.weight",
    "out_proj.bias",
)

#: The Niverel Foundation V3 block. ``headdim``, ``ngroups`` and ``chunk_size``
#: are never passed by Niverel, so upstream's defaults apply -- recording them
#: explicitly here is the whole point of pinning a reference configuration.
NIVEREL_V3 = {
    "d_model": 768,
    "d_state": 128,
    "d_conv": 4,
    "expand": 2,
    "headdim": 64,
    "ngroups": 1,
    "chunk_size": 256,
    "bias": False,
    "conv_bias": True,
}

#: Configurations the symbolic contract must reproduce exactly. Deliberately
#: exercises every conditional tensor and every derived-dimension formula,
#: including the options Niverel does not use.
VERIFICATION_SWEEP: tuple[dict[str, Any], ...] = (
    NIVEREL_V3,
    # Smallest sane block, used by the tiny maths fixture.
    {"d_model": 16, "d_state": 4, "d_conv": 4, "expand": 2, "headdim": 8, "ngroups": 1},
    # bias=True turns on in_proj.bias and out_proj.bias.
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "bias": True},
    # conv_bias=False removes conv1d.bias.
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "conv_bias": False},
    # rmsnorm=False removes norm.weight entirely.
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "rmsnorm": False},
    # D_has_hdim widens D from nheads to d_ssm.
    {"d_model": 64, "d_state": 16, "expand": 2, "headdim": 16, "D_has_hdim": True},
    # ngroups > 1 widens conv_dim and d_in_proj.
    {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "ngroups": 4},
    # d_ssm < d_inner opens the gated-MLP branch (d_mlp > 0).
    {"d_model": 128, "d_state": 32, "expand": 2, "headdim": 16, "d_ssm": 128},
    # Everything unusual at once.
    {
        "d_model": 96,
        "d_state": 8,
        "d_conv": 3,
        "expand": 4,
        "headdim": 12,
        "ngroups": 2,
        "bias": True,
        "conv_bias": False,
        "D_has_hdim": True,
        "norm_before_gate": True,
    },
)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _observed_state_dict(mamba2_cls: Any, overrides: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Instantiate the genuine upstream module and read its real shapes."""
    config = Mamba2Config(**overrides)
    module = mamba2_cls(**config.upstream_kwargs())
    return {name: tuple(tensor.shape) for name, tensor in module.state_dict().items()}


def verify_contract(contract: WeightContract, mamba2_cls: Any) -> list[str]:
    """Check the symbolic contract against the real module. Returns failures."""
    failures: list[str] = []
    for overrides in VERIFICATION_SWEEP:
        label = ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        config = Mamba2Config(**overrides)
        observed = _observed_state_dict(mamba2_cls, overrides)
        predicted = contract.expected(config)

        missing = [k for k in observed if k not in predicted]
        extra = [k for k in predicted if k not in observed]
        if missing:
            failures.append(f"[{label}] contract omits real tensors: {missing}")
        if extra:
            failures.append(f"[{label}] contract invents tensors upstream does not have: {extra}")
        for key in observed.keys() & predicted.keys():
            if observed[key] != predicted[key]:
                failures.append(
                    f"[{label}] {key}: upstream has {observed[key]}, contract predicts {predicted[key]}"
                )

        observed_order = list(observed)
        predicted_order = list(predicted)
        if observed_order != predicted_order:
            failures.append(
                f"[{label}] key order differs: upstream {observed_order} vs contract {predicted_order}"
            )
    return failures


def build_contract(source: UpstreamSource, mamba2_cls: Any) -> WeightContract:
    reference = Mamba2Config(**NIVEREL_V3)
    provenance = {
        "extracted_by": "scripts/extract_weight_contract.py",
        "extracted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "niverel_mamba_commit": _git_commit(),
        "torch_version": torch.__version__,
        "wheel_filename": source.wheel.name,
        "wheel_sha256": source.wheel_sha256(),
        "source_file_sha256": source.source_hashes,
        "verified_against_configurations": len(VERIFICATION_SWEEP),
        "note": (
            "Shapes are symbolic and were verified against the genuine upstream module "
            "instantiated for every configuration in VERIFICATION_SWEEP. Mamba2 2.3.2.post1 "
            "has no init_states tensor; the gated norm eps is hard-coded to 1e-5 in "
            "Mamba2.__init__, not 1e-6."
        ),
    }
    return WeightContract(
        schema_version=SCHEMA_VERSION,
        upstream_package="mamba-ssm",
        upstream_version=source.version,
        tensors=CONTRACT_TENSORS,
        key_order=KEY_ORDER,
        provenance=provenance,
        reference_configuration={
            "name": "niverel-foundation-v3",
            "source": "runtime_config.yaml of niverel-5b-v3-hnet-jepa-seed1337",
            "config": reference.to_dict(),
            "derived": {
                "d_inner": reference.d_inner,
                "d_ssm": reference.effective_d_ssm,
                "nheads": reference.nheads,
                "conv_dim": reference.conv_dim,
                "d_in_proj": reference.d_in_proj,
                "d_mlp": reference.d_mlp,
                "norm_group_size": reference.norm_group_size,
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=None, help="path to a mamba_ssm wheel")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the contract (default: schemas/mamba2-upstream-<version>.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only; fail if the contract on disk differs from the extracted one",
    )
    args = parser.parse_args()

    with upstream_mamba_ssm(wheel=args.wheel) as source:
        mamba2_cls = import_upstream_mamba2(source)
        print(f"upstream  mamba-ssm {source.version}  ({source.wheel.name})")
        print(f"module    {mamba2_cls.__module__}.{mamba2_cls.__name__}")

        contract = build_contract(source, mamba2_cls)

        print(f"\nverifying symbolic contract against {len(VERIFICATION_SWEEP)} real configurations")
        failures = verify_contract(contract, mamba2_cls)
        if failures:
            print("\nCONTRACT VERIFICATION FAILED\n")
            for failure in failures:
                print(f"  {failure}")
            return 1
        print("  all configurations reproduce upstream key sets, shapes and ordering")

        reference = Mamba2Config(**NIVEREL_V3)
        print(f"\nreference configuration (Niverel Foundation V3), {len(contract.expected(reference))} tensors:")
        for key, shape in contract.expected(reference).items():
            print(f"  {key:20s} {shape}")

    output = args.output or REPO_ROOT / "schemas" / f"mamba2-upstream-{source.version}.json"
    payload = json.dumps(contract.to_dict(), indent=2) + "\n"

    if args.check:
        if not output.is_file():
            print(f"\n{output} does not exist; run without --check to create it")
            return 1
        existing = json.loads(output.read_text(encoding="utf-8"))
        fresh = json.loads(payload)
        # Provenance carries timestamps and a commit id, so compare the parts
        # that actually define the contract.
        keys = ("schema_version", "upstream_package", "upstream_version", "tensors", "key_order")
        drift = [k for k in keys if existing.get(k) != fresh.get(k)]
        if drift:
            print(f"\nCONTRACT DRIFT in {drift}; regenerate schemas/ and review the diff")
            return 1
        print(f"\n{output.relative_to(REPO_ROOT)} is up to date")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"\nwrote {output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
