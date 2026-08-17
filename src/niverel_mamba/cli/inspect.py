"""``niverel-mamba inspect`` -- show the weight contract, or check a checkpoint.

    niverel-mamba inspect                          # the contract itself
    niverel-mamba inspect --config d_model=768     # resolved for a config
    niverel-mamba inspect --checkpoint ckpt.pt     # find blocks and validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..config import Mamba2Config
from ..errors import WeightContractError
from ..schema import default_contract, load_contract

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser("inspect", help="inspect the weight contract or a checkpoint")
    parser.add_argument("--contract", type=Path, default=None, help="a contract JSON to load")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="configuration override, repeatable (e.g. --config d_model=768)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="a .pt/.safetensors to examine")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(func=run)
    return parser


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--config expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        value: Any
        low = raw.strip().lower()
        if low in ("true", "false"):
            value = low == "true"
        elif low in ("none", "null"):
            value = None
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        out[key.strip()] = value
    return out


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file

        return dict(load_file(str(path)))
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "module"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not contain a state dict")
    return payload


def run(args: argparse.Namespace) -> int:
    contract = load_contract(args.contract) if args.contract else default_contract()
    overrides = _parse_overrides(args.config)

    if args.checkpoint is None:
        if not overrides:
            payload = contract.to_dict()
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"contract        {contract.schema_version}")
                print(f"upstream        {contract.upstream_package} {contract.upstream_version}")
                ref = contract.reference_configuration
                if ref:
                    print(f"reference       {ref.get('name')}")
                print()
                print("tensors:")
                for tensor in contract.tensors:
                    gate = f"  (only when {tensor.required_if}=True)" if tensor.required_if else ""
                    print(f"  {tensor.name:<20} {list(tensor.shape)}  {tensor.dtype}{gate}")
            return 0

        config = Mamba2Config(**overrides)
        expected = contract.expected(config)
        if args.json:
            print(json.dumps({"config": config.to_dict(), "tensors": {k: list(v) for k, v in expected.items()}}, indent=2))
            return 0
        print(f"d_inner={config.d_inner}  d_ssm={config.effective_d_ssm}  nheads={config.nheads}  "
              f"conv_dim={config.conv_dim}  d_in_proj={config.d_in_proj}  d_mlp={config.d_mlp}")
        print()
        for key, shape in expected.items():
            print(f"  {key:<20} {shape}")
        return 0

    # Checkpoint mode
    from ..adapters.upstream import find_blocks

    state = _load_checkpoint(args.checkpoint)
    if not overrides:
        raise SystemExit(
            "--checkpoint needs a configuration too, e.g.\n"
            "  niverel-mamba inspect --checkpoint ckpt.pt --config d_model=768 --config d_state=128"
        )
    config = Mamba2Config(**overrides)

    blocks = find_blocks(state, config)
    if args.json:
        print(json.dumps({
            "checkpoint": str(args.checkpoint),
            "tensors": len(state),
            "blocks": sorted(blocks),
            "config": config.to_dict(),
        }, indent=2))
        return 0 if blocks else 1

    print(f"checkpoint      {args.checkpoint}")
    print(f"tensors         {len(state)}")
    print(f"Mamba2 blocks   {len(blocks)}")
    for name in sorted(blocks):
        print(f"  {name}")
    if not blocks:
        print()
        print("No block satisfies the contract for this configuration.")
        return 1

    from ..weights import validate_state_dict

    first = sorted(blocks)[0]
    try:
        validate_state_dict(blocks[first], config, contract=contract)
    except WeightContractError as exc:
        print()
        print(f"{first} does NOT satisfy the contract:\n{exc}")
        return 1
    print()
    print(f"{first} satisfies the contract (strict load would succeed).")
    return 0
