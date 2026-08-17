"""``niverel-mamba doctor`` -- what can this machine actually run?

Pure inspection. Nothing is downloaded, installed or compiled.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ..capabilities import detect_environment
from ..errors import BackendUnavailableError
from ..registry import all_statuses, resolve
from ..version import __version__

__all__ = ["add_parser", "run"]


def add_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser("doctor", help="report frameworks, devices and backend status")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(func=run)
    return parser


def _fmt(info: Any, when_absent: str = "unavailable") -> str:
    if not info.available:
        return f"{when_absent}" + (f" ({info.detail})" if info.detail else "")
    parts = [info.version or "available"]
    if info.detail:
        parts.append(f"({info.detail})")
    return " ".join(parts)


def run(args: argparse.Namespace) -> int:
    env = detect_environment()
    statuses = all_statuses(env)

    try:
        recommended = resolve("auto", env=env)
    except BackendUnavailableError:
        recommended = None

    if args.json:
        payload = {
            "niverel_mamba": __version__,
            "environment": env.to_dict(),
            "backends": [s.to_dict() for s in statuses],
            "recommended": recommended.to_dict() if recommended else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"niverel-mamba {__version__}")
    print()
    print(f"{'Python':<16}{env.python_version}")
    print(f"{'Platform':<16}{env.platform_system} {env.platform_machine}")
    print(f"{'Framework':<16}torch {_fmt(env.torch, 'not installed')}")
    print(f"{'MPS':<16}{'available' if env.mps.available else _fmt(env.mps)}")
    print(f"{'MLX':<16}{_fmt(env.mlx, 'not installed')}")
    print(f"{'CUDA':<16}{_fmt(env.cuda)}")
    print(f"{'mamba-ssm':<16}{_fmt(env.upstream_mamba_ssm, 'not installed')}")
    print(f"{'causal-conv1d':<16}{_fmt(env.causal_conv1d, 'not installed')}")
    print()
    print("Available backends:")
    width = max(len(s.spec.name) for s in statuses)
    for status in statuses:
        mark = "yes" if status.available else "no "
        note = status.spec.certification.value if status.available else (status.reason or "")
        print(f"  {status.spec.name:<{width}}   {mark}   {note}")
    print()
    print("Recommended backend:")
    if recommended is None:
        print("  none -- install PyTorch or MLX")
        return 1
    print(f"  {recommended.backend} / {recommended.device}")
    return 0
