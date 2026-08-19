"""The ``niverel-mamba`` command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..errors import NiverelMambaError
from ..version import __version__

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="niverel-mamba",
        description="Portable, verifiable, multi-backend Mamba2 runtime.",
    )
    parser.add_argument("--version", action="version", version=f"niverel-mamba {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Imported lazily inside each module so that `--help` stays fast and does
    # not import torch or MLX.
    from . import doctor, inspect, install_backend, verify

    doctor.add_parser(subparsers)
    inspect.add_parser(subparsers)
    verify.add_parser(subparsers)
    install_backend.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except NiverelMambaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
