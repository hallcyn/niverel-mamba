#!/usr/bin/env python
"""Generate a static PEP 503 index for the certified CUDA wheels.

Optional. The primary distribution channel is GitHub Releases plus the
manifest-driven ``niverel-mamba install-backend cuda``, for the reasons the
brief sets out: the wheels exceed PyPI's default size limits, the full matrix
would run to gigabytes, and -- most importantly -- wheel tags do not encode the
Torch or CUDA version, so pip could happily select an ABI-incompatible build
when several look equivalent.

A PEP 503 index inherits that last problem, so if you serve one, serve **one
index per Torch/CUDA pair** and let users point at the right one:

    pip install mamba-ssm --index-url https://.../torch213-cu130/simple/
"""

from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path

TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>{title}</title>
  </head>
  <body>
    <h1>{title}</h1>
{links}
  </body>
</html>
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(name: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="where the wheels are served from")
    args = parser.parse_args()

    wheels = sorted(args.wheelhouse.glob("*.whl"))
    if not wheels:
        print(f"no wheels in {args.wheelhouse}")
        return 1

    by_project: dict[str, list[Path]] = {}
    for wheel in wheels:
        project = normalise(wheel.name.split("-")[0])
        by_project.setdefault(project, []).append(wheel)

    root_links = "\n".join(
        f'    <a href="{project}/">{project}</a><br>' for project in sorted(by_project)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(
        TEMPLATE.format(title="niverel-mamba certified wheels", links=root_links),
        encoding="utf-8",
    )

    for project, items in by_project.items():
        directory = args.output / project
        directory.mkdir(parents=True, exist_ok=True)
        links = []
        for wheel in sorted(items):
            digest = sha256_file(wheel)
            url = f"{args.base_url.rstrip('/')}/{wheel.name}#sha256={digest}"
            links.append(f'    <a href="{html.escape(url)}">{html.escape(wheel.name)}</a><br>')
        (directory / "index.html").write_text(
            TEMPLATE.format(title=project, links="\n".join(links)), encoding="utf-8"
        )
        print(f"{project}: {len(items)} file(s)")

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
