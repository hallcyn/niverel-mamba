# Contributing

## Setup

```bash
uv sync --extra torch --extra mlx --extra dev
uv run python scripts/make_golden_fixture.py
uv run pytest -q
```

The real Foundation V3 fixture needs an `HF_TOKEN` in `.env` (the checkpoint
repository is private) and downloads ~1.7 GB:

```bash
uv run python scripts/make_golden_fixture.py --niverel
```

Tests that need it skip by name when it is absent. Nothing is masked with
`continue-on-error`.

## The rules that matter

### 1. Never widen a tolerance to make a test pass

`src/niverel_mamba/certification/tolerances.yaml` holds observed, then sealed,
values. If a measurement exceeds its band, that is a finding: report it,
investigate it, and fix the cause. Raising the number is not a fix. The
`observed` blocks exist so drift is visible without re-running anything.

### 2. Never delete the sequential oracle

`torch_ops/ssd_sequential.py` is slow on purpose. It is the source of truth
every other implementation is checked against. It stays even when the chunked
path is faster in every respect.

### 3. Never introduce a silent fallback

An explicitly requested backend either works or raises. `backend="auto"` may
choose, but must report what it chose. If you find yourself writing
`except ImportError: use_cpu_instead()`, stop.

### 4. Never use `strict=False`

Weight loading is strict. A missing key, an unexpected key, a differing shape
or an incompatible configuration is refused, with a message that says which.

### 5. Never claim a certification that has not been measured

A backend's `Certification` reflects what has actually been run, and every
report names the reference it was compared against. `cuda-reference` is
`experimental` until a GPU job produces a report, however obviously correct
wrapping upstream may seem.

### 6. Keep the import graph clean

Importing `niverel_mamba` must not import torch or MLX, must not touch the
network, must not spawn a subprocess and must not compile anything. This is
enforced by `tests/release/test_packaging.py` via an audit hook.

## GitHub Actions versions

Actions are pinned to a major tag and kept current by Dependabot. Two are
deliberate exceptions:

* `pypa/gh-action-pypi-publish@release/v1` is the floating major-version branch
  PyPA documents. It is not a stale pin, and it must not be changed to a
  numeric tag.
* `astral-sh/setup-uv` is pinned to a full version (`v10.0.1`), not a major
  tag, because it does not publish a floating `v10`. Referencing `@v10` fails
  the whole job at setup time with "unable to find version". Every other action
  here does publish major tags.
* `upload-artifact` and `download-artifact` version independently, so their
  major numbers differ. Any v4-or-later pairing interoperates; the incompatible
  break was v3 to v4, when the artifact backend changed.

Before bumping an action across a major version, check that the inputs this
repository actually passes still exist in the new `action.yml`. Release notes
for a patch tag will not tell you.

## Releasing

See `RELEASING.md`. Publishing goes through PyPI Trusted Publishing; there is
no token in this repository and there should never be one.

## Adding a backend

1. Implement `backends/base.Backend`.
2. Register it in `registry.BACKENDS` with an honest `Certification` — start at
   `experimental`.
3. Add an availability probe to `registry._available_devices` whose failure
   `reason` tells the user what to install.
4. Add a test suite under `tests/<backend>/` that compares it against the
   oracle on the tiny, segmented and (where available) Niverel fixtures.
5. Only then, once the numbers exist, propose a status change.

## Changing the weight contract

The contract is not hand-edited. Change `scripts/extract_weight_contract.py`,
re-run it, and commit the regenerated `schemas/` file with the diff visible.
`--check` fails CI on drift.

A contract change that alters existing tensors is a **MAJOR** version bump.

## Style

`ruff` and `mypy --strict` on `src/`. Comments should explain *why*, especially
where the code deliberately diverges from upstream — those divergences are
documented at the point they occur, and in `THIRD_PARTY_NOTICES.md`.
