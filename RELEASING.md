# Releasing

`niverel-mamba` publishes through **PyPI Trusted Publishing** (OIDC). There is
no PyPI token in this repository, in its secrets, or on any maintainer's
machine, and there is not meant to be one: a token is a long-lived credential
that can be exfiltrated, whereas an OIDC exchange is scoped to one workflow run
of one repository.

## One-time setup

Both steps are done in a browser and cannot be automated from CI.

### 1. Register the pending publishers

The project does not exist on either index yet, so register a **pending**
publisher (PyPI calls it that for a project whose first release has not landed).

On <https://pypi.org/manage/account/publishing/> and again on
<https://test.pypi.org/manage/account/publishing/>:

| Field | Value |
|---|---|
| PyPI project name | `niverel-mamba` |
| Owner | `hallcyn` |
| Repository name | `niverel-mamba` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` on PyPI, `testpypi` on TestPyPI |

The environment name matters: `publish-pypi.yml` sets
`environment.name` from its `repository` input, and PyPI refuses the exchange
if it does not match.

### 2. Create the GitHub environments

In **Settings → Environments**, create `pypi` and `testpypi`.

Add a required reviewer to `pypi`. Publishing to the real index is
irreversible — a version number can never be reused — so it should take a
deliberate human approval, not just a green pipeline.

## Releasing

Everything before the first publish is already enforced by CI; the ordering
below exists so that nothing is published before the thing that would have
caught it wrong has run.

```bash
# 1. main is green, and the version is bumped in pyproject.toml
#    (src/niverel_mamba/version.py is checked against it by the test suite)

# 2. Tag. release.yml triggers on tags matching v*
git tag -a v0.1.0 -m "niverel-mamba 0.1.0"
git push origin v0.1.0
```

`release.yml` then runs, in order:

1. **build-core** — ruff, mypy, the full test suite, `python -m build`,
   `twine check`
2. **certify-gpu** — only when a `build-cuda-wheels` run is nominated via
   `wheel_run_id`; skipped for a core-only release
3. **github-release** — verifies every asset's SHA-256 against its manifest,
   then creates the release
4. **publish-testpypi** → cold install on Ubuntu and macOS, Python 3.10 and
   3.12, exercising `doctor`, `inspect`, `--version` and `verify`'s refusal
   path
5. **publish-pypi** → the same cold-install gate against the real index

Step 4 gating step 5 is the point. A package that builds and tests perfectly in
its own repository can still be unusable once installed: 0.1.0 nearly shipped
with a `doctor` that crashed on a missing `numpy`, and only a genuine cold
install surfaced it.

## What is *not* released this way

CUDA wheels for `mamba-ssm` and `causal-conv1d` are **never** published to
PyPI. They exceed PyPI's per-file limits, the full matrix runs to gigabytes,
and — decisively — wheel tags do not encode the Torch or CUDA version, so pip
would happily install an ABI-incompatible build when several look equivalent.

They go to GitHub Releases with a `niverel-mamba-binary-manifest-v1` document
recording each artefact's SHA-256, source commit and build workflow.
`niverel-mamba install-backend cuda` fetches that manifest, verifies the SHA
before installing, and refuses rather than compiling on the user's machine.

Those wheels must not be attached to a release until
`certify-cuda-sm80` / `certify-cuda-sm90` have produced a passing report
against those exact artefacts on real hardware. Until then the manifest carries
`certification.status: uncertified` and `cuda-reference` is published as
`experimental`.

## Version numbering

* **MAJOR** — the public API or the weight contract breaks
* **MINOR** — a new backend, PyTorch version, platform or capability
* **PATCH** — a fix with no contract change, a numerically compatible
  optimisation, or documentation

A change to the weight contract that alters an existing tensor is always MAJOR:
the whole promise of this package is that a checkpoint keeps loading.
