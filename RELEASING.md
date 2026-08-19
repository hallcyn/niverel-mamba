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

## Certifying the CUDA backend on rented hardware

`cuda-reference` stays `experimental` until a GPU has actually run it. Two
things are worth separating, because conflating them wastes money.

**Building needs no GPU.** `nvcc` and the toolkit suffice, which is why
`build-cuda-wheels.yml` compiles in Docker on free GitHub runners. Never pay
GPU-hours to compile.

**Certifying needs the exact architecture.** A wheel built with
`TORCH_CUDA_ARCH_LIST="8.0;9.0"` contains cubins for sm_80 and sm_90 and
nothing else. It will not start on anything else:

```
no kernel image is available for execution on the device
```

So an Ada card (sm_89: RTX 4090, L40S, RTX 2000 Ada) cannot certify these
wheels, however capable it is. sm_80 means an A100; sm_90 means an H100. A
certification run is roughly fifteen minutes, so renting one of each costs
about a dollar.

### Renting a pod

On RunPod, pick the **PyTorch 2.8.0** template
(`runpod/pytorch:...-cu1281-torch280-ubuntu2404`). It is the only one of the
offered templates on Ubuntu 24.04, hence the only one with **Python 3.12** —
and our wheels are tagged `cp312`, so on the py3.11 and py3.10 templates they
simply refuse to install. Its CUDA 12.8.1 also matches the cu128 reference.
The template's preinstalled torch is irrelevant; we install our own from the
pinned index.

### Registering it as a runner

`certify-cuda-sm80.yml` and `certify-cuda-sm90.yml` already target
`[self-hosted, linux, x64, cuda, sm80]` and `sm90`, so a pod carrying those
labels picks the job up with no workflow change.

```bash
# On your machine: mint a short-lived registration token.
gh api -X POST repos/hallcyn/niverel-mamba/actions/runners/registration-token --jq .token

# On the pod. RunPod containers run as root, which the runner refuses by
# default; the env var is the sanctioned override.
export RUNNER_ALLOW_RUNASROOT=1
mkdir -p /actions-runner && cd /actions-runner
LATEST=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
curl -sL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${LATEST}/actions-runner-linux-x64-${LATEST}.tar.gz"
tar xzf runner.tar.gz
./config.sh --url https://github.com/hallcyn/niverel-mamba \
  --token <TOKEN> --labels self-hosted,linux,x64,cuda,sm80 \
  --ephemeral --unattended --name runpod-a100
./run.sh
```

Then dispatch `certify-cuda-sm80` with the `build-cuda-wheels` run id to
certify. The job refuses to run if the GPU it lands on is not the architecture
the workflow claims, so a mislabelled pod fails loudly rather than producing a
report labelled `sm_80` from something else.

### Why `--ephemeral` is not optional here

**This repository is public.** GitHub advises against self-hosted runners on
public repositories, because a pull request can otherwise run arbitrary code on
your machine. Two things keep this safe, and both must stay true:

* the certification workflows trigger only on `workflow_dispatch` and
  `workflow_call` — **never** `pull_request`. `tests/release/test_workflows.py`
  is the place to add a guard if that is ever tempting to change;
* `--ephemeral` retires the runner after one job, so nothing persists between
  runs. Destroy the pod afterwards.

### Afterwards

A passing report is what promotes the backend. Until then, and this is
enforced by the tests rather than by good intentions: the binary manifest
carries `certification.status: uncertified`, `cuda-reference` is published as
`experimental`, and the `cuda_bfloat16` tolerance class in `tolerances.yaml`
carries the brief's starting values with `observed: null`. Replace that block
with the measured numbers, and only then change the published status.

## Version numbering

* **MAJOR** — the public API or the weight contract breaks
* **MINOR** — a new backend, PyTorch version, platform or capability
* **PATCH** — a fix with no contract change, a numerically compatible
  optimisation, or documentation

A change to the weight contract that alters an existing tensor is always MAJOR:
the whole promise of this package is that a checkpoint keeps loading.
