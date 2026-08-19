# Releasing

Push a tag. That is the whole procedure.

```bash
git tag -a v0.1.0 -m "niverel-mamba 0.1.0"
git push origin v0.1.0
```

`release.yml` then runs, in order:

| stage | what it does | roughly |
|---|---|---|
| `build-core` | ruff, mypy, the full suite, sdist and wheel, `twine check` | 2 min |
| `build-cuda` | three CUDA runtimes, in parallel | 70 min |
| `certify-sm80` | starts the A100 pod, certifies, stops it | 20 min |
| `certify-sm90` | the same on the H100 pod | 20 min |
| `github-release` | attaches distributions, CUDA wheels and reports | |
| `publish-testpypi` | TestPyPI, then a cold install from it | |
| `publish-pypi` | PyPI | |

**You approve once**, when the first GPU is about to start. Everything before
that is free.

The ordering is the safety property: nothing is published before the thing that
would have caught it wrong has run. Certification failing stops the release
outright — a version number on PyPI can never be reused, and an uncertified
CUDA wheel must never reach a user. `github-release` additionally reads every
certification report and refuses to attach anything if one says
`passed: false`.

To release an existing tag without re-tagging, dispatch `release.yml` and give
it the tag; every job then builds that ref rather than whatever `main` happens
to be.

---

## One-time setup

Five things, none of which can be automated from CI.

### 1. PyPI trusted publishers

The project does not exist on either index yet, so register a **pending**
publisher on <https://pypi.org/manage/account/publishing/> and again on
<https://test.pypi.org/manage/account/publishing/>:

| field | value |
|---|---|
| PyPI project name | `niverel-mamba` |
| Owner | `hallcyn` |
| Repository name | `niverel-mamba` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` on PyPI, `testpypi` on TestPyPI |

The environment name matters: `publish-pypi.yml` derives `environment.name`
from its input, and PyPI refuses the exchange on a mismatch.

There is no PyPI token anywhere in this repository and there should never be
one. A token is a long-lived credential that can be exfiltrated; an OIDC
exchange is scoped to one workflow run of one repository.

### 2. GitHub environments

Under **Settings → Environments**, create `pypi`, `testpypi` and `gpu`.

Put a **required reviewer** on `pypi` and on `gpu`. Publishing is irreversible,
and `gpu` is the gate that stops a tag push from renting hardware unattended.

### 3. Nothing, for the pods

There is nothing to create. Each certification pod is made when a release needs
it and destroyed immediately afterwards, so no volume bills between releases
and no self-hosted runner stands idle against a public repository.

Two details worth knowing, both of which came out of reading the API rather
than the docs:

**`POST /pods` has no required fields.** An empty body is accepted and RunPod
rents a GPU using its own defaults — this is not hypothetical, it happened
while probing the API and produced a live RTX 4090 at $0.74/hr. So
`_create_payload` sets every field that decides what gets rented, and a test
asserts it, name by name.

**Capacity is handled by offering the whole family, not by picking a region.**
`gpuTypeIds` is a list, so sm_80 asks for any A100 (`A100 80GB PCIe`,
`A100-SXM4-80GB`, `A100-SXM4-40GB`) and sm_90 for any H100 (`H100 80GB HBM3`,
`H100 NVL`, `H100 PCIe`). H100 stock is routinely exhausted in a given
datacenter; every H100 is sm_90, so letting RunPod place the pod is far more
robust than naming one. `countryCodes` defaults to `["US"]`, where capacity is
deepest.

Prices at the time of writing: A100 PCIe $1.19/hr, A100 SXM $1.39/hr, H100 NVL
$2.59/hr, H100 SXM $2.69/hr. A certification takes about fifteen minutes, so a
release costs roughly a dollar of GPU.

### 4. A runner, automatically

Nothing to install. The pod boots straight into a GitHub runner registered
`--ephemeral`, which takes exactly one job and retires.

> **This repository is public**, and GitHub advises against self-hosted runners
> on public repositories: a pull request could otherwise run arbitrary code on
> your machine. Three things keep this safe. The certification workflows
> trigger only on `workflow_dispatch` and `workflow_call` — **never**
> `pull_request`, and `tests/release/test_workflows.py` is where to add a guard
> if that is ever tempting to change. The runner is `--ephemeral`, so it cannot
> serve a second job. And the machine is destroyed minutes later.

### 5. Secrets

| secret | needed by | why |
|---|---|---|
| `RUNPOD_API_KEY` | `certify-cuda.yml` | creating and destroying the pods. **Read/Write**, and note that the REST API lives on `rest.runpod.io`, which is neither of the two permission groups RunPod names when you create a key. A key restricted to `api.runpod.ai` alone is refused with a bare `HTTP 401`. If a create fails that way, `python3 scripts/runpod_pod.py check-credentials` says whether reads work, which distinguishes a read-only key from one that does not cover the host at all |
| `RUNNER_PAT` | `certify-cuda.yml` | **required.** A fine-grained PAT limited to this repository with **Administration: read and write** as its only permission |

`RUNNER_PAT` cannot be avoided. Registering a self-hosted runner is a
repository-administration call, and `administration` is not among the scopes a
workflow may request — the valid ones are `actions`, `attestations`, `checks`,
`contents`, `deployments`, `discussions`, `id-token`, `issues`, `models`,
`packages`, `pages`, `pull-requests`, `repository-projects`, `security-events`
and `statuses`. So `GITHUB_TOKEN` can never mint a runner token, whatever it is
granted.

Asking for the scope anyway does not fail softly: GitHub rejects the workflow
file outright with `Unexpected value 'administration'`, which takes down every
workflow in the repository until it is removed.

Scope the PAT to this one repository and give it nothing but Administration.
That is enough to register a runner and not enough to do much else.
| `HF_TOKEN` | `certify-cuda.yml` | optional; without it the real V3 fixture skips by name rather than being silently absent |

---

## What bounds the GPU bill

Three things, because the failure that costs money is not "the pod would not
start" but "the pod started and nothing stopped it":

* approval on the `gpu` environment, so nothing begins without you;
* `timeout-minutes: 40` on each certification job, against a run that takes
  about fifteen — a runner that never registers costs minutes, not hours;
* `stop-pod` runs `if: always()`, and a `guard` job then asserts that no
  certification pod is left running. A teardown that silently failed becomes a
  red workflow rather than a slow leak.

`scripts/runpod_pod.py` holds that logic rather than the workflow YAML, so it
can be tested — and it is, including the paths that matter: stopping a pod that
does not exist, that never started, and when the API is unreachable.

Note that RunPod bills volume storage even while a pod is stopped. Two pods at
20 GB is roughly four dollars a month standing still, against about a dollar of
compute per release. If releases are rare, creating the pods on demand would
cost less than keeping them.

---

## What is not released this way

CUDA wheels for `mamba-ssm` and `causal-conv1d` are **never** published to
PyPI. They exceed PyPI's per-file limits, the full matrix runs to gigabytes,
and — decisively — wheel tags do not encode the Torch or CUDA version, so pip
would happily install an ABI-incompatible build when several look equivalent.

They go to GitHub Releases with a `niverel-mamba-binary-manifest-v1` document
recording each artefact's SHA-256, source commit and build workflow.
`niverel-mamba install-backend cuda` fetches that manifest, verifies the SHA
before installing, and refuses rather than compiling on the user's machine.

Until a certification run has passed against those exact artefacts, the
manifest carries `certification.status: uncertified`, `cuda-reference` is
published as `experimental`, and the `cuda_bfloat16` tolerance class in
`tolerances.yaml` keeps its starting values with `observed: null`. Replace that
block with the measured numbers, and only then change the published status.

---

## Version numbering

* **MAJOR** — the public API or the weight contract breaks
* **MINOR** — a new backend, PyTorch version, platform or capability
* **PATCH** — a fix with no contract change, a numerically compatible
  optimisation, or documentation

A change to the weight contract that alters an existing tensor is always MAJOR:
the whole promise of this package is that a checkpoint keeps loading.
