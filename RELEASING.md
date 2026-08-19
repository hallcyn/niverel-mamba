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

### 3. The RunPod pods

Create two pods by hand, named exactly:

* `niverel-mamba-certif-a100` — an A100, for sm_80
* `niverel-mamba-certif-h100` — an H100, for sm_90

Template: **RunPod PyTorch 2.8.0**
(`runpod/pytorch:...-cu1281-torch280-ubuntu2404`). It is the only offered
template on Ubuntu 24.04, hence the only one with **Python 3.12** — and the
wheels are tagged `cp312`, so on the py3.11 and py3.10 templates they simply
refuse to install. Its CUDA 12.8.1 also matches the cu128 reference.

**Volume: 20 GB.** Measured rather than guessed: a CUDA torch install is 3.0 GB
(2.6 GB of wheels), our two CUDA wheels add about 1 GB installed, and the
runner and checkout another 0.5 GB. That is 4.5 GB before caches; the pip cache
adds 2.6 GB and the real Foundation V3 checkpoint another 1.7 GB, which is how
10 GB runs out. 20 GB also leaves room to certify a second torch runtime later
without a resize.

The architecture is not negotiable. A wheel built with
`TORCH_CUDA_ARCH_LIST="8.0;9.0"` contains cubins for sm_80 and sm_90 and
nothing else, and will not start on anything else:

```
no kernel image is available for execution on the device
```

So an Ada card (sm_89 — RTX 4090, L40S, RTX 2000 Ada) cannot certify these
wheels, however capable it is. The certification job checks the device it
landed on and refuses to write a report labelled with an architecture it is not
running on.

### 4. A runner on each pod

Each pod carries a self-hosted runner that comes up with it. Install it once:

```bash
# On your machine: mint a short-lived registration token.
gh api -X POST repos/hallcyn/niverel-mamba/actions/runners/registration-token --jq .token

# On the pod, under /workspace so it survives a stop.
export RUNNER_ALLOW_RUNASROOT=1   # RunPod containers run as root
mkdir -p /workspace/actions-runner && cd /workspace/actions-runner
LATEST=$(curl -s https://api.github.com/repos/actions/runner/releases/latest \
         | grep -oP '"tag_name": "v\K[^"]+')
curl -sL -o runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${LATEST}/actions-runner-linux-x64-${LATEST}.tar.gz"
tar xzf runner.tar.gz

# Labels must match exactly; the workflow selects on them.
./config.sh --url https://github.com/hallcyn/niverel-mamba --token <TOKEN> \
  --labels self-hosted,linux,x64,cuda,sm80 --unattended --name certif-a100
./svc.sh install && ./svc.sh start
```

Use `sm90` and `certif-h100` on the other pod. Installing it as a service means
it re-registers when you start the pod, which is what lets the workflow do
nothing but start and stop.

> **This repository is public**, and GitHub advises against self-hosted runners
> on public repositories: a pull request could otherwise run arbitrary code on
> your machine. Two things keep this safe, and both must stay true. The
> certification workflows trigger only on `workflow_dispatch` and
> `workflow_call` — **never** `pull_request`; `tests/release/test_workflows.py`
> is where to add a guard if that is ever tempting to change. And the pods are
> stopped except during a release, so the window is minutes per release rather
> than always.

### 5. Secrets

| secret | needed by | why |
|---|---|---|
| `RUNPOD_API_KEY` | `certify-cuda.yml` | starting and stopping the pods |
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
