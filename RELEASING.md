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
| Workflow name | `release.yml` |
| Environment name | `pypi` on PyPI, `testpypi` on TestPyPI |

**The workflow name must be `release.yml`, the entry workflow -- not the file
the upload step happens to live in.** The upload used to sit in a reusable
`publish-pypi.yml`, and that cannot work. PyPI matches the trusted publisher
using `job_workflow_ref`, which names the reusable workflow, but verifies the
PEP 740 attestation against the certificate's Build Config URI, which names the
workflow that was *triggered*. With a reusable publisher those two disagree by
construction, and the upload is refused:

```
Certificate's Build Config URI (.../release.yml@refs/tags/v0.1.0)
does not match expected Trusted Publisher (publish-pypi.yml @ hallcyn/niverel-mamba)
```

That is [pypa/gh-action-pypi-publish#283][283], and it cost a release that had
already paid for both GPU certifications. The upload therefore lives inline in
`release.yml`, and a test refuses any future attempt to move it back into a
reusable workflow.

[283]: https://github.com/pypa/gh-action-pypi-publish/issues/283

The environment name matters too: the publishing jobs declare `pypi` and
`testpypi`, and PyPI refuses the exchange on a mismatch.

There is no PyPI token anywhere in this repository and there should never be
one. A token is a long-lived credential that can be exfiltrated; an OIDC
exchange is scoped to one workflow run of one repository.

## What a certification report is a claim about

Each certification run writes **two** reports per runtime, because a report
certifies exactly one backend:

| report | candidate | reference |
|---|---|---|
| `...-torch-reference.json` | `torch-reference-cpu-chunked` | its own float64 sequential oracle |
| `...-cuda-reference.json` | `cuda-reference` (upstream kernels, bfloat16) | `torch-reference` on the same GPU |

v0.1.0 shipped only the first, on both architectures. Every one passed, and not
one of them said anything about the CUDA kernels -- they were read as CUDA
certification because of the hardware they ran on and the name of the file they
landed in. `scripts/verify_certification_reports.py` now refuses a release whose
reports do not certify `cuda-reference` on both sm80 and sm90, and it refuses
v0.1.0's own reports when pointed at them.

## Measure before you gate

A tolerance that has never been observed is a guess, and scoring against a guess
on rented hardware wastes the hardware. Three certification runs were lost that
way, each to a band sized for arithmetic the machine does not perform:

| run | band assumed | what the hardware actually does |
|---|---|---|
| 1 | bfloat16 close to float32 | bfloat16 carries 8 mantissa bits; rounding the *inputs* alone already left the band |
| 2 | the same, at equal data | the kernels still accumulate in bfloat16 internally |
| 3 | float32 close to float32 | upstream's 30 `tl.dot` calls pass no `allow_tf32`, so Ampere multiplies in TF32: 10 mantissa bits |

So measurement comes first:

```console
$ gh workflow run release.yml -f tag=v0.1.0 -f wheel_run_id=32304273715 \
    -f certify_only=true -f measure_only=true
```

The campaign reports every comparison and succeeds whatever the numbers, so one
run yields evidence on both architectures. The reports are marked
`mode: measurement`, carry no verdict, and the release gate keeps refusing them
-- measuring is not certifying.

Seal the observed values into `certification/tolerances.yaml`, and the CUDA
parity tests stop skipping: `requires_measured_tolerance` keeps them dormant
until their class has an `observed` block, so no test asserts a bound nobody has
seen. Then run again without `measure_only` to certify against what was
measured.

## Not paying for the same work twice

Two costs dominate a release: three CUDA builds at roughly seventy minutes each,
and two GPU certifications at about two dollars a run. Neither should be paid
for a change that cannot affect the result.

**Reusing wheels, automatically.** The CUDA wheels are upstream's, at pinned
versions, built from Dockerfiles that change rarely. A release that only changes
this package's own code produces identical wheels, so `resolve-wheels` looks for
a run that already built them from the same inputs -- the Dockerfiles, the
target matrix, and the pinned upstream versions, each read at both commits
through the API and compared by blob SHA.

Nothing to pass, and nothing to remember:

```console
$ git tag -a v0.1.1 -m "niverel-mamba 0.1.1" && git push origin v0.1.1
```

`build-cuda` is skipped when a match is found, and the certification and the
release both read from the resolved run, so the wheels attached are the wheels
certified. It fails safe towards building: a changed Dockerfile, an expired
artifact, a missing target, an API that will not answer -- any of these and the
wheels are rebuilt.

`-f wheel_run_id=<id>` still overrides the search when you want a specific run.

**Producing evidence without releasing.** To certify a tag that is already
published -- as v0.1.0 needed, having shipped without any CUDA report:

```console
$ gh workflow run release.yml -f tag=v0.1.0 \
    -f wheel_run_id=32304273715 -f certify_only=true
```

It stops after the reports. Nothing is released, nothing is uploaded, and a
final job runs the same gate the release runs and prints every measured error to
the run summary -- so a tolerance can be sealed from evidence rather than from
the brief's starting values. Without it the run would continue to the PyPI
upload and fail there, because that version already exists.

**Rehearsing the certification for nothing.** `ci-certify-rehearsal.yml` runs
everything the pod does except the numerical comparison, on a free Linux runner,
installing the wheels from the published release. Three certification runs have
been paid for and thrown away by faults that had nothing to do with CUDA --
wheels merged into one directory, a fixture venv without torch, an upstream
package that would not import. Every one of them would have surfaced there, for
nothing, in ten minutes. It runs automatically on any pull request that touches
the certification path.

## Re-publishing a tag that was already certified

If a release fails *after* certification -- as one did, on the PyPI upload --
do not re-run the whole workflow. Certification rents two GPUs, and a run that
already produced passing reports has nothing left to prove.

Instead dispatch `release.yml` on the same tag with **publish_only** ticked:

```console
$ gh workflow run release.yml -f tag=v0.1.0 -f publish_only=true
```

Dispatch it from the default branch, which is where it runs from by default.
The workflow and the composite action it uses are read from the ref the run
starts on, while `tag` only decides which release's assets are published -- so
re-publishing an old tag does not require that tag to contain any of this.

Everything up to and including `github-release` is skipped. The distributions
are taken from the release assets rather than rebuilt, and before anything is
uploaded the certification reports attached to that release are re-read and
must all pass, covering both sm80 and sm90. `publish_only` is a way to avoid
paying twice, never a way to publish something uncertified -- the script that
enforces that is extracted from the workflow and executed by
`tests/release/test_publish_only_guard.py`.

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
| `RUNPOD_API_KEY` | `certify-cuda.yml` | creating and destroying the pods |
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
