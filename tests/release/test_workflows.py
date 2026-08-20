"""Static checks on the GitHub Actions workflows.

These exist because the two mistakes they catch are only detectable when a
release actually runs -- by which point a GitHub Release may already be cut and
TestPyPI may already hold a version number that can never be reused.

Both were real:

* `release.yml` called `publish-pypi.yml` without granting `id-token: write`.
  A called workflow can only downgrade the caller's permissions, so Trusted
  Publishing would have failed with `unable to get ACTIONS_ID_TOKEN_REQUEST_URL`
  after the release was already half-published.
* `release.yml` called `certify-cuda-sm80.yml` without granting `actions: read`,
  which made the whole file invalid -- and permission compatibility is checked
  when the file is *parsed*, so the `if:` guarding that job did not help.

Everything here is offline: it parses YAML and compares, nothing more.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is in the dev extra")

WORKFLOW_DIR = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows"

#: `on` is parsed as the boolean True by PyYAML, which reads it as YAML 1.1.
_ON_KEYS = (True, "on")


def _workflows() -> dict[str, Any]:
    return {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in WORKFLOW_DIR.glob("*.yml")}


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    for key in _ON_KEYS:
        if key in workflow:
            return workflow[key] or {}
    return {}


def _requested_permissions(callee: dict[str, Any]) -> dict[str, str]:
    """Everything a called workflow asks for, at file and job level."""
    requested = dict(callee.get("permissions") or {})
    for job in (callee.get("jobs") or {}).values():
        requested.update(job.get("permissions") or {})
    return requested


def _granted_permissions(workflow: dict[str, Any], job: dict[str, Any]) -> dict[str, str]:
    return job.get("permissions", workflow.get("permissions") or {})


def _satisfies(granted: str | None, needed: str) -> bool:
    if needed == "write":
        return granted == "write"
    if needed == "read":
        return granted in ("read", "write")
    return True


def test_every_workflow_is_valid_yaml_with_jobs():
    workflows = _workflows()
    assert workflows, f"no workflows found under {WORKFLOW_DIR}"
    for name, workflow in workflows.items():
        assert workflow, f"{name} is empty"
        assert workflow.get("jobs"), f"{name} declares no jobs"


def test_local_reusable_workflow_calls_resolve():
    workflows = _workflows()
    for name, workflow in workflows.items():
        for job_id, job in workflow["jobs"].items():
            uses = job.get("uses", "")
            if uses.startswith("./.github/workflows/"):
                target = uses.split("/")[-1]
                assert target in workflows, f"{name}:{job_id} calls missing workflow {target}"


def test_callers_grant_every_permission_their_callees_request():
    """The check that would have stopped two broken releases.

    A called workflow can only ever downgrade the caller's permissions, never
    add to them -- and this is enforced when the workflow file is parsed, so a
    job that would have been skipped still invalidates the whole file.
    """
    workflows = _workflows()
    problems = []
    for name, workflow in workflows.items():
        for job_id, job in workflow["jobs"].items():
            uses = job.get("uses", "")
            if not uses.startswith("./.github/workflows/"):
                continue
            callee = workflows[uses.split("/")[-1]]
            granted = _granted_permissions(workflow, job)
            for scope, needed in _requested_permissions(callee).items():
                if not _satisfies(granted.get(scope), needed):
                    problems.append(
                        f"{name}:{job_id} calls {uses.split('/')[-1]} which requests "
                        f"{scope}: {needed}, but the caller grants {scope}: "
                        f"{granted.get(scope, 'none')}"
                    )
    assert not problems, "\n".join(problems)


def _publishing_jobs() -> dict[str, dict[str, Any]]:
    """Every job that runs the PyPI upload action, wherever it lives."""
    found = {}
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                if "gh-action-pypi-publish" in str(step.get("uses", "")):
                    found[f"{name}:{job_id}"] = job
    return found


def test_publishing_jobs_request_an_oidc_token():
    """Trusted Publishing is the only sanctioned route; it needs id-token."""
    jobs = _publishing_jobs()
    assert jobs, "no publishing job found"
    for where, job in jobs.items():
        assert job.get("permissions", {}).get("id-token") == "write", where
        assert job.get("environment", {}).get("name"), where


def test_publishing_never_happens_from_a_reusable_workflow():
    """PyPI cannot verify an attestation produced by a reusable workflow.

    It checks the PEP 740 certificate's Build Config URI, which names the
    **entry** workflow, against the Trusted Publisher, which it matched using
    `job_workflow_ref` -- the **reusable** one. With a reusable publisher those
    two disagree by construction and the upload is refused:

        Certificate's Build Config URI (... /release.yml@refs/tags/v0.1.0)
        does not match expected Trusted Publisher (publish-pypi.yml @ ...)

    That is pypa/gh-action-pypi-publish#283, and it cost a release that had
    already paid for two GPU certifications. The upload step therefore has to
    stay in a workflow that is never `uses:`-ed by another.
    """
    workflows = _workflows()
    called = {
        job["uses"].split("/")[-1]
        for workflow in workflows.values()
        for job in workflow["jobs"].values()
        if str(job.get("uses", "")).startswith("./.github/workflows/")
    }
    offenders = [
        where for where in _publishing_jobs() if where.split(":")[0] in called
    ]
    assert not offenders, (
        f"these jobs upload to PyPI from a reusable workflow: {offenders}. "
        "PyPI will reject the attestation."
    )


def test_no_pypi_token_is_referenced_anywhere():
    """A token in a workflow would defeat the point of Trusted Publishing.

    `password:` is caught too, since that is how a token would be handed to
    twine -- with one exception. Logging in to GHCR to store the Docker layer
    cache genuinely needs a password field, and the only value tolerated there
    is `secrets.GITHUB_TOKEN`: an ambient, job-scoped credential that expires
    with the run, not a long-lived registry or index token.
    """
    offenders = []
    for path in WORKFLOW_DIR.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for marker in ("PYPI_TOKEN", "PYPI_API_TOKEN", "TWINE_PASSWORD"):
            if marker in text:
                offenders.append(f"{path.name} references {marker}")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("password:"):
                continue
            value = stripped.split(":", 1)[1].strip()
            if value != "${{ secrets.GITHUB_TOKEN }}":
                offenders.append(f"{path.name} sets password to {value!r}")
    assert not offenders, offenders


def test_cuda_wheels_are_not_built_on_every_tag():
    """Tagging a core-only release must not start hours of CUDA compilation.

    `release.yml` only attaches CUDA wheels when a build run is nominated
    explicitly via `wheel_run_id`, so a tag-triggered build produces artefacts
    nothing consumes. Note also that `paths:` filters are ignored for tag
    pushes, so a filter would not have limited it either.
    """
    push = _triggers(_workflows()["build-cuda-wheels.yml"]).get("push") or {}
    assert "tags" not in push, "build-cuda-wheels must not trigger on tags"


def test_release_publishes_to_testpypi_before_pypi():
    """The ordering is the safety property: a cold install gates the real index.

    Checked transitively, because the chain runs through the cold-install job
    that is the entire point of publishing to TestPyPI first.
    """
    jobs = _workflows()["release.yml"]["jobs"]
    reached, frontier = set(), {"publish-pypi"}
    while frontier:
        reached |= frontier
        frontier = {
            need
            for job_id in frontier
            for need in _needs(jobs[job_id])
            if need not in reached
        }
    assert "publish-testpypi" in reached
    assert any(job_id.startswith("cold-install") for job_id in reached), (
        "PyPI must be gated by a cold install from TestPyPI"
    )


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def test_descendants_of_skippable_jobs_declare_an_explicit_condition():
    """A skip propagates transitively, and `always()` only rescues its own job.

    This cost a release. `certify-gpu` is skipped for a core-only release.
    `github-release` survived because it carries `if: always() && ...`. But
    `publish-testpypi` and `publish-pypi` had no `if` of their own, so GitHub
    skipped them too -- and the run still reported **success**, having created
    an empty GitHub Release and published nothing to either index.

    Any job downstream of one that can be skipped therefore has to state its
    own condition in terms of `needs.<job>.result`.
    """
    for name, workflow in _workflows().items():
        jobs = workflow["jobs"]
        skippable = {
            job_id for job_id, job in jobs.items()
            if job.get("if") and "cancelled" not in str(job["if"])
        }
        if not skippable:
            continue
        # Walk the needs graph outwards from every skippable job.
        tainted, frontier = set(), set(skippable)
        while frontier:
            tainted |= frontier
            frontier = {
                job_id for job_id, job in jobs.items()
                if job_id not in tainted and set(_needs(job)) & tainted
            }
        problems = [
            f"{name}:{job_id} depends (transitively) on a job that can be skipped "
            f"but declares no `if`, so GitHub will skip it silently"
            for job_id in sorted(tainted - skippable)
            if "needs." not in str(jobs[job_id].get("if", ""))
        ]
        assert not problems, "\n".join(problems)


def test_nothing_publishes_on_a_cancelled_run():
    """Cancelling a release must stop it, not race it to the index.

    A job with no `if` gets the default `success()`, which already refuses to
    run on cancellation -- that is the safest form and the one to prefer. The
    danger is `always()`, which keeps going after you hit cancel; a job that
    needs to survive a skipped ancestor must say `!cancelled() && ...` instead.
    """
    jobs = _workflows()["release.yml"]["jobs"]
    for job_id in ("github-release", "publish-testpypi", "publish-pypi"):
        condition = str(jobs[job_id].get("if", ""))
        assert "always()" not in condition, f"{job_id} must not use always()"
        if condition:
            assert "cancelled" in condition, (
                f"{job_id} overrides the default `if`, so it must guard cancellation itself"
            )


def test_the_release_refuses_to_ship_without_distributions():
    """An empty release must fail loudly.

    `fail_on_unmatched_files: false` means a broken glob attaches nothing and
    still reports success -- which is exactly what happened.
    """
    steps = _workflows()["release.yml"]["jobs"]["github-release"]["steps"]
    guards = [s for s in steps if "expected a wheel and an sdist" in str(s.get("run", ""))]
    assert guards, "github-release must verify it actually has distributions to attach"
def test_the_release_names_its_tag_explicitly():
    """Both entry points must produce a release on the intended tag.

    `github.ref` means the tag on a tag push and the *branch* on a
    workflow_dispatch, so relying on the default would cut a release named
    after a branch. That matters because workflow_dispatch is the only way to
    attach CUDA wheels: `wheel_run_id` cannot be supplied by a tag push.
    """
    steps = _workflows()["release.yml"]["jobs"]["github-release"]["steps"]
    release = next(s for s in steps if "gh-release" in str(s.get("uses", "")))
    tag_name = str(release.get("with", {}).get("tag_name", ""))
    assert tag_name, "gh-release must be given an explicit tag_name"
    assert "inputs.tag" in tag_name and "ref_name" in tag_name, tag_name


def test_release_jobs_check_out_the_ref_being_released():
    """Dispatching a release for an old tag must not build main."""
    jobs = _workflows()["release.yml"]["jobs"]
    for job_id in ("build-core", "github-release"):
        checkout = next(
            s for s in jobs[job_id]["steps"] if "actions/checkout" in str(s.get("uses", ""))
        )
        ref = str(checkout.get("with", {}).get("ref", ""))
        assert "inputs.tag" in ref, f"{job_id} checks out {ref or 'the default ref'}"


#: The only permission scopes a workflow may request. `administration` is
#: conspicuously absent, which is why a self-hosted runner cannot be registered
#: with GITHUB_TOKEN at all -- it needs a PAT.
VALID_PERMISSION_SCOPES = frozenset({
    "actions", "attestations", "checks", "contents", "deployments", "discussions",
    "id-token", "issues", "models", "packages", "pages", "pull-requests",
    "repository-projects", "security-events", "statuses",
})


def test_only_real_permission_scopes_are_requested():
    """An invented scope does not degrade gracefully.

    Requesting `administration: write` -- a reasonable guess, since registering
    a runner *is* a repository-administration call -- makes GitHub reject the
    whole file with "Unexpected value 'administration'", taking down every
    workflow in the repository until it is removed.
    """
    problems = []
    for name, workflow in _workflows().items():
        blocks = [("workflow", workflow.get("permissions"))]
        blocks += [(job_id, job.get("permissions")) for job_id, job in workflow["jobs"].items()]
        for where, permissions in blocks:
            if not isinstance(permissions, dict):
                continue
            for scope in permissions:
                if scope not in VALID_PERMISSION_SCOPES:
                    problems.append(f"{name}:{where} requests unknown scope {scope!r}")
    assert not problems, "\n".join(sorted(problems))


def test_wheel_artifacts_are_never_merged_into_one_directory():
    """Each CUDA target ships identically-named files; merging loses some.

    Every `cuda-wheels-*` artifact contains a `manifest.json` and wheels named
    `mamba_ssm-...whl` and `causal_conv1d-...whl`. Downloading them with
    `merge-multiple: true` puts all three into one directory, where the last
    writer wins per filename -- so one target's manifest ends up describing
    another target's wheels.

    That stopped the first GPU certification with a SHA-256 mismatch, which was
    the lucky outcome: the interleaving is nondeterministic, so it could just as
    easily have produced a self-consistent directory and certified wheels that
    were never the ones the manifest described.
    """
    problems = []
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            for step in job.get("steps", []) or []:
                if "download-artifact" not in str(step.get("uses", "")):
                    continue
                with_ = step.get("with") or {}
                pattern = str(with_.get("pattern", ""))
                if pattern.startswith("cuda-wheels") and with_.get("merge-multiple"):
                    problems.append(
                        f"{name}:{job_id} merges {pattern}, which collides on filenames"
                    )
    assert not problems, "\n".join(problems)


def test_merged_artifacts_carry_the_architecture_in_every_filename():
    """`certification-*` *is* merged, and that is only safe by construction.

    The sm80 and sm90 certifications are two calls to the same reusable
    workflow, so they upload two artifacts whose *contents* are collected with
    `merge-multiple: true`. That is the same flattening that lost wheels --
    harmless here only because each report is written as
    `certification-<arch>-<target>.json`, so sm80's six filenames can never
    equal sm90's.

    Dropping the arch from that name would silently reintroduce the collision
    on a path that costs an H100 to discover, so the invariant is asserted
    rather than left as a comment.
    """
    steps = _workflows()["certify-cuda.yml"]["jobs"]["certify"]["steps"]

    upload = next(s for s in steps if "upload-artifact" in str(s.get("uses", "")))
    assert "inputs.arch" in str(upload["with"]["name"]), (
        "the certification artifact name must distinguish sm80 from sm90"
    )

    written = "".join(str(s.get("run", "")) for s in steps)
    # The loop builds those paths from a shell variable, so expand it: the check
    # must see the paths that are actually written, not the variable name.
    for name, value in re.findall(r'^\s*(\w+)="(reports/[^"\n]+)"\s*$', written, re.M):
        written = written.replace(f'"${name}', f'"{value}')
    # Split on whitespace would tear `${{ inputs.arch }}` in half.
    reports = set(re.findall(r"""reports/certification[^\n"']*?\.json""", written))
    assert reports, "no certification report path found in certify-cuda.yml"
    for path in reports:
        assert "inputs.arch" in path, f"{path} does not embed the architecture"
        assert "$target" in path, f"{path} does not distinguish the three runtimes"


def test_every_job_building_fixtures_installs_torch():
    """`torch` is an extra, so `.[dev]` alone cannot build a fixture.

    Keeping torch out of the core dependencies is deliberate -- an MLX user must
    not have to download it -- but that makes `pip install -e ".[dev]"` a
    perfectly reasonable-looking line that leaves `make_golden_fixture.py`
    without the one import it cannot start without.

    It failed exactly that way on the certification pod, after the GPU had been
    rented and the wheels verified -- the most expensive possible place to
    discover a missing extra.

    The check is per *environment*, not per job, and that distinction is the
    whole point: the certification job builds fixtures in `.venv-fixtures` and
    then installs a CUDA torch into a separate venv per runtime. A job-level
    scan sees those later installs and passes the broken workflow, which is
    exactly what a first version of this test did.
    """
    #: `torch`, `torch>=2.11,<2.14`, `torch~=2.12.0`, `torch==${{ matrix.torch }}`.
    requirement = re.compile(r"^torch([<>=~!].*)?$")
    venv_python = re.compile(r"(\S+)/bin/python")

    def installs_torch(lines: list[str], extra_values: str = "") -> bool:
        text = " ".join(lines) + " " + extra_values
        extras = {
            extra.strip()
            for group in re.findall(r"\[([A-Za-z0-9_,\- ]+)\]", text)
            for extra in group.split(",")
        }
        tokens = {token.strip("\"'") for token in text.split()}
        return (
            "torch" in extras
            or "--extra torch" in text
            or any(requirement.match(token) for token in tokens)
        )

    problems = []
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            steps = job.get("steps") or []
            lines = [
                line for step in steps for line in str(step.get("run", "")).splitlines()
            ]
            # A matrix can be a bare expression (${{ fromJSON(...) }}), in which
            # case there is nothing static to read.
            matrix = (job.get("strategy") or {}).get("matrix")
            values = " ".join(
                str(v)
                for entry in (matrix.get("include", []) if isinstance(matrix, dict) else [])
                for v in entry.values()
            )

            for line in lines:
                if "make_golden_fixture" not in line:
                    continue
                found = venv_python.search(line)
                if found:
                    # An explicit venv: only what was installed into *that* venv
                    # counts. `.venv-fixtures/bin/python` needs
                    # `.venv-fixtures/bin/pip`.
                    prefix = found.group(1)
                    relevant = [line for line in lines if f"{prefix}/bin/pip" in line]
                    ok = installs_torch(relevant)
                else:
                    # `uv run python ...` -- the job's ambient environment.
                    relevant = [
                        line for line in lines if "install" in line or "sync" in line
                    ]
                    ok = installs_torch(relevant, values)
                if not ok:
                    problems.append(
                        f"{name}:{job_id} runs `{line.strip()[:60]}` but never "
                        f"installs torch into the environment it uses"
                    )
                    break
    assert not problems, "\n".join(problems)


def test_certification_proves_the_backend_imports_in_a_fresh_process():
    """Installed, importable, and importable *from a clean interpreter* differ.

    Upstream's `__init__` imports `modules.mamba2` before the line that can
    fail, so a failed package import leaves that submodule in `sys.modules` and
    every later import in the same process succeeds from cache. A certification
    run passed its CUDA parity tests exactly that way, against a backend that no
    fresh process could load.

    Nothing inside the pytest session can detect that -- by the time a test
    runs, the cache may already be poisoned. Only a separate interpreter, before
    anything else touches mamba_ssm, can prove it, so the workflow must contain
    that step and must install what upstream needs to import.
    """
    steps = _workflows()["certify-cuda.yml"]["jobs"]["certify"]["steps"]
    # Comments are stripped deliberately: a first version of this assertion was
    # satisfied by the comment *explaining* why transformers is installed, and
    # so passed a workflow with the install removed.
    script = "".join(
        line
        for step in steps
        for line in str(step.get("run", "")).splitlines(keepends=True)
        if not line.strip().startswith("#")
    )

    assert any(
        "transformers" in line and ("pip" in line and "install" in line)
        for line in script.splitlines()
    ), (
        "the certification venvs must install upstream's import-time requirements; "
        "--no-deps leaves them out and the package then fails to import"
    )

    probe = "from mamba_ssm.modules.mamba2 import Mamba2"
    assert probe in script, "certification must import upstream Mamba2 in a fresh process"
    assert script.index(probe) < script.index("-m pytest"), (
        "the fresh-process import must run before pytest, which can poison sys.modules"
    )


def test_jobs_using_a_local_action_do_not_check_out_an_older_ref():
    """A local composite action is resolved from the working directory.

    The workflow file itself comes from the ref the run was started on, but
    `uses: ./.github/actions/...` is read from whatever `checkout` left on disk.
    Pinning that checkout to the tag being released therefore breaks the moment
    the tag predates the action:

        Can't find 'action.yml' ... under .github/actions/collect-distributions

    which is how re-publishing v0.1.0 failed. Jobs that use a local action must
    check out the ref the workflow came from, and take the thing they are
    actually publishing from artifacts or release assets instead.
    """
    problems = []
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            steps = job.get("steps") or []
            if not any(str(s.get("uses", "")).startswith("./.github/actions/") for s in steps):
                continue
            for step in steps:
                if "actions/checkout" not in str(step.get("uses", "")):
                    continue
                ref = str((step.get("with") or {}).get("ref", ""))
                if "inputs.tag" in ref:
                    problems.append(
                        f"{name}:{job_id} uses a local action but checks out {ref}, "
                        f"which may predate that action"
                    )
    assert not problems, "\n".join(problems)


def test_certification_actually_certifies_the_cuda_backend():
    """Running on a GPU is not the same as certifying the GPU backend.

    v0.1.0's six reports were produced on an A100 and an H100 and every one
    passed, while none of them measured the CUDA kernels: `verify` was invoked
    without `--certify`, so the campaign compared the portable chunked
    implementation against its own float64 oracle and labelled the report
    accordingly. Nothing about the hardware makes that a CUDA result.
    """
    steps = _workflows()["certify-cuda.yml"]["jobs"]["certify"]["steps"]
    script = "".join(
        line
        for step in steps
        for line in str(step.get("run", "")).splitlines(keepends=True)
        if not line.strip().startswith("#")
    )
    assert "--certify cuda-reference" in script, (
        "certification must run a campaign whose candidate_backend is cuda-reference"
    )


def test_the_release_checks_what_the_reports_certify():
    """`passed: true` is not the question; `candidate_backend` is."""
    checks = []
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                if "verify_certification_reports.py" in str(step.get("run", "")):
                    checks.append(f"{name}:{job_id}")
    assert checks, "no job verifies the certification reports"
    for name, workflow in _workflows().items():
        for job_id, job in workflow["jobs"].items():
            for step in job.get("steps") or []:
                run = str(step.get("run", ""))
                if "verify_certification_reports.py" not in run:
                    continue
                assert "--require-backend" in run and "cuda-reference" in run, (
                    f"{name}:{job_id} does not require the CUDA backend to be certified"
                )


def test_the_release_publishes_the_index_install_backend_needs():
    """`install-backend` cannot work from the per-runtime build manifests.

    They carry `"url": null` -- at build time nothing knows where the artefact
    will be published -- so the release has to write an index that does know,
    and attach it.
    """
    steps = _workflows()["release.yml"]["jobs"]["github-release"]["steps"]
    script = "".join(str(step.get("run", "")) for step in steps)
    assert "build_release_manifest.py" in script, "the release must build the CUDA index"

    release_step = next(s for s in steps if "gh-release" in str(s.get("uses", "")))
    files = str(release_step["with"]["files"])
    assert "assets/*" in files, "the index is written into assets/, which must be attached"


def test_a_release_can_reuse_wheels_it_did_not_build():
    """Seventy minutes per runtime, for an artefact that cannot differ.

    The wheels are upstream's, at pinned versions, from unchanged Dockerfiles.
    Rebuilding them for a release that only changes this package's own code is
    pure cost, so `wheel_run_id` nominates an existing build -- and the
    certification and the release must both read from that run, or the release
    would attach wheels other than the ones certified.
    """
    jobs = _workflows()["release.yml"]["jobs"]
    assert "wheel_run_id" in str(jobs["build-cuda"].get("if")), (
        "build-cuda must be skipped when wheels are nominated"
    )
    for job_id in ("certify-sm80", "certify-sm90"):
        assert "inputs.wheel_run_id" in str(jobs[job_id]["with"].get("wheel_run_id")), job_id

    download = next(
        s for s in jobs["github-release"]["steps"]
        if str((s.get("with") or {}).get("pattern", "")).startswith("cuda-wheels")
    )
    assert "inputs.wheel_run_id" in str(download["with"]["run-id"]), (
        "the release must attach the wheels from the run that was certified"
    )


def test_certify_only_stops_before_anything_is_released_or_published():
    """Evidence about a tag that is already out, without a doomed second upload.

    Re-certifying v0.1.0 to produce the CUDA reports it never had would
    otherwise run all the way to the PyPI upload and fail there, because the
    version already exists. The mode exists so the run ends where the answer is.
    """
    jobs = _workflows()["release.yml"]["jobs"]
    assert "certify_only" in str(jobs["github-release"].get("if")), (
        "github-release must be skipped in certify_only mode"
    )
    # Publishing hangs off github-release, so it must not have a route around it.
    for job_id in ("publish-testpypi", "cold-install-testpypi", "publish-pypi"):
        condition = str(jobs[job_id].get("if"))
        assert "needs." in condition, f"{job_id} must depend on an upstream result"
    testpypi = str(jobs["publish-testpypi"]["if"])
    assert "needs.github-release.result == 'success'" in testpypi, (
        "publishing must require github-release to have actually run"
    )
    assert "certify-sm90" in str(jobs["certification-summary"]["needs"]), (
        "the summary must wait for both architectures"
    )


def test_the_certification_summary_runs_the_same_gate_as_the_release():
    """A certify_only run must answer the question it was started to answer."""
    steps = _workflows()["release.yml"]["jobs"]["certification-summary"]["steps"]
    script = "".join(str(step.get("run", "")) for step in steps)
    assert "verify_certification_reports.py" in script
    assert "--require-backend" in script and "cuda-reference" in script


def test_the_cuda_gate_is_float32_and_bfloat16_is_only_measured():
    """The verdict must be about the algorithm, not about a number format.

    Two A100 runs established that a bfloat16 comparison sits against the limit
    of bfloat16 whatever the implementation: at equal data the residue was
    mean_abs 2.88e-03 on outputs of RMS one, against bfloat16's own 0.39% of
    relative precision. Gating on it certifies nothing and fails correct code.

    So `comparisons` -- which is what `passed` is computed from -- must carry the
    float32 campaign, and the bfloat16 numbers must live in metadata, where they
    inform without deciding.
    """
    source = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "niverel_mamba" / "cli" / "verify.py"
    ).read_text(encoding="utf-8")

    gated = re.findall(r'tolerance="(cuda_[a-z0-9_]+)"', source)
    assert gated, "the CUDA campaign scores nothing"
    assert set(gated) == {"cuda_float32"}, (
        f"the CUDA gate must be float32 only, found {sorted(set(gated))}"
    )
    assert 'report.metadata["bfloat16_measured"]' in source, (
        "bfloat16 must still be measured and reported, just not gated"
    )
