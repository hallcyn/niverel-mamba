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


def test_publishing_jobs_request_an_oidc_token():
    """Trusted Publishing is the only sanctioned route; it needs id-token."""
    workflows = _workflows()
    publish = workflows["publish-pypi.yml"]["jobs"]["publish"]
    assert publish.get("permissions", {}).get("id-token") == "write"
    assert publish.get("environment", {}).get("name")


def test_no_pypi_token_is_referenced_anywhere():
    """A token in a workflow would defeat the point of Trusted Publishing."""
    offenders = []
    for path in WORKFLOW_DIR.glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        for marker in ("PYPI_TOKEN", "PYPI_API_TOKEN", "TWINE_PASSWORD", "password:"):
            if marker in text:
                offenders.append(f"{path.name} references {marker}")
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
    """The ordering is the safety property: a cold install gates the real index."""
    jobs = _workflows()["release.yml"]["jobs"]
    needs = jobs["publish-pypi"]["needs"]
    needs = [needs] if isinstance(needs, str) else needs
    assert "publish-testpypi" in needs


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
    """`always()` would keep publishing after you hit cancel. `!cancelled()` does not."""
    jobs = _workflows()["release.yml"]["jobs"]
    for job_id in ("github-release", "publish-testpypi", "publish-pypi"):
        condition = str(jobs[job_id].get("if", ""))
        assert "cancelled" in condition, f"{job_id} must guard against cancellation"
        assert "always()" not in condition, f"{job_id} must not use always()"


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
