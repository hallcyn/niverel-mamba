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
