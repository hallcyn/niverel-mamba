"""`install-backend cuda`, against a local index and a local archive.

The command was broken from the moment the release started shipping one zip per
runtime instead of loose wheels: the per-runtime build manifests carry
`"url": null`, because at build time nothing knows where the artefact will be
published, so there was nothing to fetch. It was noticed and deferred, and
deferring it would have cost a whole extra release cycle to discover again.

Everything here runs offline: the index points at `file://` URLs, and pip is
never actually invoked -- what is asserted is which commands *would* run, and
that nothing gets installed unless every SHA-256 matched first.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from niverel_mamba.cli.main import main
from tests.conftest import requires_torch

pytestmark = requires_torch

WHEELS = {
    "mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl": b"not really a wheel, but hashed like one",
    "causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl": b"nor is this one",
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def release(tmp_path: Path) -> dict[str, object]:
    """A runtime archive and an index describing it, matching this machine."""
    import sys

    import torch

    archive = tmp_path / "niverel-mamba-cuda-local.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, payload in WHEELS.items():
            bundle.writestr(name, payload)

    capability = None
    if torch.cuda.is_available():  # pragma: no cover - CI has no GPU
        major, minor = torch.cuda.get_device_capability(0)
        capability = f"sm_{major}{minor}"

    index = {
        "schema_version": "niverel-mamba-cuda-index-v1",
        "release": "v0.0.0-test",
        "runtimes": [
            {
                "name": "local",
                "torch_version": torch.__version__.split("+")[0],
                "torch_cuda": torch.version.cuda,
                "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "architectures": [capability] if capability else ["sm_80", "sm_90"],
                "source_repository": "https://github.com/state-spaces/mamba",
                "source_commit": "deadbeef",
                "archive": {
                    "filename": archive.name,
                    "url": str(archive),  # _fetch accepts a local path as well as a URL
                    "sha256": _sha(archive.read_bytes()),
                    "size_bytes": archive.stat().st_size,
                },
                "wheels": [
                    {
                        "package": name.split("-")[0].replace("_", "-"),
                        "package_version": name.split("-")[1],
                        "filename": name,
                        "sha256": _sha(payload),
                    }
                    for name, payload in WHEELS.items()
                ],
            }
        ],
    }
    path = tmp_path / "cuda-manifest.json"
    path.write_text(json.dumps(index))
    return {"index": path, "archive": archive, "payload": index}


@pytest.fixture
def pip_calls(monkeypatch):
    """Record what would be installed instead of installing it."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(
        "niverel_mamba.cli.install_backend.subprocess.run",
        lambda command, **kwargs: (calls.append(list(command)), _Result())[1],
    )
    return calls


def test_the_plan_downloads_and_installs_nothing(release, capsys, tmp_path):
    code = main(["install-backend", "cuda", "--manifest", str(release["index"]),
                 "--dest", str(tmp_path / "cache")])
    out = capsys.readouterr().out
    assert code == 0
    assert "Nothing was downloaded or installed" in out
    assert not (tmp_path / "cache").exists(), "the plan must not even create the cache"


def test_a_matching_runtime_is_verified_then_installed(release, pip_calls, capsys, tmp_path):
    code = main(["install-backend", "cuda", "--yes", "--manifest", str(release["index"]),
                 "--dest", str(tmp_path / "cache")])
    out = capsys.readouterr().out
    assert code == 0, out
    for name in WHEELS:
        assert f"verified    {name}" in out

    assert len(pip_calls) == 2, pip_calls
    wheels_command, requirements_command = pip_calls
    assert "--no-deps" in wheels_command, "torch must not be replaced"
    assert sum(arg.endswith(".whl") for arg in wheels_command) == len(WHEELS)
    # Upstream's import-time requirement, installed *with* its own dependencies.
    assert "transformers" in requirements_command
    assert "--no-deps" not in requirements_command


def test_a_tampered_archive_is_refused_before_anything_is_unpacked(
    release, pip_calls, tmp_path, capsys
):
    """The archive hash is checked before the zip is opened at all."""
    archive = release["archive"]
    assert isinstance(archive, Path)
    archive.write_bytes(archive.read_bytes() + b"tampered")

    code = main(["install-backend", "cuda", "--yes", "--manifest", str(release["index"]),
                 "--dest", str(tmp_path / "cache")])
    assert code != 0, capsys.readouterr().out
    assert not pip_calls, "nothing may be installed after a hash mismatch"
    assert not (tmp_path / "cache" / "local").exists(), "the archive must not be unpacked"


def test_a_tampered_wheel_inside_a_good_archive_is_refused(release, pip_calls, tmp_path, capsys):
    """The archive hash proves what was downloaded, not what comes out of it."""
    payload = dict(release["payload"])  # type: ignore[arg-type]
    payload["runtimes"][0]["wheels"][0]["sha256"] = "0" * 64  # type: ignore[index]
    index = tmp_path / "tampered.json"
    index.write_text(json.dumps(payload))

    code = main(["install-backend", "cuda", "--yes", "--manifest", str(index),
                 "--dest", str(tmp_path / "cache")])
    assert code != 0
    assert not pip_calls


def test_an_environment_with_no_matching_runtime_is_told_what_exists(release, capsys, tmp_path):
    payload = dict(release["payload"])  # type: ignore[arg-type]
    payload["runtimes"][0]["torch_version"] = "1.0.0"  # type: ignore[index]
    index = tmp_path / "mismatch.json"
    index.write_text(json.dumps(payload))

    code = main(["install-backend", "cuda", "--yes", "--manifest", str(index)])
    out = capsys.readouterr().out
    assert code == 1
    assert "will not compile" in out
    assert "Available runtimes in this release:" in out
    assert "torch 1.0.0" in out
