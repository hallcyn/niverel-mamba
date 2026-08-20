"""The CLI surface."""

from __future__ import annotations

import json

import pytest

from niverel_mamba.cli.main import main
from tests.conftest import HAVE_MLX, HAVE_TORCH, requires_torch


def test_doctor_json(capsys):
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["niverel_mamba"]
    assert {b["backend"] for b in payload["backends"]} == {
        "torch-reference", "cuda-reference", "mlx"
    }


def test_doctor_human_output_matches_the_brief_shape(capsys):
    """The documented doctor layout, whatever is installed.

    The exit code is deliberately meaningful rather than always zero: doctor
    reports failure when no backend at all is usable, which is exactly the
    situation a user running it would need to know about. ci-core exercises
    that path, since it installs neither torch nor MLX.
    """
    code = main(["doctor"])
    out = capsys.readouterr().out
    for expected in ("Python", "Platform", "MPS", "MLX", "CUDA",
                     "Available backends:", "Recommended backend:"):
        assert expected in out

    if HAVE_TORCH or HAVE_MLX:
        assert code == 0
        assert "none -- install PyTorch or MLX" not in out
    else:
        assert code == 1
        assert "none -- install PyTorch or MLX" in out


def test_doctor_reports_certification_per_backend(capsys):
    main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    for backend in payload["backends"]:
        assert backend["certification"] in (
            "reference", "numerically-certified", "experimental", "unsupported"
        )
        assert "official_reference" in backend


def test_inspect_prints_the_contract(capsys):
    assert main(["inspect"]) == 0
    out = capsys.readouterr().out
    assert "in_proj.weight" in out
    assert "2.3.2.post1" in out
    assert "init_states" not in out


def test_inspect_resolves_a_configuration(capsys):
    assert main(["inspect", "--config", "d_model=768", "--config", "d_state=128"]) == 0
    out = capsys.readouterr().out
    assert "(3352, 768)" in out
    assert "nheads=24" in out


@requires_torch
def test_install_backend_plans_without_installing(capsys, tmp_path):
    """Without --yes the command must never download or install anything."""
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "schema_version": "niverel-mamba-cuda-index-v1",
        "runtimes": [],
    }))
    code = main(["install-backend", "cuda", "--manifest", str(manifest)])
    out = capsys.readouterr().out
    assert "Detected:" in out
    assert code == 1  # no matching artifact for this machine
    assert "will not compile" in out


@requires_torch
def test_install_backend_rejects_an_unknown_manifest_schema(capsys, tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"schema_version": "something-else", "artifacts": []}))
    assert main(["install-backend", "cuda", "--manifest", str(manifest)]) == 1
    assert "Unknown manifest schema" in capsys.readouterr().out


@pytest.mark.skipif(HAVE_TORCH, reason="only meaningful when torch is absent")
def test_install_backend_without_torch_says_so(capsys):
    """The CUDA extensions are built against a specific torch ABI, so there is
    nothing sensible to install before torch itself exists."""
    assert main(["install-backend", "cuda"]) == 1
    assert "PyTorch is not installed" in capsys.readouterr().out


def test_sha256_mismatch_is_refused(tmp_path):
    from niverel_mamba.cli.install_backend import verify_sha256
    from niverel_mamba.errors import AssetVerificationError

    target = tmp_path / "wheel.whl"
    target.write_bytes(b"not the real wheel")
    with pytest.raises(AssetVerificationError, match="SHA-256 mismatch"):
        verify_sha256(target, "0" * 64)


@requires_torch
def test_verify_runs_the_tiny_fixture(capsys):
    assert main(["verify", "--fixture", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "PASSED" in out


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()
