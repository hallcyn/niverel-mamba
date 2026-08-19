#!/usr/bin/env python
"""Phase 0 -- build golden fixtures, including one from the real V3 checkpoint.

Three fixture tiers:

``tiny``
    B=1 L=8 D=16 N=4 headdim=8. Small enough to run in float64 and to reason
    about by hand. The mathematical ground truth.

``segmented``
    B=2 L=257 with irregular ``seq_idx`` boundaries. Exercises strict reset,
    internal chunk padding and multi-document batching.

``niverel``
    One real Mamba2 block lifted out of
    ``thibaud-perrin/niverel-5b-v3-hnet-jepa-seed1337``. The full checkpoint is
    never committed: we download it, verify its SHA-256 against the SHA the
    bundle manifest already recorded, extract a single block (~15 MB), and
    keep only that plus the provenance needed to prove where it came from.

Usage::

    python scripts/make_golden_fixture.py                 # tiny + segmented
    python scripts/make_golden_fixture.py --niverel       # also the real block

``--niverel`` needs ``HF_TOKEN`` (read from ``.env`` if present) because the
checkpoint repository is private.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from niverel_mamba.config import Mamba2Config  # noqa: E402
from niverel_mamba.errors import FixtureError  # noqa: E402
from niverel_mamba.schema import default_contract  # noqa: E402

FIXTURE_ROOT = REPO_ROOT / "fixtures"
MANIFEST_ROOT = FIXTURE_ROOT / "golden-manifests"

NIVEREL_REPO = "thibaud-perrin/niverel-5b-v3-hnet-jepa-seed1337"
NIVEREL_REVISION = "5da95e264026d80fd6d8debb50c4ca4c40277483"
NIVEREL_CKPT = "ckpt.pt"

TINY_CONFIG = {
    "d_model": 16,
    "d_state": 4,
    "d_conv": 4,
    "expand": 2,
    "headdim": 8,
    "ngroups": 1,
    "chunk_size": 4,
}

SEGMENTED_CONFIG = {
    "d_model": 32,
    "d_state": 8,
    "d_conv": 4,
    "expand": 2,
    "headdim": 8,
    "ngroups": 2,
    "chunk_size": 16,
}


#: Variables whose values name a filesystem location and must therefore be
#: expanded rather than passed through literally.
_PATH_LIKE = frozenset({"HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"})


def load_dotenv(path: Path) -> None:
    """Minimal .env reader. The file is gitignored and holds HF_TOKEN.

    ``~`` and ``$VAR`` are expanded for path-like variables. A shell would do
    that; a naive reader would not, and passing a literal ``$HOME/...`` to
    ``huggingface_hub`` makes it cheerfully create a directory *named*
    ``$HOME`` in the working directory.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in _PATH_LIKE:
            value = os.path.expanduser(os.path.expandvars(value))
        os.environ.setdefault(key, value)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's exact bytes, in a layout-independent way."""
    contiguous = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def save_state_dict(state: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))


def write_manifest(name: str, payload: dict[str, Any]) -> Path:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_ROOT / f"{name}.json"
    payload = {
        "schema_version": "niverel-mamba-golden-fixture-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def synth_weights(config: Mamba2Config, seed: int) -> dict[str, torch.Tensor]:
    """Deterministic weights matching the contract exactly, in float64.

    Values are drawn to be *representative*, not merely non-zero: A_log spans
    the upstream init range so decay rates vary across heads, and dt_bias is
    the true inverse-softplus of a plausible dt.
    """
    generator = torch.Generator().manual_seed(seed)
    contract = default_contract()
    expected = contract.expected(config)
    state: dict[str, torch.Tensor] = {}

    def randn(*shape: int, scale: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64) * scale

    for key, shape in expected.items():
        if key == "A_log":
            a = torch.rand(shape, generator=generator, dtype=torch.float64) * 15.0 + 1.0
            state[key] = torch.log(a)
        elif key == "dt_bias":
            dt_sample = torch.exp(
                torch.rand(shape, generator=generator, dtype=torch.float64)
                * (torch.log(torch.tensor(0.1, dtype=torch.float64)) - torch.log(torch.tensor(0.001, dtype=torch.float64)))
                + torch.log(torch.tensor(0.001, dtype=torch.float64))
            ).clamp(min=1e-4)
            state[key] = dt_sample + torch.log(-torch.expm1(-dt_sample))
        elif key == "D":
            state[key] = torch.ones(shape, dtype=torch.float64) + randn(*shape, scale=0.1)
        elif key == "norm.weight":
            state[key] = torch.ones(shape, dtype=torch.float64) + randn(*shape, scale=0.05)
        elif key.endswith(".bias"):
            state[key] = randn(*shape, scale=0.1)
        else:
            fan_in = shape[-1] if len(shape) > 1 else shape[0]
            state[key] = randn(*shape, scale=fan_in**-0.5)
    return state


def build_tiny() -> dict[str, Any]:
    config = Mamba2Config(**TINY_CONFIG)
    weights = synth_weights(config, seed=1337)
    generator = torch.Generator().manual_seed(20260817)
    x = torch.randn(1, 8, config.d_model, generator=generator, dtype=torch.float64)

    directory = FIXTURE_ROOT / "tiny"
    save_state_dict(weights, directory / "weights.safetensors")
    save_state_dict({"x": x}, directory / "inputs.safetensors")
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "fixture": "tiny",
        "purpose": "float64 mathematical ground truth (brief 11.1, tier 1)",
        "config": config.to_dict(),
        "input_shape": list(x.shape),
        "seeds": {"weights": 1337, "input": 20260817},
        "tensor_sha256": {k: tensor_sha256(v) for k, v in weights.items()},
    }


def build_segmented() -> dict[str, Any]:
    config = Mamba2Config(**SEGMENTED_CONFIG)
    weights = synth_weights(config, seed=4242)
    generator = torch.Generator().manual_seed(20260818)
    batch, length = 2, 257
    x = torch.randn(batch, length, config.d_model, generator=generator, dtype=torch.float64)

    # Deliberately awkward boundaries: a length-1 document, two consecutive
    # boundaries, a boundary exactly on a chunk edge (chunk_size=16), and a
    # document that lives entirely inside one chunk.
    seq_idx = torch.zeros(batch, length, dtype=torch.int32)
    row0 = [0] * 16 + [1] + [2] + [3] * 15 + [4] * 96 + [5] * 128
    row0 = (row0 + [5] * length)[:length]
    row1 = [0] * 1 + [1] * 63 + [2] * 64 + [3] * 5 + [4] * 124
    row1 = (row1 + [4] * length)[:length]
    seq_idx[0] = torch.tensor(row0, dtype=torch.int32)
    seq_idx[1] = torch.tensor(row1, dtype=torch.int32)

    directory = FIXTURE_ROOT / "segmented"
    save_state_dict(weights, directory / "weights.safetensors")
    save_state_dict({"x": x, "seq_idx": seq_idx}, directory / "inputs.safetensors")
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "fixture": "segmented",
        "purpose": "strict-reset and internal-padding coverage (brief 11.1, tier 2)",
        "config": config.to_dict(),
        "input_shape": list(x.shape),
        "seeds": {"weights": 4242, "input": 20260818},
        "documents_per_row": [int(seq_idx[i].max()) + 1 for i in range(batch)],
        "tensor_sha256": {k: tensor_sha256(v) for k, v in weights.items()},
    }


# --------------------------------------------------------------------------
# The real Niverel Foundation V3 block
# --------------------------------------------------------------------------


def _find_state_dict(obj: Any, depth: int = 0) -> dict[str, torch.Tensor]:
    """Locate the parameter mapping inside a training checkpoint."""
    if depth > 4:
        raise FixtureError("could not locate a state dict inside the checkpoint")
    if isinstance(obj, dict):
        tensor_items = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if tensor_items and len(tensor_items) > 10:
            return tensor_items
        for key in ("model", "state_dict", "module", "ema", "net"):
            if key in obj:
                try:
                    return _find_state_dict(obj[key], depth + 1)
                except FixtureError:
                    continue
        for value in obj.values():
            if isinstance(value, dict):
                try:
                    return _find_state_dict(value, depth + 1)
                except FixtureError:
                    continue
    raise FixtureError("could not locate a state dict inside the checkpoint")


def find_mamba2_blocks(
    state: dict[str, torch.Tensor], config: Mamba2Config
) -> dict[str, dict[str, torch.Tensor]]:
    """Find every prefix whose sub-keys satisfy the weight contract."""
    contract = default_contract()
    expected = contract.expected(config)
    required = set(expected)

    candidates: dict[str, dict[str, torch.Tensor]] = {}
    suffix = "in_proj.weight"
    for key in state:
        if not key.endswith(suffix):
            continue
        prefix = key[: -len(suffix)]
        block = {}
        ok = True
        for name in required:
            full = f"{prefix}{name}"
            if full not in state:
                ok = False
                break
            block[name] = state[full]
        if ok:
            candidates[prefix.rstrip(".")] = block
    return candidates


def build_niverel(force: bool = False) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise FixtureError(
            "HF_TOKEN is not set. The V3 checkpoint repository is private; put the token in .env."
        )

    expected_sha = None
    manifest_path = hf_hub_download(
        repo_id=NIVEREL_REPO,
        filename="bundle_manifest.json",
        revision=NIVEREL_REVISION,
        token=token,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected_sha = manifest.get("files", {}).get(NIVEREL_CKPT)
    if not expected_sha:
        raise FixtureError("bundle_manifest.json does not record a SHA-256 for ckpt.pt")

    print(f"  downloading {NIVEREL_REPO}/{NIVEREL_CKPT} (~1.7 GB) ...")
    ckpt_path = Path(
        hf_hub_download(
            repo_id=NIVEREL_REPO,
            filename=NIVEREL_CKPT,
            revision=NIVEREL_REVISION,
            token=token,
        )
    )

    print("  verifying SHA-256 against the bundle manifest ...")
    actual_sha = sha256_file(ckpt_path)
    if actual_sha != expected_sha:
        raise FixtureError(
            f"checkpoint SHA-256 mismatch:\n  manifest {expected_sha}\n  actual   {actual_sha}"
        )
    print(f"  ok  {actual_sha}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _find_state_dict(checkpoint)

    config = Mamba2Config(
        d_model=768,
        d_state=128,
        d_conv=4,
        expand=2,
        metadata={
            "source": "niverel-5b-v3-hnet-jepa-seed1337",
            "note": "headdim/ngroups/chunk_size are never passed by Niverel, so upstream defaults apply",
        },
    )
    blocks = find_mamba2_blocks(state, config)
    if not blocks:
        raise FixtureError(
            "no Mamba2 block in the checkpoint satisfies the weight contract; "
            f"checkpoint has {len(state)} tensors"
        )

    chosen = sorted(blocks)[0]
    block = blocks[chosen]
    print(f"  found {len(blocks)} Mamba2 blocks; extracting {chosen!r}")

    directory = FIXTURE_ROOT / "niverel"
    save_state_dict(block, directory / "block.safetensors")

    generator = torch.Generator().manual_seed(1337)
    x = torch.randn(1, 96, config.d_model, generator=generator, dtype=torch.float32)
    seq_idx = torch.zeros(1, 96, dtype=torch.int32)
    seq_idx[0, 37:] = 1
    seq_idx[0, 70:] = 2
    save_state_dict({"x": x, "seq_idx": seq_idx}, directory / "inputs.safetensors")
    (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "fixture": "niverel",
        "purpose": "real Foundation V3 Mamba2 block (brief 11.1, tier 3)",
        "source_repository": NIVEREL_REPO,
        "source_revision": NIVEREL_REVISION,
        "source_file": NIVEREL_CKPT,
        "source_sha256": actual_sha,
        "run_id": manifest.get("receipt", {}).get("run_id"),
        "extracted_block": chosen,
        "blocks_available": sorted(blocks),
        "config": config.to_dict(),
        "input_shape": list(x.shape),
        "input_seed": 1337,
        "tensor_dtypes": {k: str(v.dtype) for k, v in block.items()},
        "tensor_shapes": {k: list(v.shape) for k, v in block.items()},
        "tensor_sha256": {k: tensor_sha256(v) for k, v in block.items()},
        "note": (
            "The full checkpoint is never committed. Only this single block and its provenance "
            "are versioned; block.safetensors itself is gitignored and rebuilt by this script."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niverel", action="store_true", help="also extract the real V3 block")
    parser.add_argument("--only-niverel", action="store_true", help="skip the synthetic fixtures")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not args.only_niverel:
        print("tiny fixture")
        path = write_manifest("tiny", build_tiny())
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

        print("segmented fixture")
        path = write_manifest("segmented", build_segmented())
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    if args.niverel or args.only_niverel:
        print("niverel fixture (real Foundation V3 block)")
        path = write_manifest("niverel", build_niverel())
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
