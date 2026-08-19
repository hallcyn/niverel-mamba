# niverel-mamba

**A Mamba2 checkpoint should not be a prisoner of the CUDA backend it was trained with.**

`niverel-mamba` loads the *same* Mamba2 weights on Linux + NVIDIA CUDA, Linux CPU,
macOS Apple Silicon with PyTorch MPS, and macOS Apple Silicon with MLX — with one
weight contract, no silent fallbacks, and an explicit certification status per backend.

It is not a new model and it does not change the Mamba2 equations. It is a portability
and certification layer around them.

```
                     canonical Mamba2 config
                                +
                     canonical weight contract
                                |
         ┌──────────────────────┼──────────────────────┐
         v                      v                      v
  torch-reference         cuda-reference              mlx
  CPU / CUDA / MPS      upstream kernels        Apple Silicon
         |                      |                      |
         └──────────────────────┼──────────────────────┘
                                |
                      certification reports
```

## Install

```bash
pip install niverel-mamba                # core only: no torch, no MLX
pip install "niverel-mamba[torch]"       # + PyTorch
pip install "niverel-mamba[mlx]"         # + MLX (macOS arm64)
```

PyTorch is deliberately *not* a required dependency: an MLX-only user must not have to
download it. In a CUDA environment, install your CUDA build of PyTorch **first** — the
CUDA variants come from a separate index.

## Use

```python
import torch
from niverel_mamba import Mamba2Config
from niverel_mamba.torch import Mamba2

config = Mamba2Config(d_model=768, d_state=128, d_conv=4, expand=2)

model = Mamba2(config, device="mps", dtype=torch.float32)
model.load_state_dict(weights, strict=True)      # strict is the only mode

y = model(x, seq_idx=seq_idx)
```

It is a real `torch.nn.Module` with upstream's parameter names, so it drops into an
existing H-Net in place of `mamba_ssm.Mamba2` without touching a checkpoint.

MLX deliberately does *not* imitate `nn.Module`:

```python
from niverel_mamba.mlx import Mamba2

model = Mamba2(config)
model.load_canonical_weights(weights)
y = model(x, seq_idx=seq_idx)
```

### Autoregressive decoding

```python
state = model.allocate_inference_state(batch_size=1)
y_t, state = model.step(x_t, state)          # functional: no mutation
```

`step` returns a new state rather than mutating the one it was given, so PyTorch and MLX
share one contract. `concat(step(x_t)) == forward(sequence)` is certified.

### `seq_idx` strict reset

At every change of `seq_idx`, both the convolution state and the SSM state reset to zero:

```python
model(x, seq_idx=segments) == concat(model(doc_1), model(doc_2), ...)
```

Document boundaries do not need to align with chunk boundaries. This is certified across
nine structurally distinct cases (boundary at position 0, length-1 documents, consecutive
boundaries, boundary on a chunk edge, a document entirely inside one chunk, documents
misaligned across batch rows, terminal padding, non-zero first id, many short documents)
and at L=8192.

## Diagnose

```console
$ niverel-mamba doctor
niverel-mamba 0.1.0

Python          3.12.9
Platform        Darwin arm64
Framework       torch 2.13.0
MPS             available
MLX             0.32.0
CUDA            unavailable (no CUDA device visible to torch)
mamba-ssm       not installed
causal-conv1d   not installed

Available backends:
  torch-reference   yes   numerically-certified
  cuda-reference    no    CUDA backend not installed (mamba-ssm is absent)
  mlx               yes   experimental

Recommended backend:
  torch-reference / mps
```

```console
$ niverel-mamba inspect --config d_model=768 --config d_state=128
$ niverel-mamba verify --fixture niverel --device mps --mlx
$ niverel-mamba install-backend cuda          # prints a plan; --yes to install
```

## No silent fallback

An explicit request either succeeds or raises. It never quietly becomes something else:

```python
>>> load_mamba2(config, backend="cuda-reference")
BackendUnavailableError: backend 'cuda-reference' was requested explicitly but is not
available: CUDA backend not installed (mamba-ssm is absent). This package does not
silently fall back to another backend.
```

`backend="auto"` may choose, but it always reports what it chose:

```json
{
  "backend": "torch-reference",
  "framework": "torch",
  "device": "mps",
  "certification": "numerically-certified",
  "official_reference": false
}
```

## Certification status

Four statuses, used strictly:

| status | meaning |
|---|---|
| `reference` | the exact backend used to produce or certify a result |
| `numerically-certified` | compared against a reference under sealed tolerances, and passed |
| `experimental` | functional, not yet certified |
| `unsupported` | explicitly refused |

### Where 0.1.0 actually stands

| backend | status | certified against |
|---|---|---|
| `torch-reference` CPU | `numerically-certified` | the float64 sequential oracle, itself checked against upstream's `ssd_minimal_discrete` |
| `torch-reference` MPS | `numerically-certified` | `torch-reference` CPU, real Foundation V3 weights |
| `mlx` | `experimental` | measured against `torch-reference` CPU, but not yet CI-enforced on every release |
| `cuda-reference` | `experimental` | **nothing yet** — no NVIDIA GPU has run it |

`cuda-reference` wraps the upstream kernels, but wrapping is not certifying. It becomes
`reference` only once `certify-cuda-sm80.yml` has produced a real report on real
hardware. The `cuda_bfloat16` tolerance class in `tolerances.yaml` is correspondingly
marked unverified.

Every report names the reference it was measured against, because "torch-reference
passed" means nothing on its own.

**Where each claim is actually reproduced.** GitHub-hosted macOS runners cannot do MPS —
and they lie about it: `torch.backends.mps.is_available()` returns `True`, then every
allocation fails with *MPS backend out of memory … tried to allocate 21.00 KiB*. So
`capabilities` decides MPS by attempting a real allocation rather than trusting the flag,
which also stops `doctor` advertising a device that cannot hold a tensor and stops `auto`
selecting it. `ci-macos-mps` then splits in two: `macos-cpu` runs the macOS CPU suite on
hosted runners, while `macos-mps` targets a self-hosted Apple Silicon machine and refuses
to emit a report labelled `mps` if that runner cannot use MPS either. The MPS figures below were measured on a local M1; enable the self-hosted job (set
the repository variable `HAS_SELF_HOSTED_MAC`) to reproduce them on every push.

The two hosted macOS jobs share a serialising concurrency group so at most one runs at a
time: macOS minutes count 10× against the included Actions quota and the two have no
reason to overlap. Both still gate every pull request. Linux carries the torch matrix
(2.11 / 2.12 / 2.13 across Python 3.10–3.13); macOS runs a single torch version, because
what it adds is Apple-specific device behaviour rather than another torch version.

### Measured tolerances

Sealed in `src/niverel_mamba/certification/tolerances.yaml`. Observed on Apple M1,
macOS 26.0.1, torch 2.13.0, MLX 0.32.0:

| comparison | max abs error | tolerance class |
|---|---|---|
| oracle vs upstream `ssd_minimal_discrete` (fp64) | 3.6e-15 | `cpu_float64` |
| chunked vs sequential oracle (fp64) | 1.1e-14 | `cpu_float64` |
| `forward` vs `concat(step)` (fp64) | 1.6e-15 | `cpu_float64` |
| segmented vs separate documents (fp64) | 4.4e-16 | `cpu_float64` |
| CPU vs MPS, real V3 weights, L=8192 | 1.9e-05 | `mps_float32` |
| torch vs MLX, real V3 weights | 1.1e-05 | `mlx_float32` |

Tolerances are observed and then sealed. Widening one to make a test pass is not a
permitted move — a measurement outside its band is a finding to report.

### What is *not* promised

Same weights, same equations, same logical operation order, and **measured** numerical
equivalence. Not bit-for-bit identity between CUDA bf16, CPU fp32, MPS and MLX —
reductions, fusions and rounding genuinely differ.

Even the strict-reset invariant holds to float64 rounding rather than bitwise: the
intra-chunk decay is evaluated as `exp(cs_l - cs_s)` over a cumulative sum spanning the
whole chunk, and that cancellation is exact in real arithmetic but not in floating point.
Upstream's Triton kernels compute the same segsum, so the CUDA path carries it too.

## The weight contract

One contract, `niverel-mamba2-weights-v1`, extracted from the genuine
`mamba_ssm.Mamba2` 2.3.2.post1 and *verified* against it across nine configurations
(including `bias`, `conv_bias=False`, `rmsnorm=False`, `D_has_hdim`, `ngroups>1` and the
gated-MLP `d_ssm < d_inner` branch).

For the Niverel Foundation V3 configuration
(`d_model=768, d_state=128, d_conv=4, expand=2`, with upstream defaults
`headdim=64, ngroups=1, chunk_size=256`):

```
dt_bias           (24,)
A_log             (24,)
D                 (24,)
in_proj.weight    (3352, 768)
conv1d.weight     (1792, 1, 4)
conv1d.bias       (1792,)
norm.weight       (1536,)
out_proj.weight   (768, 1536)
```

Two things the extraction settled that guesswork would have got wrong:

* **there is no `init_states`** in Mamba2 2.3.2.post1 — no learnable initial state exists;
* **`norm.eps` is `1e-5`**, hard-coded in `Mamba2.__init__`, not the `1e-6` default of
  upstream's own `rms_norm_ref`.

Loading is always equivalent to `load_state_dict(state_dict, strict=True)`. A missing
key, an unexpected key, a differing shape, an incompatible configuration or an unknown
contract version is refused. For MLX the conversion is reversible and proven so: the
round-trip `PyTorch → canonical → MLX → canonical → PyTorch` reproduces identical
SHA-256 digests.

## Support matrix (0.1)

| backend | torch | python | OS | device | status |
|---|---|---|---|---|---|
| `torch-reference` | 2.11–2.13 | 3.10–3.13 | Linux | CPU | certified |
| `torch-reference` | 2.11–2.13 | 3.10–3.13 | macOS arm64 | CPU/MPS | certified |
| `cuda-reference` | 2.11.0+cu128 | 3.12 | Linux x86_64 | sm80/sm90 | experimental, pending GPU certification |
| `mlx` | n/a | 3.10–3.13 | macOS arm64 ≥14 | Apple GPU | experimental |

Not supported in 0.1: Windows CUDA, ROCm, Intel XPU, Jetson, musl/Alpine, PyTorch
nightly, Python 3.14+ for CUDA extensions, MLX training.

## Development

```bash
uv sync --extra torch --extra mlx --extra dev

uv run python scripts/extract_weight_contract.py      # Phase 0 gate
uv run python scripts/make_golden_fixture.py          # tiny + segmented fixtures
uv run python scripts/make_golden_fixture.py --niverel  # real V3 block (needs HF_TOKEN)

uv run pytest -q
uv run pytest -q -m "not slow"
uv run ruff check . && uv run mypy src
```

The sequential oracle in `torch_ops/ssd_sequential.py` is the source of truth for every
numerical test. It is slow by design and is never to be deleted, however fast the chunked
path becomes.

## Licence

Apache-2.0. See `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md`.

`mamba-ssm` (Apache-2.0) and MLX (MIT) are used as documented in `THIRD_PARTY_NOTICES.md`.
No third-party source is vendored into this repository.
