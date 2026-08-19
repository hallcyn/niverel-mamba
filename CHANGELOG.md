# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows the scheme in the project brief:

* **MAJOR** — a break in the public API or the weight contract
* **MINOR** — a new backend, PyTorch version, platform or capability
* **PATCH** — a fix with no contract change, a numerically compatible
  optimisation, or documentation

## [Unreleased]

### Fixed

* A cold `pip install niverel-mamba` followed by `niverel-mamba doctor` crashed
  with `ModuleNotFoundError: numpy`. `build_parser()` imports every subcommand
  so `--help` can list them, so `verify`'s transitive numpy import was paid for
  by `doctor` and `inspect` too. numpy is now a core dependency (certification
  utilities are core content per brief section 13, and both `compare()` and
  `tensor_digest()` need it), and `verify` defers its heavy imports so the
  diagnostic commands stay usable in a partially-installed environment.
* `niverel-mamba verify` without PyTorch printed a raw traceback instead of a
  clean, actionable error. It now exits 2 with an explanation.

## [0.1.0] — 2026-08-17

First release. Portable Mamba2 with one weight contract and honest per-backend
certification.

### Added

* **Weight contract `niverel-mamba2-weights-v1`**, extracted from the genuine
  `mamba_ssm.Mamba2` 2.3.2.post1 by `scripts/extract_weight_contract.py` and
  verified against it across nine configurations covering `bias`,
  `conv_bias=False`, `rmsnorm=False`, `D_has_hdim`, `ngroups>1` and the
  gated-MLP `d_ssm < d_inner` branch. Shapes are symbolic, so one contract
  covers every configuration.
* **`torch-reference` backend** — pure PyTorch, running on Linux CPU, Linux
  CUDA without Mamba kernels, macOS CPU and macOS MPS.
  * `ssd_sequential`, the explicit float64 recurrence oracle.
  * `ssd_chunked`, the production chunked SSD: internal padding, no global
    L×L matrix, autograd-compatible, `initial_states` and final-state support.
  * `per_segment`, an independent second oracle sharing no masking code with
    the chunked path.
  * Strict `seq_idx` reset in the chunked path via four boundary masks.
  * Functional stateful API: `allocate_inference_state` / `step`.
* **`mlx` backend** for Apple Silicon, with a reversible weight conversion
  proven byte-identical by SHA-256 round-trip.
* **`cuda-reference` backend** wrapping the upstream kernels, which raises
  rather than falling back when its wheels are absent.
* **CLI**: `doctor`, `inspect`, `verify`, `install-backend`.
* **Certification**: golden fixtures (tiny / segmented / real Foundation V3
  block), elementwise comparison against sealed tolerances, JSON reports that
  always name the reference backend they were measured against.
* **Niverel integration**: `build_mamba2(config, backend)` factory and
  `niverel_v3_config()`, with `state_dict` identical across backends.

### Certification status

* `torch-reference` CPU and MPS: **numerically-certified** against the float64
  sequential oracle, which is itself checked against upstream's
  `ssd_minimal_discrete` at 3.6e-15.
* `mlx`: **experimental**. Parity against `torch-reference` CPU is measured
  (1.1e-05 on real V3 weights) but not yet enforced on every release.
* `cuda-reference`: **experimental**, not `reference`. No NVIDIA GPU has run
  it. The `cuda_bfloat16` tolerance class is marked unverified and carries the
  brief's starting values unchanged. It becomes the reference only once
  `certify-cuda-sm80.yml` produces a report on real hardware.
* No backend claims `training: true`. That requires a gradient comparison
  against CUDA which has not been performed.

### Known limitations

* CUDA wheels are not yet built or published; `build-cuda-wheels.yml` and the
  `certify-cuda-*` workflows exist but have never run.
* Not published to PyPI yet. The name is available on both PyPI and TestPyPI.
* Metal kernels for MLX are deliberately out of scope until parity is
  CI-enforced — correctness first, then profiling, then optimisation.
* The strict-reset invariant holds to float64 rounding rather than bitwise,
  because the chunked decay uses a cumulative-sum difference. Upstream's
  kernels share this property.

### Notes on upstream

Two findings from the contract extraction that contradict assumptions worth
recording:

* Mamba2 2.3.2.post1 has **no `init_states` tensor** — there is no learnable
  initial state.
* The gated norm's `eps` is **1e-5**, hard-coded in `Mamba2.__init__`, and not
  the `1e-6` default of upstream's own `rms_norm_ref`.

[Unreleased]: https://github.com/hallcyn/niverel-mamba/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hallcyn/niverel-mamba/releases/tag/v0.1.0
