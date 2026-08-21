# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows the scheme in the project brief:

* **MAJOR** — a break in the public API or the weight contract
* **MINOR** — a new backend, PyTorch version, platform or capability
* **PATCH** — a fix with no contract change, a numerically compatible
  optimisation, or documentation

## [Unreleased]

## [0.1.1] — 2026-08-21

`cuda-reference` is certified. Everything else here is what it took to certify
it honestly.

### Certified

* `cuda-reference` moves from `experimental` to `numerically-certified`,
  measured on an A100 (sm_80) and an H100 (sm_90) across all three CUDA
  runtimes: max_abs **2.5e-03** against the portable backend in float32,
  cosine similarity **0.999999988**, bit-identical on both architectures.
  It is deliberately not `reference` -- that names the backend which *produces
  or certifies* a result, and here that is `torch-reference`.
* `cuda_float32` is sealed from that measurement, with the derivation recorded:
  upstream's `tl.dot` calls pass no `allow_tf32`, so Ampere and Hopper multiply
  in TF32, ten mantissa bits, one epsilon of 4.9e-04.
* `cuda_bfloat16` is recorded as a **measurement and never a gate**. Gating on
  it measures the number format: rounding inputs and weights to bfloat16
  deviates by 3.0e-02 with the portable implementation alone, no CUDA involved.

### Fixed

* `install-backend cuda` works again. It had been broken since the release
  started shipping one archive per runtime, because the build manifests carry
  `"url": null`. The release now publishes an index naming each archive, its
  URL and SHA-256, and the SHA-256 of every wheel inside it; the command
  verifies the archive before unpacking and each wheel before installing.
* Publishing moved out of a reusable workflow. PyPI verifies a PEP 740
  attestation against the entry workflow while matching the Trusted Publisher
  against the reusable one, so no configuration satisfied both.
* The certification pod is required to carry a CUDA 13.0 driver, and refuses
  early if it does not. Two of the three runtimes are cu130 and a 12.x driver
  cannot run them.
* `cuda-reference` no longer reports as available on the strength of installed
  metadata alone: upstream's `__init__` needs `transformers`, which `--no-deps`
  does not install, and a package that will not import is not available.

### Added

* `verify --certify cuda-reference` produces a report whose candidate really is
  the CUDA backend. The reports shipped with 0.1.0 were produced on GPUs and
  certified the portable CPU implementation.
* `--measure` reports every comparison without passing judgement, so a run on
  rented hardware yields evidence instead of stopping at the first band nobody
  has observed.
* `ci-certify-rehearsal` runs everything the certification pod does except the
  numerical comparison, on a free Linux runner.
* `wheel_run_id`, `certify_only` and `measure_only` on the release workflow, so
  a release need not rebuild wheels that cannot differ or re-rent GPUs that
  have already answered.


## [0.1.0] — 2026-08-17

First release. Portable Mamba2 with one weight contract and honest per-backend
certification.

### Packaging

* `numpy` is a core dependency. Certification utilities are part of the core
  surface, and both `compare()` and `tensor_digest()` need it. The CLI keeps
  its heavy imports inside `verify.run()`, so `doctor` and `inspect` stay
  usable on a core-only install -- a cold `pip install` followed by
  `niverel-mamba doctor` must never crash on a dependency it does not use.
* `niverel-mamba verify` without PyTorch exits 2 with an actionable message
  rather than a traceback.

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
