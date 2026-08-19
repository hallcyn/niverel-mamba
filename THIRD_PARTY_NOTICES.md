# Third-party notices

`niverel-mamba` is licensed under the Apache License 2.0 (see `LICENSE`).
This file records the third-party work it builds on and how that work is used.

---

## mamba-ssm — Apache License 2.0

* Project: <https://github.com/state-spaces/mamba>
* Copyright (c) 2023-2024, Tri Dao, Albert Gu
* License: Apache License 2.0
* Version this project is pinned to: **2.3.2.post1**

### How it is used

**No source file from `mamba-ssm` is vendored into this repository.**

`mamba-ssm` is used in three distinct and clearly separated ways:

1. **As the source of the weight contract.**
   `scripts/extract_weight_contract.py` reads the pure-Python tree out of a
   `mamba_ssm` wheel at *build time*, instantiates the genuine
   `mamba_ssm.modules.mamba2.Mamba2`, and records the resulting parameter
   names, shapes and configuration into `schemas/`. Only the derived metadata
   is committed — never upstream source code.

2. **As the numerical reference.**
   The portable implementations in `niverel_mamba/torch_ops/` and
   `niverel_mamba/mlx_ops/` are **reimplementations of the published Mamba2
   equations**, written to reproduce upstream's operation order and its
   float32 upcasting rules. They are certified against upstream's own
   `ssd_minimal_discrete` (the paper's Listing 1) and, on GPU hardware,
   against the upstream kernels themselves.

   Where this project deliberately diverges from upstream behaviour, the
   divergence is documented in the source at the point it occurs. The
   substantive ones are:

   * `step()` is functional (returns a new state) rather than mutating in
     place, so that the PyTorch and MLX backends share one contract;
   * `step()` computes `dt` in float32 and applies `dt_limit`, aligning it
     with upstream's own *forward* path, which its non-kernel `step` does not;
   * `step()` supports `ngroups > 1`, which upstream's non-kernel path
     asserts against;
   * the SSM state is kept in float32 regardless of weight dtype;
   * masked decay exponents use `-1e30` rather than `-inf` (value-identical
     under `exp`, but keeps `inf` out of the autograd graph and out of
     MPS/MLX code paths).

3. **As an optional runtime dependency.**
   The `cuda-reference` backend imports and calls the real upstream `Mamba2`
   when it is installed. `mamba-ssm` is never bundled, never vendored, and
   never compiled by this package on a user's machine.

A copy of the Apache License 2.0 under which `mamba-ssm` is distributed is
included in this repository as `LICENSE` (the same licence this project uses).

---

## causal-conv1d — Apache License 2.0

* Project: <https://github.com/Dao-AILab/causal-conv1d>
* Copyright (c) 2022-2024, Tri Dao
* License: Apache License 2.0
* Version pinned by the Niverel reference runtime: **1.6.2.post1**

Used only as an optional runtime dependency of the `cuda-reference` backend.
No source is vendored. The portable causal convolution in
`niverel_mamba/torch_ops/causal_conv.py` is an independent implementation
expressed as a sum of masked shifts.

---

## MLX — MIT License

* Project: <https://github.com/ml-explore/mlx>
* Copyright (c) 2023 Apple Inc.
* License: MIT

Used as an optional runtime dependency of the `mlx` backend. No MLX source is
vendored; `niverel_mamba/mlx_ops/` uses only MLX's public Python API.

MIT License terms:

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## PyTorch — BSD-3-Clause

* Project: <https://github.com/pytorch/pytorch>
* License: BSD-3-Clause

Optional runtime dependency. No source vendored.

---

## Papers

The Mamba2 algorithm implemented here is described in:

> Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient
> Algorithms Through Structured State Space Duality.* ICML 2024.

The chunked decomposition follows Listing 1 of that paper.
