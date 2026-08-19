# Build the upstream CUDA extensions against a pinned Torch/CUDA pair.
#
# A physical GPU is not required to compile: nvcc and the toolkit suffice. The
# resulting wheels must still be certified on real hardware before publication
# -- see certify-cuda-sm80.yml. Nothing here is ever published directly.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS build

ARG TORCH_CUDA_ARCH_LIST="8.0;9.0"
ARG MAMBA_SSM_VERSION=2.3.2.post1
ARG CAUSAL_CONV1D_VERSION=1.6.2.post1
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FORCE_CUDA=1 \
    CCACHE_DIR=/ccache \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates git curl ccache \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip wheel setuptools ninja packaging

# Torch must be installed BEFORE the extensions: upstream's setup.py reads the
# installed torch to decide its CUDA and C++11-ABI dimensions. That is also why
# --no-build-isolation is required below -- with isolation, the build would pick
# a different torch than the one these wheels must match.
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Capped deliberately: nvcc is memory-hungry and an unbounded parallel build
# reliably exhausts a CI runner.
ENV MAX_JOBS=4

RUN --mount=type=cache,target=/ccache \
    pip wheel --no-build-isolation --no-deps \
        "causal-conv1d==${CAUSAL_CONV1D_VERSION}" -w /wheelhouse

RUN --mount=type=cache,target=/ccache \
    pip install --no-build-isolation --no-deps /wheelhouse/causal_conv1d-*.whl && \
    pip wheel --no-build-isolation --no-deps \
        "mamba-ssm==${MAMBA_SSM_VERSION}" -w /wheelhouse

RUN python - <<'PY'
import glob
wheels = glob.glob("/wheelhouse/*.whl")
assert any("mamba_ssm" in w for w in wheels), wheels
assert any("causal_conv1d" in w for w in wheels), wheels
print("\n".join(wheels))
PY

FROM scratch AS export
COPY --from=build /wheelhouse/ /
