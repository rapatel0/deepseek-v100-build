# CUDA build of the antirez DeepSeek V4 Flash llama.cpp fork, with the
# DSv4 HyperConnection ops backfilled into the CUDA backend by the
# cuda-patches/ tree.
#
# Targets sm_70 (V100). Build this on the GPU host so the runtime image
# matches the driver version. CMAKE_CUDA_ARCHITECTURES is set explicitly
# to 70-real (skip JIT, ship binary code).

ARG CUDA_VERSION=12.2.2
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG FORK_REPO=https://github.com/antirez/llama.cpp-deepseek-v4-flash.git
ARG FORK_REF=main
ARG CUDA_ARCH=70-real

ENV DEBIAN_FRONTEND=noninteractive

# libcuda.so (CUDA Driver API) ships only as a build-time stub in the devel
# image. cmake's find_package(CUDAToolkit) doesn't always add the stubs dir
# to the link path, so the final llama-server link fails on cuMemMap etc.
# Putting it in LIBRARY_PATH covers every link step in the build.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        libcurl4-openssl-dev \
        libgomp1 \
        pkg-config \
        patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${FORK_REF}" "${FORK_REPO}" llama.cpp

# Drop in the CUDA implementations of the DSv4 HC ops (auto-globbed by the
# existing ggml-cuda CMakeLists), then patch ggml-cuda.cu to dispatch them.
COPY cuda-patches/ggml-cuda/dsv4-hc.cu  /src/llama.cpp/ggml/src/ggml-cuda/
COPY cuda-patches/ggml-cuda/dsv4-hc.cuh /src/llama.cpp/ggml/src/ggml-cuda/
COPY cuda-patches/ggml-cuda/concat.cu   /src/llama.cpp/ggml/src/ggml-cuda/concat.cu
COPY cuda-patches/ggml-cuda/concat.cuh  /src/llama.cpp/ggml/src/ggml-cuda/concat.cuh
COPY cuda-patches/0001-wire-dsv4-hc-cuda.patch /tmp/0001.patch
COPY cuda-patches/0002-concat-f16-bf16.patch /tmp/0002.patch
COPY cuda-patches/tests/concat_bitcompare.cu /tmp/concat_bitcompare.cu

WORKDIR /src/llama.cpp
RUN git apply --verbose /tmp/0001.patch \
    && git apply --verbose /tmp/0002.patch \
    && grep -q "ggml_cuda_op_dsv4_hc_split_sinkhorn" ggml/src/ggml-cuda/ggml-cuda.cu \
    && grep -q "GGML_OP_DSV4_HC_SPLIT_SINKHORN" ggml/src/ggml-cuda/ggml-cuda.cu \
    && grep -q "GGML_TYPE_BF16" ggml/src/ggml-cuda/concat.cu \
    && grep -q "GGML_OP_CONCAT" ggml/src/ggml-cuda/ggml-cuda.cu \
    && echo "patches applied OK"

# --no-as-needed -lcuda forces libggml-cuda.so to declare libcuda.so as
# a NEEDED dep, which it otherwise doesn't (this snapshot of the fork
# doesn't link CUDA::cuda_driver in its target_link_libraries call).
# Without this, llama-server's link step fails on cuMemMap, cuGetErrorString,
# and other CUDA Driver API symbols.
# GGML_SCHED_MAX_SPLIT_INPUTS=128 (default 30): DSv4 with 4-way pipeline
# parallelism + the fused Gated Delta Net path produces more cross-GPU
# inputs than the default array can hold. The constant is `#ifndef`-guarded
# in ggml-backend.cpp so a compile flag is enough.
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -lcuda" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -lcuda" \
        -DCMAKE_CXX_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=128" \
        -DCMAKE_CUDA_FLAGS="-DGGML_SCHED_MAX_SPLIT_INPUTS=128" \
    && cmake --build build -j"$(nproc)" --target llama-server llama-cli

# Sanity check: verify the new CUDA HC ops are present in libggml-cuda.so.
# (They live in the shared library, not in llama-server itself, since
# llama-server links against ggml-cuda dynamically.) Symbols are
# C++-mangled, so we grep loosely for "dsv4_hc".
RUN set -e; \
    SO=$(find build -name 'libggml-cuda.so*' -type f | head -1); \
    test -n "$SO" || { echo "ERROR: libggml-cuda.so not found in build/" >&2; exit 1; }; \
    echo "checking $SO"; \
    nm -D "$SO" 2>/dev/null | grep -E "dsv4_hc" \
      || nm    "$SO" 2>/dev/null | grep -E "dsv4_hc" \
      || { echo "ERROR: dsv4_hc_* CUDA symbols missing from $SO" >&2; exit 1; }; \
    echo "dsv4_hc CUDA symbols present"

RUN nvcc -std=c++17 -O2 -arch=sm_70 /tmp/concat_bitcompare.cu -o /tmp/concat-bitcompare

# ---- runtime stage ----
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libcurl4 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 1001 --home /app llama

COPY --from=build /src/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/llama.cpp/build/bin/llama-cli    /usr/local/bin/llama-cli
COPY --from=build /src/llama.cpp/build/bin/*.so         /usr/local/lib/
COPY --from=build /tmp/concat-bitcompare                /usr/local/bin/concat-bitcompare

RUN ldconfig

USER llama
WORKDIR /app

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/llama-server"]
