# SPRINT-020 P1.3 — Minimal turbomind carve-out for SPRINT-020 gemm_bench.
#
# This CMakeLists is the minimal harness that lets us build turbomind's
# `gemm_bench` standalone without invoking lmdeploy's setup.py pipeline
# (pod has no python3). It pulls in just enough of the turbomind tree to
# link gemm_bench against the SM70 INT8 GEMM kernels.
#
# Gated by parent `tools/tc-grid/CMakeLists.txt`'s
# `TCGRID_ENABLE_TURBOMIND_GEMM=ON`. Default OFF preserves the existing
# tc-grid build.
#
# Expected to fail on first invocation; iteration plan in
# tools/tc-grid/docs/SPRINT-020-P1-gemm_bench-audit.md §"Risks for P1.5".

cmake_minimum_required(VERSION 3.18)

include(FetchContent)

if(NOT DEFINED LMDEPLOY_SRC)
    set(LMDEPLOY_SRC "${CMAKE_SOURCE_DIR}/../../research/lmdeploy/src")
endif()
# turbomind sources use `#include "src/turbomind/.../..."` (root-relative).
# The upstream lmdeploy CMakeLists adds `${PROJECT_SOURCE_DIR}` to the
# include path so that resolves; we mirror that here by adding the
# directory ONE LEVEL ABOVE src/.
get_filename_component(LMDEPLOY_ROOT "${LMDEPLOY_SRC}" DIRECTORY)
include_directories(${LMDEPLOY_ROOT})

# Turbomind sources call constexpr __host__ functions from device code; the
# upstream lmdeploy build enables --expt-relaxed-constexpr globally. Mirror
# that here. Also --expt-extended-lambda for the same reason.
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    # turbomind/kernels/core/mma.h uses nv_bfloat16 in overload signatures
    # without including cuda_bf16.h itself. Force-include here so all
    # turbomind sources see the type without patching every include site.
    $<$<COMPILE_LANGUAGE:CUDA>:-include>
    $<$<COMPILE_LANGUAGE:CUDA>:cuda_bf16.h>
)

if(NOT EXISTS "${LMDEPLOY_SRC}/turbomind/kernels/gemm/CMakeLists.txt")
    message(FATAL_ERROR "LMDEPLOY_SRC=${LMDEPLOY_SRC} does not contain "
            "turbomind/kernels/gemm/CMakeLists.txt — set LMDEPLOY_SRC correctly")
endif()

# ---- fmt (header-only) ----------------------------------------------------
# core, parser, cuda_utils all link against fmt::fmt. Use a recent stable
# release in header-only mode to avoid linking a separate fmt library.
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.0.2
    GIT_SHALLOW    TRUE
)
set(FMT_HEADER_ONLY ON CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fmt)

# Make fmt headers visible globally so turbomind sources that include
# <fmt/format.h> compile cleanly without per-target target_link_libraries
# changes.
include_directories(SYSTEM ${fmt_SOURCE_DIR}/include)

# ---- concurrentqueue is FetchContent'd inside core/CMakeLists.txt — we
# can rely on that.

# ---- CUTLASS alias for turbomind's expected target name -------------------
# Upstream turbomind links `nvidia::cutlass::cutlass`. tc-grid's parent
# CMakeLists uses `cutlass_SOURCE_DIR` directly. Provide an alias so
# add_subdirectory(turbomind/kernels/gemm) finds what it expects.
if(NOT TARGET nvidia::cutlass::cutlass)
    add_library(turbomind_cutlass_alias INTERFACE)
    target_include_directories(turbomind_cutlass_alias INTERFACE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    add_library(nvidia::cutlass::cutlass ALIAS turbomind_cutlass_alias)
endif()

# ---- Defaults turbomind's CMakeLists expect -------------------------------
# These would otherwise come from the parent lmdeploy CMakeLists.
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_PYT OFF CACHE BOOL "" FORCE)

# ---- Pull in the turbomind libraries needed by gemm_bench ----------------
# Order matters: utils → core → kernels/gemm
add_subdirectory(${LMDEPLOY_SRC}/turbomind/utils
                 ${CMAKE_BINARY_DIR}/turbomind_utils
                 EXCLUDE_FROM_ALL)
add_subdirectory(${LMDEPLOY_SRC}/turbomind/core
                 ${CMAKE_BINARY_DIR}/turbomind_core
                 EXCLUDE_FROM_ALL)
add_subdirectory(${LMDEPLOY_SRC}/turbomind/kernels/gemm
                 ${CMAKE_BINARY_DIR}/turbomind_gemm
                 EXCLUDE_FROM_ALL)

# ---- Surface gemm_bench at a discoverable path ---------------------------
# Use a symlink-ish copy after build so users find it at:
#   tools/tc-grid/build/gemm_bench_turbomind
if(TARGET gemm_bench)
    set_target_properties(gemm_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_NAME "gemm_bench_turbomind"
    )
endif()

# ---- SPRINT-020 P1.5 path (b) — nvbench-free fallback --------------------
# Gated separately so we can build EITHER the upstream gemm_bench (needs
# nvbench + CMake 3.23+) OR our simple harness, depending on environment.
option(TM_GEMM_BENCH_SIMPLE "Build the nvbench-free turbomind gemm bench (SPRINT-020 P1.5 path b)" OFF)
if(TM_GEMM_BENCH_SIMPLE)
    add_executable(gemm_bench_simple gemm_bench_simple.cu)
    target_include_directories(gemm_bench_simple PRIVATE
        ${LMDEPLOY_SRC}
    )
    target_link_libraries(gemm_bench_simple PRIVATE gemm2 core CUDA::cublas)
    set_target_properties(gemm_bench_simple PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()

# ---- SPRINT-021 P0 — packed-weight ceiling bench ------------------------
# Measures turbomind's FP8/U4/FP4 paths in addition to FP16. Requires
# additional libs (kernels + gpt_kernels) because we call Convert /
# extend_to_u16 / invokeTransposeAxis01 directly.
option(TM_GEMM_BENCH_PACKED "Build the U4/FP8/FP4/FP16 ceiling bench (SPRINT-021 P0)" OFF)
if(TM_GEMM_BENCH_PACKED)
    # Pull in gpt_kernels.cu directly (invokeTransposeAxis01 lives there).
    # Avoids adding the whole kernels/ subdir which has many other deps.
    add_executable(gemm_bench_packed
        gemm_bench_packed.cu
        ${LMDEPLOY_SRC}/turbomind/kernels/gpt_kernels.cu
    )
    target_include_directories(gemm_bench_packed PRIVATE
        ${LMDEPLOY_SRC}
    )
    target_link_libraries(gemm_bench_packed PRIVATE gemm2 core CUDA::cublas)
    set_target_properties(gemm_bench_packed PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
endif()
