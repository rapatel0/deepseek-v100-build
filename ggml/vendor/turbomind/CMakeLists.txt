# SPRINT-023 P1 — libggml-turbomind.so carve-out
#
# Builds a shared library that exposes the grouped MoE GEMM C ABI defined in
# ggml-turbomind-api.h, wrapping turbomind's Gemm::Run + LlamaLinear semantics
# for V100 sm70.
#
# This is the dlopen target that libggml-cuda.so loads at runtime when the
# CUDA_TURBOMIND buffer type is requested. All CUTLASS / fmt / turbomind
# internals are linked statically into this .so with hidden visibility.
#
# Build pattern is lifted from tools/tc-grid/turbomind_minimal/CMakeLists.txt
# (SPRINT-020 P1.3 / SPRINT-021 carve-out) which is the only pattern that
# successfully built turbomind gemm2 + core + cuda_utils on the V100 build
# environment (CUDA 12.2, cmake 3.22, gcc 11.4).
#
# Gated by parent ggml/CMakeLists.txt's GGML_TURBOMIND=ON. Default OFF so
# stock builds remain unchanged.

cmake_minimum_required(VERSION 3.18)

# Standalone project — not add_subdirectory'd from llama.cpp. The cmake
# policy state lmdeploy expects (VERSION 3.11) conflicts with llama.cpp's
# modern CUDA build; isolating into its own cmake invocation sidesteps the
# whole compat surface.
project(ggml-turbomind LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

# LMDEPLOY_SRC must point at the lmdeploy/src tree containing the turbomind
# subdirectory.
if(NOT DEFINED LMDEPLOY_SRC)
    set(LMDEPLOY_SRC "${CMAKE_SOURCE_DIR}/../../../research/lmdeploy/src" CACHE PATH
        "Path to lmdeploy/src — contains turbomind/ subdirectory")
endif()

if(NOT EXISTS "${LMDEPLOY_SRC}/turbomind/kernels/gemm/CMakeLists.txt")
    message(FATAL_ERROR
        "GGML_TURBOMIND=ON but LMDEPLOY_SRC=${LMDEPLOY_SRC} does not contain "
        "turbomind/kernels/gemm/. Set -DLMDEPLOY_SRC=/path/to/lmdeploy/src "
        "or place lmdeploy at the default location.")
endif()

# Add include path so turbomind sources can use `#include "src/turbomind/..."`.
get_filename_component(LMDEPLOY_ROOT "${LMDEPLOY_SRC}" DIRECTORY)
include_directories(${LMDEPLOY_ROOT})

# Hide internal symbols so the .so only exports our C ABI (and standard CUDA
# runtime requirements). Smaller surface, no ODR collisions when dlopen'd by
# libggml-cuda.so.
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# ---- fmt (header-only) ----------------------------------------------------
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
include_directories(SYSTEM ${fmt_SOURCE_DIR}/include)
# Ensure fmt static lib (if built despite FMT_HEADER_ONLY) is PIC so it
# can link into our .so without R_X86_64_PC32 relocation errors.
if(TARGET fmt)
    set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

# ---- CUTLASS 2.11.0 (header-only used by turbomind) ----------------------
# turbomind links against the nvidia::cutlass::cutlass alias name. CUTLASS
# 2.x has mature sm_70 support; 3.x focuses on sm_80+.
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v2.11.0
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

if(NOT TARGET nvidia::cutlass::cutlass)
    add_library(turbomind_cutlass_alias INTERFACE)
    target_include_directories(turbomind_cutlass_alias INTERFACE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    add_library(nvidia::cutlass::cutlass ALIAS turbomind_cutlass_alias)
endif()

# ---- Defaults turbomind's CMakeLists expect ------------------------------
set(BUILD_TEST OFF CACHE BOOL "" FORCE)
set(BUILD_PYT OFF CACHE BOOL "" FORCE)

# turbomind builds need these CUDA flags. Set them ONLY for turbomind targets,
# not for the global scope (which would propagate to FetchContent subbuilds
# of fmt/cutlass that don't enable CUDA).
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    # mma.h uses nv_bfloat16 without including cuda_bf16.h — force-include.
    $<$<COMPILE_LANGUAGE:CUDA>:-include>
    $<$<COMPILE_LANGUAGE:CUDA>:cuda_bf16.h>
)

# ---- Pull in turbomind libraries needed by the grouped-MoE GEMM path ----
# lmdeploy CMakeLists set cmake_minimum_required(VERSION 3.11) which resets
# policies — CMP0146 (FindCUDA module disabled), CMP0104 (CUDA architectures),
# and others that the parent llama.cpp build relies on. Override them here
# so the policy state we hand to subdirectories matches what cmake 3.22+
# would natively use.
cmake_policy(VERSION 3.18)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

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

# Force these object libraries to be position-independent so they can be
# combined into a shared object cleanly.
foreach(t IN ITEMS gemm2 core parser cuda_utils)
    if(TARGET ${t})
        set_target_properties(${t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
endforeach()

# ---- The shared library ---------------------------------------------------
# `api.cc` implements the 6-function C ABI in ggml-turbomind-api.h. We also
# need gpt_kernels.cu for invokeTransposeAxis01 (used during weight packing).
add_library(ggml-turbomind SHARED
    api.cc
    ggml-turbomind-deinterleave.cu
    ${LMDEPLOY_SRC}/turbomind/kernels/gpt_kernels.cu
)

target_include_directories(ggml-turbomind PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(ggml-turbomind PRIVATE
    ${LMDEPLOY_SRC}
)
target_link_libraries(ggml-turbomind PRIVATE gemm2 core CUDA::cudart CUDA::cublas)

# Position-independent code + hide internals.
set_target_properties(ggml-turbomind PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "ggml-turbomind"
)

# Don't propagate CUTLASS / fmt headers to consumers — they're build-time
# only. Only the C ABI header in include/ is consumer-visible.

# ---- P1.4 standalone test -------------------------------------------------
# Tiny dlopen-based test that exercises the C ABI without linking against the
# .so directly. Validates the export surface.
option(GGML_TURBOMIND_TEST "Build dlopen smoke test for libggml-turbomind.so" ON)
if(GGML_TURBOMIND_TEST)
    add_executable(test_ggml_turbomind test_dlopen.cpp)
    target_include_directories(test_ggml_turbomind PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_ggml_turbomind PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
    add_dependencies(test_ggml_turbomind ggml-turbomind)

    # SPRINT-023 P2.3 — round-trip correctness gate.
    add_executable(test_ggml_turbomind_correctness test_correctness.cpp)
    target_include_directories(test_ggml_turbomind_correctness PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_ggml_turbomind_correctness PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
    add_dependencies(test_ggml_turbomind_correctness ggml-turbomind)
endif()
