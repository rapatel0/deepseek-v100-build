cmake_minimum_required(VERSION 3.18)
project(tc-grid LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "V100 default; override at configure")

# --- CUTLASS (header-only) for SPRINT-018 V100 INT8 GEMM hedge -------------
# v2.11.0: last 2.x release, mature sm_70 support. CUTLASS 3.x focuses on
# sm_80+. We only need headers; no add_subdirectory to avoid pulling their
# whole build system.
option(TCGRID_ENABLE_CUTLASS "Pull CUTLASS via FetchContent and enable cutlass_int8 kernel" ON)
if(TCGRID_ENABLE_CUTLASS)
    include(FetchContent)
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v2.11.0
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(cutlass)
    if(NOT cutlass_POPULATED)
        FetchContent_Populate(cutlass)
    endif()
    message(STATUS "CUTLASS source dir: ${cutlass_SOURCE_DIR}")
endif()

set(TCGRID_SOURCES
    src/main.cu
    src/data_gen.cu
    src/reference.cu
    src/launch_int8.cu
    src/launch_int4.cu
    src/launch_fp4.cu
    src/launch_fp8.cu
)

# SPRINT-020 P2: turbomind GEMM bridge. The launcher skeleton always
# compiles; the actual integration depends on TCGRID_ENABLE_TURBOMIND_GEMM
# AND on P1's gemm_bench build producing a linkable turbomind gemm2 lib.
# Default OFF preserves non-SPRINT-020 builds.
option(TCGRID_ENABLE_TURBOMIND_GEMM "Build the turbomind GEMM bridge launcher (SPRINT-020 P2)" OFF)
list(APPEND TCGRID_SOURCES src/launch_turbomind_int8.cu)

add_executable(tc-grid ${TCGRID_SOURCES})

if(TCGRID_ENABLE_TURBOMIND_GEMM)
    target_compile_definitions(tc-grid PRIVATE TCGRID_ENABLE_TURBOMIND_GEMM=1)
    # SPRINT-020 P1.3 — pull in the minimal turbomind carve-out (gemm2 + core
    # + parser + cuda_utils + fmt + concurrentqueue).
    # Requires LMDEPLOY_SRC override (defaults to ../../research/lmdeploy/src).
    # TM_ENABLE_GEMM_BENCH and TM_GEMM_BENCH_SIMPLE are intentionally NOT
    # forced here — pass them via cmake -D to opt in to a bench target.
    # (default both OFF → no nvbench FetchContent attempt → safe in
    # offline pods)
    add_subdirectory(turbomind_minimal)
    target_link_libraries(tc-grid PRIVATE gemm2 core)
endif()

target_include_directories(tc-grid PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

if(TCGRID_ENABLE_CUTLASS)
    target_include_directories(tc-grid PRIVATE
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    target_compile_definitions(tc-grid PRIVATE TCGRID_HAS_CUTLASS=1)
endif()

target_link_libraries(tc-grid PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(tc-grid PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(tc-grid PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>
)

# P0 build-verification executable: minimal CUTLASS Gemm template
# instantiation that proves CUTLASS headers compile and link for sm_70.
if(TCGRID_ENABLE_CUTLASS)
    add_executable(test_cutlass_build tests/test_cutlass_build.cu)
    target_include_directories(test_cutlass_build PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/include
        ${cutlass_SOURCE_DIR}/tools/util/include
    )
    target_link_libraries(test_cutlass_build PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_options(test_cutlass_build PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
endif()

# V11 Step 2a: SMEM → fragment round-trip test for turbomind SmemCopy_MMA_884
# lane mappings on sm_70.
add_executable(test_smem_to_frag_sm70 tests/test_smem_to_frag_sm70.cu)
target_include_directories(test_smem_to_frag_sm70 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(test_smem_to_frag_sm70 PRIVATE CUDA::cudart)

# V11 Step 2b: single 884 tile (m8n32k8) with manual Lds + SM70_MMA_884::fma.
add_executable(test_mma_884_tile_sm70 tests/test_mma_884_tile_sm70.cu)
target_include_directories(test_mma_884_tile_sm70 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(test_mma_884_tile_sm70 PRIVATE CUDA::cudart)

# SPRINT-019 P1.1: FP16-accumulator m8n8k4 atom — empirical lane-mapping probe
# + adversarial-input verification. Tier-1 isolated correctness test.
add_executable(test_mma_884_acc_f16_sm70 tests/test_mma_884_acc_f16_sm70.cu)
target_include_directories(test_mma_884_acc_f16_sm70 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_link_libraries(test_mma_884_acc_f16_sm70 PRIVATE CUDA::cudart)
