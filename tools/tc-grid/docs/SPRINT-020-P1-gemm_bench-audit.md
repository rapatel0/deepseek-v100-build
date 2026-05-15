# SPRINT-020 P1.1 — Turbomind gemm_bench transitive-dep audit

## Disabled target location

`research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt:108-137`
The `gemm_bench` target plus its NVBench `FetchContent_Declare` block
are commented out under `if(BUILD_TEST)`. The block reads:

```cmake
add_executable(gemm_bench
        test/gemm_bench.cu
        test/quantization.cu
        test/reference.cu)
target_link_libraries(gemm_bench PRIVATE gemm2 core nvbench::nvbench cublas)
```

with an NVBench FetchContent block immediately above (tag
`d8dced8a64d9ce305add92fa6d274fd49b569b7e`).

## Transitive dependency tree

```
gemm_bench
├── gemm2 (local, kernels/gemm/CMakeLists.txt)
│   ├── parser (local, turbomind lib)
│   ├── nvidia::cutlass::cutlass (external — need vendor or FetchContent)
│   ├── CUDA::cuda_driver
│   └── gemm2_sm90 (sm_90 only; gated by GEMM2_ARCH_90_ENABLED — does NOT
│       impact sm_70 build)
├── core (local, turbomind/core/CMakeLists.txt)
│   ├── cuda_utils (local)
│   ├── CUDA::cudart
│   ├── CUDA::cuda_driver
│   ├── fmt::fmt (external — need vendor or FetchContent)
│   └── concurrentqueue (already FetchContent'd inside core's CMakeLists,
│       from https://github.com/cameron314/concurrentqueue.git v1.0.4)
├── nvbench::nvbench (external FetchContent from upstream)
└── cublas (system / CUDA toolkit)
```

Local turbomind sources `test/quantization.cu` and `test/reference.cu` are
small CUDA files; no additional deps beyond cudart.

## External libraries needed

| Library | Source | Status |
|---|---|---|
| nvbench | `FetchContent_Declare(repo-nvbench, GITHUB)` | Disabled-out — needs to be re-enabled with the `TM_ENABLE_GEMM_BENCH` guard. Tag: `d8dced8a64d9ce305add92fa6d274fd49b569b7e`. |
| CUTLASS 2.x | tc-grid already FetchContent's at `_deps/cutlass-src/` v2.11.0 | Reusable. Turbomind imports as `nvidia::cutlass::cutlass`; need to verify the imported-target alias matches what tc-grid registers. |
| fmt | Not yet found in tree | Likely external; either FetchContent or system package. |
| concurrentqueue | Already FetchContent'd inside `core/CMakeLists.txt` | No action needed. |

## Local turbomind libraries needed

Beyond `gemm2` and `core` listed above, building the `gemm2` library will
pull in `parser` (the turbomind config parser, used by some kernel
dispatcher logic). Its CMakeLists location is
`research/lmdeploy/src/turbomind/utils/CMakeLists.txt` (probably) — not
yet inspected; flagged as a P1.3 sub-task.

## Build complexity assessment

This is a **3× gut estimate** target per memory
`feedback_effort_estimation_undocumented_hardware`:

- Multiple FetchContent'd deps with potentially version-pinned semantics
- `fmt::fmt` external dep status unclear
- The parent llama.cpp / lmdeploy build system has its own conventions
  that may conflict with simply guarding the gemm_bench target
- nvbench itself has a CMake structure (gated `NVBench_ENABLE_EXAMPLES`,
  `NVBench_ENABLE_TESTING`, `BUILD_SHARED_LIBS`) that needs careful
  parent-build interaction

Honest estimate for clean reproducible build on gpu-01:
**6–12 hours**, matching SPRINT-020 §11 P1 ETA range.

## Recommended P1 execution order

1. **P1.2 first**: add `option(TM_ENABLE_GEMM_BENCH OFF)` guard. Default OFF preserves
   non-V100 builds.
2. **P1.3a — verify fmt availability**: check if fmt is vendored in
   `research/lmdeploy/` or available system-wide. If not, add
   FetchContent for fmt v11.x (header-only mode preferred).
3. **P1.3b — verify CUTLASS imported-target alias**: confirm tc-grid's
   `_deps/cutlass-src/` exports `nvidia::cutlass::cutlass` interface
   target or add a wrapper.
4. **P1.3c — re-enable NVBench FetchContent**: uncomment the block,
   gate on `TM_ENABLE_GEMM_BENCH`.
5. **P1.3d — verify parser/utils**: inspect
   `research/lmdeploy/src/turbomind/utils/CMakeLists.txt` for `parser`
   target dependencies.
6. **P1.4 — add DSv4 shapes** to `test/models.h`.
7. **P1.5 — build smoke test**: `cmake -B build-tm
   -DTM_ENABLE_GEMM_BENCH=ON` + `cmake --build build-tm --target
   gemm_bench -j 8`. If build fails, the failure mode dictates next steps.
8. **P1.6 — registry verification**: `gemm_bench --list` (or equivalent)
   to confirm `sm70_884_{4,8,16}` registrations are visible on the V100.

## Hard time-box

Per SPRINT-020 §11 + P1 decision gate: if P1 build engineering exceeds
12 hr without a working gemm_bench binary, close the sprint early at the
ceiling-proof branch (P5 → DSv4 deployment integration without turbomind
comparison data).

## Status

Audit complete (this document). P1.2 (guard option) is the next concrete
work item.

---

## P1.3 minimal-carve-out plan (post-investigation)

**Decision** (recorded in SPRINT-020-FOLLOWUPS): option (b) minimal
CMake carve-out, because the pod has no python3 to drive lmdeploy's
setup.py path (option a).

### Minimum dep graph required to link gemm_bench

```
gemm_bench (test/gemm_bench.cu + test/quantization.cu + test/reference.cu)
  ← gemm2  (kernels/gemm/*.cu, ~17 files + SM70 kernel list — already in patch 0006)
      ← parser           (utils/parser.cc + fmt)
      ← CUTLASS           (already in tc-grid's _deps/cutlass-src/ v2.11.0)
      ← CUDA::cuda_driver
  ← core   (core/*.cc + core/*.cu, 14 files)
      ← cuda_utils        (utils/cuda_utils.cc + fmt)
      ← concurrentqueue   (FetchContent cameron314/concurrentqueue v1.0.4
                           — already in core/CMakeLists.txt)
      ← fmt::fmt          (need to FetchContent fmtlib/fmt v10.x or
                           v11.x; not yet in tree)
  ← nvbench::nvbench      (FetchContent already in patch 0006 guarded
                           block)
  ← cublas
```

### Concrete carve-out file plan

Create `tools/tc-grid/turbomind_minimal/CMakeLists.txt` that does:

1. `FetchContent fmtlib/fmt v11.0.2` (header-only mode: define FMT_HEADER_ONLY)
2. `FetchContent cameron314/concurrentqueue` v1.0.4 (same tag the
   upstream uses)
3. `add_subdirectory(${LMDEPLOY_SRC}/turbomind/utils)` — pulls in
   `parser`, `cuda_utils`, etc.
4. `add_subdirectory(${LMDEPLOY_SRC}/turbomind/core)` — pulls in `core`
5. `add_subdirectory(${LMDEPLOY_SRC}/turbomind/kernels/gemm)` with
   `TM_ENABLE_GEMM_BENCH=ON` — builds `gemm2` + `gemm_bench`

Then in `tools/tc-grid/CMakeLists.txt`:
```cmake
if(TCGRID_ENABLE_TURBOMIND_GEMM)
    set(LMDEPLOY_SRC "${CMAKE_SOURCE_DIR}/../../research/lmdeploy/src")
    add_subdirectory(turbomind_minimal)
    target_compile_definitions(tc-grid PRIVATE TCGRID_ENABLE_TURBOMIND_GEMM=1)
    target_link_libraries(tc-grid PRIVATE gemm2 core)
endif()
```

### Risks for P1.5 build smoke-test

1. **Upstream CMakeLists assume parent context**. The lmdeploy
   superbuild defines variables like `CMAKE_CUDA_ARCHITECTURES`,
   `_archs_100` filter, `_has_sm100` that may not be set when
   subdirectory'd. Need to set sensible defaults before add_subdirectory.
2. **SM90/SM100 paths**: `kernel/sm90_*.cu` and `tma.cu` may try to
   compile even on V100 if architectures aren't filtered. Currently
   the upstream gates these with `if (sm90 in CMAKE_CUDA_ARCHITECTURES)`;
   verify this gating works when subdirectory'd.
3. **fmt version**: lmdeploy may have been tested against a specific
   fmt version. v11.0.2 is a safe pick (recent stable, header-only mode
   supported).
4. **Catch2 dep in core's BUILD_TEST block**: core has test_* targets
   gated under BUILD_TEST. Ensure BUILD_TEST is OFF in the carve-out
   so we don't pull Catch2.
5. **Per-source `Xptxas=-v`** in core. Will trigger build warnings on
   every core source. Cosmetic only.

### Smoke-test command for P1.5

```bash
cd /src/tools/tc-grid
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_CUDA_ARCHITECTURES=70 \
              -DTCGRID_ENABLE_TURBOMIND_GEMM=ON \
              -DTM_ENABLE_GEMM_BENCH=ON \
              -DBUILD_TEST=OFF
cmake --build build --target gemm_bench -j 8
./build/turbomind_minimal/.../gemm_bench --help  # smoke
```

Expected first-attempt failures:
- Missing fmt: add FetchContent first
- Missing Catch2: confirm BUILD_TEST=OFF
- Architecture filter issues on sm90 kernels: add `-DGEMM2_ARCH_90_ENABLED=OFF`
- CUTLASS namespace mismatch (`nvidia::cutlass::cutlass`): may need
  a small alias `add_library(nvidia::cutlass::cutlass ALIAS ...)`

### Effort estimate for P1.3 + P1.5 combined

Honest: **6–10 hr** of iterative CMake debugging. Matches the
memory `feedback_effort_estimation_undocumented_hardware` 3× pattern.

---

## P1.5 BUILD COMPLETE — gemm_bench_simple runs ✅ (commit fb1a2b74b)

The minimal carve-out succeeded in ~8 hr of iterative debug. Final
working flags:

1. `include_directories(${LMDEPLOY_ROOT})` — turbomind sources use
   `#include "src/turbomind/..."` (project-root-relative).
2. `include_directories(SYSTEM ${fmt_SOURCE_DIR}/include)` — gemm2
   sources include `<fmt/format.h>` without explicit fmt linkage.
3. `add_compile_options(-include cuda_bf16.h)` — turbomind's
   `kernels/core/mma.h` uses `nv_bfloat16` in overload signatures
   without including cuda_bf16 itself.
4. `add_compile_options(--expt-relaxed-constexpr --expt-extended-lambda)`
   — turbomind sources call constexpr __host__ from device.
5. `nvidia::cutlass::cutlass` INTERFACE alias to tc-grid's
   `_deps/cutlass-src/`.
6. Full turbomind subtree (`utils/`, `core/`, `kernels/{core,gemm,
   attention,...}/`, top-level `macro.h` etc.) synced to pod —
   cross-tree includes are pervasive.

Resulting binary: `tools/tc-grid/build-tm/gemm_bench_simple` (23 KB).
Currently outputs SKIP stubs in tc-grid CSV format; smoke run on V100:
```
device: Tesla V100-SXM2-32GB, cc=7.0, smem=48KiB/cta
format,path,dist,M,N,K,tile,status,detail
INT8,LUT,U(-1,1),64,7168,7168,turbomind_TBD,SKIP,P1.5 stub
INT8,LUT,U(-1,1),2048,7168,7168,turbomind_TBD,SKIP,P1.5 stub
```

---

## P1.6 + P2 continuation: Gemm::Run API

The upstream `test/testbed_v3.h` (used by `test/gemm_bench.cu`) is
NOT importable into `gemm_bench_simple.cu` — it transitively pulls
in `models/llama/LlamaLinear.h`, `models/linear_weight.h`,
`kernels/gpt_kernels.h`, etc. — the entire LLaMA model code tree.
That's way outside the minimal carve-out scope.

Instead, gemm_bench_simple.cu must call `turbomind::gemm::Gemm::Run`
**directly**, bypassing testbed_v3:

```cpp
// src/turbomind/kernels/gemm/gemm.h:
[[nodiscard]] int Run(const Operation&    operation,
                      float               alpha,
                      const void*         A,             // half[M, K] activations
                      const MatrixLayout& Adesc,
                      const void*         U,             // (unused for INT8 w/ FP16 A)
                      const MatrixLayout& Udesc,
                      const void*         B,             // int8_t[N, K] weights
                      const MatrixLayout& Bdesc,
                      const void*         V,             // half[N, K/group_size] scales
                      const MatrixLayout& Vdesc,
                      float               beta,
                      const void*         C,             // unused (beta=0)
                      const MatrixLayout& Cdesc,
                      void*               D,             // float[M, N] output
                      const MatrixLayout& Ddesc,
                      const Workspace&    workspace,     // ~1 MB scratch
                      cudaStream_t        stream);
```

Concrete next-session task: write ~100-200 lines in
`gemm_bench_simple.cu` that:
1. cudaMalloc the buffers (A, B, V, D).
2. Generate uniform_small data + INT8 quantize B with group_size=128.
3. Build the Operation + MatrixLayout descriptors. Operation fields
   to populate: `dispatch`, `epilogue`, `quant_a`, `quant_b`, plus
   the dtype tags. The `MatrixLayout` carries `{type, order, rows,
   cols, ld}`. See `kernels/gemm/types.h` for field definitions.
4. Median-of-5 timed `Gemm::Run` invocations via cudaEventRecord.
5. Emit tc-grid CSV row with TF computation.

Effort: 4-6 hr including debug. Matches sprint plan §11's P1.6
estimate.
