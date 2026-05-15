# SPRINT-023 — Intent

**Sprint number**: 023
**Date**: 2026-05-15
**Predecessor**: SPRINT-022 (operational baseline shipped; 4.28 pp / 4.73 tg t/s)
**Seed prompt**: `/sprint-plan - Execute`

---

## North Star

Lift DSv4-Flash V100 decode throughput from the SPRINT-022 baseline by
moving MoE expert weights from CPU RAM to V100 VRAM and dispatching them
through turbomind's sm70 `Config_MXF4` / `Config_E4M3` kernels — leveraging
the 59 TF FP8 + 65 TF U4 ceilings already proven in REPORT-15.

This is the Tier 2 work explicitly identified in SPRINT-022-OPERATIONAL.md.

---

## Orientation summary

- **SPRINT-022 baseline**: DSv4-Flash-256e (284.33B, MXFP4 experts + F8_E4M3_B128
  dense) running on V100 + 251GB host. `-ngl 99 -ot exps=CPU`. pp128 = 4.28 t/s,
  tg32 = 4.73 t/s. The pp ≈ tg ratio of 0.9 proves we're CPU-expert bound on
  every forward pass.
- **REPORT-15 (SPRINT-021 P0) ceilings**: turbomind sm70 `Config_E4M3` (FP8) =
  59.07 TF at M=2048 N=K=7168, `Config_U4_g` = 64.67 TF, `Config_MXF4` exists
  with group_size=32 constraint. ncu HMMA active% 55-61% for these configs vs
  31% for our INT8 v12_ms3.
- **Integration surface**: `ggml/src/ggml-cuda/mmvq.cu` (lines 947-960) and
  `mmq.cu` (lines 23+, 276+) already have `GGML_TYPE_MXFP4` / `GGML_TYPE_NVFP4` /
  `GGML_TYPE_F8_E4M3_B128` cases. nisparks's `Bring up native FP4 FP8 quant
  support` commit (a8062c0a9) wired these to scalar `vec_dot_*_q8_1` paths.
- **Carve-out infrastructure**: `tools/tc-grid/turbomind_minimal/` builds
  turbomind's `gemm2 + core + parser + cuda_utils + fmt` as a standalone library.
  Proven in SPRINT-020 P1.5. The same pattern can be lifted into the llama.cpp
  build to expose `turbomind::gemm::Gemm::Run` to ggml-cuda dispatch.
- **VRAM budget**: 32494 MiB total - 7500 MiB SPRINT-022 dense baseline - 1024 MiB
  context - 1500 MiB compute buffers = **~22.5 GiB free** for hot expert weights.

---

## Relevant codebase areas

| Path | Why |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu` | mat-vec dispatch (decode hot path) — add turbomind dispatch case for MXFP4 + F8_E4M3 |
| `ggml/src/ggml-cuda/mmq.cu` | mat-mul dispatch (prefill hot path) — same |
| `ggml/src/ggml-cuda/convert.cu` | Tensor format conversion at upload time |
| `ggml/src/ggml-cuda/common.cuh` | Type definitions |
| `tools/tc-grid/turbomind_minimal/CMakeLists.txt` | Existing build glue for gemm2 + core; reuse pattern |
| `research/lmdeploy/src/turbomind/kernels/gemm/` | turbomind gemm2 source (untracked, in `cuda-patches/`) |
| `src/llama-memory-deepseek4.{cpp,h}` | Memory layout for DSv4; may need hot-expert allocation logic |
| `src/models/deepseek4.cpp` | Model graph that calls mul_mat for experts |

---

## Constraints

- **V100 sm_70 only** for now. Turbomind's `Config_MXF4` / `Config_E4M3` we've
  proven work on this target. Cross-arch (sm75/80/90) is out of scope.
- **No correctness regression**: must match nisparks's existing CPU/scalar path
  within the SPRINT-015 P2 tolerance contract (col-parallel 2e-2/1e-2,
  row-parallel 1e-2/5e-3 for MXFP4 on sm70).
- **Bit-identical to GGUF source**: weight repacking is a one-time conversion;
  no in-place GGUF mutation. The mmap-loaded data stays canonical.
- **Backwards-compatible default**: if turbomind path is disabled (env var or
  build flag), fall back to nisparks's existing scalar dispatch. No silent
  behavior change for anyone not opting in.
- **gpu-01 V100 environment**: single GPU, CUDA 12.2, gcc 11.4, cmake 3.22.1.
  Build pod (`llamacpp-build`) is provisioned and ready.
- **Production model is 145 GiB** with 256 experts × ~60 layers. Hot-expert
  selection at runtime needs offline profiling (a JSON file).

---

## Success criteria

| Goal | Target |
|---|---|
| **Build**: `GGML_TURBOMIND_GEMM=ON` flag pulls in `gemm2` carve-out and links | Builds clean on V100 sm70, no spill warnings |
| **MXFP4 GPU path correctness** | Bit-equivalent (within tolerance) to scalar CPU path; sanity test on 8 expert mul_mats at M ∈ {1, 128} N=K=7168 |
| **F8_E4M3_B128 GPU path correctness** | Same contract for dense layers |
| **Decode TPS uplift** | `tg32` ≥ **20 tok/s** (current 4.73; target = 4.2× over baseline). Decode-time MoE moves from CPU to GPU for hot experts |
| **Prefill TPS uplift** | `pp128` ≥ **20 tok/s** (current 4.28; target 4.7× over baseline). Stretch: pp ≥ 50 if we hit the ceiling |
| **VRAM stays under 30 GiB** | Leaves 2.5 GiB headroom for KV cache growth, compute buffers, allocator fragmentation |
| **Fallback works** | `GGML_TURBOMIND_GEMM=OFF` runs identical to SPRINT-022 baseline |

---

## Verification strategy

1. **Per-kernel correctness tests** — extend `gemm_bench_packed.cu` style harness
   to compare turbomind output vs nisparks scalar output for 5 random
   `(M, N, K, weight_bytes)` cases per type. rel ≤ 1e-2 gate.
2. **End-to-end TPS via `llama-bench`** — same flags as SPRINT-022 baseline run.
   Compare `-DGGML_TURBOMIND_GEMM=OFF` to `=ON`.
3. **VRAM budget validation** — `nvidia-smi` snapshot after model load + 32-token
   generation. Confirm under 30 GiB.
4. **Hot-expert selection sanity** — load with a synthetic "always-uniform"
   expert distribution; verify TPS regresses gracefully (still beats baseline)
   when hot-experts don't match actual routing.
5. **ncu metric pack** on 1 mul_mat per type at one shape — confirm HMMA active%
   tracks the REPORT-15 ceilings (≥50% for MXFP4, ≥55% for FP8).

---

## Uncertainty assessment

| Dimension | Level | Rationale |
|---|---|---|
| **Correctness** | Medium | Turbomind kernel correctness proven (gemm_bench_packed runs). New: layout conversion at GGUF load, multi-tensor dispatch wiring. Easy to bit-compare. |
| **Scope** | Medium-High | Three sub-goals (MXFP4 path, FP8 path, hot-expert selection) — each could grow. Tight scope discipline required. |
| **Architecture** | Medium | Two new dispatch paths in ggml-cuda + one-time weight upload converter. Mirrors existing `convert.cu` patterns. Carve-out CMake pattern already validated. |

---

## Open questions

1. **Weight conversion location**: do we repack inside `ggml-cuda/convert.cu`
   at tensor-upload time, or in a llama-load hook before tensors hit ggml? The
   former is more consistent with ggml-cuda's existing format conversions;
   the latter has lower risk of breaking other backends.
2. **Hot-expert profile source**: need an offline JSON file (per
   `llama-deepseek4-hot.h` convention). Where does it come from in SPRINT-023
   scope — synthetic profile, or run a profile pass first? Affects schedule.
3. **MXFP4 group_size constraint**: turbomind's `Config_MXF4` registry only
   has `group_size=32` entries (we hit this in SPRINT-021). DSv4's GGUF MXFP4
   uses `QK_MXFP4=32` (matches). Sanity check needed: do the registered tile
   sizes (BM ∈ {8, 16, 32, 64, 128}) cover the production shapes (N=K=7168)?
4. **NVFP4 deferred or included?** GGML supports `GGML_TYPE_NVFP4` (block of 4
   FP4 values with E4M3 scale). DSv4 model uses MXFP4 not NVFP4. We could
   defer NVFP4 to SPRINT-024 — confirm scope.
5. **F8_E4M3_B128 vs FP8 E4M3 native**: GGML's F8 block has 128 e4m3 values
   plus one E8M0 scale. Turbomind's Config_E4M3 expects per-128-K-group scales
   in a specific packed layout. Conversion mapping needs careful spec.

---

## Deferred / follow-up items from prior sprints (now actionable)

From `SPRINT-022-DEFERRED-PORTS.md`:
- **WMMA-MMVQ MoE port (sprint-017 P2+P3, 61f9ebebe + 5ec3a9b41)** — deferred
  until baseline. SPRINT-023 may complete this AS PART OF the turbomind dispatch
  hook, or hold until SPRINT-024. Decide during planning.

From `SPRINT-020-FOLLOWUPS.md` (cleared most via SPRINT-021):
- ncu metric pack script `/tmp/run_ncu3.sh` — promote to repo
- U4 asymmetric N≠K illegal memory access — fix if we hit it

From REPORT-15 §6 (SPRINT-021 P0 followups):
- MXFP4 sm70 needs `group_size=32` — already noted above as scope constraint
- Asymmetric DSv4 shapes for U4/FP8 head-to-head — relevant if we go beyond
  N=K=7168

---

## Out of scope (will be in deferred doc)

- **Multi-GPU TP**: SPRINT-024+ when single-GPU is at ceiling
- **Cross-architecture (sm75/80/90)**: This sprint = sm70 only
- **NVFP4**: DSv4 model uses MXFP4; NVFP4 support is dead code for our case
- **In-place GGUF re-quantization**: keep canonical mmap, convert at upload
- **Dynamic hot-expert reselection**: load-time JSON only, no online
  promotion/demotion
- **Long-context optimizations**: 2K context same as SPRINT-022

---

## Vision context

No `docs/sprints/VISION.md`. Planning from scratch but SPRINT-022-OPERATIONAL.md
§3 explicitly named this as the next move. The implicit vision:

1. **Operational** (✅ SPRINT-022) — DSv4-Flash runs on V100, any TPS
2. **Production-grade** (SPRINT-023) — TPS ≥ 20 t/s decode via GPU experts
3. **Optimized** (SPRINT-024+) — push toward FP8 ceiling, multi-GPU, prod workloads
