# SPRINT-023 — Turbomind MoE integration infrastructure (V100 sm70)

**Status:** PLANNED 2026-05-15
**Predecessor:** SPRINT-022 (operational; 4.28 pp / 4.73 tg tok/s on V100 with `exps=CPU`)
**Successor:** SPRINT-024 (perf landing — grouped MoE if needed, hot-expert selection, perf gates)

---

## 1. Overview

SPRINT-023 builds the integration mechanism between llama.cpp ggml-cuda and
turbomind's sm70 quantized GEMM kernels (`Config_E4M3`, `Config_MXF4`) without
committing to a performance landing. Outcome of this sprint is a working
dispatch path with measured TPS and a survey of turbomind's MoE primitives —
enough data to decide SPRINT-024's perf strategy with evidence, not speculation.

**Explicit non-goal: no decode TPS target.** SPRINT-022's 4.73 tg t/s is the
baseline. Whatever this sprint produces is the new measurement; the perf gate
arrives in SPRINT-024.

---

## 1.1 Why this sprint exists (decision context)

Three architectural questions stayed unanswered at end of SPRINT-022:

1. **Is turbomind's `Config_MXF4` viable at M=1?** REPORT-15 measured 50+ TF at
   M=2048 (prefill regime). M=1 (decode) is HBM-bandwidth-bound, not compute-
   bound — those ceilings don't transfer. Real M=1 number is unknown.
2. **Does turbomind have a grouped/batched MoE primitive?** ~464 GEMMs per
   decoded token at top-8 routing × 58 layers. If turbomind already supports
   batched-per-expert dispatch, launch overhead drops 8× and 20 t/s becomes
   physically reachable. If not, SPRINT-024 designs one.
3. **Is the GGML-block ↔ turbomind-packed conversion stable for MXFP4 and
   F8_E4M3_B128?** Layout, scale type, K-alignment, and `Convert` rc semantics
   need to be exercised against the production tensors, not just synthetic
   shapes from the SPRINT-021 bench harness.

SPRINT-023 answers (1) and (2) via microbench in P0, then builds the integration
that makes (3) verifiable end-to-end via P1-P4.

---

## 2. Use cases

Each phase delivers something useful even if subsequent phases slip:

| Phase | Useful output even if sprint stops here |
|---|---|
| P0 | Decision data: do we know turbomind at M=1 is competitive? Is grouped-MoE available? Informs SPRINT-024 architecture. |
| P1 | Standalone `libggml-turbomind.so` artifact. Reusable for tc-grid lab work going forward. |
| P2 | GGML-block ↔ turbomind packer utility — testable, reusable in any future tensor-format work. |
| P3 | New `CUDA_TURBOMIND` buffer type — sets up Optional kernel substitution for future precision regimes. |
| P4 | End-to-end working dispatch (no perf claim). Validates the abstraction. |
| P5 | Measurement + REPORT-17. Closes the loop. |

---

## 3. Architecture

### 3.1 Plug-in surface

We do NOT modify ggml's public ABI. We do NOT introduce a new `GGML_OP_*`.
The integration uses two existing extension points:

1. **A new buffer type** `CUDA_TURBOMIND` registered as a ggml-cuda backend
   buffer type alongside `CUDA0`. Users opt in tensor-by-tensor via the
   existing `-ot 'pattern=CUDA_TURBOMIND'` regex (already supported in
   llama-cli/server). Weights routed to this buffer type are packed via
   turbomind's `Convert()` at upload time.

2. **A dispatch predicate inside the existing case branches** in `mmq.cu`
   and `mmvq.cu` for `GGML_TYPE_MXFP4` and `GGML_TYPE_F8_E4M3_B128`. If the
   tensor is on a `CUDA_TURBOMIND` buffer, route to a new launch wrapper.
   Else fall through to the existing scalar/vec_dot path.

### 3.2 Out-of-process build isolation

Turbomind links gemm2 + core + parser + cuda_utils + fmt + CUTLASS into a
separate `libggml-turbomind.so` with a small **C ABI** (4 entry points):

```c
// 1. Library lifecycle
int ggml_turbomind_init(int cuda_device);
void ggml_turbomind_shutdown(void);

// 2. Layout conversion (one-time at upload)
int ggml_turbomind_pack_weight(
    const void* src,         // ggml-format weight (MXFP4 / F8_E4M3_B128)
    int ggml_type,           // GGML_TYPE_MXFP4 or GGML_TYPE_F8_E4M3_B128
    int N, int K,            // logical dims
    int group_size,
    void* dst,               // pre-allocated packed output buffer
    void* packed_scales,
    int* k_pack_value,       // OUT: pack value for descriptor reconstruction
    cudaStream_t stream);

// 3. Bytes needed for packed weight (sizing pre-allocation)
size_t ggml_turbomind_packed_bytes(int ggml_type, int N, int K, int group_size);

// 4. Mul-mat launch
int ggml_turbomind_mul_mat(
    const void* A_fp16,      // activations (FP16 row-major)
    const void* B_packed,    // packed weights from #2
    const void* V_packed,    // packed scales from #2
    int ggml_type,           // determines which Config to use
    int M, int N, int K,
    int group_size,
    int k_pack_value,
    void* D_fp16,            // output (FP16 row-major)
    cudaStream_t stream);
```

`libggml-cuda.so` dlopen's `libggml-turbomind.so` on first request to a
`CUDA_TURBOMIND` buffer. If dlopen fails (library missing in deployment),
falls back to existing dispatch with a one-time warning.

This pattern matches the existing ggml-cuda + ggml-cpu split (separate `.so`'s
already exist). No new build infrastructure pattern.

### 3.3 Tensor lifecycle

```
GGUF mmap (host RAM)
    │
    │  llama-load: tensor matched by `-ot 'exps=CUDA_TURBOMIND'`
    ▼
ggml_backend_cuda_turbomind_buffer.alloc_tensor()  ── reserves VRAM
    │
    │  ggml_backend_cuda_turbomind_buffer.set_tensor()
    ▼
ggml_turbomind_pack_weight()  ── converts in-place into packed layout
    │                            stores: packed weight + packed scales
    │                                   + k_pack value (stashed in tensor extra)
    ▼
Tensor ready; dispatch into mmq.cu / mmvq.cu uses
ggml_turbomind_mul_mat() instead of the scalar path
```

### 3.4 Decision tree for kernel selection

```
mul_mat called with src0 (weight) and src1 (activation):
│
├── src0->type == MXFP4 OR F8_E4M3_B128?
│   ├── No  → existing dispatch (unchanged)
│   └── Yes
│       ├── src0->buffer->buft == CUDA_TURBOMIND?
│       │   ├── No  → existing scalar dispatch (unchanged)
│       │   └── Yes
│       │       └── ggml_turbomind_mul_mat(...)
│       │           (calls into libggml-turbomind.so)
```

No tensor that's not explicitly opted-in via `-ot` is affected.

---

## 4. Implementation

### P0 — Microbench + survey (2 days, ship-blocker for P1+)

**Goal**: get the data we need to commit to P1+ architecture.

1. **P0.1 — Turbomind M=1 decode regime perf**
   - Modify `tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` to sweep
     M ∈ {1, 4, 8, 16, 32, 64} at N=K=7168 for `fp8` and `u4` (proxy for MXFP4).
   - Run on V100, capture TFLOPS + ms/iter.
   - **Gate**: if turbomind M=1 < 3× our CPU expert baseline TPS-equivalent
     (~5 t/s × CPU-expert-bytes-per-token), we abort the sprint and revisit.
2. **P0.2 — `Config_MXF4` group_size=32 validation**
   - Confirm registry entries for production shapes (BN ∈ {7168, 18944}).
     SPRINT-021 REPORT-15 §6 flagged this as untested.
   - Test packing of a real MXFP4 expert tensor through `Convert()`.
3. **P0.3 — Survey lmdeploy/turbomind for grouped MoE primitives**
   - Read `research/lmdeploy/src/turbomind/kernels/gemm/` for batched/grouped
     GEMM. Check `Gemm::Run` parameters for batch/group dims.
   - Check `research/lmdeploy/src/turbomind/models/moe_ffn_layer.cc` for
     dispatch patterns.
   - Document findings as `docs/sprints/SPRINT-023-P0-moe-survey.md`.
4. **P0.4 — Launch-overhead microbench (informational only)**
   - Empty-kernel launch latency on V100 (helps SPRINT-024 design budgeting).
   - Document in survey doc.

**P0 Gate**:
- ✅ turbomind M=1 numbers exist for MXFP4 + FP8
- ✅ `Config_MXF4` packs and runs real DSv4 tensor without crash
- ✅ Survey concludes "grouped primitive available" or "not available"
- ✅ Decision: proceed with single-launch plug-in (this sprint) OR pivot to
  use turbomind's grouped primitive if found

If P0.1 gate fails: stop sprint, document why, return to SPRINT-022 baseline +
plan SPRINT-024 as new design sprint.

### P1 — `libggml-turbomind.so` carve-out + C ABI (3-4 days)

1. **P1.1 — Lift `tools/tc-grid/turbomind_minimal/` pattern to ggml/vendor/turbomind/**
   - CMake target `ggml-turbomind` produces `libggml-turbomind.so`
   - Links: gemm2, core, parser, cuda_utils, fmt, CUTLASS (carved from
     research/lmdeploy/)
   - Uses the same patches as SPRINT-020 (cuda-patches/0006-*.patch)
2. **P1.2 — Define 4-function C ABI** in `ggml-turbomind/ggml-turbomind-api.h`
   (see §3.2 above)
3. **P1.3 — Implement C ABI in `ggml-turbomind/api.cc`**, wrapping `Gemm::Run`
4. **P1.4 — Standalone test**: link a small `test_turbomind_api.cpp` against
   `libggml-turbomind.so`, exercise the 4 functions, gate on success.

**P1 Gate**: `libggml-turbomind.so` builds; standalone test passes; size
under 200 MiB.

### P2 — GGML block ↔ turbomind packed weight conversion utility (2-3 days)

1. **P2.1 — Spec the conversion** for `block_mxfp4` (16 fp4 values + e8m0 scale)
   and `block_f8_e4m3_b128` (1 e8m0 + 128 fp8) → turbomind packed layout
2. **P2.2 — Implement** `ggml_turbomind_pack_weight()` as C ABI function
   that wraps the existing turbomind `Convert()` API used in
   `gemm_bench_packed.cu`
3. **P2.3 — Unit test**: a Python-driven conversion that round-trips a known
   tensor through `ggml_turbomind_pack_weight` then `Gemm::Run` and compares
   against `vec_dot_*` scalar output, gating at SPRINT-015 P2 tolerance
   (col-parallel 2e-2/1e-2 for MXFP4)

**P2 Gate**: round-trip test passes for 5 random (N, K) pairs at production
shapes for both MXFP4 and F8_E4M3_B128.

### P3 — `CUDA_TURBOMIND` buffer type + upload hook (2-3 days)

1. **P3.1 — Define the buffer type** in
   `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu`:
   - `ggml_backend_cuda_turbomind_buffer_type()` returns a static struct
   - `init_tensor` / `set_tensor` / `free_buffer` callbacks
2. **P3.2 — Register the buffer type** with the ggml-cuda backend so
   `-ot 'pattern=CUDA_TURBOMIND'` matches it
3. **P3.3 — Implement `set_tensor`**: for MXFP4 / F8_E4M3_B128 tensors,
   call `ggml_turbomind_pack_weight()` then store result + metadata (k_pack
   value) in the tensor's `extra` field
4. **P3.4 — Lifecycle test**: load a tiny test GGUF with 2 MoE tensors
   pinned via `-ot 'exps=CUDA_TURBOMIND'`, verify VRAM is allocated and
   the conversion ran

**P3 Gate**: a smoke-test GGUF loads with the new buffer type and no crash.

### P4 — Dispatcher integration (2-3 days)

1. **P4.1 — Add dispatch predicate** at top of `ggml_cuda_mul_mat_q`
   (`mmq.cu`) and `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`):
   - Check `src0->type == MXFP4 || F8_E4M3_B128`
   - Check `src0->buffer->buft == CUDA_TURBOMIND`
   - If both: route to `ggml_turbomind_mul_mat`, else fall through
2. **P4.2 — `mul_mat_id` path** (the MoE-specific dispatcher): same predicate.
   Note: this preserves single-launch-per-expert semantics. Grouped-MoE
   integration deferred to SPRINT-024.
3. **P4.3 — Smoke test**: load DSv4-Flash-256e with a small subset of expert
   tensors on `CUDA_TURBOMIND`, run a 32-token generation, compare output
   against `exps=CPU` baseline within `SPRINT-015 P2` tolerance

**P4 Gate**: end-to-end generation runs without crash, output within
tolerance vs the SPRINT-022 baseline.

### P5 — Measurement + REPORT-17 (2 days)

1. **P5.1 — End-to-end TPS** via `llama-bench` at `-p 128 -n 32` for:
   - Baseline (`exps=CPU`) — confirm SPRINT-022 numbers reproduce
   - All experts on `CUDA_TURBOMIND` (if VRAM permits — likely won't for
     full 256e but document what fits)
   - First-N-layers on `CUDA_TURBOMIND` for N ∈ {4, 8, 16, 24}
2. **P5.2 — ncu metric pack** on the new path: HMMA active%, DRAM%,
   launch overhead per expert call
3. **P5.3 — REPORT-17**: write the close-out narrative. Include findings
   from P0 survey + microbench. Recommendations for SPRINT-024.
4. **P5.4 — Memory updates**: capture key learnings as memory entries

**P5 Gate**: REPORT-17 written; memory updated; SPRINT-024 planning has
quantitative basis.

### P6 — Close-out

1. Commit, tag `sprint-023-close`
2. Write SPRINT-023-FOLLOWUPS.md if anything emerged during execution

---

## 5. Files Summary

### New files

| Path | Purpose |
|---|---|
| `ggml/vendor/turbomind/CMakeLists.txt` | Builds `libggml-turbomind.so` (carve-out of `tools/tc-grid/turbomind_minimal/`) |
| `ggml/vendor/turbomind/ggml-turbomind-api.h` | C ABI definition (4 functions) |
| `ggml/vendor/turbomind/api.cc` | C ABI implementation, wraps `Gemm::Run` |
| `ggml/vendor/turbomind/test_turbomind_api.cpp` | Standalone unit test (P1.4) |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | `CUDA_TURBOMIND` buffer type registration + upload hook + dispatch wrapper |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | Header for the above |
| `docs/sprints/SPRINT-023-P0-moe-survey.md` | P0 survey output |
| `tools/tc-grid/docs/REPORT-17.md` | Sprint close report |

### Modified files

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Register the new buffer type; dlopen libggml-turbomind on first use |
| `ggml/src/ggml-cuda/mmq.cu` | Add dispatch predicate (5-line addition at top of `ggml_cuda_mul_mat_q`) |
| `ggml/src/ggml-cuda/mmvq.cu` | Same |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Add `ggml-cuda-turbomind.cu` + dlopen linkage |
| `ggml/CMakeLists.txt` | Add `add_subdirectory(vendor/turbomind)` (opt-in via cmake option) |
| `tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` | Add `--m-list 1,4,8,16,32` for P0.1 microbench |

---

## 6. Definition of Done

### Ship-blockers (must pass to close the sprint)

1. ✅ P0 survey doc exists (`SPRINT-023-P0-moe-survey.md`) with a clear answer
   on grouped-MoE primitive availability
2. ✅ P0 M=1 microbench numbers captured for MXFP4 + FP8 at N=K=7168
3. ✅ `libggml-turbomind.so` builds; standalone test passes; size under 200 MiB
4. ✅ MXFP4 conversion round-trip test passes within SPRINT-015 P2 tolerance
5. ✅ F8_E4M3_B128 conversion round-trip test passes within tolerance
6. ✅ `CUDA_TURBOMIND` buffer type registered and visible to `-ot` regex
7. ✅ End-to-end DSv4-Flash-256e generation runs without crash with a subset
   of expert tensors on `CUDA_TURBOMIND`
8. ✅ Output of P4 path matches SPRINT-022 baseline within tolerance for a
   common short prompt
9. ✅ REPORT-17 written with TPS numbers + ncu data
10. ✅ Memory entries updated

### Explicit non-gates

- ❌ **No decode TPS target.** Whatever the new path produces is the result.
  If it's slower than SPRINT-022 baseline, that's data for SPRINT-024.
- ❌ **No "everything on GPU" requirement.** VRAM budget may force partial
  deployment. Document what fits and what doesn't.
- ❌ **No grouped-MoE deliverable.** Survey only — design + landing is
  SPRINT-024 if survey finds no existing primitive.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | turbomind M=1 perf is worse than CPU expert at decode regime | Medium | HIGH (sprint pivot needed) | P0.1 gate catches this; sprint aborts cleanly |
| 2 | `Config_MXF4` group_size=32 doesn't cover production shapes | Low | High | P0.2 verifies before P1; if missing, abort or pivot to FP8-only |
| 3 | `Convert()` API has undocumented constraint that breaks for real tensors | Medium | Medium | P2 unit tests stress this with real DSv4 tensors, not synthetic |
| 4 | Build size of `libggml-turbomind.so` > 200 MiB | Low | Low | Carve out only what's used; CMake LTO; size budget enforced in P1.4 gate |
| 5 | dlopen failures in deployment containers (missing library) | Medium | Low | Graceful fallback to scalar dispatch with one-time warning |
| 6 | VRAM duplication (mmap'd source + packed copy) > free budget | Medium | Medium | P3 documents the doubling cost; deploy with smaller `-ot` patterns initially |
| 7 | Per-expert launch overhead dominates at decode (Gemini's concern) | High | High (perf reveal, not failure) | Measured in P5 metric pack — informs SPRINT-024 grouped-MoE design |
| 8 | Sprint scope creeps to include perf optimization | Medium | High (timeline overrun) | Explicit "no perf gate" policy; reviewers reject if scope drifts |
| 9 | `-ot` regex matches more tensors than intended | Low | Medium | Documentation + test pattern matching against full tensor name list at P3.4 |
| 10 | The 4-function C ABI is insufficient for some edge case | Medium | Medium | API additions are forward-compatible (deprecation-free); add as needed |

---

## 8. Security

- **No new attack surface**: turbomind code is invoked through a narrow C ABI;
  no user data crosses the boundary directly (only tensor data already in-process)
- **dlopen path is fixed**: only `libggml-turbomind.so` from the same directory
  as `libggml-cuda.so` is loaded — no environment-variable controlled paths
- **Memory safety**: the C ABI takes pointer-size pairs and returns int rc;
  every entry point has a length check and returns error on overflow
- **No deserialization**: turbomind operates on already-validated ggml tensors;
  GGUF parsing happens upstream

---

## 9. Dependencies

### Upstream

- `nisparks/experiment/deepseek-v4-dynamic-graph` (branched from in SPRINT-022)
- `research/lmdeploy/` (gitignored; contains turbomind source as a patched local
  checkout per `cuda-patches/0006-*.patch`)

### Hardware

- gpu-01 V100-SXM2-32GB
- 251 GiB host RAM (for model mmap + page cache)
- CUDA 12.2 toolchain (in build pod `llamacpp-build` already provisioned)

### Sprint preconditions

- `libggml-cuda.so` builds and works (SPRINT-022 P1 confirmed)
- DSv4-Flash-256e GGUF available at `/models/` (SPRINT-022 P0 confirmed)
- Turbomind `gemm_bench_packed.cu` builds and runs on V100 (SPRINT-021 P1.5
  confirmed)

---

## 10. Open Questions

These are deliberately deferred — not blockers for SPRINT-023, but answers
shape SPRINT-024:

1. **Hot-expert selection mechanism**: synthetic ("first N layers"), JSON
   profile, or runtime-promoted? Deferred — SPRINT-023 uses a hand-authored
   `-ot` regex; mechanism design happens in SPRINT-024 if perf says hot-only
   matters.
2. **Grouped MoE in turbomind**: P0.3 survey resolves this. If "yes," P4.2
   should use the grouped API directly. If "no," SPRINT-024 designs one.
3. **MXFP4 in production**: do we trust this format end-to-end? The HF model
   `nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF` is the test vehicle. SPRINT-023
   doesn't change its quantization recipe.
4. **What if M=1 turbomind loses to CPU?** Then SPRINT-024 considers
   PCIe-streaming hot experts (Gemini's contrarian Path B) or pivots to
   custom v13_rf_v6 grouped MoE (Gemini's Path A). P0.1 informs this.
5. **Future amortization of M=1**: the M=1 decode regime is a worst-case for
   compute-bound kernels (turbomind ceilings only apply at M ≥ ~32). Two
   user-suggested levers eventually lift M:
   - **Multi-slot continuous batching** (N parallel sequences = effective M=N
     per forward pass). Already supported by llama-server `--parallel`.
   - **Speculative decoding** (draft-model proposals, target-model verification
     in batches of K candidates = effective M=K per forward pass).
   Both move the operating point closer to where turbomind's prefill ceilings
   apply. SPRINT-024 should consider whether to optimize for M=1 or for the
   batched regime as the production target.

---

## 11. Sequencing — what if P0 fails?

| P0 outcome | SPRINT-023 plan |
|---|---|
| Turbomind M=1 ≥ 3× CPU equivalent, grouped primitive found | Use turbomind grouped path in P4.2 — biggest possible win |
| Turbomind M=1 ≥ 3× CPU equivalent, no grouped primitive | Single-launch plug-in (current plan) — ship infra, perf in SPRINT-024 |
| Turbomind M=1 < 3× CPU equivalent | ABORT sprint. Document why. SPRINT-024 = pivot to v13_rf_v6 grouped MoE or PCIe streaming |
| `Config_MXF4` group_size=32 doesn't cover production shapes | Scope down to FP8-only path; defer MXFP4 to SPRINT-024 |

---

## 12. Estimated effort

| Phase | Days |
|---|---:|
| P0 — Microbench + survey | 2 |
| P1 — libggml-turbomind.so + C ABI | 3-4 |
| P2 — GGML ↔ turbomind weight pack | 2-3 |
| P3 — CUDA_TURBOMIND buffer type | 2-3 |
| P4 — Dispatcher integration | 2-3 |
| P5 — Measurement + REPORT-17 | 2 |
| **Total** | **13-17 days** |

Per the `feedback_effort_estimation_undocumented_hardware.md` memory: multiply
by 3 for opaque-hardware-work signals. SPRINT-023 has some such signals
(turbomind internal APIs, sm70 packing layout), so realistic worst case is
30+ days. The phase boundaries are designed so the sprint can close
incrementally if scope pressure mounts — each P gate produces independent
value.
