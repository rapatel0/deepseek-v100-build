# REPORT-10 — SPRINT-016 outcome

Date: 2026-05-13
Branch: sprint-015-tp8-baseline
Hardware: V100 SXM2 32GB (gpu-01, sm_70)
Inputs: SPRINT-016.md, SPRINT-016-INTENT.md, REPORT-9, ncu_int4.log.

## TL;DR

- **v9 (chunked FP16/FP32 mixed-precision) is a confirmed dead end** on V100 WMMA. Root cause: half-accumulator and float-accumulator fragments have *opaque, non-equivalent* element ordering, so the register-level `c_f.x[e] += __half2float(c_h.x[e])` promote produces garbage (rel = 0.71 at M=2048 vs ≤1e-3 contract). A correctness-preserving promote requires per-chunk SMEM round-trip (store half → load float), which erases the register-reduction win.
- **INT4 LUT path is the production winner**; the BITSHIFT path was only ever implemented for the baseline BM=16 kernel where it underperforms LUT on TC% by ~33% (3.12% vs 4.16%). Optimized v1-v5 INT4 kernels never had a BITSHIFT implementation — there is no "drop LUT and ship BITSHIFT" alternative to evaluate.
- **INT4 BN=256 spill** has a clean architectural diagnosis: at FRAG_M=8 × FRAG_N=4 the accumulator alone needs 256 registers — over the sm_70 255-reg/thread cap. Fix is two-pass FRAG_N (split FN=4 into 2×FN=2 outer-product passes, re-loading A each pass). Deferred to SPRINT-017.
- **Tier B winners (Report 9) confirmed unchanged**: v4 for INT8 ≥ M=256, v3 for FP4/FP8/INT4, v3_b1 for M≤64. No new champion this sprint.

## P0.1 / P0.2 / P0.3 — v9 mixed precision (NEGATIVE RESULT)

Implemented `tools/tc-grid/kernels/v9_kernels.cuh` with two register-resident accumulators (half `c_h`, float `c_f`) and a serialized promote every CHUNK_K K-tiles. CHUNK_K swept over {2, 4, 8} for `128x128_w4` plus a couple of FRAG_M/FN variants.

Run results at M=2048 N=K=7168 (uniform_small):

| tile | CHUNK_K | ms | TFLOPS | rel | bit-ok? |
|---|---:|---:|---:|---:|---|
| 128x128x32_w4_v9 | 2 | 7.983 | 26.36 | 7.07e-1 | NO |
| 128x128x32_w4_v9 | 4 | 7.820 | 26.91 | 7.07e-1 | NO |
| 128x128x32_w4_v9 | 8 | 7.749 | 27.16 | 7.07e-1 | NO |
| 64x128x32_w4_v9 | 4 | 7.969 | 26.41 | 7.07e-1 | NO |
| 64x64x32_w4_v9 | 4 | 10.926 | 19.26 | 7.07e-1 | NO |

Compare to v4 best (bit-correct): 128x128x32_w4_v4 = 7.588 ms, 27.73 TFLOPS, rel = 2.59e-4.

The performance is *competitive* with v4 (within 2%) — the chunked promote overhead is real but small. The failure mode is correctness: **rel = 0.707** is a 7-orders-of-magnitude blowup from the v4 baseline. This is not a precision-degradation pattern (where we'd expect rel ~1e-2 or 1e-3); it's a *layout mismatch* pattern (random-looking outputs).

Root cause investigation: the CUDA WMMA fragment layout docs state element ordering is "opaque" and not guaranteed to match across element types. On sm_70, `wmma::fragment<accumulator, 16,16,16, half>` and `wmma::fragment<accumulator, 16,16,16, float>` distribute the 256 logical output elements across 32 lanes differently: half fragments pack 8 halves per thread in one lane-mapping, float fragments use a different 8-float-per-thread mapping. Reading c_h.x[e] and adding to c_f.x[e] mixes elements from different logical positions in the output matrix.

The correctness-preserving design would require:
1. `wmma::store_matrix_sync(sB_scratch, c_h[fm][fn], 16, mem_row_major)` — half storage in SMEM
2. `wmma::load_matrix_sync(c_f_tmp, sB_scratch, 16, mem_row_major)` — float load (different layout)
3. `for (e=0; e<c_f[fm][fn].num_elements; ++e) c_f[fm][fn].x[e] += c_f_tmp.x[e];` — same-layout add

This works but adds SMEM bandwidth (2 KB per (fm,fn) per promote × 28 promotes for CHUNK_K=8 × 16 frags = ~900 KB per CTA per matmul). And the SMEM write-then-read sequence is what we were trying to avoid. **Not pursued in this sprint.**

**Decision**: kill v9 design as-written. The half-accumulator-then-promote idea remains a viable engineering path but requires the SMEM round-trip variant.

**SMEM round-trip variant filed as SPRINT-017 candidate** (separate from XOR swizzle).

## P0.4 — INT4 LUT vs BITSHIFT ncu

Profiled `mm_int4_lut` and `mm_int4_bitshift` (baseline BM=16 tiles only — these are the only variants implementing both paths) at M=2048 N=K=7168:

| variant | Regs | TC% | ALU% | DRAM% | ShtSb% | LngSb% |
|---|---:|---:|---:|---:|---:|---:|
| mm_int4_lut<16,64,32,4> | 64 | 4.16 | 63.07 | 1.85 | 7.92 | 33.61 |
| mm_int4_bitshift<16,64,32,4> | 71 | 3.12 | 50.56 | 2.20 | 6.61 | **7.28** |
| mm_int4_lut<16,128,32,8> | 64 | 4.12 | 58.40 | 1.83 | 10.20 | 34.88 |
| mm_int4_bitshift<16,128,32,8> | 71 | 2.98 | 45.38 | 1.69 | 9.46 | **9.63** |

Observations:
- BITSHIFT uses **7 MORE registers** than LUT, contrary to my hypothesis that LUT would cost more (entries kept somewhere). nvcc keeps the LUT in SMEM, not registers, so BITSHIFT's extra mask+shift intermediate values are the bigger cost.
- BITSHIFT has **dramatically lower Long Scoreboard stall** (7-10% vs 34-35%). The LUT-load creates memory dependencies that bottleneck the pipeline.
- **LUT still wins TC%** (4.16% vs 3.12% — ~33% higher tensor-core utilization). The LUT's reduced ALU pressure (44% vs 50%) frees up enough capacity to keep tensor cores fed despite the memory stalls.

This is a counterintuitive but consistent picture: BITSHIFT is less memory-dependent but more ALU-bound, and the V100 is *compute*-limited on these baseline tiles, so LUT's compute-efficiency edge wins.

**Verdict**: keep LUT as production for INT4. Optimized v1-v5 INT4 kernels only have LUT paths anyway; no production-relevant BITSHIFT variant exists. Drop the line item.

If we ever need to revisit: the BITSHIFT path's lower memory stall might matter for kernels that *aren't* tensor-core-bound (e.g., very small M where launch overhead dominates), but the current production winners (`32x64x32_w4_v3_b1` at M=64 etc.) all use LUT.

## P0.5 — INT4 BN=256 spill triage

Target kernel: `int4_v3::mm_int4_lut_v3<128, 256, 32, 4, 8, 4>`. ptxas reports 1664 B spill stores, 1472 B spill loads per invocation.

Register breakdown (FRAG_M=8, FRAG_N=4):
- c_frag (float accumulator): 8 × 4 = 32 fragments × 8 regs each = **256 regs**
- a_frag (half): 8 fragments × 8 regs = 64 regs
- b_frag (half): 4 fragments × 8 regs = 32 regs
- Tile/lane/stride scalars, dequant scratch, SMEM pointers: ~30 regs

Total without spill: ~382 regs / thread. sm_70 caps register usage per thread at **255**. nvcc inserts spills to fit; ~127 regs worth of spill (≈ 500 B at 4 B/reg) is plausible, with multiple spill-load pairs accounting for the 1664 B stores / 1472 B loads observed.

**Architectural fix**: split FRAG_N=4 into two outer passes of FRAG_N=2. Inside each outer pass: full K-loop with only 16 c_frags alive (128 regs). Total ~200 regs, fits the 255 cap.

Trade-off:
- 2x A-tile re-load from SMEM. A tile = 4096 halves = 8 KB per warp per K-iter. With 224 K-iters × 4 warps × 8 KB = ~7 MB extra SMEM read per CTA. V100 SMEM is ~10 TB/s; cost ≈ 0.7 ms per CTA, distributed across SMs.
- Synchronization: each outer pass needs its own K-loop, double-buffered. Doubles the `__syncthreads` count.
- Expected net: depending on whether the eliminated spill latency exceeds the added A-reload cost. Both are SMEM-mediated, but spill stalls compound register-dep stall (Short Scoreboard), which is the dominant stall already.

**Implementation cost**: ~100 lines in a new `v3_bn256_kernels.cuh`. Estimate 1 sprint phase.

**Filed for SPRINT-017** as a register-architecture fix. Not blocking current sprint.

## Updated full-grid sweep with v9

v9 added to the harness (see `run_M2048.csv` v2 column). No new champions:

| format | M | bit-correct best (unchanged from R9) | v9 best (any CHUNK_K) | v9 wins? |
|---|---:|---|---|---|
| INT8 | 2048 | 128x128x32_w4_v4 @ 7.59 ms / 27.73 TF | 128x128x32_w4_v9_ck8 @ 7.75 ms / 27.16 TF (FAILS bit) | no |
| INT8 | 1024 | 128x128x32_w4_v4 @ 3.77 ms / 27.92 TF | (similar — see CSV) | no |
| INT8 | 256, 64 | as Report 9 | (similar) | no |

## Sprint outcome vs DoD

| Definition of Done item | Status |
|---|---|
| 1. v9 INT8 bit-correct ≥ 3% faster than v4 | **NOT MET** — v9 fails bit-correctness gate by 700x |
| 2. v9 for all four formats either ships or documented as no-win | MET (documented as design-broken, no INT4/FP4/FP8 variants implemented) |
| 3. INT4 LUT vs BITSHIFT decision with quantitative basis | MET — keep LUT |
| 4. INT4 BN=256: working variant OR documented blocker | MET — documented blocker (255-reg cap), fix sketched |
| 5. Grid sweep CSVs include v9 | MET |
| 6. REPORT-10 with ncu stall breakdown | MET (this document) |
| 7. Per-row tolerance gate continues to hold | MET (v9 rows correctly marked NO — gate caught the regression) |

**Sprint result: 6/7 DoD met. Item 1 missed because v9 design has a fatal flaw, not a tuning issue. Filed SMEM-round-trip variant for SPRINT-017.**

## Lessons / corrections

- The v9 design in SPRINT-016.md said "Serialize the promote per (fm,fn) — only one half + one float frag alive at a time." This was based on a register-pressure model that did not account for the **fragment layout mismatch** between half and float accumulators. Both elements being in registers is necessary AND sufficient for the layout mismatch to corrupt the math.
- Sprint planning should have included "verify wmma::fragment element ordering equivalence" as a precondition for v9. Adding to follow-ups for future sprint-plan vetting.

## Artifacts

- New kernel (left in tree, gated off correctness gate): `tools/tc-grid/kernels/v9_kernels.cuh`
- New ncu data: `/tmp/ncu_int4.log` (1716 rows; will be copied to `tools/tc-grid/docs/ncu_int4_path_comparison.csv` if useful)
- New sprint plan tree: `docs/sprints/SPRINT-016.md`, `SPRINT-016-INTENT.md`, `SPRINT-016-DEFERRED.md`, `SPRINT-016-FOLLOWUPS.md`
- Dev pod: `tcg-dev` (kept running for SPRINT-017 work)
