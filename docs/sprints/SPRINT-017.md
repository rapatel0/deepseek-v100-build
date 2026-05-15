# SPRINT-017 — v11: m8n8k4 PTX + XOR-swizzled SMEM (path to 50 TFLOPS)

**Status: ACTIVE.** Started 2026-05-13.

**Execution plan**: see [`tools/tc-grid/docs/V11-EXECUTION-PLAN.md`](../../tools/tc-grid/docs/V11-EXECUTION-PLAN.md)
for the wave-by-wave decision-gated breakdown with links to supporting docs.

Sprint goal: exceed **50 TFLOPS** on V100 INT8 tensor-core GEMM at M=2048, N=K=7168,
bit-correct (`rel ≤ 1e-3`). Current production champion: v10 row-major B at **29.49 TFLOPS**.
Gap: 1.7×.

All work on a single V100 (gpu-01) against the live `tcg-dev` pod.

## Why this sprint exists

SPRINT-016 closed at 29.49 TFLOPS (v10 row-major B + HMUL dequant). REPORTs 9/10/11 show
the `wmma::*` API path is exhausted — Short Scoreboard 21.5%, Long Scoreboard 22.7%,
MIO 10.9%, TC% only 23.6%. Bank conflicts on B loads remain at 334M and cannot be reduced
further with `wmma::load_matrix_sync` because:

1. **Padding cannot zero bank conflicts** — V100's `ldm` constraint forces gcd(BK_PAD/2, 32) ≥ 4,
   minimum 4-way conflict (see memory file `v100_wmma_smem_conflict_constraint.md`).
2. **`wmma::load_matrix_sync` assumes linear stride** — incompatible with XOR-swizzled SMEM.

The only paths to 50 TF+ require dropping the `wmma::*` API entirely.

## Reference implementation (turbomind)

Local clone: `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/`. Full design
review with code excerpts and ranked porting candidates in
**`tools/tc-grid/docs/TURBOMIND-INSIGHTS.md`**.

Key files (in order to study):

1. `core/mma.h:11-30` — `mma_m8n8k4_row_col` inline PTX wrapper. Copy verbatim.
2. `core/layout.h:8-19` — `Swizzle<Bits, Base, Shift>` template.
3. `gemm/arch/mma_sm70.h` — `SM70_MMA_884` fragment shapes.
4. `gemm/arch/smem_copy_sm70.h:21-65` — `SmemCopy_MMA_884_{A,B}` lane→element offsets.
5. `gemm/iterator_sm70.h:134-256` — phase-table swizzle precomputation (the trick).
6. `gemm/mainloop_sm70.h:196-351` — full pipelined mainloop (structural reference).
7. `gemm/test/gemm_bench.cu` — standalone nvbench harness for independent comparison.

## Implementation phases (from V11-DESIGN.md, sharpened by TURBOMIND-INSIGHTS.md)

| Phase | Task | Acceptance | ETA |
|---|---|---|---|
| P-bench | **Deferred** — turbomind `gemm_bench` target is commented out in upstream (`gemm/CMakeLists.txt:108-137`), needs 1–2 days of build engineering to resurrect. Revisit ONLY if v11 Steps 1–3 land below 40 TF and we want a sanity check on whether 50 TF is reachable. | — | deferred |
| P0 | **Step 1**: bare m8n8k4 PTX replacement. `v11a_kernels.cuh` = v10 + extract-halves + 4× `mma_m8n8k4_row_col` for each `wmma::mma_sync`. Keep `wmma::load_matrix_sync`. | Bit-correct vs v10. No perf change expected. | 2 hr |
| P1 | **Step 2**: manual `Lds` (ld.shared.b32) fragment loads matching turbomind's `SmemCopy_MMA_884_{A,B}::unique` lane offsets. Drop `wmma::load_matrix_sync`. SMEM layout unchanged. | Bit-correct vs v10. No perf change expected. | 3 hr |
| P1.5 | **Step 2.5**: try CTA_K=16 in the v11 codepath (BK=16 vs current BK=32). m8n8k4 advances K by 4, so BK=16 = 4 inner mma loop iterations — interleaves dequant more frequently. | Bit-correct. Compare 16 vs 32 perf. Keep winner. | 1 hr |
| P1.75 | **Step 2.6**: cache-policy `Stream` (cs-evict) on B-side gmem load. Replace `__ldg` with `ld.global.cs` inline PTX for the W_qs load. B has no reuse and is the largest gmem tensor; evicting from L1 frees capacity for A. | Bit-correct. Compare vs plain Ldg + L2-prefetch combo. Est +0–3%. | 1 hr |
| P2 | **Step 3**: add XOR swizzle to B-side SMEM store + load. Use `Swizzle<3, 3, 3>` canonical pattern for half-typed access. **Precompute phase table at iterator construction** per turbomind `iterator_sm70.h:134-139` — saves AND+SHIFT+XOR per SMEM store at cost of small register table. | Bit-correct. `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` drops ≥10× on B-loads. Target perf: 35–45 TFLOPS @ M=2048. | 3 hr |
| P3 | **Step 4**: apply XOR swizzle + phase table to A-side. | Bit-correct. Further perf if A-side conflicts non-trivial. | 1 hr |
| P4 | **Step 5**: tile / warp / stages grid sweep with new SMEM layout. Configurations to try (mirroring turbomind's shipped sm_70 tiles): `(BM, BN, BK, WARPS, FRAG_M, FRAG_N)` ∈ {(128, 256, 16, 8, 8, 16), (128, 128, 16, 8, 8, 8), (64, 128, 32, 4, 4, 8), (32, 128, 32, 4, 2, 8)}. | Final champion identified per M ∈ {64, 256, 1024, 2048, 4096}. | 2 hr |
| Done | REPORT-12 + ledger update + SPRINT-017 close | Numerical comparison vs v10 AND vs turbomind across full M sweep, ncu stall budget for v11-best, decision on 50 TF status. | — |

## Verification gates (apply at every step)

- `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1` against the reference matmul (existing harness).
- First launch of each new kernel: `compute-sanitizer --tool memcheck`.
- Bank-conflict measurement: `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`.
- TC%: `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed`.
- Per-row tolerance gate must hold for every OK row in the final sweep CSV.

## Risks

- **R1 — m8n8k4 lane→element mapping is opaque per PTX docs.** Turbomind's exact lane
  offsets are validated on their codebase, not ours. Mitigation: bit-compare after every
  step; never compose Steps 1+2 without verifying Step 1 first.
- **R2 — XOR swizzle must be applied symmetrically.** Easy to typo at store vs load.
  Mitigation: write a SMEM-roundtrip unit test in the harness *before* integrating with mma.
- **R3 — Compiler may not optimize hand-rolled `ld.shared` as well as `wmma`.** Mitigation:
  inspect SASS via `cuobjdump --dump-sass` if Step 2 regresses unexpectedly.
- **R4 — DCGM-exporter accidentally re-enabled.** Mitigation: re-check
  `kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'`
  before each ncu run; pause with `--overwrite` if needed.

## Fallback — UPDATED 2026-05-14 after SPRINT-018 P0+P1

**SPRINT-018 P0+P1 measured the ceiling** at **85.93 TFLOPS at M=2048** via CUTLASS with
pre-dequant W to FP16. This is NOT a viable production path — pre-dequant doubles VRAM
(28 GB DSv4 weights) and doubles HBM bandwidth, defeating INT8 quantization
(see [feedback_pre_dequant_defeats_int8.md](../../.claude/projects/-Users-ravi-repos-deepseek/memory/feedback_pre_dequant_defeats_int8.md)).
**CUTLASS sm_70 has no fused-dequant template** (mixed-input GEMM is sm_80+ only), so
SPRINT-018 P2+P3+P4 were scoped out.

**V11 IS the production path on V100 INT8.** With fused dequant in load_tile (gmem stays
INT8, SMEM holds FP16, mma on FP16), the target is to **close to the 85 TF ceiling** that
P1 proved is reachable on this hardware/shape.

New decision rule (post-SPRINT-018):

- V11 closes to **≥ 60 TF** (within ~30% of ceiling): production-quality, ship.
- V11 lands 50–60 TF: still meets sprint goal (50 TF), worth shipping with documented gap.
- V11 lands 35–50 TF: misses goal but beats v10. Ship + investigate why ceiling gap is
  larger than turbomind achieves.
- V11 stalls < 35 TF: revisit fundamentals — likely hitting a hardware limit we don't
  yet understand. Possibly back to investigating gmem bandwidth ceiling for INT8 reads.

Secondary fallback (if V11 stalls hard): the 85 TF P1 result is published as the academic
ceiling; v10s SplitK ships for small M; v10 (29 TF) ships for large M as fallback. Document
the engineering tradeoff explicitly.

## Definition of Done

1. v11-final bit-correct: `rel ≤ 1e-3` at M ∈ {64, 256, 1024, 2048, 4096}.
2. ncu confirms `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` drops ≥10× vs v10.
3. Per-M champion table written into REPORT-12.
4. Goal verdict: explicit pass/fail on 50 TFLOPS at M=2048. If fail, fallback option chosen.
5. Updated grid sweep CSVs including v11 variants.
6. SPRINT-017-FOLLOWUPS.md captures any deferred work.

## Files summary

New:
- `tools/tc-grid/kernels/v11a_kernels.cuh` (Step 1 — bare m8n8k4)
- `tools/tc-grid/kernels/v11b_kernels.cuh` (Step 2 — manual Lds)
- `tools/tc-grid/kernels/v11_kernels.cuh` (Step 3+ — swizzle + tuned)
- `tools/tc-grid/docs/REPORT-12.md`

Modified:
- `tools/tc-grid/src/launch_int8.cu` (LAUNCH_V11/RERUN_V11, smem calc)
- `tools/tc-grid/src/main.cu` (v11 tile entries)

## Dependencies

- `tcg-dev` pod live on gpu-01.
- CUDA 12.2.2 in pod.
- Turbomind clone at `/tmp/turbomind` (already pulled).
- No external APIs / credentials.

## Open questions

1. Should v11 also lift to INT4 / MXFP4 / F8 in this sprint, or defer to a follow-up after
   INT8 lands? Likely defer — m8n8k4 PTX wrapper is format-agnostic, but each format's
   dequant front-end needs separate validation.
2. Multi-shape MoE generalization still pending from SPRINT-016 P1. Pair with v11 final
   if time allows.
3. **MoE dispatcher integration into DSv4 production**: turbomind has `moe_utils_v2.cu`
   and an MoE-aware GEMM dispatcher. Our tc-grid winners are scalar GEMMs; integrating
   into DSv4's MoE call site is a separate concern from the TFLOPS goal. Filed as
   follow-up; do NOT block sprint closure on this.

## Counterfactual notes

`tools/tc-grid/docs/TURBOMIND-INSIGHTS.md` §L documents a full counterfactual analysis:
what we have that they don't (L2 prefetch, fine quant granularity, launch_bounds tuning,
multi-format kernels, bit-correctness audit harness) and what they have that we miss
(see L2). Our v10 row-major B SMEM win comes from a DIFFERENT mechanism than their XOR
swizzle (changing WMMA's lane access pattern vs eliminating bank conflicts). Both attack
the same problem from different angles, so v11's swizzle may not stack additively with
v10's row-major win.
