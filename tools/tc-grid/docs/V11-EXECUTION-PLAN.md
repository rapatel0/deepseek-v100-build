# v11 Execution Plan — path to 50 TFLOPS on V100 INT8

Date: 2026-05-14
Owner: tc-grid optimization loop
Status: Active plan; pre-implementation.

Concrete, decision-gated order of operations for the v11 work. Total estimated ~12 hr
engineering across 3 waves. Each wave has a decision gate that lets us bail to a fallback
without burning sunk cost.

## Context (read these first)

| Doc | Purpose |
|---|---|
| [REPORT-11.md](./REPORT-11.md) | Current production state — v10 ships at 29.49 TFLOPS, full-spectrum bit-correctness audit, why `wmma::*` API is exhausted |
| [REPORT-10.md](./REPORT-10.md) | SPRINT-016 negative result — v9 mixed-precision broken-by-design (sm_70 frag layout mismatch) |
| [REPORT-9.md](./REPORT-9.md) | Tier B retrospective with corrected ncu interpretation (Short Scoreboard 30%, BankConf 390M) |
| [V11-DESIGN.md](./V11-DESIGN.md) | Original 5-step v11 roadmap (m8n8k4 PTX → manual Lds → XOR swizzle → A-side → tune) |
| [TURBOMIND-INSIGHTS.md](./TURBOMIND-INSIGHTS.md) | Source-walk of turbomind sm_70 GEMM, ranked porting candidates, counterfactual analysis (§L) |
| [../../docs/sprints/SPRINT-017.md](../../../docs/sprints/SPRINT-017.md) | Active sprint doc with phase table |
| [v10_kernels.cuh](../kernels/v10_kernels.cuh) | Current production champion kernel |

## Memory references

| File | Constraint |
|---|---|
| `~/.claude/projects/-Users-ravi-repos-deepseek/memory/v100_wmma_smem_conflict_constraint.md` | Padding cannot zero bank conflicts on V100 col-major B; real lever was SMEM access pattern (v10), not register count |
| `~/.claude/projects/-Users-ravi-repos-deepseek/memory/v100_wmma_half_float_frag_layout_mismatch.md` | sm_70 half- and float-typed `wmma::fragment` accumulators use different lane→element mappings — broke v9 |

## Turbomind reference files (for porting)

| Concept | File | Lines |
|---|---|---|
| m8n8k4 PTX wrapper | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/core/mma.h` | 11–30 |
| Swizzle template | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/core/layout.h` | 8–19 |
| MMA_884 fragment shapes | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/arch/mma_sm70.h` | full file |
| SmemCopy A/B lane offsets | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h` | 21–65 |
| Phase-table iterator | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h` | 134–256 |
| Mainloop pipeline | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h` | 196–351 |
| Shipped tile registry | `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` | full file |

---

## Wave 1 — Cheap counterfactual validation (≤1 hr)

**Premise**: turbomind ships `SplitK=true` in every sm_70 config; we deferred it in
SPRINT-016 because M=2048 didn't seem to need it. Worth a quick test against that
assumption before committing to v11.

### A0. SplitK trial on existing v10

- **What**: Add a SplitK launcher variant for v10 at M=2048 with K-split factor ∈ {2, 4, 8}.
  Use atomic-add accumulation into the C tile.
- **Why**: Turbomind's reliance on SplitK in every config suggests it's doing work at every
  M, not just small M. Cheap test of the counterfactual hypothesis.
- **Files**: `tools/tc-grid/src/launch_int8.cu` (new `LAUNCH_V10_SPLITK` macro), small kernel
  wrapper in `v10_kernels.cuh`.
- **Acceptance**: Bit-correct (`rel ≤ 1e-3`). Numerical comparison v10 vs v10-splitK at
  M ∈ {64, 256, 1024, 2048, 4096}.
- **Decision**: ship if ≥3% win at any production M; shelve otherwise.
- **Reference**: See `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu:18`
  for their SplitK=true config example.

**Decision gate after Wave 1**: log result; proceed to Wave 2 regardless. SplitK is
orthogonal to v11 — independent perf lever.

---

## Wave 2 — v11 structural prep (4–6 hr, no perf change expected)

**Premise**: get the m8n8k4 PTX and manual fragment loads working in our build, bit-correct
against v10. Foundation for the perf attack in Wave 3. Zero performance change expected at
this wave — we're swapping out the wmma API while preserving the exact SMEM layout and
operation count.

### A1. v11 Step 1 — bare m8n8k4 PTX swap

- **What**: `tools/tc-grid/kernels/v11a_kernels.cuh` = copy of v10 with `wmma::mma_sync`
  replaced by 4× `mma_m8n8k4_row_col` per `16×16×16` accumulator. Keep
  `wmma::load_matrix_sync` for A and B — same SMEM layout, same fragment shapes.
- **PTX wrapper**: copy verbatim from
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/core/mma.h:11-30`.
- **Why**: Validates that inline PTX works in our build (compiler version, sm_70 target,
  asm-volatile flags) before introducing the harder manual-Lds lane mapping.
- **Acceptance**: Bit-correct vs v10. No perf change.
- **Estimated effort**: 2 hr.

### A2. v11 Step 2 — manual `ld.shared.b32` fragment loads

- **What**: `tools/tc-grid/kernels/v11b_kernels.cuh` = v11a + replace `wmma::load_matrix_sync`
  with manual lane-indexed SMEM loads.
- **Lane offsets**: replicate turbomind's `SmemCopy_MMA_884_{A,B}::unique` from
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h:21-65`.
- **Why**: XOR-swizzled SMEM (Wave 3) cannot use `wmma::load_matrix_sync` because that
  function assumes a linear stride. Manual Lds is the prerequisite.
- **Acceptance**: Bit-correct vs v10. No perf change.
- **Estimated effort**: 3 hr (lane mapping is finicky).

### A3. SMEM-roundtrip swizzle unit test

- **What**: A small standalone CUDA test that fills SMEM with a known pattern, applies
  `Swizzle<3, 3, 3>::apply` at store, reads back via the same mapping, and confirms data
  integrity. Run in isolation, no mma.
- **Why**: XOR swizzle bugs are notoriously hard to debug when composed with mma — values
  silently corrupt one lane at a time. A dedicated unit test catches typos cheaply.
- **Files**: new `tools/tc-grid/tests/swizzle_roundtrip.cu`.
- **Acceptance**: Reads-back match writes for every offset in the swizzled range.
- **Estimated effort**: 1 hr.

**Decision gate after Wave 2**:
- ✅ Steps 1+2+3 all bit-clean → proceed to Wave 3.
- ❌ Unexpected build / correctness issues → pause and assess CUTLASS-direct fallback
  (see Wave 4 fallback options).

---

## Wave 3 — The actual perf attack (4–5 hr, target 35–45 TF)

**Premise**: with the wmma API gone, we can apply XOR swizzle and finer K-staging. This
is where the perf wins live.

### A4. v11 Step 2.5 — try BK=16 inner K-stage

- **What**: Add a BK=16 variant of v11b. m8n8k4 advances K by 4, so BK=16 = 4 inner-loop
  iterations per K-stage. Interleaves dequant + mma more tightly than our current BK=32.
- **Reference**: Turbomind's shipped large tiles all use CTA_K=16 (see
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`).
- **Acceptance**: Bit-correct. Compare BK=16 vs BK=32 perf; keep winner.
- **Estimated impact**: +2–4%.
- **Estimated effort**: 1 hr.

### A5. v11 Step 2.6 — `ld.global.cs` (Stream cache policy) on B-side

- **What**: Replace `__ldg(W_qs)` with `ld.global.cs` inline PTX. B has no reuse; evict
  from L1 to free capacity for A.
- **Why**: This is the counterfactual to our L2-prefetch strategy. Turbomind explicitly
  uses `Stream` cache policy for B-side; we explicitly prefetch B. Worth comparing.
- **Reference**: Turbomind's `Policy` template parameter in
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h`. See
  also `S = cache_policy::Stream` in `kernel/sm70_884_4.cu:11`.
- **Acceptance**: Bit-correct. Measure vs plain Ldg + L2-prefetch.
- **Estimated impact**: +0–3%.
- **Estimated effort**: 1 hr.

### A6. v11 Step 3 — XOR swizzle on B-side with phase-table precomputation

- **What**: `tools/tc-grid/kernels/v11c_kernels.cuh` = v11b + `Swizzle<3, 3, 3>` applied
  to B-side SMEM store AND load. Use **phase-table precomputation** to amortize the
  AND+SHIFT+XOR math.
- **Phase table**: precompute swizzled offsets at iterator construction (one register
  table per warp), apply via lookup at each store. Reference:
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/iterator_sm70.h:134-139`
  (constructor-time precompute) and lines 237–256 (per-store apply).
- **Acceptance**: Bit-correct. `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  drops ≥10× on B-loads (measure with ncu).
- **Estimated impact**: +6–8% on top of v10 (UNCERTAIN — see counterfactual test below).
- **Estimated effort**: 3 hr.

### A6-CF. Critical counterfactual test inside Step 3

- **What**: Build v11c TWO ways:
  - (a) v11c on v10's row-major B SMEM base
  - (b) v11c on col-major B SMEM base (the pre-v10 layout)
- **Why**: Our v10 row-major B SMEM is a DIFFERENT mechanism than XOR swizzle (changes
  WMMA's lane access pattern vs eliminates bank conflicts at the load). The two may not
  stack additively. If (a) ≈ v10, swizzle is redundant given row-major. If (b) > (a),
  the two mechanisms are additive. Either result is informative — captured in
  [TURBOMIND-INSIGHTS.md §L3](./TURBOMIND-INSIGHTS.md#l3-notable-counterfactual-surprises).
- **Acceptance**: Both bit-correct. Numerical comparison.

**Decision gate after Wave 3**:
- ✅ 35+ TF and path to 50 visible → proceed to Wave 4.
- ⚠️ 30–35 TF, gap to 50 unclear → build turbomind `gemm_bench` as a sanity ceiling
  (see [TURBOMIND-INSIGHTS.md §I](./TURBOMIND-INSIGHTS.md#i-standalone-bench-feasibility--harder-than-initially-thought)
  — 1–2 days build effort, only do this if needed).
- ❌ 25–30 TF, less than expected → debug; swizzle should have moved us more.

---

## Wave 4 — Closing the gap (2–4 hr, if needed)

### A7. v11 Step 4 — XOR swizzle + phase table on A-side

- **What**: Apply Swizzle to A SMEM as well as B.
- **Estimated impact**: Smaller than B-side win since A-side conflict count is lower.
- **Estimated effort**: 1 hr.

### A8. v11 Step 5 — tile sweep with turbomind-mirrored configs

- **What**: Grid sweep `(BM, BN, BK, WARPS, FRAG_M, FRAG_N)` over:
  - `(128, 256, 16, 8, 8, 16)` — turbomind's largest tile
  - `(128, 128, 16, 8, 8, 8)`
  - `(64, 128, 32, 4, 4, 8)`
  - `(32, 128, 32, 4, 2, 8)` — tall-thin variant
- **Reference**: shipped sm_70 configs in
  `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/gemm/kernel/sm70_884_4.cu`.
- **Acceptance**: Final champion identified per M ∈ {64, 256, 1024, 2048, 4096}.
- **Estimated effort**: 2 hr.

### A9. Goal verdict + REPORT-12

- **What**: Write REPORT-12 with v11 final numbers, ncu stall budget for v11-best vs v10,
  fallback decision if 50 TF goal missed.
- **Fallback options** (if v11 < 35 TF after Wave 4):
  1. **CUTLASS direct integration** (~4 hr). Pull V100 INT8 GEMM as a library. Ceiling
     60–75 TF.
  2. **Pre-dequant W → FP16 + cuBLAS** — works for bench, blows memory budget for full DSv4
     inference (~28 GB FP16 weights). Prefill / single-layer bench only.
  3. **Accept ~30 TFLOPS ceiling** — document the engineering tradeoff explicitly,
     ship v10 + HMUL + (whichever Wave items shipped) as final.

---

## Recommended immediate action

Start with **Wave 1 / A0 (SplitK trial on v10)**. Reasons:

1. Cheapest test (≤1 hr).
2. Independent of v11 work — no risk of contaminating v11 if it fails.
3. Validates the counterfactual hypothesis from [TURBOMIND-INSIGHTS.md §L2](./TURBOMIND-INSIGHTS.md#l2-what-turbomind-has-that-were-missing-sharper-than-the-h-ranked-list).
4. Possible immediate production win on v10 before we even start the harder v11 work.

If A0 shows a clear win, ship and update REPORT-11 in-place; then start v11 Step 1
afterward. If A0 is noise, log and move on to Step 1.

## Risks (cumulative from V11-DESIGN.md §Risks + new)

- **R1 — m8n8k4 lane→element mapping is opaque per PTX docs.** Mitigation: bit-compare
  after every step.
- **R2 — XOR swizzle must be applied symmetrically.** Mitigation: A3 unit test before A6.
- **R3 — Compiler may not optimize hand-rolled `ld.shared` as well as wmma.** Mitigation:
  check SASS via `cuobjdump --dump-sass` if Step 2 regresses.
- **R4 — v10 row-major B already extracted the bank-conflict win.** Captured as A6-CF
  test. If swizzle = row-major reaches the same local optimum, document it; don't double
  up on a non-additive mechanism.
- **R5 — SplitK win may NOT generalize past M=2048.** Mitigation: A0 tests at multiple M.
- **R6 — DCGM-exporter accidentally re-enabled during ncu runs.** Mitigation: check
  `kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'`
  before each ncu pass.

## Definition of Done (sprint-level, from SPRINT-017)

1. v11-final bit-correct: `rel ≤ 1e-3` at M ∈ {64, 256, 1024, 2048, 4096}.
2. ncu confirms `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` drops ≥10× vs v10.
3. Per-M champion table written into REPORT-12.
4. Explicit pass/fail on 50 TFLOPS at M=2048. If fail, fallback option chosen.
5. Updated grid sweep CSVs including v11 variants.
6. SPRINT-017-FOLLOWUPS.md captures any deferred work.
