# SPRINT-019 — v11 → v12: methodical push to 50 TF with full no-skip discipline

**Status:** ACTIVE (planned). **Predecessor:** SPRINT-017 (v11 closed at
35.08 TF M=2048, bit-correct).

**Headline goal:** cross **50 TF at M=2048** AND reach **≥ 20 TF at M=64**
via v11+SplitK port. Both bit-correct (`rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤
0.1`), ncu-evidenced, CUTLASS-Gemm70-ratio-tracked.

**Discipline:** every kernel change passes a four-rung correctness ladder
BEFORE production wiring (isolated CPU-reference test → `compute-sanitizer`
→ tc-grid sweep → baseline regression). Every commit-gate measurement is
median-of-5 runs. Every phase sweeps ≥ 12 tile shapes across the extended
M-list {1, 8, 32, 64, 256, 1024, 2048, 4096}. **No exceptions, no skipping.**

Cross-references:
- [SPRINT-019-INTENT.md](./drafts/SPRINT-019-INTENT.md) — sprint motivation
  and user directives.
- [SPRINT-019-MERGE-NOTES.md](./drafts/SPRINT-019-MERGE-NOTES.md) — how the
  three drafts and three critiques were synthesized.
- [SPRINT-019-DEFERRED.md](./SPRINT-019-DEFERRED.md) — items explicitly
  scoped out for SPRINT-020.
- [REPORT-12.md](../../tools/tc-grid/docs/REPORT-12.md) — sprint-017 close,
  benchmark learnings, six concrete §6.1–§6.6 levers.
- Memory: `v100_wmma_half_float_frag_layout_mismatch`,
  `feedback_pre_dequant_defeats_int8`,
  `feedback_dont_skip_plan_steps`,
  `feedback_effort_estimation_undocumented_hardware`.

---

## 1. Overview

V11 ships at **35.08 TF at M=2048** vs CUTLASS pre-dequant **85.93 TF**
ceiling. REPORT-12 §3 characterizes the gap: 21.45% long-scoreboard
(unhidden HBM latency), 18.45% short-scoreboard (mma write→read), 15.56%
mio-throttle (SMEM bandwidth), 1.35% math-pipe throttle. **The tensor
cores are idle 98.6% of the time when math-pipe is the bottleneck — we are
gmem-latency-bound, not compute-bound.**

Six concrete levers from REPORT-12 §6:

| § | Lever | Expected Δ | Risk |
|---|---|---:|---|
| 6.1 | FP16 accumulator + SMEM round-trip epilogue | **+30–50%** | HIGH |
| 6.2 | Per-shape 3-stage pipeline | +5–8% | MEDIUM |
| 6.3 | SplitK port to v11/v12 | +50–80% at M=64 | LOW |
| 6.4 | Larger CTA tile (BM=192/256) w/ c_frag SMEM spill | +5–10% | MEDIUM |
| 6.5 | PRMT-vectorized A-side load | +0.5–2% | LOW |
| 6.6 | Multi-shape MoE validation | orthogonal (rigor) | LOW |

**Execution order**: P0 (reproduce) → P1 (§6.1) → P2 (§6.2) → P3 (§6.3)
→ P4 (§6.4) → P5 (§6.5) → P6 (§6.6) → P7 (close). Rationale (Claude §3.3):
§6.1 halves c_frag register pressure, which is the prerequisite for §6.2
3-stage to fit on BM=128 shapes (sprint-017 regressed -29% on the champion)
AND for §6.4 to be possible at BM ≥ 192.

### 1.1 Headline targets

- **M=2048**: v12 (or successor) at **≥ 50 TF**, bit-correct, ncu-evidenced.
- **M=64**: v12s (SplitK) at **≥ 20 TF** (parity with v10s_ks8 = 20.31 TF).
- **All M ∈ {1, 8, 32, 64, 256, 1024, 2048, 4096}**: no regression > 2%
  vs sprint-017 per-M champion at the shared M values.
- **CUTLASS ratio**: each shipped champion's TF / CUTLASS Gemm70 TF
  documented at M=2048 and M=4096.
- **Multi-shape robustness**: P6 sweep across DSv4 layer dims confirms
  the champion isn't 7168×7168-specific.

### 1.2 The no-skip rule (applied at every phase, every commit)

A change is **shipped** only when **all** of the following hold:

1. Tier-1 CPU-reference correctness test passes (`rel = 0` for identity
   inputs, `rel ≤ 1e-3` for uniform_small).
2. `compute-sanitizer --tool memcheck` returns 0 errors on first launch.
   For atomic kernels: `--tool racecheck` and `--tool initcheck` also
   return 0.
3. Full tc-grid M-sweep bit-compare against v10 passes
   (`rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`) at every M ∈
   {1, 8, 32, 64, 256, 1024, 2048, 4096}.
4. The new kernel's `rel` does not exceed the prior champion's `rel` for
   the same row (any drift documented but must stay within tolerance).
5. ncu shows the **target stall** (per phase) actually dropped vs the
   prior champion, measured at M=2048 AND M=4096.
6. Headline TF (**median of 5 runs**) improved at ≥ 3 of 8 M values.
7. No M value regresses by > 2% (median of 5 runs).
8. CUTLASS Gemm70 ratio measured at M=2048 AND M=4096.
9. Grid sweep of ≥ 12 tile shapes recorded in
   `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-N.csv`.
10. For pipeline-restructuring phases (P1, P2): Nsight Systems timeline
    captured, overlap confirmed visually, PNG committed.

If any of 1–10 fails, the change is **reverted from default dispatch**
(kept as a `#if 0` artifact in source for documentation). The reason is
logged in REPORT-13 with ncu before/after evidence.

**No phase-specific relaxations exist.** The PRMT A-side phase (§6.5)
gets the same 10-item bar as the FP16-acc phase (§6.1). The 3-stage
phase (§6.2) gets a CPU-reference correctness test even though it is
"only" a mainloop rearrangement.

---

## 2. Use cases

### 2.1 Production workloads addressed

- **DSv4 INT8 inference, decode-style batches (M ∈ [1, 64])**: §6.3
  SplitK port closes the small-M production gap. v11 today is 7.52 TF at
  M=64; v10s_ks8 is 20.31 TF. Target: v12s (or v11s if §6.1 abandoned)
  ≥ 20 TF.
- **DSv4 INT8 inference, prefill / large-batch (M ∈ [1024, 4096])**:
  §6.1 FP16 acc is the primary lever; §6.2 3-stage and §6.4 large-BM
  are stack-ons. Target: ≥ 50 TF at M=2048.
- **MoE expert dispatch (variable shapes per expert)**: §6.6 in-sprint
  multi-shape validation ensures the champion generalizes to
  DSv4-relevant N/K combinations.

### 2.2 Non-production use cases

- **Kernel-development infrastructure**: every `tests/test_<phase>.cu`
  becomes a reusable template.
- **CUTLASS ratio tracking** at every phase shrinks the
  "captured-fraction" gap of the 85 TF ceiling.

### 2.3 Out of scope (see SPRINT-019-DEFERRED.md)

- Turbomind `gemm_bench` standalone build (1–2 days build engineering).
- INT4 BN=256 spill fix (conditional on §6.4 success).
- v5 persistent-CTA revisit (conditional on §6.1 success).
- MoE-aware dispatcher integration into DSv4 inference (separate sprint).
- Cache-policy `Stream` revisit (sprint-017 negative result).

### 2.4 Canonical ncu metric set

Used identically at every phase, M=2048 AND M=4096, exported to
`tools/tc-grid/docs/ncu/SPRINT-019-P<N>-<variant>-M<m>.csv`:

```
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__t_sector_hit_rate.pct
l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum       # atomic count (P3 only, but recorded everywhere)
launch__registers_per_thread
launch__shared_mem_per_block_static
launch__waves_per_multiprocessor
```

### 2.5 Standardized profiling protocol

Copy-pasteable commands. Every phase runs them identically.

**ncu**:
```bash
ncu --kernel-id ::mm_int8_lut_v<X>:1 \
    --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sector_hit_rate.pct,\
l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__waves_per_multiprocessor \
    --csv \
    ./build/tc-grid --m-list 2048 --nk 7168 --dist uniform_small \
    > tools/tc-grid/docs/ncu/SPRINT-019-P<N>-<variant>-M2048.csv
```

Same for M=4096 (`--m-list 4096`).

**Nsight Systems**:
```bash
nsys profile --capture-range cudaProfilerApi --trace cuda,nvtx \
     --output tools/tc-grid/docs/nsys/SPRINT-019-P<N>-<variant>-M2048 \
     ./build/tc-grid --m-list 2048 --nk 7168 --dist uniform_small
# Export PNG screenshot of the timeline section showing LDG/mma overlap.
```

**DCGM-exporter pre-flight (mandatory before every ncu run)**:
```bash
kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'
# expected output: paused
# if not: kubectl label --overwrite nodes gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=paused
```

**Median-of-5 measurement script** (new helper, P0 prerequisite):
```bash
# scripts/bench-median.sh — runs tc-grid 5 times, prints median TF per shape.
# Used in every commit gate where headline TF is being compared.
```

---

## 3. Architecture

### 3.1 Current state (sprint-017 close)

```
v11<128, 128, 16, 4, 8, 2>
├── SMEM: B col-major (n outer, k inner), BK_PAD = BK + 8 = 24 halves
│         A row-major, stride BK
├── Mainloop: 2-stage (LDG → STS → mma), L2 prefetch (no cp.async on sm_70)
├── mma atom: m8n8k4_row_col, FP32 acc, c_frag = float[8] per lane
│             2× back-to-back mma (K=0..3, K=4..7) accumulating same c_frag
├── Dequant: PRMT bias-trick (XOR 0x80 + prmt 0x64646464 + sub 1152), half2
└── Epilogue: cvt.rn.f16 + STG fp16
```

ncu stall breakdown at M=2048: long_scoreboard 21.45%, short_scoreboard
18.45%, mio_throttle 15.56%, math_pipe_throttle 1.35%.

### 3.2 Target architecture (sprint-019 close)

```
v12<BM, BN, BK, W, ATOMS_M, ATOMS_N>                       (large-M champion)
├── FP16 accumulator (c_frag = half[4] per lane, regs halved)
├── SMEM round-trip epilogue (sidesteps f16/f32 lane→element mismatch)
├── Optional 3-stage pipeline (P2 dispatched per shape via reg budget)
├── Optional larger BM (P4; partial c_frag SMEM spill rotation)
└── PRMT-vectorized A-side fp32→half (P5)

v12s<BM, BN, BK, W, ATOMS_M, ATOMS_N, KSPLIT>              (small-M champion)
├── v12 mainloop replicated grid.z times
├── fp32 scratch + atomic-add accumulation
└── Separate fp32→fp16 epilogue kernel (NOT a "last grid.z slice does it"
    trick — that has no global-sync guarantee on sm_70)
```

v11 remains as the FP32-acc fallback in case §6.1 (FP16) fails its
correctness gate. v10 remains as the deepest fallback.

### 3.3 Why the dependency order is §6.1 → 6.2 → 6.3 → 6.4 → 6.5 → 6.6

1. **§6.1 first**: FP16 acc halves c_frag register pressure (~128 fp32 →
   64 fp16 = 32 regs per lane). This is the prerequisite for §6.2
   (3-stage rmem buffer needs register headroom) AND §6.4 (BM=192 needs
   c_frag count reduction).
2. **§6.2 next**: per-shape 3-stage pipeline. Sprint-017 hit 36.68 TF on
   one shape (+9.5%) but -29% on the champion. With §6.1 cutting c_frag
   in half, the regression should narrow.
3. **§6.3 SplitK**: low-risk, mechanical. Inherits the §6.1 base if it
   exists, otherwise forks v11.
4. **§6.4 large BM**: only worthwhile if §6.1 succeeded. With FP32 acc,
   BM=192 needs 192 fp32 c_frag per lane — over budget.
5. **§6.5 PRMT A**: tail lever; least dependency.
6. **§6.6 multi-shape**: final robustness check across DSv4 layer dims.
   Does not change the kernel; validates dispatch.

---

## 4. Implementation

Every phase uses the same five-tier verification structure:

- **Tier 1 — isolated CPU-reference correctness** (`tests/test_<phase>.cu`)
  with `compute-sanitizer --tool memcheck` (and for atomics: `racecheck`,
  `initcheck`).
- **Tier 2 — production integration**: wire into `launch_int8.cu` + add
  kTiles[] entries in `main.cu`.
- **Tier 3 — full M-sweep bit-compare**: tc-grid against v10 across the
  extended M-list.
- **Tier 4 — performance characterization**: median-of-5 tc-grid + ncu at
  M ∈ {2048, 4096} + CUTLASS-Gemm70 ratio + grid sweep ≥ 12 shapes.
- **Tier 5 — Nsight Systems timeline** (P1, P2 only; recommended for P4).

### Phase P0 — Reproduce baseline; build helpers; lock comparison set

**Goal:** before any kernel change, prove the harness reproduces sprint-017
on the live pod; create the median-of-5 benchmark helper; lock the
comparison set (v10, v11, CUTLASS Gemm70).

**Pre-conditions:**
- `tcg-dev` pod live on gpu-01.
- CUDA 12.2.2 in pod, driver pinned to sprint-017's version.
- DCGM-exporter paused on gpu-01.
- `_deps/cutlass-src/` populated.

**Steps:**

1. **P0.1 Build clean**: `cmake -B build && cmake --build build -j 8`.
   Record build time.
2. **P0.2 Build the median-of-5 helper**: `scripts/bench-median.sh` that
   runs tc-grid 5 times and prints median TF per shape per M.
3. **P0.3 Reproduce v11 35.08 TF M=2048**: median-of-5 sweep at the
   extended M-list. Expected: `128x128x16_w4_v11` at M=2048 within ±2%
   of 35.08 TF, rel = 2.594e-04. If not: stop, investigate.
4. **P0.4 Record baselines for v10, v11, v10s_ks8, CUTLASS Gemm70**.
   Output: `tools/tc-grid/docs/baseline-SPRINT-019-P0.csv`. Median-of-5.
5. **P0.5 Capture baseline ncu** for v11 at M=2048 and M=4096 with §2.4
   canonical metrics. Outputs to `tools/tc-grid/docs/ncu/`.
6. **P0.6 Capture baseline Nsight Systems trace** for v11 at M=2048.
   `.nsys-rep` gitignored; PNG export committed.
7. **P0.7 Tag the working tree**: `git tag sprint-019-baseline`.

**Decision gate P0:**
- ✅ baseline reproduced within ±2%, helper works → proceed.
- ❌ drift → investigate (pod env, driver, DCGM-exporter).

**ETA:** 2–3 hr.

---

### Phase P1 — §6.1: FP16 accumulator + SMEM round-trip epilogue

**Goal:** halve c_frag register pressure (128 fp32 → 64 fp16 = 32 regs per
lane) and unlock the **2× tensor-pipe ceiling** (FP16-acc m8n8k4 = 125 TF
peak on V100 vs 62 TF for FP32 acc). Expected: 45–50 TF at M=2048.

**Risk:** the m8n8k4 lane→element mapping for FP16 accumulator is
undocumented in PTX ISA and DIFFERS from FP32 acc (memory file
`v100_wmma_half_float_frag_layout_mismatch`). v9 (SPRINT-016) shipped a
register-resident promote and got rel = 0.71.

**User directive (interview)**: NO time-box on this phase. Invest whatever
it takes. Abandon ONLY if a correctness gate proves the lane mapping is
unrecoverable; not on effort alone.

#### P1.1 Tier 1 — isolated atom correctness

1. Create `tests/test_mma_884_acc_f16_sm70.cu` as a sibling of
   `test_mma_884_tile_sm70.cu`. One CTA, one warp, one mma call.
2. Test inputs: A=I (8×8 half identity), B=I (8×4 half packed), expect
   D=I (4 halves per lane in c_frag).
3. **Empirically probe** the FP16-acc `thread_offset_C` and
   `static_offset_C` formulas. The FP32 formulas
   (`((lane&1) + (lane/16)*4, (lane&2) + (lane&12)*2)` and
   `{(0,0), (2,0), (0,4), (2,4)}`) may NOT transfer. Print per-lane c_frag
   values for a sequence of known inputs; reverse-engineer the mapping.
4. **Adversarial inputs**: zeros, sign-heavy, saturation-adjacent
   (|x| > 65500), and patterns that surface partial-tile correctness
   bugs that aggregate `rel` would mask.
5. `compute-sanitizer --tool memcheck` on first launch.
6. CPU reference compare. Tolerance: bit-exact for integer/identity
   inputs; rel ≤ 1e-3 for distributions.
7. Document the derived FP16-acc layout in
   `tools/tc-grid/docs/V12-DESIGN.md` (new file).

**Decision gate P1.1**:
- ✅ Lane mapping figured + bit-exact across all input patterns → proceed.
- ❌ A correctness gate (specifically: any input pattern where rel >
  1e-3 in the isolated test, or maxabs > 0.1 even on small inputs)
  proves the lane mapping is unrecoverable → abandon §6.1, log in
  REPORT-13, jump to P2 with v11 as base. No effort-based abandonment.

#### P1.2 Tier 2 — v12 kernel skeleton

1. Fork `v11_kernels.cuh` → `v12_kernels.cuh`. Template
   `mm_int8_lut_v12<BM, BN, BK, W, ATOMS_M, ATOMS_N>`.
2. `float c_frag[ATOMS_M][ATOMS_N][8]` → `half c_frag[ATOMS_M][ATOMS_N][4]`.
3. Replace `mma_m8n8k4_row_col_acc` with the FP16-acc wrapper from
   `mma_sm70.cuh`. Verify `+r` constraints on the 2-uint32 (= 4-half) C arg.
4. **Epilogue: SMEM round-trip**. Allocate `__shared__ half c_scratch[BM][BN_PAD]`
   (or per-warp tile if SMEM footprint risks dropping occupancy). Lanes
   scatter c_frag halves to c_scratch per the lane mapping derived in
   P1.1. `__syncthreads()`. Reload as fp16, cvt to fp32, STG to gmem.
   **Concrete occupancy budget**: total SMEM ≤ 64 KB at the champ shape so
   `launch_bounds(2)` still fits.
5. **ptxas --verbose** check: `launch__registers_per_thread` ≤ 96, no
   new spill warnings vs v11.
6. `compute-sanitizer --tool memcheck` on production-shape first launch.
7. Wire into `launch_int8.cu`: `LAUNCH_V12(b_m, b_n, b_k, w)` and
   `RERUN_V12`. Dispatcher version=50.
8. Add tile entries to `main.cu` kTiles[]: start with
   `(128, 128, 16, 4, 8, 2, v12)`.

#### P1.3 Tier 3 — full M-sweep bit-compare

1. Build clean. Record ptxas warnings.
2. `./build/tc-grid --m-list 1,8,32,64,256,1024,2048,4096 --nk 7168 --dist uniform_small | grep v12`.
3. Tolerance gate per row: `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`.
4. New kernel's `rel` per row ≤ v11's `rel` per row (FP16 acc can shift
   rounding noise; this is acceptable if within tolerance, but documented).

**Decision gate P1.3:** all 8 M values pass tolerance → proceed; else revert.

#### P1.4 Tier 4 — performance characterization

1. **Median-of-5 sweep** across the extended M-list using
   `scripts/bench-median.sh`.
2. **ncu at M=2048 AND M=4096** using §2.5 commands. Compare to P0.5
   baseline:
   - `math_pipe_throttle.pct`: should RISE (tensor cores busier).
   - `long_scoreboard.pct`: should drop modestly.
   - `short_scoreboard.pct`: should drop (FP16-acc shorter mma latency).
   - `launch__registers_per_thread`: should fall to ≤ 96.
3. **CUTLASS Gemm70 ratio**: measure version=40 at M=2048 AND M=4096.
   Report v12/CUTLASS ratio.
4. **Grid sweep ≥ 12 tile shapes**:
   ```
   BM ∈ {64, 96, 128, 192, 256}
   BN ∈ {128, 256}
   BK ∈ {16, 32}
   W  ∈ {2, 4, 8}
   constrained: N_PER_WARP ≥ 32, BM = 8k, register budget fits
   ```
   At least 12 valid shapes registered in `main.cu`. Output:
   `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-1.csv`.

#### P1.5 Tier 5 — Nsight Systems timeline

1. `nsys profile` over one v12 invocation at M=2048.
2. Confirm LDG and mma blocks overlap in the SM trace (not just adjacent).
3. Commit `.nsys-rep` (gitignored) + PNG screenshot to `tools/tc-grid/docs/nsys/`.

#### Decision gate P1

Ship v12 as the new champion if **all** 10 no-skip-rule items hold (§1.2).
Else revert (keep code as `#if 0` artifact), log in REPORT-13, proceed to P2
with v11 as base.

**ETA:** open-ended per user directive. Realistic range: 8–24 hr (3× the
REPORT-12 estimate; could extend on lane mapping). The user accepts this
phase may consume the entire sprint.

---

### Phase P2 — §6.2: Per-shape 3-stage pipeline

**Goal:** add 3-stage mainloop (LDG → rmem → STS → SMEM → mma) for shapes
whose register budget can absorb the persistent rmem buffer.

**Risk:** sprint-017 mistake — shipped without per-shape register-budget
verification. Mitigation: per-shape dispatch + global ship rule.

#### P2.1 Tier 1 — isolated correctness

Per the no-skip rule, even though 3-stage is "only" a mainloop
rearrangement, it gets a CPU-reference test.

1. Create `tests/test_v12_ms3_sm70.cu`. Single warp, single CTA, smallest
   workable shape (BM=8, BN=32, BK=16, 2 K-tiles).
2. CPU reference matmul vs the 3-stage GPU output. rel = 0 expected for
   integer inputs.
3. `compute-sanitizer --tool memcheck`.

**Decision gate P2.1**: bit-exact + 0 memcheck errors → proceed.

#### P2.2 Tier 2 — kernel variant

1. Build the 3-stage as a separate template `mm_int8_lut_v12_ms3<...>`
   (or `mm_int8_lut_v11_ms3<...>` if P1 abandoned). Persistent rmem
   `next_A_chunk[]`, `next_B_chunk[]`, `next_S_scale[]` arrays per
   thread.
2. Mainloop structure (per REPORT-12 §4.2): prelude loads tile 0 and 1;
   steady state runs mma(K), STS(K+1 from rmem), LDG(K+2 into rmem)
   concurrently; drain handles last 2 tiles.
3. `compute-sanitizer --tool memcheck` on first production-shape launch.
4. Wire into `launch_int8.cu`: `LAUNCH_V12_MS3` + `RERUN_V12_MS3`,
   dispatcher version=51.
5. Add kTiles[] entries.

#### P2.3 Tier 3 — full M-sweep bit-compare

Per Tier 3 template. ≥ 8 M values × full grid.

#### P2.4 Tier 4 — performance characterization

1. **ncu at M=2048 AND M=4096**. **Target metric**:
   `long_scoreboard.pct` must drop vs §6.1 (or §6.0) baseline. If it
   does NOT drop → revert.
2. **Register-pressure tracking** — build a per-shape table:
   ```
   shape         | regs/thread | within budget? | 3-stage candidate?
   --------------+-------------+----------------+-------------------
   64x256x16_w8  |   ~80       | YES            | YES
   128x128x16_w4 |   ~110      | YES (FP16 acc) | YES
   128x256x16_w8 |   ~90       | YES            | YES
   192x128x16_w4 |   ~140      | NO             | NO
   ```
3. **Grid sweep ≥ 12 shapes**, both 2-stage AND 3-stage variants. Output:
   `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-2.csv`.
4. **CUTLASS ratio** at the new per-M champion.

#### P2.5 Tier 5 — Nsight Systems timeline

MANDATORY for §6.2 (overlap is the whole point). ncu can't distinguish
"long_scoreboard drop from smaller K-tile" from "long_scoreboard drop from
real LDG↔mma overlap." Only nsys can.

#### P2.6 Per-shape dispatcher

Encode the 2-stage vs 3-stage choice per shape in kTiles[]. The
dispatcher chooses by tile-shape match.

#### Decision gate P2

Per the global no-skip rule (§1.2). Per-shape: ship 3-stage for that shape
if all 10 items hold for that shape's measurements.

**ETA:** 6–10 hr.

---

### Phase P3 — §6.3: SplitK port to v12 (or v11 if P1 abandoned)

**Goal:** close the small-M production gap. v11 at M=64 = 7.52 TF;
v10s_ks8 = 20.31 TF. Target: v12s at M=64 ≥ 20 TF; ≥ 30 TF at M=256.

**Atomic correctness model**: fp32 scratch + atomicAdd into C scratch +
separate fp32→fp16 epilogue kernel. **NO single-kernel "last grid.z slice
does the cast" trick** — sm_70 has no grid-wide global ordering guarantee
without `cooperative_groups::grid_group`.

#### P3.1 Tier 1 — isolated SplitK correctness

1. Create `tests/test_v12s_splitk_sm70.cu`. M=64 N=K=128, KSPLIT=4.
   CPU-reference compares atomic-add accumulation against serial sum.
2. `compute-sanitizer --tool memcheck` AND `--tool racecheck` AND
   `--tool initcheck` on first launch. **All three mandatory** for atomic
   kernels.
3. **Adversarial KSPLIT factors**: test KSPLIT=2, 4, 8, 16, AND
   non-power-of-two (KSPLIT=3, 5) to surface tile-boundary bugs.
4. **Repeated-launch initcheck**: ensure scratch buffer is reset between
   launches.

**Decision gate P3.1**: all sanitizers clean + bit-exact across KSPLIT ∈
{2..16, plus 3, 5} → proceed.

#### P3.2 Tier 2 — kernel variant

1. Fork `v10splitk_kernels.cuh` → `v12splitk_kernels.cuh` (or
   `v11splitk_kernels.cuh` if P1 abandoned).
2. Replace the v10 mainloop body with v12/v11 mainloop.
3. **grid.z carries KSPLIT**. Each CTA handles K_TILE × (K / (TILES_K
   × KSPLIT)) slice. Fp32 scratch C accumulation via atomicAdd.
4. **Separate fp32→fp16 epilogue kernel** launches after the gemm grid
   completes. Cast + STG fp16.
5. Wire into `launch_int8.cu`: `LAUNCH_V12S(b_m, b_n, b_k, w, ksplit)`
   + `RERUN_V12S`. Dispatcher version=60.
6. Add kTiles[] entries for KSPLIT ∈ {2, 4, 8, 16} at the small-M shapes.

#### P3.3 Tier 3 — full M-sweep bit-compare

Per Tier 3 template. Special focus on M ∈ {1, 8, 32, 64, 256} — the
SplitK target range. SplitK may have slightly higher `rel` due to wider
reduction tree; still must be ≤ 1e-3.

#### P3.4 Tier 4 — performance characterization

1. **ncu at M=64 AND M=256**. Metrics:
   - `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` (utilization).
   - `l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum` (atomic-collision
     proxy; should scale with KSPLIT but stay < some threshold).
   - `launch__waves_per_multiprocessor` (occupancy).
2. **KSPLIT × shape sweep ≥ 12 shapes**:
   ```
   M=64:  BM=64 BN=128 BK=16 W ∈ {2, 4} × KSPLIT ∈ {4, 8, 16}      (6 shapes)
   M=256: BM=64 BN=128 BK=16 W ∈ {2, 4} × KSPLIT ∈ {2, 4, 8}       (6 shapes)
   M=1: separate launch (single-row degenerate case)                (verify-only)
   ```
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-3.csv`.
3. **CUTLASS ratio at M=64** (note: CUTLASS Gemm70 not tuned for small-M;
   record as documentation only).
4. **Atomic-collision check**: at the highest KSPLIT, verify
   `l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum` doesn't dominate the
   total cycle count. If it does, that's the SplitK ceiling for that shape.
5. **Dispatcher mis-selection check**: verify v12s is NEVER chosen at M ≥
   1024 by the per-M dispatcher.

#### Decision gate P3

Per the global no-skip rule. Specifically also:
- v12s at M=64 ≥ 20 TF (parity with v10s_ks8).
- No regression > 2% at M ≥ 1024 (dispatcher should pick v12 there).

Else: keep v10s_ks8 as the M=64 fallback.

**ETA:** 4–6 hr (sanitizers + KSPLIT × shape sweep + epilogue-kernel
plumbing).

---

### Phase P4 — §6.4: Larger CTA tile with c_frag SMEM spill

**Goal:** push BM to 192 or 256 by partially spilling c_frag to SMEM.
Conditional on P1 (FP16 acc) succeeding — with FP32 acc, BM=192 needs
192 fp32 c_frag per lane; with FP16 acc, BM=192 needs 96 halves = 48
regs per lane.

**If P1 abandoned**: P4 reduces to a register-pressure investigation only
(no new kernel). Document findings, no ship.

#### P4.1 Tier 1 — isolated correctness

1. Create `tests/test_v12_bm192_spillrotate_sm70.cu`. Small M=192 N=K=128
   problem. Confirms the partial-SMEM-spill rotation puts the right
   values in the right c_frag slots.
2. `compute-sanitizer --tool memcheck` AND `--tool racecheck` (spill/load
   to SMEM is a shared-state operation, racecheck applies).
3. Bank-conflict-adversarial shapes: BM=192 with BN_PAD that creates
   pathological column-stride patterns.

**Decision gate P4.1**: bit-exact + 0 sanitizer errors → proceed.

#### P4.2 Tier 2 — kernel variant

1. Fork `v12_kernels.cuh` → `v12_bm192_kernels.cuh` OR add `LARGE_BM`
   template flag.
2. **Partial c_frag SMEM rotation**: between K-tiles, half c_frag stays
   in registers, half writes to SMEM. Next K-tile flips. Adds ~2 ld.shared
   + 2 st.shared per K-iter; saves on register-pressure inner-loop
   serialization.
3. **ptxas --verbose**: `launch__registers_per_thread` ≤ 96.
4. **SMEM footprint sanity**: `launch__shared_mem_per_block_static` ≤
   the per-CTA budget that preserves 2 CTAs/SM.
5. Wire into `launch_int8.cu`: `LAUNCH_V12_LBM(b_m, b_n, b_k, w)` +
   `RERUN_V12_LBM`. Dispatcher version=52.

#### P4.3 Tier 3 — full M-sweep bit-compare

Per Tier 3 template.

#### P4.4 Tier 4 — performance characterization

1. **ncu at M=2048 AND M=4096**. **Target metric**: `short_scoreboard.pct`
   must drop. If it rises → SMEM rotation is the new bottleneck → revert.
2. **SMEM footprint occupancy gate**: `launch__shared_mem_per_block_static`
   × CTAs/SM ≤ 96 KB. If a winning shape drops to 1 CTA/SM, it's a hidden
   regression — flag in REPORT-13.
3. **Grid sweep ≥ 12 shapes**:
   ```
   BM ∈ {128, 160, 192, 224, 256}
   BN ∈ {128, 256}
   BK = 16        (locked; BK=32 uniformly worse per sprint-017)
   W  ∈ {4, 8}
   ```
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-4.csv`.
4. **CUTLASS ratio** at the champion shape.

#### Decision gate P4

Per global rule.

**ETA:** 5–7 hr.

---

### Phase P5 — §6.5: PRMT-vectorized A-side load

**Goal:** vectorize fp32→half on A. Replace 4× `__floats2half_rn` per A
chunk with `__float22half2_rn` or inline PTX `cvt.rn.f16x2.f32x2`.
Expected: +0.5–2%.

**No exception to no-skip rule** despite the small expected lift.

#### P5.1 Tier 1 — isolated correctness

1. Create `tests/test_v12_a_prmt_sm70.cu`. Sequence of fp32 input → half
   output through both the old and new path. Bit-exact comparison.
2. `compute-sanitizer --tool memcheck`.
3. **Round-mode adversarial inputs**: values near half-resolution
   boundaries (e.g., 1024.5 vs 1024.499...) to catch a stray `.rz`
   instead of `.rn`.

**Decision gate P5.1**: new path bit-equivalent to old `__floats2half_rn`
within rounding equivalence (NOT bit-exact at the binary level for
intermediates, but produces the same final half value for all fp32 inputs
in the DSv4 activation range) → proceed.

#### P5.2 Tier 2 — production swap

1. In `v12_kernels.cuh` (or current champion), replace per-element
   `__floats2half_rn` with `__float22half2_rn` (or PTX cvt packed).
2. **SASS verification**: `cuobjdump --dump-sass` — confirm compiler
   emitted the vectorized cvt. If it didn't, inline PTX explicitly.

#### P5.3 Tier 3 — full M-sweep bit-compare

Per Tier 3 template.

#### P5.4 Tier 4 — performance characterization

1. **ncu at M=2048 AND M=4096**. Expected: `mio_throttle.pct` flat,
   `math_pipe_throttle.pct` mildly rises.
2. **Grid sweep ≥ 12 shapes** (same grid as P4; no-skip rule applies).
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-5.csv`.
3. **CUTLASS ratio**.

#### Decision gate P5

Per global no-skip rule. Even a marginal lever must drop its target
stall AND improve at ≥ 3 of 8 M values.

**ETA:** 2–3 hr.

---

### Phase P6 — §6.6: Multi-shape MoE validation [IN-SPRINT per user interview]

**Goal:** validate that the final champion isn't 7168×7168-specific. DSv4
MoE expert layers have diverse N/K combinations; the champion needs to
generalize.

#### P6.1 Tier 1 — shape catalog

1. Collect representative DSv4 layer dimensions:
   ```
   Shape catalog (N × K):
   - 7168 × 18944  (DSv4 expert FFN up-projection)
   - 18944 × 7168  (DSv4 expert FFN down-projection)
   - 2048 × 7168   (attention output projection)
   - 4096 × 4096   (attention QKV projection)
   - 7168 × 7168   (sprint-017 baseline shape)
   - 8192 × 8192   (square control)
   ```
2. Document in `tools/tc-grid/docs/V12-DESIGN.md`.

#### P6.2 Tier 3 — multi-shape sweep

1. Run the final champion (whatever it is post-P5) across all 6 shapes
   at M ∈ {1, 8, 32, 64, 256, 1024, 2048, 4096}.
2. Median-of-5 per cell.
3. Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-6.csv`.
4. Identify shapes where the champion underperforms by > 5% vs a
   secondary tile in the sweep. Document.

#### P6.3 Tier 4 — per-shape per-M champion table

1. For each (shape, M) cell, identify the optimal kernel template.
2. If the optimal kernel varies by shape, the dispatcher needs shape-aware
   logic. **Decision**: per-M champion is OK to ship; per-(M, shape)
   champion gets deferred to SPRINT-020 unless the perf delta is > 10%
   (in which case ship per-shape logic this sprint).

#### Decision gate P6

- ✅ Champion is within 5% of optimal across all 6 shapes → ship as is.
- ⚠️ Champion underperforms by > 10% on some shape → ship per-shape
  dispatch logic this sprint.
- ❌ Champion fails on some shape (correctness or > 20% regression) →
  treat as a correctness bug, debug before sprint close.

**ETA:** 3–5 hr.

---

### Phase P7 — Close-out (REPORT-13 + per-M dispatcher + memory updates)

#### P7.1 Per-M (and possibly per-shape) dispatcher

1. Build the final per-M champion table:
   ```
   M     | champion kernel              | TF       | vs v10  | vs CUTLASS
   ------+------------------------------+----------+---------+-----------
   1     | v12s_64x128x16_w4_ks16       | ?        | ?       | ?
   8     | v12s_64x128x16_w4_ks16       | ?        | ?       | ?
   32    | v12s_64x128x16_w4_ks8        | ?        | ?       | ?
   64    | v12s_64x128x16_w4_ks8        | ?        | ?       | ?
   256   | v12s or v12 (TBD)            | ?        | ?       | ?
   1024  | v12_128x128x16_w4_ms3        | ?        | ?       | ?
   2048  | v12_128x128x16_w4            | ?        | ?       | ?
   4096  | v12_128x128x16_w4            | ?        | ?       | ?
   ```
2. Encode the dispatch rule in `launch_int8.cu` as a per-M switch.
3. **Threshold-adjacent verification**: explicitly test M ∈ {63, 65,
   255, 257, 1023, 1025} to confirm no dispatcher mis-selection at
   boundaries.
4. **Stale kTiles[] audit**: verify every kTiles[] entry that was used
   during the sprint is either (a) the chosen champion at some M or
   (b) explicitly marked as historical/reverted with a comment.

#### P7.2 REPORT-13.md

Mirror REPORT-12 structure:
1. Headline + per-M champion table.
2. Commit history.
3. Per-phase ncu stall breakdown (P0 baseline + P1–P6 deltas).
4. Lever-by-lever experiment log: which §6.x shipped, which reverted, why.
5. Grid-search outcomes — one summary table per phase + links to committed
   CSVs.
6. Key architectural insights (FP16-acc lane mapping if §6.1 succeeded,
   3-stage register-budget rule, SplitK fp32-scratch pattern, multi-shape
   findings).
7. CUTLASS ratio at every phase; final v12/CUTLASS gap.
8. **Headline 50 TF goal**: explicit pass/fail at M=2048. If fail,
   document ncu-evidenced rationale for the unclosable gap AND propose
   the architectural break for SPRINT-020.
9. What's left on the table (forward levers for SPRINT-020).

#### P7.3 Memory updates

- If §6.1 succeeded: append the empirical FP16-acc lane mapping AND the
  SMEM-round-trip workaround to
  `v100_wmma_half_float_frag_layout_mismatch.md`.
- New memory files as warranted: e.g., `v100_splitk_atomic_pattern.md`,
  `v100_3stage_register_budget_rule.md`.
- Update `MEMORY.md` index.

#### P7.4 Ledger

`scripts/ledger.py` does NOT exist (FOLLOWUPS-016). Skip; document in
REPORT-13.

#### P7.5 SPRINT-019-FOLLOWUPS.md

Capture items discovered during execution:
- Bugs found.
- Gaps in assumptions.
- New deferred items.

**ETA:** 3–4 hr.

---

## 5. Files Summary

### New files

| Path | Purpose |
|---|---|
| `tools/tc-grid/kernels/v12_kernels.cuh` | FP16-acc base kernel (§6.1) |
| `tools/tc-grid/kernels/v12_bm192_kernels.cuh` | Large-BM variant (§6.4) |
| `tools/tc-grid/kernels/v12splitk_kernels.cuh` | SplitK port (§6.3) |
| `tools/tc-grid/tests/test_mma_884_acc_f16_sm70.cu` | FP16-acc atom (§6.1) |
| `tools/tc-grid/tests/test_v12_ms3_sm70.cu` | 3-stage correctness (§6.2) |
| `tools/tc-grid/tests/test_v12s_splitk_sm70.cu` | SplitK correctness (§6.3) |
| `tools/tc-grid/tests/test_v12_bm192_spillrotate_sm70.cu` | Spill rotation (§6.4) |
| `tools/tc-grid/tests/test_v12_a_prmt_sm70.cu` | PRMT A-side (§6.5) |
| `tools/tc-grid/docs/V12-DESIGN.md` | FP16-acc lane mapping + epilogue + MoE shape catalog |
| `tools/tc-grid/docs/REPORT-13.md` | Sprint close report |
| `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-{1..6}.csv` | Per-phase grid sweep |
| `tools/tc-grid/docs/baseline-SPRINT-019-P0.csv` | P0 baseline |
| `tools/tc-grid/docs/ncu/SPRINT-019-P{0..6}-*.csv` | Per-phase ncu exports |
| `tools/tc-grid/docs/nsys/SPRINT-019-P{0,1,2}-*.{nsys-rep,png}` | Per-phase nsys |
| `scripts/bench-median.sh` | Median-of-5 measurement helper |
| `docs/sprints/SPRINT-019-FOLLOWUPS.md` | Discovered follow-ups |

### Modified files

| Path | Change |
|---|---|
| `tools/tc-grid/kernels/mma_sm70.cuh` | Wire `mma_m8n8k4_row_col_acc_f16`; add atoms as needed |
| `tools/tc-grid/kernels/v11_kernels.cuh` | (Conditional) port PRMT A-side from §6.5 back if v12 didn't ship |
| `tools/tc-grid/src/launch_int8.cu` | New LAUNCH/RERUN macros for v12, v12_ms3, v12_bm192, v12s; per-M dispatcher |
| `tools/tc-grid/src/main.cu` | New kTiles[] entries; cleanup of reverted entries |
| `tools/tc-grid/docs/ncu/.gitignore` | Exclude `.ncu-rep`, commit CSVs |
| `tools/tc-grid/docs/nsys/.gitignore` | Exclude `.nsys-rep`, commit PNGs |
| Memory file `v100_wmma_half_float_frag_layout_mismatch.md` | Append empirical FP16-acc mapping if §6.1 lands |

---

## 6. Definition of Done

Applied per phase AND at sprint close. **No phase-specific relaxations.**

### Per phase (every commit gate)

The 10-item rule from §1.2. Restated:

1. CPU-reference test passes (Tier 1).
2. `compute-sanitizer --tool memcheck` clean. Atomic kernels: `racecheck`
   + `initcheck` also clean.
3. Full M-sweep bit-compare against v10 passes at every M ∈ {1, 8, 32,
   64, 256, 1024, 2048, 4096}.
4. New kernel's `rel` per row ≤ prior champion's `rel` per row.
5. ncu at M=2048 AND M=4096 shows the target stall actually dropped.
6. Median-of-5 headline TF improved at ≥ 3 of 8 M values.
7. No M regresses by > 2% (median of 5).
8. CUTLASS Gemm70 ratio measured at M=2048 AND M=4096.
9. Grid sweep ≥ 12 tile shapes committed as CSV.
10. P1, P2, P4: Nsight Systems timeline captured + PNG committed.

Commit message includes: headline TF at M=2048 (median-of-5), target stall
before/after %, v12/CUTLASS ratio.

If any item fails: change is reverted from default dispatch, reason
logged in REPORT-13.

### Sprint close

1. **Headline 50 TF goal**: pass/fail at M=2048 with median-of-5 evidence.
   If fail: REPORT-13 contains ncu-evidenced rationale.
2. **M=64 goal**: v12s ≥ 20 TF OR v10s_ks8 remains the M=64 fallback.
3. **Per-M dispatcher** in `launch_int8.cu` encoding the champion rule;
   threshold-adjacent M values verified.
4. **Multi-shape robustness** (P6): champion verified across DSv4 shape
   catalog.
5. **REPORT-13.md** published.
6. **Memory updates** committed.
7. **SPRINT-019-FOLLOWUPS.md** captures discovered items.
8. **No new ptxas spill warnings** in any shipped kernel.
9. **All kTiles[] entries audited** (champion or marked historical).

---

## 7. Risks

### R1 — FP16 accumulator lane mapping undocumented (§6.1)
PTX ISA doesn't specify the f16-acc layout; differs from f32-acc.
**Mitigation**: P1.1 mandates empirical lane probing with adversarial
inputs. SMEM round-trip epilogue sidesteps register-resident promote.
**Severity**: HIGH (gates the biggest ROI lever).

### R2 — 3-stage asymmetric regression (§6.2)
Sprint-017 hit +9.5% on one shape, -29% on another.
**Mitigation**: Per-shape register-budget table; per-shape dispatcher;
global no-skip rule. nsys timeline mandatory to verify actual overlap.
**Severity**: MEDIUM.

### R3 — SplitK atomic correctness (§6.3)
Atomic-add into fp32 scratch introduces accumulator-order non-determinism.
**Mitigation**: `racecheck` + `initcheck` mandatory. Separate fp32→fp16
epilogue kernel (no single-kernel "last grid.z slice" trick).
**Severity**: LOW.

### R4 — Large BM register pressure (§6.4)
Conditional on §6.1 success.
**Mitigation**: Partial-SMEM-spill rotation. SMEM footprint occupancy gate.
ptxas spill output is hard gate.
**Severity**: MEDIUM.

### R5 — Grid sweep build-time explosion
≥ 12 tile shapes × 7 phases × multiple kernel families.
**Mitigation**: Per-phase incremental additions, not bulk dumps.
Build-time threshold: if > 5 min, prune obviously-bad shapes.
**Severity**: LOW.

### R6 — DCGM-exporter re-enabled
ncu unreliable if DCGM-exporter reads concurrently.
**Mitigation**: Pre-flight check in every ncu run (§2.5).
**Severity**: LOW.

### R7 — gpu-01 contention
Other workloads scheduling onto gpu-01 invalidates measurements.
**Mitigation**: `kubectl get pods -n llm -o wide` before each sweep.
gpu-02-4090rtx is NOT a substitute (qwen3-moe-rotorquant lives there).
**Severity**: LOW.

### R8 — Scope creep / time-box overrun
User chose no time-box on §6.1 + full 6-lever execution + 5-run median +
extended M-list. Realistic effort: 30–50 hr session-time.
**Mitigation**: Per-phase decision gates allow partial-ship. Sprint can
land with 3–4 levers shipped if §6.1 consumes excessive time.
**Severity**: MEDIUM.

### R9 — Measurement noise on 1–2% gates
Several decision gates depend on small percentage moves.
**Mitigation**: Median-of-5 per gate. If run-to-run variance exceeds
1%, escalate to median-of-9 for that phase.
**Severity**: LOW (mitigated by user-directed median-of-5).

### R10 — Benchmark harness drift
Adding many tile registrations could change selection behavior.
**Mitigation**: P0.7 git tag at sprint start; periodic re-baseline at
phase boundaries.
**Severity**: LOW.

### R11 — Dispatcher mis-selection at M-thresholds
Per-M dispatcher could pick the wrong kernel at threshold-adjacent M
values.
**Mitigation**: P7.1 explicit test of M ∈ {63, 65, 255, 257, 1023, 1025}.
**Severity**: LOW.

### R12 — Grid-search overfitting to 7168×7168
The champion may not generalize to DSv4 MoE shapes.
**Mitigation**: P6 multi-shape validation in-sprint.
**Severity**: LOW (mitigated by P6 inclusion).

### R13 — Partial correctness masked by aggregate rel
Particular tiles or lanes wrong while `rel` passes.
**Mitigation**: Tier-1 tests use adversarial inputs (identity, zeros,
sign-heavy, saturation-adjacent) per P1.1 step 4.
**Severity**: MEDIUM.

### R14 — SMEM-footprint occupancy regression
Fix one occupancy limiter (registers) but trip another (SMEM).
**Mitigation**: §2.4 canonical metric set includes
`launch__shared_mem_per_block_static` and
`launch__waves_per_multiprocessor`; explicit gate in P4.4.
**Severity**: MEDIUM.

### R15 — Atomic collision scaling for high KSPLIT
Atomic-add contention on C scratch grows with KSPLIT.
**Mitigation**: P3.4 measures
`l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum` per KSPLIT.
**Severity**: LOW.

### R16 — CUTLASS apples-to-oranges at small-M
CUTLASS Gemm70 not tuned for small-M; ratio at M=64 is documentation only.
**Mitigation**: P3.4 explicit note.
**Severity**: LOW.

### R17 — SASS codegen instability for PRMT A-side
Compiler may not emit `cvt.rn.f16x2.f32x2` even with vectorized intrinsic.
**Mitigation**: P5.2 explicit SASS verification via `cuobjdump`. Inline
PTX fallback.
**Severity**: LOW.

### R18 — Stale kTiles[] entries selected by accident
Reverted variants left in kTiles[] could be benchmarked or selected.
**Mitigation**: P7.1 stale-entry audit at sprint close.
**Severity**: LOW.

---

## 8. Security

Same surface area as sprint-017. Summary:
- No external network. Build offline; CUTLASS vendored.
- No credentials. kubectl via pre-authenticated kubeconfig.
- No data exfil. Synthetic distributions only.
- Sanitizer coverage: memcheck on every new kernel; racecheck + initcheck
  on atomic kernels.
- Inline-asm constraints: every new PTX wrapper has explicit operand
  constraints, validated by Tier-1 CPU-reference test.

---

## 9. Dependencies

### Hardware
- gpu-01 V100 32GB sm_70, sole V100 in cluster.
- `tcg-dev` pod in `llm` namespace, source at `/src/tools/tc-grid`.
- CUDA 12.2.2, driver pinned to sprint-017 version.

### Tooling
- `ncu` (Nsight Compute), `nsys` (Nsight Systems).
- `compute-sanitizer` with memcheck, racecheck, initcheck tools.
- `cuobjdump` for SASS inspection.
- `ptxas --verbose` for register/spill report.

### Code
- `_deps/cutlass-src/` at v2.11.0 (sprint-018 pinned).
- `kernels/v10splitk_kernels.cuh` as template for §6.3.
- `kernels/mma_sm70.cuh` (FP16-acc wrapper scaffolded).
- `tests/test_mma_884_tile_sm70.cu` as template for §6.1 atom test.

### Workflow
- Laptop → pod sync via rsync.
- DCGM-exporter pause before every ncu run.
- No `scripts/ledger.py` — skip.
- gpu-02-4090rtx unavailable (qwen3-moe-rotorquant).

---

## 10. Open questions (post-interview)

All four interview questions answered. Remaining open items:

1. **Dispatcher version-number allocation.** Reserved: 50 (v12), 51
   (v12_ms3), 52 (v12_bm192), 60 (v12s). Verify no collisions with
   sprint-018 version=40 (CUTLASS) before P1.2 wiring.
2. **REPORT-13 numbering**. Assumed REPORT-13 is the close report
   (matches REPORT-9/10/11/12 sequence). Confirm if any SPRINT-018
   follow-up doc has consumed that number.
3. **§6.1 fallback if abandoned**. If P1 abandons (lane mapping
   unrecoverable), P2/P3/P4 proceed with v11 as base. Decision: even
   without §6.1, the sprint may still hit 40–45 TF via §6.2 + §6.3 +
   §6.4 + §6.5. Re-evaluate the 50 TF goal at the P1 close-out, not
   sprint close.
4. **MoE shape catalog completeness**. The 6 shapes in P6.1 are a
   starting point; if DSv4 profiling reveals more critical shapes,
   add them mid-sprint.
5. **SPRINT-020 architectural break candidates** (preview, for REPORT-13).
   - Wholesale port of turbomind's sm_70 GEMM library.
   - Direct CUTLASS extension with custom dequant prologue (CUTLASS 2.x
     doesn't expose this cleanly; may require CUTLASS 3.x upgrade).
   - Accept the v11/v12 ceiling as the production answer and shift to
     deployment-integration work.

---

## 11. Estimated total effort

User chose: all 6 levers methodical, no time-box on §6.1, median-of-5,
extended M-list. Realistic effort:

| Phase | ETA | Notes |
|---|---:|---|
| P0 (reproduce + median-of-5 helper) | 2–3 hr | One-time setup |
| P1 (§6.1 FP16 acc) | **open-ended** | User accepts; realistic 8–24 hr |
| P2 (§6.2 3-stage) | 6–10 hr | per-shape register table + nsys |
| P3 (§6.3 SplitK) | 4–6 hr | sanitizers + KSPLIT × shape sweep |
| P4 (§6.4 large BM) | 5–7 hr | conditional on §6.1 |
| P5 (§6.5 PRMT A) | 2–3 hr | mechanical + SASS verify |
| P6 (§6.6 multi-shape) | 3–5 hr | 6 shapes × 8 M values × median-of-5 |
| P7 (close + dispatcher + report) | 3–4 hr |  |
| **Total** | **33–62 hr (~4–6 sessions)** |  |

---

## 12. Success / partial-success / failure summary

**Sprint succeeds if:**
- v12 (or successor) at M=2048 ≥ 50 TF, median-of-5, bit-correct.
- v12s at M=64 ≥ 20 TF.
- No M regresses > 2% vs sprint-017 champion.
- Multi-shape validation confirms champion generalizes.
- REPORT-13 published.

**Sprint partially succeeds if:**
- v12 at M=2048 lands in [40, 50) TF.
- Every phase's no-skip gates were applied (no skipping).
- REPORT-13 documents the ncu-evidenced rationale for the residual gap
  AND a proposed architectural break for SPRINT-020.
- v12s at M=64 ≥ 20 TF.

**Sprint fails if:**
- v12 at M=2048 < 35 TF (regression from sprint-017). Full revert; REPORT-13
  documents failure mode + re-attempt strategy.
- OR: any phase skipped without explicit user authorization (per
  `feedback_dont_skip_plan_steps`). Sprint methodology failure regardless
  of TF outcome.
