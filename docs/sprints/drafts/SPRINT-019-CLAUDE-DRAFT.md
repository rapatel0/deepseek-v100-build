# SPRINT-019 — Methodical v11 closure to 50 TF (REPORT-12 §6.1–6.5)

**Status:** DRAFT (independent plan, not yet executed).
**Predecessor:** SPRINT-017 — v11 closed at **35.08 TF M=2048**, bit-correct vs v10
(`rel = 2.594e-04`).
**Headline goal:** cross **50 TF at M=2048** AND reach **≥ 20 TF at M=64** by porting
SplitK to v11, both bit-correct (`rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`) and
ncu-evidenced.
**Discipline:** every phase has its own CPU-reference correctness test, compute-sanitizer
run, full M-sweep bit-compare against v10, ncu stall breakdown, CUTLASS-ratio measurement,
and grid sweep. **No phase is shipped if its target stall didn't drop or if the prior
champion regresses by >2% at any M.**

Cross-references:
- [SPRINT-019-INTENT.md](./SPRINT-019-INTENT.md) — sprint motivation, methodical
  discipline, deferred items.
- [REPORT-12.md](../../tools/tc-grid/docs/REPORT-12.md) — sprint-017 close, lever-by-lever
  log, six concrete §6.1–6.6 candidates.
- [SPRINT-017.md](../SPRINT-017.md) — v11 execution and lessons.
- [SPRINT-018-CUTLASS.md](../SPRINT-018-CUTLASS.md) — CUTLASS 85.93 TF pre-dequant ceiling.
- Memory: `v100_wmma_half_float_frag_layout_mismatch`,
  `feedback_pre_dequant_defeats_int8`, `feedback_dont_skip_plan_steps`,
  `feedback_effort_estimation_undocumented_hardware`.

---

## 1. Overview

V11 currently ships at **35.08 TF at M=2048** vs CUTLASS pre-dequant **85.93 TF**
ceiling. The gap is well-characterized in REPORT-12 §3: 21% long-scoreboard stalls
(unhidden HBM latency), 18% short-scoreboard (mma write→read latency), 15% mio-throttle
(SMEM bandwidth), and only **1.35% math-pipe throttle** — the tensor cores are idle
98.6% of the time when the math pipe is the bottleneck. We are **gmem-latency-bound,
not compute-bound**.

Six concrete levers are enumerated in REPORT-12 §6:

| § | Lever | Expected | Risk |
|---|---|---:|---|
| 6.1 | FP16 accumulator + SMEM round-trip epilogue | **+30–50%** | HIGH (undocumented lane mapping) |
| 6.2 | Per-shape 3-stage pipeline | +5–8% | MEDIUM (asymmetric regression in SPRINT-017) |
| 6.3 | SplitK port to v11 | +50–80% at M=64 | LOW (well-understood from v10s) |
| 6.4 | Larger CTA tile (BM=192/256) w/ c_frag SMEM spill | +5–10% | MEDIUM (register pressure) |
| 6.5 | PRMT-vectorized A-side load | +0.5–2% | LOW (mechanical) |
| 6.6 | Multi-shape MoE validation | orthogonal | — (deferred to SPRINT-020) |

This sprint executes §6.1–6.5 in **dependency order**, with a hard gate at every
phase. §6.6 stays out of scope per INTENT.

### 1.1 Headline targets

- **M=2048**: v11 successor at **≥ 50 TF**, bit-correct, ncu-evidenced.
- **M=64**: v11+SplitK at **≥ 20 TF** (parity with v10s_ks8 = 20.31 TF).
- **All M ∈ {64, 256, 1024, 2048, 4096}**: no regression > 2% vs the sprint-017
  per-M champion (v11 at M ∈ {256, 1024, 2048, 4096}; v10s at M=64).
- **CUTLASS ratio**: each shipped champion's TF / CUTLASS Gemm70 TF documented at
  M=2048 and M=4096.

### 1.2 Methodical discipline (applied at every phase boundary)

Borrowed verbatim from SPRINT-019-INTENT §"Methodical gates":

1. **CPU-reference** bit-correctness on a small isolated kernel test BEFORE any
   integration into `launch_int8.cu` / `main.cu`.
2. **`compute-sanitizer --tool memcheck`** on the first launch of every new
   kernel template.
3. **Full M-sweep bit-compare**: `tc-grid --m-list 64,256,1024,2048,4096 --nk 7168`
   against v10 reference: `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`.
4. **ncu stall breakdown** at M=2048 and M=4096 with the seven canonical metrics
   (§2.4).
5. **CUTLASS comparison** at M=2048: `int8_cutlass::Gemm70` (version=40, pre-dequant
   FP16 path) — report absolute TF and v11/CUTLASS ratio.
6. **Nsight Systems timeline capture** for any pipeline-restructuring change
   (§6.1, §6.2).
7. **Grid sweep** of **≥ 12 tile shapes** per phase, results in
   `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-N.csv`.
8. **Decision rule**: a change is shipped only if (a) target stall dropped per
   ncu, (b) headline TF improved at ≥ 3 of 5 M values, (c) no M regresses by
   > 2% vs prior champion. Otherwise it is **reverted**, with the reason logged.

---

## 2. Use cases

### 2.1 Production workloads addressed

- **DSv4 INT8 inference, generate-phase batches (M ∈ [1, 64])**: SplitK port
  (§6.3) closes the M=64 production gap. Current v11 is 7.52 TF at M=64; v10s_ks8
  is 20.31 TF. Goal: v11+SplitK at ≥ 20 TF.
- **DSv4 INT8 inference, prefill / large-batch (M ∈ [1024, 4096])**: FP16-acc
  v11 (§6.1) is the primary lever. Current v11 at 35.08 TF; goal ≥ 50 TF.
- **MoE expert dispatch (variable M per expert)**: addressed implicitly by
  per-M champion table. Multi-shape MoE validation is deferred to SPRINT-020
  (§6.6 in REPORT-12).

### 2.2 Non-production use cases

- **Kernel-development infrastructure**: every new isolated correctness test
  (`tests/test_<phase>.cu`) is reusable for future kernel work. The 884 atom
  fragment-layout testbed (`tests/test_mma_884_tile_sm70.cu`) gets a FP16-acc
  sibling in §6.1.
- **CUTLASS ratio tracking**: each phase produces a CUTLASS-vs-v11 ratio,
  shrinking the "how much of the 85 TF ceiling have we captured" gap toward
  the architectural break decision.

### 2.3 Out of scope

- **§6.6 multi-shape MoE sweep**: orthogonal; depends on DSv4 layer-dim profile
  not yet collected. Defer to SPRINT-020.
- **MoE-aware dispatcher integration into live DSv4 inference**: a separate
  systems-integration sprint, not a kernel-tuning task.
- **INT4 BN=256 spill fix** (FOLLOWUPS-016): only re-considered if §6.4 succeeds
  AND time permits.
- **v5 persistent-CTA revisit** (DEFERRED-016): only re-considered if §6.1 FP16
  acc opens occupancy headroom.
- **Turbomind `gemm_bench` standalone build**: 1–2 days build engineering for a
  sanity check already partially answered by CUTLASS Gemm70. Defer.

### 2.4 ncu metrics — the canonical set (used in every phase)

All measurements at the production shape (N=K=7168), `uniform_small`
distribution, M ∈ {2048, 4096}:

```
--metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
          smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
          smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,\
          smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,\
          smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
          sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
          l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
          l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
          l1tex__t_sector_hit_rate.pct,\
          launch__registers_per_thread,\
          launch__shared_mem_per_block_static
```

These are exported to CSV per-phase as
`tools/tc-grid/docs/ncu/SPRINT-019-PHASE-N-Mxxxx.csv`. The binary `.ncu-rep`
is gitignored; the CSV is committed.

---

## 3. Architecture

### 3.1 Current state (sprint-017 close)

```
v11<128, 128, 16, 4, 8, 2>
├── SMEM layout: B col-major (n outer, k inner), BK_PAD = BK + 8 = 24 halves
│                A row-major,  AM_PAD = AM + 8 = 136 halves
├── Mainloop: 2-stage (LDG → STS → mma), pipelined via cp.async-style
│             prefetch (no actual cp.async on sm_70)
├── mma atom: m8n8k4_row_col, FP32 accumulator (c_frag = float[8] per lane)
│             per-K-tile: 2× back-to-back mma (K=0..3, K=4..7) accumulating
│             into the same c_frag
├── Dequant: PRMT bias-trick (XOR 0x80 + prmt.b32 0x64646464 + sub 1152.0f)
│            half2 vectorized; in-SMEM (gmem stays INT8)
└── Epilogue: c_frag fp32 → cvt.rn.f16 → STG fp16
```

Sprint-017 baseline at M=2048: **35.08 TF**, `rel = 2.594e-04`, ncu stall
breakdown:
- long_scoreboard 21.45%, short_scoreboard 18.45%, mio_throttle 15.56%,
  math_pipe_throttle 1.35%.

### 3.2 Target architecture (sprint-019 close)

Two kernel families coexist:

```
v12<BM, BN, BK, W, ATOMS_M, ATOMS_N>     (large-M champion)
├── FP16 accumulator (c_frag = half[4] per lane, register pressure halved)
├── SMEM round-trip epilogue (sidesteps f16/f32 lane→element mismatch;
│                              see v100_wmma_half_float_frag_layout_mismatch memory)
├── Optional 3-stage pipeline (§6.2; dispatched per shape via register-budget
│                              decision)
├── Optional larger BM (§6.4; partial c_frag SMEM spill rotation)
└── PRMT-vectorized A-side fp32→half conversion (§6.5; __float22half2_rn or
                                                  cvt.rn.f16.f32)

v12s<BM, BN, BK, W, ATOMS_M, ATOMS_N, KSPLIT>     (small-M champion)
├── v12 mainloop replicated grid.z times
├── Each CTA handles K_TILE * KSPLIT slice; atomic-add into C tile in fp32
│   scratch, then a separate fp32→fp16 epilogue
└── KSPLIT ∈ {2, 4, 8, 16}; sweep per phase 3
```

`v11` (sprint-017) stays in the dispatcher as the FP32-acc fallback in case
§6.1 fails its correctness gate. v10 remains as the deepest fallback. The
dispatcher gains a per-M champion lookup table once §6.3 lands SplitK.

### 3.3 Why the dependency order is §6.1 → 6.2 → 6.3 → 6.4 → 6.5

1. **§6.1 first** because FP16 acc halves c_frag register pressure, which is
   the prerequisite for §6.2 (3-stage pipeline regressed at large-M because
   the rmem buffer overflowed register budget) AND §6.4 (larger BM hits
   c_frag register-lifetime limit). Without §6.1, §6.2 and §6.4 are largely
   blocked.
2. **§6.3 (SplitK) can be done in parallel** — it touches a fresh kernel file
   (`v11splitk_kernels.cuh` or `v12splitk_kernels.cuh` once §6.1 lands).
   We schedule it after §6.1 so it inherits the FP16-acc base if §6.1
   succeeds, but it's checkpoint-able to start earlier if §6.1 stalls.
3. **§6.4 and §6.5 are tail levers** — neither is expected to be more than
   ~10% individually. They land after the §6.1/§6.2 register-budget picture
   is settled.

---

## 4. Implementation

Phases below use **P0 (reproduce)**, **P1–P5 (one per §6.1–6.5)**, and
**P6 (close)** numbering. Every phase has the same five-tier verification
structure (see §1.2 and the per-phase Verification sub-sections).

### Phase P0 — Reproduce baseline; confirm harness; lock the comparison set

**Goal:** before changing any kernel, prove the harness reproduces sprint-017
numbers on the live pod. Lock the comparison set (v10, v11, CUTLASS Gemm70)
that every phase will be measured against.

**Pre-conditions:**
- `tcg-dev` pod live on gpu-01 (`kubectl get pods -n llm tcg-dev`).
- CUDA 12.2.2 in pod (`nvcc --version`).
- DCGM-exporter paused on gpu-01 (`kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'` returns `paused`).
- `_deps/cutlass-src/` populated from SPRINT-018 FetchContent.

**Steps:**

1. **P0.1 Build clean.** `cmake -B build && cmake --build build -j 8` in
   `tools/tc-grid` on the pod. Record build time as a baseline (regressions
   matter for inner-loop iteration speed).
2. **P0.2 Reproduce v11 35.08 TF M=2048.** Run:
   ```
   ./build/tc-grid --m-list 64,256,1024,2048,4096 --nk 7168 --dist uniform_small \
                   | tee tools/tc-grid/docs/baseline-SPRINT-019-P0.csv
   ```
   Expected: `128x128x16_w4_v11` at M=2048 within ±2% of 35.08 TF, `rel = 2.594e-04`.
   If not: stop. Investigate before any kernel change.
3. **P0.3 Record v10 + v10s + CUTLASS baselines.** Same command also captures
   v10 (29.50 TF M=2048), v10s_ks8 (20.31 TF M=64), CUTLASS Gemm70 version=40
   (85.93 TF M=2048). All four go in
   `tools/tc-grid/docs/baseline-SPRINT-019-P0.csv`.
4. **P0.4 Capture baseline ncu** for v11 at M=2048 and M=4096 using the
   canonical metric set (§2.4). Output:
   `tools/tc-grid/docs/ncu/SPRINT-019-P0-v11-M2048.csv` and `-M4096.csv`.
5. **P0.5 Capture baseline Nsight Systems trace** for v11 at M=2048
   (`nsys profile --capture-range cudaProfilerApi --trace cuda,nvtx ... `).
   Output: `tools/tc-grid/docs/nsys/SPRINT-019-P0-v11-M2048.nsys-rep`
   (gitignored; export PNG to `tools/tc-grid/docs/nsys/SPRINT-019-P0-v11-M2048.png`,
   commit the PNG).

**Decision gate P0:**
- ✅ baseline reproduced → proceed to P1.
- ❌ baseline drift > 2% → stop. Investigate (pod env, driver, DCGM-exporter
  not actually paused, etc.).

**ETA:** 1–2 hr (mostly waiting for the sweep).

---

### Phase P1 — §6.1: FP16 accumulator + SMEM round-trip epilogue [HIGH RISK]

**Goal:** halve c_frag register pressure (128 fp32 → 64 fp16 = 32 regs per lane)
and unlock the **2× tensor-pipe ceiling** (FP16-acc m8n8k4 is 125 TF peak on V100
vs 62 TF for FP32 acc). Expected post-phase TF: **45–50 TF at M=2048**.

**Risk:** the m8n8k4 lane→element mapping for FP16 accumulator is undocumented
in PTX ISA docs and differs from FP32 acc. Memory file
`v100_wmma_half_float_frag_layout_mismatch.md` records the v9 failure mode
(register-resident promote corrupted output; `rel = 0.71` across all variants).
**Mitigation: an isolated CPU-reference correctness test BEFORE any integration.**

#### P1.1 Pre-flight (correctness scaffolding — Tier 1)

1. **Build `tests/test_mma_884_acc_f16_sm70.cu`** as a sibling of
   `tests/test_mma_884_tile_sm70.cu`. Inputs: small fixed A (8×8 half), B (8×4
   half packed), expected C (8×4 half from CPU reference). One CTA, one warp,
   one mma call.
2. **Empirically derive the FP16-acc `thread_offset_C` and `static_offset_C`
   formulas** by running the kernel with known A=I, B=I, C=0 and reading out
   c_frag per lane. The FP32-acc formulas (`thread_offset_C = ((lane&1) +
   (lane/16)*4, (lane&2) + (lane&12)*2)`, `static_offset_C = {(0,0), (2,0),
   (0,4), (2,4)}`) may or may not transfer. **Do NOT assume they do.**
3. **`compute-sanitizer --tool memcheck`** on first launch. Expected: 0 errors.
4. **Bit-compare CPU reference** vs GPU output. Tolerance: bit-exact for
   identity matrices; `rel ≤ 1e-3` for the small uniform_small distribution.
5. **Document the derived FP16-acc layout** in
   `tools/tc-grid/docs/V12-DESIGN.md` (new file). Update memory file
   `v100_wmma_half_float_frag_layout_mismatch.md` with the empirical mapping
   if (and only if) Step 4 passes.

**Decision gate P1.1:**
- ✅ FP16-acc lane mapping figurable + bit-exact → proceed to P1.2.
- ❌ Lane mapping cannot be made bit-exact within **8 hr** (per the 3× effort
  multiplier for undocumented hardware work) → abandon §6.1, jump to P2.
  Document the abandonment in `tools/tc-grid/docs/V12-DESIGN.md` and
  REPORT-13.

#### P1.2 v12 kernel skeleton (Tier 2 integration)

1. **Fork `v11_kernels.cuh` → `v12_kernels.cuh`.** Rename template
   `mm_int8_lut_v12<BM, BN, BK, W, ATOMS_M, ATOMS_N>`.
2. **Swap c_frag type:** `float c_frag[ATOMS_M][ATOMS_N][8]` →
   `half c_frag[ATOMS_M][ATOMS_N][4]` (4 halves per lane per atom; lane mapping
   per P1.1).
3. **Swap mma wrapper:** `mma_m8n8k4_row_col` (f32 acc) →
   `mma_m8n8k4_row_col_acc_f16` (already scaffolded in `mma_sm70.cuh`).
   Verify the `+r` constraint on the 2-uint32 (= 4-half) C arg.
4. **Epilogue: SMEM round-trip.** Allocate `__shared__ half c_scratch[BM][BN]`
   (or per-warp tile to limit SMEM). Lanes scatter c_frag halves to c_scratch
   per the lane mapping. `__syncthreads()`. Reload c_scratch as fp16 via
   plain `__ldsh` (or fp32 via on-the-fly `__half2float` + `__ldsh` pair) and
   STG-write to gmem. This sidesteps the register-resident f16/f32 promote
   that bit `v9`.
5. **Register-budget verification.** `ptxas --verbose` output: expect
   ≤ 96 regs/thread (vs v11's 128). If > 128, the SMEM-scratch path saved
   nothing — investigate.
6. **`compute-sanitizer --tool memcheck`** on first launch with the production
   shape (M=2048, N=K=7168).
7. **Wire into `launch_int8.cu`:** add `LAUNCH_V12(b_m, b_n, b_k, w)` +
   `RERUN_V12` macros. Add `s.version == 50` dispatch.
8. **Add tile entries to `main.cu` kTiles[]**: start with the v11 champion
   shape `(128, 128, 16, 4, 8, 2, v12)` for the first sweep.

#### P1.3 Tier 2 correctness — full M-sweep bit-compare

1. Build clean. Record any new ptxas spill warnings.
2. Run `./build/tc-grid --m-list 64,256,1024,2048,4096 --nk 7168
   --dist uniform_small | grep v12`.
3. Tolerance gate per row: `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`.
4. **Regression check against v11 champion:** every row's `rel` against v10
   reference must be ≤ v11's `rel` for the same row (FP16 acc can shift
   accumulator-order noise; this is fine if within tolerance, but document it).

**Decision gate P1.3:**
- ✅ All five M values pass tolerance → proceed to P1.4.
- ❌ Any M fails → revert to v11. Investigate via maxabs and p99 first to
  distinguish accumulator-order noise (acceptable) from a real bug
  (unacceptable).

#### P1.4 Tier 3 performance — ncu + CUTLASS + grid sweep

1. **ncu stall breakdown at M=2048, M=4096** using §2.4 metrics. Export to
   `tools/tc-grid/docs/ncu/SPRINT-019-P1-v12-Mxxxx.csv`.
2. **Compare to baseline (P0.4 v11):**
   - `math_pipe_throttle.pct`: expect to **rise** (proxy for tensor cores
     working harder).
   - `long_scoreboard.pct`: expect modest drop (FP16 acc cuts c_frag writes
     in half).
   - `short_scoreboard.pct`: expect drop (shorter mma latency on FP16-acc atom).
   - `launch__registers_per_thread`: expect ≤ 96 (vs v11 128).
3. **CUTLASS ratio:** measure `int8_cutlass::Gemm70` version=40 at the same
   shape (already in dispatcher). Report v12 / CUTLASS ratio at M=2048 and
   M=4096.
4. **Grid sweep (≥ 12 tile shapes).** Sweep the same v11-family grid:
   ```
   BM ∈ {64, 96, 128, 192, 256}
   BN ∈ {128, 256}
   BK ∈ {16, 32}              (BK=16 has been the consistent winner; keep 32 as control)
   W  ∈ {2, 4, 8}
   constrained: N_PER_WARP ≥ 32, BM = 8k, ATOMS_M*ATOMS_N within register budget
   ```
   Minimum 12 valid shapes registered in `main.cu`. Run full M-sweep for each.
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-1.csv`.
5. **Identify per-M champion.** Build per-M table in
   `tools/tc-grid/docs/V12-DESIGN.md`.

#### P1.5 Tier 4 — Nsight Systems timeline

1. `nsys profile --capture-range cudaProfilerApi` over one v12 invocation at
   M=2048.
2. Inspect timeline: confirm LDG and mma instruction blocks **overlap** in
   the SM trace (not just adjacent). Save `.nsys-rep` + screenshot PNG to
   `tools/tc-grid/docs/nsys/SPRINT-019-P1-v12-M2048.{nsys-rep,png}`.

#### Decision gate P1 (ship or revert)

Ship v12 as the new champion if **all** of:
1. Bit-correct at all 5 M values.
2. Headline TF improved at ≥ 3 of 5 M values vs v11.
3. No M regresses > 2% vs v11.
4. ncu shows `math_pipe_throttle.pct` rose AND `short_scoreboard.pct`
   dropped at M=2048.
5. CUTLASS ratio improved vs P0 baseline.

Else: **revert** v12 from default dispatch (keep code as commented future
work), log reason in REPORT-13, proceed to P2.

**ETA:** 8–12 hr (apply 3× multiplier to the REPORT-12 5–8 hr estimate;
undocumented hardware path).

---

### Phase P2 — §6.2: Per-shape 3-stage pipeline [MEDIUM EFFORT]

**Goal:** add 3-stage mainloop (LDG → rmem → STS → SMEM → mma) for shapes whose
register budget can absorb the persistent rmem buffer. Sprint-017 hit 36.68 TF
on 64x256x16_w8 (+4.6%) but -29% on 128x128x16_w4 — asymmetric because of
register-budget overflow. With §6.1 cutting c_frag in half, the asymmetry
should narrow.

**Risk:** repeating the SPRINT-017 mistake — shipping without per-shape
register-budget verification. **Mitigation: the decision rule explicitly
requires ncu-verification AND the no-regression-by-2% gate.**

**Pre-condition:** §6.1 either succeeded (and v12 is the new base) or failed
gracefully (and v11 remains the base; 3-stage is added on top of v11).
**Branch in the plan**, not a hard prerequisite.

#### P2.1 Pre-flight (correctness scaffolding — Tier 1)

1. **Reuse the existing test harness.** No new isolated correctness test
   required — the 3-stage change is purely a mainloop rearrangement; the mma
   atom is unchanged. (If §6.1 succeeded and we're on v12, the test from
   P1.1 covers atom correctness.)
2. **Build the 3-stage variant as a separate template:**
   `mm_int8_lut_v12_ms3<...>` (or `mm_int8_lut_v11_ms3<...>` if P1 was
   abandoned). Persistent rmem `next_A_chunk`, `next_B_chunk` arrays per
   thread. Structure mirrors REPORT-12 §4.2 (the SPRINT-017 commented-out
   experiment).
3. **`compute-sanitizer --tool memcheck`** on first launch.

#### P2.2 Tier 2 — full M-sweep bit-compare

Same procedure as P1.3, but with `v{11,12}_ms3` template name.

#### P2.3 Tier 3 — ncu + grid sweep

1. **ncu stall breakdown at M=2048, M=4096.** Target metric:
   `long_scoreboard.pct` must **drop** vs the §6.1 (or §6.0) baseline on
   shapes where 3-stage is supposed to help. If it doesn't drop, the
   prefetch isn't actually overlapping → **revert this kernel**.
2. **Register-pressure tracking:** `launch__registers_per_thread` for every
   tile shape. Build a per-shape table:
   ```
   shape         | regs/thread | within budget? | 3-stage candidate?
   --------------+-------------+----------------+-------------------
   64x256x16_w8  |   ~80       |     YES        |       YES
   128x128x16_w4 |   ~110      |     YES (FP16) |       YES (was overflow on FP32)
   128x256x16_w8 |   ~90       |     YES        |       YES
   192x128x16_w4 |   ~140      |     NO         |       NO
   ...
   ```
3. **Grid sweep (≥ 12 tile shapes):** same grid as P1.4, but with both 2-stage
   AND 3-stage variants registered for each shape (so up to 24 entries in
   main.cu kTiles[]; the dispatcher picks per shape).
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-2.csv`.
4. **CUTLASS ratio** at the new per-M champion shape, M=2048 and M=4096.

#### P2.4 Tier 4 — Nsight Systems timeline

Same procedure as P1.5. Mandatory for §6.2 because the whole point is to
verify LDG↔mma overlap, which **only the timeline shows**. ncu can't
distinguish a "long_scoreboard drop because we found a smaller K-tile" from
"long_scoreboard drop because LDG and mma actually overlap in time."

#### P2.5 Per-shape dispatcher

Add to `launch_int8.cu`: per-shape choice of 2-stage vs 3-stage. Encode the
choice in the kTiles[] entry itself (e.g., `version=51` for v12_ms3,
`version=50` for v12). The dispatcher chooses by tile-shape match.

#### Decision gate P2

Per-shape ship decision:
- For each shape, if `long_scoreboard.pct` dropped AND TF improved at all M,
  ship that shape's 3-stage variant.
- For each shape, if either metric failed, ship the 2-stage variant for that
  shape.

**ETA:** 4–6 hr.

---

### Phase P3 — §6.3: SplitK port to v11/v12 [LOW EFFORT]

**Goal:** close the small-M production gap. v11 at M=64 is 7.52 TF; v10s_ks8
is 20.31 TF. Expected: v11+SplitK at ≥ 20 TF at M=64; ≥ 30 TF at M=256.

**Risk:** LOW. v10s SplitK is well-understood. The atomic-add into the C tile
needs fp32 scratch (cannot atomic-add fp16 portably on sm_70), so the
epilogue path is two-step: fp32 scratch accumulation, then a separate fp32→fp16
cast+STG.

**Pre-condition:** P1 has settled (either v12 or v11 is the base kernel).

#### P3.1 Pre-flight (correctness scaffolding — Tier 1)

1. **Build `tests/test_v12s_splitk_sm70.cu`** (or `test_v11s_splitk_sm70.cu`).
   Inputs: small M=64 N=K=128 problem, KSPLIT=4. CPU-reference compares the
   atomic-add accumulation against a serial reference.
2. **`compute-sanitizer --tool memcheck` AND `--tool racecheck`** on first
   launch. SplitK introduces atomic operations — racecheck is mandatory.
3. **`compute-sanitizer --tool initcheck`**: confirm the C scratch buffer is
   zero-initialized per launch (or per atomic-add semantics).

**Decision gate P3.1:**
- ✅ atomic-add bit-exact + zero races → proceed.
- ❌ any sanitizer error → fix before any production wiring.

#### P3.2 v12s (or v11s) kernel skeleton (Tier 2)

1. **Fork `v10splitk_kernels.cuh` → `v12splitk_kernels.cuh`.** Or
   `v11splitk_kernels.cuh` if P1 was abandoned.
2. **Replace the v10 mainloop body with the v11/v12 mainloop body**
   (m8n8k4 PTX path).
3. **Grid.z carries KSPLIT.** Each CTA handles K_TILE * (K / (TILES_K * KSPLIT))
   slice. KSPLIT ∈ {2, 4, 8, 16}; sweep in P3.4.
4. **Epilogue: atomic-add fp32 to C scratch, then global-sync barrier, then
   a second kernel (or grid.z==KSPLIT-1 only) does the fp32→fp16 cast + STG**.
5. **Wire into `launch_int8.cu`:** `LAUNCH_V12S(b_m, b_n, b_k, w, ksplit)` +
   `RERUN_V12S`. Dispatcher version=60.
6. **Add kTiles[] entries** for the SplitK shapes at M ∈ {64, 256}:
   `(64, 128, 16, 4, ..., ksplit=8)`, `(64, 128, 16, 4, ..., ksplit=4)`, etc.

#### P3.3 Tier 2 — full M-sweep bit-compare

Same procedure as P1.3, with v12s in the output rows. Tolerance same.
SplitK accumulates in fp32 then casts; expect `rel` slightly higher than the
straight v11 path due to wider reduction tree (still within `rel ≤ 1e-3`).

#### P3.4 Tier 3 — ncu + KSPLIT sweep + CUTLASS

1. **ncu at M=64.** Metrics: `sm__cycles_active.avg` (utilization),
   `l1tex__t_sector_pipe_lsu_mem_global_op_atom.sum` (atomic count),
   `launch__waves_per_multiprocessor` (occupancy).
2. **KSPLIT sweep** at M=64: KSPLIT ∈ {2, 4, 8, 16}. Pick winner per
   (BM, BN, BK, W) shape combo.
3. **Grid sweep (≥ 12 shapes total across KSPLIT and tile shape):**
   ```
   M=64:  BM=64 BN=128 BK=16 W ∈ {2, 4} × KSPLIT ∈ {4, 8, 16}     (6 shapes)
   M=256: BM=64 BN=128 BK=16 W ∈ {2, 4} × KSPLIT ∈ {2, 4, 8}      (6 shapes)
   plus 2 high-BM controls (BM=128 BN=128 W=4 × KSPLIT ∈ {2, 4}) (2 shapes)
   ```
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-3.csv`.
4. **CUTLASS ratio at M=64.** Note: CUTLASS Gemm70 isn't tuned for small-M;
   this comparison is for documentation only.

#### Decision gate P3

Ship v12s as the M ∈ {64, 256} champion if:
1. Bit-correct at all 5 M values.
2. v12s at M=64 ≥ 20 TF (parity target).
3. v12s at M=64 ≥ v10s_ks8 (20.31 TF).
4. No M ≥ 1024 regresses (v12s shouldn't be the dispatcher's choice there,
   but verify).

Else: keep v10s as the M=64 fallback in the dispatcher.

**ETA:** 3–5 hr (mechanical, but sanitizer + KSPLIT sweep add time).

---

### Phase P4 — §6.4: Larger CTA tile with c_frag SMEM spill [MEDIUM EFFORT]

**Goal:** push BM to 192 or 256 by spilling half the c_frag to SMEM and
rotating. Sprint-017 grid-sweep showed BM=192 W=4 at 32.87 TF with a small
spill; eliminating the spill via partial SMEM offload may unlock +5–10%.

**Pre-condition:** §6.1 succeeded. With FP32 acc, BM=192 needs 24 × 8 = 192
fp32 per lane in c_frag — well over 128-reg budget. With FP16 acc, it's
24 × 4 = 96 halves = 48 regs per lane — fits comfortably. **If §6.1 was
abandoned, P4 is reduced to a register-pressure investigation only, with no
new kernel.**

#### P4.1 Pre-flight (correctness scaffolding — Tier 1)

1. **Reuse the v12 atom test from P1.1.** The mma atom is unchanged; only the
   c_frag organization changes.
2. **Build a CPU-reference test** in
   `tests/test_v12_bm192_spillrotate_sm70.cu`. Small M=192 N=K=128. Confirms
   the partial-SMEM-spill rotation lands the right values in the right
   c_frag slots.
3. **`compute-sanitizer --tool memcheck`** on first launch.

#### P4.2 v12_bm192 kernel skeleton (Tier 2)

1. **Fork `v12_kernels.cuh` → `v12_bm192_kernels.cuh`** or add a `LARGE_BM`
   template flag.
2. **Partial c_frag SMEM rotation:** between K-tiles, half of c_frag stays
   in registers, half writes to SMEM. Next K-tile flips which half is
   resident. Increases instruction count by ~2 ld.shared + 2 st.shared per
   K-iter, but the saving is on register-pressure-induced inner-loop
   serialization.
3. **Register-budget verification.** Expect ≤ 96 regs/thread.
4. **Wire into `launch_int8.cu`:** `LAUNCH_V12_LBM(b_m, b_n, b_k, w)` +
   `RERUN_V12_LBM`. Dispatcher version=52.

#### P4.3 Tier 2 — full M-sweep bit-compare

Same procedure as P1.3.

#### P4.4 Tier 3 — ncu + grid sweep

1. **ncu at M=2048, M=4096.** Target metric: `short_scoreboard.pct` (the
   register-resident c_frag write→read dep) must **drop**. If it rises,
   the SMEM rotation is itself the bottleneck → revert.
2. **Grid sweep (≥ 12 shapes):**
   ```
   BM ∈ {128, 192, 256}
   BN ∈ {128, 256}
   BK = 16  (locked, BK=32 was uniformly worse in sprint-017)
   W  ∈ {4, 8}
   ```
   Output: `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-4.csv`.
3. **CUTLASS ratio** at the new champion shape.

#### Decision gate P4

Ship the large-BM variant if all of:
1. Bit-correct.
2. TF improved at M ∈ {1024, 2048, 4096} vs the P3 champion.
3. `short_scoreboard.pct` dropped vs P3 baseline.
4. M=64 and M=256 don't regress (the dispatcher should still pick v12s there,
   but verify).

Else: keep BM=128 as the canonical large-M shape.

**ETA:** 4–6 hr.

---

### Phase P5 — §6.5: PRMT-vectorized A-side load [LOW EFFORT]

**Goal:** vectorize the fp32→half conversion on A. Each thread currently does
4× `__floats2half_rn` per A chunk. Switch to `__float22half2_rn` (1 instr per
pair) or inline PTX `cvt.rn.f16.f32`. Expected: +0.5–2%.

**Risk:** LOW. Pure instruction-count optimization.

#### P5.1 Pre-flight (correctness scaffolding — Tier 1)

1. **No new isolated test needed.** The conversion is over a known-good
   input range (DSv4 activations, scaled). The existing harness's CPU
   reference matmul covers correctness.
2. **`compute-sanitizer --tool memcheck`** on first launch.

#### P5.2 Implementation (Tier 2)

1. **In `v12_kernels.cuh` (the current champion)**, replace the per-element
   `__floats2half_rn` calls in the A-side gmem→SMEM dequant with
   `__float22half2_rn(fp32_pair_struct)`. Alternative: inline PTX
   `cvt.rn.f16.f32` packed into a single uint32.
2. **Verify the SASS** via `cuobjdump --dump-sass` to confirm the compiler
   emitted the vectorized cvt instruction.

#### P5.3 Tier 2 — full M-sweep bit-compare

Same procedure. Tolerance: `rel ≤ 1e-3`. fp32→half rounding mode must stay
`rn` (round-to-nearest-even); a stray `.rz` would shift `rel`.

#### P5.4 Tier 3 — ncu + grid sweep

1. **ncu at M=2048**: `mio_throttle.pct` should not rise (the SMEM-store side
   of the dequant is unchanged). `math_pipe_throttle.pct` should mildly rise
   (more tensor cycles per total cycles).
2. **Grid sweep is optional at this phase** — but run it anyway for the
   final per-M champion table. **≥ 6 shapes** is sufficient given the
   marginal expected lift.
3. **CUTLASS ratio** at the champion shape.

#### Decision gate P5

Ship if:
1. Bit-correct.
2. TF improved at any M (even +0.5% counts; this is the tail of the
   optimization curve).
3. No regression > 0.5% at any M.

Else: revert (the change is mechanical; if it regressed, something is wrong
with the SASS — investigate before moving on).

**ETA:** 1–2 hr.

---

### Phase P6 — Close-out (REPORT-13 + per-M champion table + ledger)

**Goal:** consolidate the per-M dispatcher; write the sprint close report;
update memory files with new lessons.

#### P6.1 Per-M dispatcher

1. Build the final per-M champion table:
   ```
   M     | champion kernel           | TF        | vs v10  | vs CUTLASS
   ------+---------------------------+-----------+---------+-----------
   64    | v12s_64x128x16_w4_ks8     | ?         | ?       | ?
   256   | v12s_64x128x16_w4_ks4     | ?         | ?       | ?
   1024  | v12_128x128x16_w4_ms3     | ?         | ?       | ?
   2048  | v12_128x128x16_w4         | ?         | ?       | ?
   4096  | v12_128x128x16_w4         | ?         | ?       | ?
   ```
2. Encode the dispatch rule in `launch_int8.cu` as a per-M switch.
3. Verify the dispatcher with one final full-M sweep.

#### P6.2 REPORT-13.md

Mirror REPORT-12 structure:
1. Headline + per-M champion table.
2. Commit history.
3. Per-phase ncu stall breakdown (P0 baseline + P1–P5 deltas).
4. Lever-by-lever experiment log (which §6.x shipped, which reverted, why).
5. Grid-search outcomes — at minimum one summary table per phase
   referencing the committed CSVs.
6. Key architectural insights (FP16-acc lane mapping if §6.1 succeeded,
   3-stage register-budget rule, SplitK fp32-scratch atomic pattern).
7. CUTLASS ratio at every phase; final v12 / CUTLASS gap.
8. What's left on the table (forward levers for SPRINT-020).
9. Decision-rule status — pass/fail on the 50 TF goal.

#### P6.3 Memory updates

- If §6.1 succeeded: update
  `v100_wmma_half_float_frag_layout_mismatch.md` with the empirical FP16-acc
  lane mapping AND the SMEM-round-trip workaround.
- New memory file (if a new lesson surfaced): e.g.,
  `v100_splitk_atomic_pattern.md`, `v100_3stage_register_budget_rule.md`,
  etc.
- Verify all changed memory files appear in the MEMORY.md index.

#### P6.4 Ledger

INTENT.md notes `scripts/ledger.py` does NOT exist (per FOLLOWUPS-016). **Skip
this step.** Document in REPORT-13 that the ledger sync is deferred.

#### P6.5 Follow-ups

Write `SPRINT-019-FOLLOWUPS.md` capturing:
- §6.6 multi-shape MoE validation (for SPRINT-020).
- INT4 BN=256 spill fix (if §6.4 succeeded).
- v5 persistent-CTA revisit (if §6.1 succeeded and opens occupancy).
- Turbomind `gemm_bench` standalone (only if the sprint missed 50 TF and a
  third-party comparison is wanted).
- MoE-aware dispatcher integration into DSv4 inference.

**ETA:** 2–3 hr.

---

## 5. Files summary

### New files

| Path | Purpose |
|---|---|
| `tools/tc-grid/kernels/v12_kernels.cuh` | FP16-acc base kernel (§6.1) |
| `tools/tc-grid/kernels/v12_bm192_kernels.cuh` | Large-BM variant (§6.4) |
| `tools/tc-grid/kernels/v12splitk_kernels.cuh` | SplitK port (§6.3) |
| `tools/tc-grid/tests/test_mma_884_acc_f16_sm70.cu` | FP16-acc atom correctness (§6.1) |
| `tools/tc-grid/tests/test_v12s_splitk_sm70.cu` | SplitK atomic-add correctness (§6.3) |
| `tools/tc-grid/tests/test_v12_bm192_spillrotate_sm70.cu` | Partial SMEM spill (§6.4) |
| `tools/tc-grid/docs/V12-DESIGN.md` | Empirical FP16-acc lane mapping + epilogue design |
| `tools/tc-grid/docs/REPORT-13.md` | Sprint close report |
| `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-{1..5}.csv` | Per-phase grid sweep |
| `tools/tc-grid/docs/baseline-SPRINT-019-P0.csv` | P0 reproduce baseline |
| `tools/tc-grid/docs/ncu/SPRINT-019-P{0..5}-*.csv` | Per-phase ncu exports |
| `tools/tc-grid/docs/nsys/SPRINT-019-P{0,1,2}-*.png` | Per-phase Nsight Systems screenshots |
| `docs/sprints/SPRINT-019.md` | Final adopted sprint plan (this draft's successor) |
| `docs/sprints/SPRINT-019-FOLLOWUPS.md` | Deferred items |

### Modified files

| Path | Change |
|---|---|
| `tools/tc-grid/kernels/mma_sm70.cuh` | Wire `mma_m8n8k4_row_col_acc_f16` (already scaffolded); add atom variants as needed |
| `tools/tc-grid/src/launch_int8.cu` | New LAUNCH/RERUN macros for v12, v12_ms3, v12_bm192, v12s; per-M dispatch rule |
| `tools/tc-grid/src/main.cu` | New kTiles[] entries for v12 family; per-phase grid additions |
| `tools/tc-grid/docs/ncu/.gitignore` | Ensure `.ncu-rep` binaries excluded; CSVs committed |
| `tools/tc-grid/docs/nsys/.gitignore` | Ensure `.nsys-rep` binaries excluded; PNGs committed |
| Memory file `v100_wmma_half_float_frag_layout_mismatch.md` | Append empirical FP16-acc mapping if §6.1 lands |

---

## 6. Definition of Done

Applied per phase AND at sprint close.

### Per phase (every commit)

1. CPU-reference test passes (Tier 1).
2. `compute-sanitizer --tool memcheck` clean on first launch (Tier 1).
3. For SplitK: `--tool racecheck` and `--tool initcheck` clean.
4. Full M-sweep bit-compare against v10 reference passes
   (`rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`) at M ∈ {64, 256, 1024, 2048, 4096}.
5. ncu stall breakdown captured at M=2048 AND M=4096 (§2.4 metrics) and
   committed under `tools/tc-grid/docs/ncu/`.
6. CUTLASS Gemm70 (version=40) ratio measured at M=2048.
7. Grid sweep of ≥ 12 tile shapes (≥ 6 for §6.5) committed as
   `tools/tc-grid/docs/grid-sweep-SPRINT-019-PHASE-N.csv`.
8. For §6.1 and §6.2: Nsight Systems timeline captured and screenshot
   committed under `tools/tc-grid/docs/nsys/`.
9. Commit message includes: headline TF at M=2048, the target stall that
   moved (with before/after %), and the v12/CUTLASS ratio.
10. **If the target stall did NOT drop or the no-regression-by-2% gate
    failed**: change is reverted, decision logged in REPORT-13.

### Sprint close

1. **Headline 50 TF goal**: explicit pass/fail at M=2048. If fail, REPORT-13
   contains the ncu-evidenced rationale for why the gap is unclosable in
   this kernel family AND a proposed architectural break for SPRINT-020.
2. **M=64 goal**: v12s (or v11s) at M=64 ≥ 20 TF (parity with v10s_ks8) OR
   v10s_ks8 remains the M=64 dispatcher choice.
3. **Per-M champion table** in REPORT-13 with: kernel name, TF,
   Δ vs sprint-017 champion, v12/CUTLASS ratio.
4. **Dispatcher** in `launch_int8.cu` encodes the per-M champion rule and
   is verified by one final full-M sweep.
5. **REPORT-13.md** published with all sections from §P6.2 above.
6. **Memory updates** committed (any new V100-specific lesson).
7. **SPRINT-019-FOLLOWUPS.md** captures deferred items.
8. **No new ptxas spill warnings** introduced by any shipped kernel
   (existing 4-byte spill on v11_192x128 is accepted, but new spills require
   justification in REPORT-13).

---

## 7. Risks

### R1 — FP16 accumulator lane mapping is undocumented (§6.1)

**Description:** PTX ISA does not specify the lane→element layout of m8n8k4
with f16 accumulator. v9 (SPRINT-016) shipped a register-resident
f16/f32 promote and got `rel = 0.71` across all variants (memory:
`v100_wmma_half_float_frag_layout_mismatch`).

**Mitigation:**
- P1.1 mandates an isolated CPU-reference test with empirical lane probing
  BEFORE any integration.
- SMEM round-trip epilogue (not register-resident promote) sidesteps the
  layout mismatch even if the per-lane mapping is non-trivial.
- 8-hr time-box on figuring the lane mapping (3× multiplier on REPORT-12's
  5–8 hr estimate). Beyond that: abandon §6.1, proceed to P2.

**Severity:** HIGH (gates the biggest expected ROI lever).

### R2 — 3-stage pipeline asymmetric regression (§6.2)

**Description:** Sprint-017 shipped 3-stage without per-shape register-budget
verification and got +9.5% on one shape, -29% on the champion at M=4096.

**Mitigation:**
- P2.3 explicitly tabulates `launch__registers_per_thread` per shape and
  identifies 3-stage candidates by register budget.
- §6.1's FP16 acc halves c_frag, which is the prerequisite for 3-stage to fit
  on shapes where it previously overflowed.
- Per-shape dispatcher (P2.5) allows shipping 3-stage on candidate shapes
  AND 2-stage on non-candidate shapes simultaneously.

**Severity:** MEDIUM.

### R3 — SplitK atomic-add correctness (§6.3)

**Description:** atomic-add into fp32 scratch + separate fp32→fp16 epilogue
introduces accumulator-order non-determinism. `rel` may exceed v11's
`2.594e-04` baseline. If it crosses `1e-3`, the harness fails.

**Mitigation:**
- `--tool racecheck` mandatory in P3.1.
- Tolerance is `1e-3`, generous enough to absorb fp32-wide-reduction noise.
  v10s SplitK already passes this gate, so the pattern is known-good.
- If `rel` crosses `1e-3`, increase scratch precision (fp32 → double scratch,
  Kahan summation) before declaring failure.

**Severity:** LOW (well-trodden territory).

### R4 — Larger BM register pressure (§6.4)

**Description:** BM=192 needs careful c_frag layout to avoid spills.
Sprint-017 grid-sweep already showed 660-byte spill at 192x128_w4 on FP32 acc.

**Mitigation:**
- §6.4 is conditional on §6.1 succeeding (FP16 acc cuts c_frag in half).
- Partial-SMEM-spill rotation is a register-pressure-trade, not a spill
  acceptance. ptxas spill output is a hard gate.
- If §6.1 fails, §6.4 is reduced to a register-pressure investigation only.

**Severity:** MEDIUM.

### R5 — Grid sweep build-time explosion

**Description:** ≥ 12 tile shapes per phase × multiple phases × kTiles[]
template instantiations could push build time from ~30s to 5+ min per
build. Slows the inner-loop iteration speed.

**Mitigation:**
- Per-phase incremental kTiles[] additions, not bulk dumps.
- Build via `-j 8` in pod.
- If build time exceeds 3 min, prune obviously-bad shapes (BK=32 is
  uniformly worse per sprint-017) before committing.

**Severity:** LOW.

### R6 — DCGM-exporter re-enabled

**Description:** `ncu` runs are unreliable if DCGM-exporter is reading hardware
counters concurrently. SPRINT-017 risk R4 documents this.

**Mitigation:**
- Pre-flight check in P0:
  `kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'`.
  Must return `paused`.
- If re-enabled mid-sprint: `kubectl label --overwrite nodes gpu-01
  nvidia.com/gpu.deploy.dcgm-exporter=paused`.

**Severity:** LOW.

### R7 — gpu-01 contention with other workloads

**Description:** if any other pod schedules onto gpu-01, our perf numbers
become noise.

**Mitigation:**
- `kubectl get pods -n llm -o wide --field-selector spec.nodeName=gpu-01`
  before each `tc-grid` run.
- gpu-02-4090rtx remains occupied by `qwen3-moe-rotorquant`; do NOT migrate
  v100 work onto it per INTENT.

**Severity:** LOW.

### R8 — Scope creep / time-box overrun

**Description:** six phases × methodical discipline = 20–30 hr of session
time. The user explicitly demands no skipping.

**Mitigation:**
- Per-phase ETAs sum to ~22–34 hr (P0: 1–2; P1: 8–12; P2: 4–6; P3: 3–5;
  P4: 4–6; P5: 1–2; P6: 2–3).
- Phase decision gates allow partial shipping (e.g., if §6.1 abandons, the
  sprint still ships §6.2–§6.5 wins).
- Sprint goal allows a "documented rationale" fallback if 50 TF proves
  unreachable.

**Severity:** MEDIUM.

---

## 8. Security

This sprint is **kernel-tuning on a private V100 dev pod**. Surface area:

- **No external network access**: build is offline; CUTLASS already vendored
  in `_deps/cutlass-src/` from SPRINT-018.
- **No credentials introduced**: all kubectl access is via the existing
  pre-authenticated kubeconfig.
- **No data exfiltration risk**: all benchmarks use synthetic
  `uniform_small` distributions, no real model weights.
- **Sanitizer coverage**: every new kernel template runs `compute-sanitizer
  --tool memcheck` on first launch; SplitK additionally runs `racecheck`
  and `initcheck`. This is the primary defense against silent memory-safety
  bugs in hand-rolled PTX.
- **PTX wrappers** in `mma_sm70.cuh` use inline assembly. Each new wrapper
  must specify operand constraints (`+r`, `+f`, `r`, `f`) explicitly. The
  CPU-reference correctness gate (Tier 1) catches constraint-typo bugs that
  the compiler won't.
- **Atomic operations** in SplitK touch a shared C scratch buffer.
  Racecheck mandatory in P3.1.

No new attack surface vs sprint-017.

---

## 9. Dependencies

### Hardware / environment

- **gpu-01** (V100, 32 GB HBM2, sm_70). Sole V100 in cluster.
- **`tcg-dev` pod** in `llm` namespace. Source mounted at `/src/tools/tc-grid`.
- **CUDA 12.2.2** in pod (matches sprint-017 baseline; do not upgrade
  mid-sprint).
- **Driver**: whatever the node currently runs (sprint-017 used 535.x; do
  not upgrade).

### Tooling

- `nsight-compute` (`ncu`) and `nsight-systems` (`nsys`) available in pod.
- `compute-sanitizer` (memcheck, racecheck, initcheck tools) in pod.
- `cuobjdump` for SASS inspection.
- `ptxas --verbose` for register/spill report.

### Code dependencies

- `_deps/cutlass-src/` at SPRINT-018's pinned tag (v2.11.0). Reused for the
  CUTLASS-ratio comparison; not modified.
- `kernels/v10splitk_kernels.cuh` as template for §6.3 port.
- `kernels/mma_sm70.cuh` already has the FP16-acc wrapper scaffolded.
- `tests/test_mma_884_tile_sm70.cu` as template for §6.1 atom test.

### Workflow

- Laptop → pod sync via `rsync` to
  `ubuntu@192.168.102.5:/srv/dev/dsv4-cuda/deepseek-sprint017/...`
  (per INTENT — note path still says sprint017, leave as-is unless the user
  renames).
- DCGM-exporter pause before every `ncu` run.
- No `scripts/ledger.py` — skip ledger sync.

### Out-of-band

- **gpu-02-4090rtx** is occupied by `qwen3-moe-rotorquant` — do NOT evict.
- **No turbomind `gemm_bench` build** — defer to a follow-up sprint.

---

## 10. Open questions

1. **Time-box vs methodical exhaustion.** INTENT §"Open questions for
   interview" Q1 asks: execute all 5 in-scope levers (estimated 20–30 hr)
   OR cap at 2–3 levers and ship at 40 TF? This draft assumes "execute all
   5 methodically" with per-phase gates allowing early termination if any
   lever's correctness gate fails. **Confirm with user before P1.**
2. **§6.1 abandonment rule.** INTENT Q2 asks: does "methodical" preclude
   abandoning §6.1 mid-sprint if the FP16-acc lane mapping isn't figurable?
   This draft proposes an 8-hr time-box (REPORT-12 5–8 hr × 1.6 floor under
   the 3× undocumented-hardware multiplier), after which §6.1 is abandoned
   and we proceed to P2. **Confirm the time-box.**
3. **CUTLASS comparison depth.** INTENT Q3 asks: build turbomind's
   `gemm_bench` standalone (1–2 days) for an independent comparison? This
   draft defers per INTENT's "out of scope" list. **Confirm deferral.**
4. **SplitK sprint boundary.** INTENT Q4 asks: split §6.3 into its own
   sprint? This draft keeps it in-sprint because it's low-risk and the M=64
   parity goal is part of the sprint goal in INTENT. **Confirm.**
5. **Grid sweep size per phase.** INTENT Q5 asks: 12+ shapes per phase or
   ~6 per phase + one comprehensive sweep at sprint close? This draft uses
   12 for §6.1, §6.2, §6.4 (the structural changes) and 6 for §6.5 (the
   tail lever). §6.3 uses 12 across KSPLIT and shape combos. **Confirm
   per-phase counts.**
6. **Nsight Systems gating.** INTENT Q6 asks: nsys for every pipeline
   change or only §6.1 and §6.2? This draft mandates nsys for §6.1 and §6.2
   only (the two phases that restructure overlap pattern). **Confirm
   scope.**
7. **Dispatcher version-number allocation.** This draft reserves
   version=50 (v12), 51 (v12_ms3), 52 (v12_bm192), 60 (v12s). Conflicts
   with SPRINT-018's version=40 (CUTLASS) and sprint-017's existing version
   IDs? Verify against the current `launch_int8.cu` switch table.
8. **REPORT-13 ↔ INTENT naming.** The INTENT references "REPORT-13"; this
   draft consistently uses REPORT-13 for the close report. Confirm
   numbering doesn't collide with any SPRINT-018 follow-up doc.
9. **Memory file update protocol.** If §6.1 succeeds and we learn the
   empirical FP16-acc lane mapping, do we update the existing
   `v100_wmma_half_float_frag_layout_mismatch.md` (renaming it would be
   confusing since the file is referenced from INTENT and from this draft)
   or append a separate `v100_wmma_acc_f16_lane_mapping.md`? This draft
   assumes "append to existing." Confirm.

---

## 11. Estimated total effort

Per `feedback_effort_estimation_undocumented_hardware.md` — 3× multiplier
applies wherever the work involves opaque ISA mappings (§6.1), sticky build
caches, or commented-out upstream targets. Phases below already incorporate
the multiplier where appropriate.

| Phase | REPORT-12 estimate | Sprint-019 estimate (w/ multiplier) |
|---|---:|---:|
| P0 (reproduce) | — | 1–2 hr |
| P1 (§6.1 FP16 acc) | 5–8 hr | **8–12 hr** (3× on lane-mapping; remainder of phase ~5 hr) |
| P2 (§6.2 3-stage) | 3–4 hr | **4–6 hr** |
| P3 (§6.3 SplitK) | 1–2 hr | **3–5 hr** (sanitizer + KSPLIT sweep) |
| P4 (§6.4 large BM) | 4–6 hr | **4–6 hr** |
| P5 (§6.5 PRMT A) | 1 hr | **1–2 hr** |
| P6 (close) | — | **2–3 hr** |
| **Total** | **14–21 hr** | **23–36 hr (~3 sessions)** |

This is consistent with INTENT §"Uncertainty assessment" — "methodical
execution may take 2–3 weeks of session-time."

---

## 12. Success / partial-success / failure summary

**Sprint succeeds if:**
- v12 (or successor) at M=2048 ≥ 50 TF, bit-correct, ncu-evidenced.
- v12s at M=64 ≥ 20 TF, bit-correct.
- No M regresses > 2% vs sprint-017 champion.
- REPORT-13 published with per-M champion table and CUTLASS ratio.

**Sprint partially succeeds if:**
- v12 at M=2048 lands in [40, 50) TF AND every phase's gates were applied
  methodically (no skipping). REPORT-13 documents the ncu-evidenced
  rationale for the remaining gap AND a proposed architectural break for
  SPRINT-020.
- v12s at M=64 ≥ 20 TF.

**Sprint fails if:**
- v12 at M=2048 < 35 TF (regression from sprint-017 baseline). In this
  case: full revert to v11 sprint-017 state; REPORT-13 documents the
  failure mode and proposed re-attempt strategy.
- OR: any phase was skipped without explicit user authorization (per
  `feedback_dont_skip_plan_steps`). Sprint methodology failure, regardless
  of TF outcome.
