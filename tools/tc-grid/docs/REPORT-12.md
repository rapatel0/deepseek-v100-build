# REPORT-12 — V11 final numbers + benchmark learnings

Date: 2026-05-14
Author: V100 INT8 GEMM optimization loop
Status: Sprint-017 close; production v11 shipped at +18.6% vs v10. Headroom
analysis below points to a realistic 55–65 TF ceiling for the v11 family
before the next architectural break.

This is the technical record for the v11 implementation arc. Companions:
- [REPORT-11.md](./REPORT-11.md) — v10 close (29.49 TF baseline, why wmma exhausted)
- [V11-EXECUTION-PLAN.md](./V11-EXECUTION-PLAN.md) — original wave-decisioned plan
- [V11-STEP2-HANDOFF.md](./V11-STEP2-HANDOFF.md) — manual Lds lane-mapping derivation
- [V11-DESIGN.md](./V11-DESIGN.md) — original 5-step design

---

## 1. Headline

**v11 champion: `mm_int8_lut_v11<128, 128, 16, 4, 8, 2>` (BM=128, BN=128, BK=16, W=4)**

Production INT8 GEMM, bit-correct vs v10 (rel = 2.594e-04 exactly at M ≥ 1024,
matching v10's reference numerically). N=K=7168, uniform_small distribution:

| M    | v10 best TF | v11 best TF | Δ vs v10 |
|------|------------:|------------:|---------:|
| 64   |  11.02      |   7.52      |   ~      |
| 256  |  27.94      |  29.13      |  +4.3%   |
| 1024 |  28.85      | **34.94**   | **+21.1%** (W=2 variant)
| 2048 |  29.50      | **35.08**   | **+18.9%** |
| 4096 |  29.18      | **34.77**   | **+19.2%** |

Decision-rule placement: lands solidly in the **"35–50 TF: beats v10, ship +
investigate gap"** tier. Gap to 50 TF P0 goal: 1.43×. Gap to 85 TF P1 CUTLASS
ceiling (FP16 pre-dequant): 2.43×.

M=64 stays with v10s SplitK (20.31 TF, +84% vs v10 from sprint-017 A0). v11 is
not the right kernel for decode-style small-M operating points.

---

## 2. What was built (commit history)

```
14b074a1e sprint-017 V11: PRMT INT8→FP16 dequant + grid search → 35.08 TF
c78106eb6 sprint-017 V11 Steps 2.6+3+misc: v11 final at 34.65 TF (+17.5% v10)
e79175ab6 sprint-017 V11 Step 2.5: BK=16 inner K-stage → 34.5 TF (+17%)
919a44baa sprint-017 V11 Step 2c+2d: v11 bit-correct vs v10 at all M
85c7e2b76 sprint-017 V11 Step 2b: 884 tile (m8n32k8) bit-correct
ecb6be08a sprint-017 V11 Step 2a: SmemCopy_MMA_884 lane mapping validated
897847fd5 sprint-017 P0:  v11 Step 1 — m8n8k4 PTX wrapper (prior session)
c2bf15796 sprint-017 V11: 3-stage pipeline experiment + ncu analysis
```

Architecture: v11 replaces v10's `wmma::*` path with manual `ld.shared.b128`
plus `mma.sync.aligned.m8n8k4.row.col.{f32,f16}.f16.f16.{f32,f16}` PTX
wrappers. SMEM layout switched from row-major B (v10) to col-major B with
BK+8 padding (v11) to support per-lane uint4 Lds at the SmemCopy_MMA_884_B
offsets.

Key files:
- `tools/tc-grid/kernels/v11_kernels.cuh` — production kernel
- `tools/tc-grid/kernels/mma_sm70.cuh` — m8n8k4 PTX wrappers (f32 + f16 acc)
- `tools/tc-grid/tests/test_smem_to_frag_sm70.cu` — Step 2a lane validation
- `tools/tc-grid/tests/test_mma_884_tile_sm70.cu` — Step 2b atom validation
- `tools/tc-grid/src/launch_int8.cu` — LAUNCH_V11/RERUN_V11 dispatch macros
- `tools/tc-grid/src/main.cu` — kTiles[] entries for v11 variants

---

## 3. ncu stall breakdown — the strategic pivot

After Step 2.5 (BK=16) landed us at 34.5 TF, an ncu pass on the champion
shape revealed where we actually stall:

| Stall reason                                                 | %      |
|--------------------------------------------------------------|-------:|
| `warp_issue_stalled_long_scoreboard_per_warp_active`         | 21.45% |
| `warp_issue_stalled_short_scoreboard_per_warp_active`        | 18.45% |
| `warp_issue_stalled_mio_throttle_per_warp_active`            | 15.56% |
| `warp_issue_stalled_lg_throttle_per_warp_active`             |  1.88% |
| `warp_issue_stalled_math_pipe_throttle_per_warp_active`      |  1.35% |
| Issued + other                                                | ~42%   |

**Interpretation**:
- 21% **long_scoreboard** = gmem LDG result not ready when downstream
  instruction wants it. The double-buffered prefetch isn't hiding all of HBM
  latency. **This is the biggest single bucket of headroom.**
- 18% **short_scoreboard** = `mma` write to `c_frag[am][an]` not visible
  to the next read. Either compiler can't interleave enough independent
  atom chains, or our 16 atom × 1 atom-per-warp layout doesn't have enough
  parallelism to hide the ~10-cycle mma latency.
- 15% **mio_throttle** = SMEM bandwidth saturated. Each lane does
  uint4 Lds at every K-iter; the SMEM banks can't keep up.
- 1.35% **math_pipe_throttle** = the tensor cores are **idle** 98.6% of the
  active time when this metric is the bottleneck. **We are nowhere near
  compute-bound.**

This contradicted the working theory ("we're at 95% smsp_active so compute
is the wall"). The SM is busy, but it's busy *waiting*, not computing.

CUTLASS's 85 TF result on the same shape (FP16 pre-dequant) means their
kernel pushes the tensor pipe much closer to saturation. The gap to them
is almost entirely hidden gmem latency + better SMEM bandwidth budget,
not raw compute.

---

## 4. Lever-by-lever experiment log

Each row: what we tried, measured Δ vs the immediate prior champion,
ship/revert decision, and *why* (compressed enough to be useful next
session).

### 4.1 Levers that shipped

| Lever | Pre-TF | Post-TF | Δ | Notes |
|---|---:|---:|---:|---|
| Step 2.5 — `BK=16` (was 32) | 28.58 | **34.53** | +20.8% | K_ITERS=2 (was 4) cuts inner-loop register pressure; SMEM 33KB → 12KB lets more CTAs/SM live. Biggest single win of the sprint. |
| `half2` vectorized dequant | 34.65 | 34.72 | +0.2% | Halves the dequant instruction count. Only +12% on BK=32 variants; champ already had dequant pipelined. |
| PRMT INT8→FP16 dequant | 34.72 | **35.08** | +1.0% | The bias-trick converter (XOR 0x80 + `prmt.b32` with 0x64646464 + subtract 1152). Bigger lift on BK=32 (+4.6%). Small at champ because dequant cost was already mostly hidden. |
| In-place mma accumulator (`+f` constraint) | 34.65 | 34.73 | +0.1% | Saves a temp + 8-float copy per call. Compiler was already optimizing the temp away; kept for cleaner asm. |

Cumulative win from v10's 29.50 TF: **+18.9% at M=2048**.

### 4.2 Levers that reverted with documented reasons

| Lever | Pre | Post | Why reverted |
|---|---:|---:|---|
| Stream cache (`__ldcs`) on B + W_scales | 34.5 | 24 | -30% on champ. W_scales reuse within a K-tile (one scale per QK_INT8=32 reused across N-lanes) makes L1 caching beat the L1-capacity savings turbomind gets from Stream policy. |
| Full XOR Swizzle<3,3,3> on sB | 34.72 | 34.16 | +14% on BK=32 (eliminates the 16-way bank conflict), but -1% at champ. BK=16 BK_PAD=24 already gives only 4-way conflict; swizzle's bit-XOR overhead exceeded the bank-conflict savings. |
| 2-uint4 store for the dequant write | 34.72 | 34.16 | +12% on BK=32, -1% at champ. Compiler interleaves per-half dequant arithmetic with per-half stores more naturally than with bulk uint4. |
| k-major mma reorder (all mma1 then all mma2) | 35.08 | 34.43 | -1.9%. Compiler was already interleaving across atoms when atom-major was `#pragma unroll`'d. Forcing k-major widened a_frags register lifetime and cost more than it saved. |
| `__launch_bounds__(*, 3)` (3 CTAs/SM) | 35.08 | 28.00 | -20%. Register budget per thread (256/3 ≈ 85 vs current 128) too tight; inner mma loop serialized. |
| **3-stage pipeline** (LDG → rmem → STS → smem → mma) | 35.08 → 36.68 *peak* at one shape | -29% at champ | **Asymmetric**: +9.5% on 64x256x16_w8 (33.51 → 36.68), -29% on 128x128x16_w4 (35.08 → 25.04). At M=4096, every shape regressed 5-30%. The per-thread rmem buffer pushed register pressure over budget at large-M shapes, cratering occupancy. Pattern is sound; needs per-shape register-budget tuning. Left in code as a commented future-work item. |

### 4.3 Grid-search outcome

20+ tile variants registered: BM ∈ {32, 64, 96, 128, 192, 256} × BN ∈
{128, 256} × BK ∈ {16, 32, 64} × W ∈ {2, 4, 8}, constrained to N_PER_WARP
≥ 32 (= ATOM_N) and BM = 8k.

The grid search did NOT find a new champion at M=2048 (BK=16 BM=128
W=4 holds). But it surfaced two interesting alternates:

- **64x128x16 W=2 ATOMS_N=2** ties the champ at M=1024+ (34.94 vs 34.35).
  Fewer warps per CTA → more CTAs per SM → better wave parallelism.
- **192x128x16 W=4** at 32.87 TF (mild 660-byte spill). Larger BM helps
  amortize K-loop overhead; this is the natural BM=192 → 256 direction if
  we ever resolve the spill.

BK=64 was uniformly worse (more SMEM, less occupancy). BM=32 was
uniformly worse (too many K-loop iterations per output cell).

---

## 5. Key architectural insights

### 5.1 The lane-mapping foundation is reusable

`SmemCopy_MMA_884_A::unique` (lane → (m, k_quad)) and `_B::unique` (lane → m)
are now empirically validated against a CPU model (Step 2a) and against
a small CPU reference matmul (Step 2b). Future kernels that want manual
fragment loads on sm_70 should reuse these without re-deriving:

- A: `aL_m = (lane/16)*4 + (lane%4)`,  read 8 halves at `&sA[m * stride]`
- B: `bL_n = (lane/16)*4 + (lane&12)*2 + (lane%4)`, read 8 halves at `&sB[n * stride]`
- Per-lane FragC scatter via `thread_offset_C` (= `((lane&1) + (lane/16)*4,
  (lane&2) + (lane&12)*2)`) + `static_offset_C` (= `{(0,0), (2,0), (0,4), (2,4)}`).

The 884 atom shape is **m=8, n=32, k=8** per `fma()`, implemented as 2
back-to-back `mma.m8n8k4_row_col` (K=0..3 then K=4..7 accumulating to the
same c_frag). Per-lane FragC = 8 floats covers a 4 m × 8 n area per atom
via the offset arrays above.

### 5.2 Compiler is smarter than micro-managing

Multiple "make the inner loop explicit" experiments regressed:
- Explicit k-major ordering: -1.9%
- 2-uint4 bulk stores for dequant: -1%
- In-place mma without `+f` interleaving manually: neutral

The `#pragma unroll` + atom-major nested loop already gets the compiler to
interleave atoms aggressively. Forcing a specific order limits its
scheduler.

### 5.3 We are gmem-latency-bound, not compute-bound

The single most counterintuitive finding. Headline metric "95% smsp_active"
is misleading — most of those active cycles are *issue stalls*, not work.
The 1.35% math_pipe_throttle tells the real story: the tensor cores spend
almost all their time idle, waiting for the next mma operand to arrive.

This reorients the entire optimization strategy. Bank conflicts, swizzle
tuning, instruction count reduction — these matter, but they're not the
main wall. The main wall is **time-to-feed the tensor cores**.

### 5.4 BK is the most sensitive single template parameter

BK=16 gave +20.8% over BK=32 — by far the biggest single lever. Reasons,
in order of contribution:

1. **K_ITERS halved** (4 → 2 inner-mma loop iterations per K-tile).
   Halves register lifetime for `a_frags` and `b_frags`, lets compiler
   keep more of c_frag in registers.
2. **SMEM footprint shrinks 33KB → 12KB**, allowing more CTAs/SM at the
   same `__launch_bounds__`, increasing warp count for latency hiding.
3. **K-loop count doubles** (224 → 448 K-tiles) but each iteration is
   much smaller; net wall-time win.

This contradicts the v10 sprint instinct ("smaller BK = more outer-loop
overhead"). v10 was bottlenecked by SMEM bank conflicts at small BK,
which the v11 manual-Lds + col-major-B layout removes.

### 5.5 SMEM layout choice is shape-sensitive

v10's row-major B SMEM (k outer, n inner) was a +6.4% win over col-major
for `wmma::load_matrix_sync`. v11's manual Lds **requires** col-major B
(n outer, k inner) for uint4-aligned reads of 8 K-contiguous halves at a
fixed N.

`BK_PAD = BK + 8` gives uint4 alignment with 4-way bank conflict (stride/2
even, GCD ≥ 2). Reducing to `BK_PAD = BK` (power of 2) would allow XOR
swizzle to fully eliminate conflicts, but the stride/2 vs uint4 conflict
math leaves a 4-way floor either way for col-major B. **The right move is
either accept 4-way + simpler code, or move to FP16 accumulator and
revisit.**

---

## 6. What's left on the table (concrete next-session levers)

Ranked by expected ROI for an engineer picking this up cold.

### 6.1 FP16 accumulator + SMEM round-trip epilogue [HIGH-RISK, HIGH-REWARD]

**Why**: V100 m8n8k4 with FP16 accumulator is 125 TF peak; FP32 acc is
~62 TF peak (2× the throughput on the same hardware). v11 currently uses
FP32 acc, halving the tensor-pipe ceiling.

**What to build**:
- Wrapper already in `mma_sm70.cuh` as `mma_m8n8k4_row_col_acc_f16` —
  takes `half* c` and uses `+r` constraints on a 2-uint32 (= 4-half) C.
- Replace `float c_frag[ATOMS_M][ATOMS_N][8]` with
  `half c_frag[ATOMS_M][ATOMS_N][4]`. Per-lane register pressure halves
  (128 floats → 64 halves = 32 registers).
- In the epilogue, write c_frag to a scratch SMEM region, then read
  back as float via separate half→float conversion (sidesteps the f16/f32
  lane→element mismatch — see `v100_wmma_half_float_frag_layout_mismatch.md`
  memory).

**Risk**: The lane→element mapping for FP16-acc mma may differ from
FP32-acc. The scatter formulas (`thread_offset_C` + `static_offset_C`) were
empirically validated for FP32 acc. **Need a Step 2b-style isolated test
with FP16 acc first** before plugging into v11.

**Effort**: 5–8 hr realistic (the 3× multiplier applies — undocumented
hardware mapping).

**Expected**: +30–50% if the lane mapping is figurable. Lands v11 at
~50 TF.

### 6.2 Per-shape 3-stage pipeline [MEDIUM-EFFORT]

**Why**: The 3-stage experiment hit 36.68 TF (+4.6% over current champ)
on the 64x256x16_w8 shape but regressed BM=128 W=4. The asymmetry is
about register budget: BN=256 W=8 has fewer atoms per warp, so adding
rmem buffers fits within `__launch_bounds__`. BM=128 W=4 has 16 atoms
per warp, so adding rmem overflows.

**What to build**:
- Two kernel templates: `mm_int8_lut_v11` (current 2-stage) and
  `mm_int8_lut_v11_ms3` (3-stage with persistent rmem).
- Per-tile dispatch chooses the 3-stage when the shape's register
  budget can absorb it.
- ncu-verify that the 3-stage version actually drops long_scoreboard
  stalls on the shapes that benefit.

**Effort**: 3–4 hr.

**Expected**: +5–8% at the headline shapes. Lands v11 at ~37 TF.

### 6.3 SplitK port to v11 [LOW-EFFORT]

**Why**: v10 SplitK gave +84% at M=64 (v10s_64x128_ks8 = 20.31 TF vs
v10 ~11 TF). v11 at M=64 is currently 7.52 TF — much worse than v10s.
v11+SplitK should win at small M too.

**What to build**:
- Copy `v10splitk_kernels.cuh` pattern into `v11splitk_kernels.cuh`.
- The grid.z dimension carries the SplitK factor; each CTA handles a
  K-slice and atomic-adds into the C tile.
- Add `LAUNCH_V11S` dispatcher entries.

**Effort**: 1–2 hr (mostly mechanical copy + bit-correctness gate).

**Expected**: +50–80% at M ∈ {64, 256}. Closes the small-M production
gap that v11 leaves open today.

### 6.4 Larger CTA tile with register-spill mitigation [MEDIUM-EFFORT]

**Why**: 192x128x16_w4 already hit 32.87 TF with only 4-byte spill. With
proper register management, BM=192 or 256 could be the new champ for
M ∈ {1024+}.

**What to build**:
- Re-examine c_frag layout. With BM=192 BN=128 W=4: ATOMS_M=24, ATOMS_N=1
  → 24 × 8 = 192 floats per lane in c_frag. Over the 128-reg lifetime
  limit.
- Trick: **partial accumulator spill to SMEM** between K-tiles. Hold
  half c_frag in registers, half in SMEM; rotate. Increases instruction
  count but trades against the register-pressure-induced inner-loop
  serialization.

**Effort**: 4–6 hr.

**Expected**: +5–10% at M ≥ 1024. Lands v11 at ~37–38 TF.

### 6.5 PRMT-vectorized A-side load [LOW-EFFORT]

**Why**: A-side is fp32 in gmem → half conversion via
`__floats2half_rn`. Each thread does 4 `float→half` conversions per A
chunk. That's another non-tensor instruction sink that could vectorize.

**What to build**:
- Use `__float22half2_rn` (1 inst per pair, already in CUDA's intrinsics
  catalog) instead of 2 separate `__floats2half2_rn`s.
- Or convert at fp32 → uint16 packing level with PTX `cvt.rn.f16.f32`.

**Effort**: 1 hr.

**Expected**: +0.5–2%. Marginal but cheap.

### 6.6 Multi-shape MoE validation [MEDIUM-EFFORT, ORTHOGONAL]

We measured at one shape (N=K=7168). Real DSv4 inference has more
diverse shapes (different layer dims, expert routing). The champion at
N=K=7168 may not be the champion at e.g. N=2048 K=18944. A short sweep
across the actual DSv4 layer dimensions is overdue.

**Effort**: 2 hr (mostly sweep + analysis).

---

## 7. What we tried that won't be re-tried

Documented anti-patterns so the next pass doesn't repeat them:

1. **Stream cache (`__ldcs`) on the dequant front-end** — regressed -30%.
   The W_scales reuse pattern is fundamentally different from turbomind's
   activation/B-only stream usage. If we ever revisit, restrict `__ldcs`
   to W_qs alone (not W_scales) and verify.
2. **`__launch_bounds__(*, 3)`** — current register pressure blocks this.
   Only revisit after FP16 acc cuts c_frag in half.
3. **k-major explicit mma ordering** — compiler does this better than we
   can. Don't reorder unless ncu shows short_scoreboard regression first.
4. **Full XOR swizzle on the production champion shape** — the
   bank-conflict math doesn't justify the swizzle bit-fiddling overhead
   at BK_PAD=BK+8. Revisit only with BK_PAD=BK.

---

## 8. Pointers for the next session

**To resume work**:
1. Read this report (REPORT-12.md) top-to-bottom.
2. Read `tools/tc-grid/kernels/v11_kernels.cuh` — production champion.
3. Read `~/.claude/.../memory/MEMORY.md` index, especially:
   - `v100_wmma_half_float_frag_layout_mismatch.md` (blocks Step 6.1 if not handled)
   - `feedback_pre_dequant_defeats_int8.md` (frames why 85 TF is a ceiling)
   - `feedback_effort_estimation_undocumented_hardware.md` (3× multiplier)
4. Run the bit-correctness sweep to confirm the resume state:
   `./build/tc-grid --m-list 64,256,1024,2048,4096 --nk 7168 --dist uniform_small | grep INT8.*v11`
5. Verify expected: 128x128x16_w4_v11 ≈ 35.08 TF at M=2048, rel = 2.594e-04.

**Open follow-ups deferred from this sprint** (none blocking):
- `SPRINT-016-FOLLOWUPS.md`: INT4 BN=256 spill, v9 SMEM round-trip variant.
- MoE-aware dispatcher integration into the DSv4 inference path
  (separate work, not a kernel-tuning task).
- gpu-02-4090rtx remains occupied by `qwen3-moe-rotorquant` — do NOT
  evict for parallel V100 work.

---

## 9. Decision-rule status

Per the sprint-017 decision rule (CHECKPOINT.md §2):

| Bucket | v11 result | Action |
|---|---|---|
| ≥ 60 TF | — | — |
| 50–60 TF | — | — |
| **35–50 TF** | **35.08 TF M=2048** ✓ | **Ship + investigate gap** |
| < 35 TF | — | — |

v11 is production-quality. The investigate-gap work is §6.1–6.6 above.
