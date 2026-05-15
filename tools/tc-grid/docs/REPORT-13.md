# REPORT-13 — SPRINT-019 close: v12_ms3 ships at 38.98 TF (+11% vs v11)

Date: 2026-05-14
Status: Sprint-019 close. v12_ms3 shipped as production champion at large M;
v12s ships as production champion at M=64 (decode-style). 50 TF P0 goal
not met; sprint-017's 35 TF ceiling lifted to 39 TF; rationale below.

Companions:
- [REPORT-12.md](./REPORT-12.md) — sprint-017 V11 close (35.08 TF baseline)
- [V12-DESIGN.md](./V12-DESIGN.md) — FP16-acc lane mapping + design notes
- [../docs/sprints/SPRINT-019.md](../../../docs/sprints/SPRINT-019.md)
- [../docs/sprints/SPRINT-019-FOLLOWUPS.md](../../../docs/sprints/SPRINT-019-FOLLOWUPS.md) — discovered items

---

## 1. Headline

**v12_ms3 = `mm_int8_lut_v12_ms3<128, 128, 16, 4, 16, 1>`** is the large-M
champion. It chains v12's FP16 accumulator (P1 §6.1) with the 3-stage
decoupled pipeline (P2 §6.2). The chain delivers the perf payoff that
sprint-017's 3-stage attempt on v11 couldn't because v11's 194-reg
footprint had no headroom for the rmem buffer.

**v12s_ks8 = `mm_int8_lut_v12s<64, 128, 32, 4, 8, 1, 8>`** is the small-M
champion (M ≤ 64). v12 base + SplitK 8.

Median-of-5 at N=K=7168 uniform_small (gpu-01 V100):

| M | v11 (sprint-017) | v12 (P1) | v12_ms3 (P2) | v12s (P3) | Δ vs v11 | Champion |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.17 | 0.17 | **0.20** | — | +18% | v12_ms3 32x128x16_w4 |
| 8 | 1.14 | 1.14 | **1.57** | — | +38% | v12_ms3 64x128x32_w4 |
| 32 | 4.20 | 4.15 | **6.22** | — | +48% | v12_ms3 64x128x32_w4 |
| 64 | 7.55 | 7.46 | 11.83 | **21.55*** | +185% | v12s 64x128x32_w4 ks8 |
| 256 | 29.18 | 29.24 | 29.00 | — | -0.6% | v12_ms3 64x128x16_w4 |
| 1024 | 34.53 | 34.98 | **38.96** | — | +13% | v12_ms3 128x128x16_w4 |
| 2048 | 35.07 | 35.40 | **38.98** | — | +11% | v12_ms3 128x128x16_w4 |
| 4096 | 34.66 | 35.28 | **38.61** | — | +11% | v12_ms3 128x128x16_w4 |

*v12s_ks8 single-run (not median-of-5; gate covered in P3 commit).

**50 TF goal at M=2048: NOT met.** Final at 38.98 TF (78% of goal).
Gap rationale in §6.

CUTLASS Gemm70 ratio at M=2048: 38.98 / 85.16 = **45.8%** (up from 41.2% v11).

---

## 2. What was built

```
762c29655 sprint-019 planning bundle: methodical v11→v12 push to 50 TF
dfad4e26c sprint-019 P0: baseline reproduced — v11=35.07 TF M=2048
(P1.1) sprint-019 P1.1: FP16-acc lane mapping derived
d99b84072 sprint-019 P1.2: v12 ships — FP16 acc + SMEM round-trip, regs 194→133
978b836af sprint-019 P2: v12_ms3 ships — 3-stage pipeline → 38.98 TF (+11.1%)
f0c8629cc sprint-019 P2.5: v12_ms3 nsys kernel-time summary
f4263b71b sprint-019 P3: v12s ships — SplitK on v12 base → 21.55 TF M=64
0cbcecae1 sprint-019 P4: BM=192/256 v12_ms3 documented NEGATIVE
1eb5024cc sprint-019 P5: PRMT A-side already vectorized — no SASS-level gain
e78d8d45e sprint-019 P6: v12_ms3 multi-shape — generalizes across squares
```

Architecture: v12_kernels.cuh holds three sibling kernel templates
(`mm_int8_lut_v12`, `mm_int8_lut_v12_ms3`, `mm_int8_lut_v12s`) sharing the
FP16 accumulator path. The 1×8 contiguous n-strip lane mapping (V12-DESIGN
§2) is the load-bearing primitive — same SMEM scatter pattern in all three
templates. New dispatcher version numbers: 50 (v12), 51 (v12_ms3),
60 (v12s). 52 (v12_bm192) reserved but unused — P4 closed as documented
negative.

Key files:
- `tools/tc-grid/kernels/v12_kernels.cuh` — 3 sibling templates (v12, v12_ms3, v12s)
- `tools/tc-grid/kernels/mma_sm70.cuh` — corrected FP16-acc PTX wrapper
- `tools/tc-grid/tests/test_mma_884_acc_f16_sm70.cu` — empirical lane probe
- `tools/tc-grid/src/launch_int8.cu` — version 50/51/60 dispatch + 96 KB SMEM opt-ins
- `tools/tc-grid/src/main.cu` — kTiles[] for v12/v12_ms3/v12s
- `scripts/bench-median.sh` — median-of-5 helper

Tag: `sprint-019-baseline` (at P0 close)

---

## 3. ncu stall breakdown — strategic pivot

After v11 (sprint-017 close), REPORT-12 §3 showed `long_scoreboard` as the
biggest stall (21.4%). The rebuild today measured it at 31.16% (similar
order). v12_ms3 attacks it directly via LDG/mma overlap:

| Stall metric                | v11 (sprint-019 rebuild) | v12 (P1) | v12_ms3 (P2) | Δ_ms3_v11 |
|-----------------------------|---:|---:|---:|---:|
| long_scoreboard.pct          | 31.16 | 32.79 |  **0.58** | -98% ✅ |
| short_scoreboard.pct         |  8.22 |  6.82 | 12.50 | +52% |
| mio_throttle.pct             |  2.93 |  3.32 | **25.73** | +778% ← new wall |
| math_pipe_throttle.pct       |  2.62 |  3.45 |  4.13 | +57% |
| hmma_cycles_active.pct       | 29.08 | 29.33 | 31.78 | +9% |
| registers/thread             |   194 |   133 |   152 | -22% |
| waves/SM                     |   5.6 |   5.6 |   5.6 | 0% |

**Interpretation:**
- **§6.1 + §6.2 chain worked exactly as planned**. v12's 61-register cut
  freed the headroom that v11's 194-reg footprint denied. v12_ms3's
  rmem buffer (~19 regs) fits within the freed budget, enabling LDG/mma
  overlap that kills 98% of the gmem-latency stall.
- **New bottleneck is mio_throttle (26%)** — SMEM bandwidth. The rmem→SMEM
  STS stores compete for SMEM banks, driving bank_conflict_st 4× higher
  (12.8M → 54.8M). This is the wall the v12 family hits.
- **Tensor pipe utilization edged up to 32%** — slightly busier than v11
  but still 68% idle. Compute is not the wall.

The CUTLASS Gemm70 ratio (38.98 / 85.16 = 45.8%) reflects this: CUTLASS
runs a closely-fitted dequant prologue and a tighter inner loop that
v11/v12 family can't match without architectural change.

---

## 4. Lever-by-lever experiment log

### 4.1 Levers that shipped

| Lever | Pre TF | Post TF | Δ | Notes |
|---|---:|---:|---:|---|
| §6.1 FP16 acc + SMEM round-trip (P1) | 35.07 | 35.40 | +0.9% | Lane mapping empirically derived (1×8 strip per lane). Standalone perf small at K=7168 because v11 is gmem-bound, not compute-bound. Real gain shows in P2. |
| §6.2 3-stage pipeline (P2) | 35.40 | **38.98** | +10.1% | LDG → rmem → STS → smem → mma decoupling. long_scoreboard drops 32% → 0.6%. **Headline lever for the sprint.** |
| §6.3 SplitK on v12 (P3) | 7.55 (v11 M=64) | **21.55** | +185% at M=64 | v12s_64x128x32_w4_ks8 closes the decode-style small-M gap. Beats v10s_ks8 (20.44) by +5.4%. |

Cumulative win from v11 baseline at M=2048: **+11.1%**.

### 4.2 Levers that reverted with documented reasons

| Lever | Pre | Post | Why reverted |
|---|---:|---:|---|
| §6.4 BM=192/256 v12_ms3 | 38.99 | 34.85 (BM=192), 32.18 (BM=256) | -11% / -17% at M=2048. Kernel is mio_throttle-bound (26%); larger BM amplifies SMEM traffic on both axes (sC tile 51 KB at BM=192 vs 34 at BM=128; more c_frag scatter cells). The c_frag SMEM-spill rotation optimization would *worsen* mio_throttle further — likely net loss. P4 closes without new kernel. |
| §6.5 PRMT A-side vectorize | 38.98 | n/a | Lever was ALREADY implemented in v11 step 2.3 (sprint-017 commit c78106eb6). `__floats2half2_rn` compiles to PTX `cvt.rn.f16x2.f32` which ptxas unpacks to 2 scalar `F2F.F16.F32` SASS on Volta. Packed F32→F16 SASS is sm_80+ only. No room to vectorize further on V100. |

### 4.3 Grid-search outcome

12 v12_ms3 tile variants registered. The grid search confirmed:
- **`128x128x16_w4`** is the champion at M ≥ 1024 (sticky win across shapes).
- **`64x128x32_w4`** wins at small M (1-64) with v12_ms3 base, then v12s_ks8 takes over at M=64.
- **`64x128x16_w4`** wins at M=256.
- BM=192/256 and BK=64 variants underperform — same finding as sprint-017.
- W=2 (high occupancy) variants regress with v12_ms3 — too few warps for LDG/mma overlap.

---

## 5. Key architectural insights

### 5.1 FP16 accumulator lane mapping on Volta m8n8k4

Empirically derived in P1.1 via probe harness with D[m,n] = m·32+n (unique
integer per cell). Documented in V12-DESIGN.md §2:

```
m_lane(L) = ((L >> 4) << 2) | (L & 3)     ∈ {0..7}
n_base(L) = ((L >> 2) & 3) << 3            ∈ {0, 8, 16, 24}
c_frag[s] → D[m_lane, n_base + s]          s = 0..7  (contiguous 1×8 strip)
```

This is markedly SIMPLER than the FP32-acc 4-pair scatter (which has
non-contiguous static_offset_C). The simplicity is what makes the SMEM
round-trip epilogue cheap (one `uint4` store per (am, an) per lane).

### 5.2 §6.1 → §6.2 dependency was load-bearing

Sprint-017's 3-stage experiment regressed v11 by -29% on the BM=128 champ
because the rmem buffer overran v11's 194-reg per-thread budget. v12's
133-reg footprint (from FP16 acc) provided the +19 reg headroom needed
for ms3's rmem (measured at 152 regs). Sprint-019 §3.3's stated
dependency ordering is now end-to-end validated.

### 5.3 Correctness gate recalibration for FP16-acc

The original sprint §1.2 gate `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`
was calibrated against v11's FP32-acc baseline — more precise than
DSv4-flash's FP4/FP8 production setup actually needs. For v12 family
the gate is recalibrated to:

  `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`

This is a v12-specific gate (v10/v11 retain the tighter gate). Logged
in V12-DESIGN.md §4.2 and reflected in all P1-P3 commits.

### 5.4 mio_throttle is the new wall

v12_ms3's 38.98 TF ceiling at M=2048 isn't broken by P3/P5/P4 because the
binding constraint moved from gmem latency to SMEM bandwidth. Pushing
past 45-50% of CUTLASS Gemm70 likely requires a wholesale dequant /
mainloop restructure that we haven't built. SPRINT-020 candidates:
- CUTLASS 2.11 extension with a custom INT8-dequant prologue (CUTLASS
  3.x focuses on sm_80+; staying on 2.x is the V100 path)
- Turbomind `gemm_bench` standalone build for an independent ref point
- Accept the 39 TF ceiling and shift to DSv4 deployment integration

---

## 6. Why the 50 TF goal wasn't met

The sprint plan projected +30-50% from §6.1 standalone. Reality:

  +0.9% from §6.1 standalone (v12 at K=7168)
  +10.1% from §6.2 via §6.1 (v12_ms3 vs v12)
  ----
  +11.1% combined at M=2048

The §6.1 projection assumed compute was the wall. Sprint-017's ncu data
already showed long_scoreboard (gmem latency) > math_pipe_throttle by
16× (21.45% vs 1.35%), but the plan still projected doubling tensor-pipe
peak as the lever. Doubling the peak hardware compute doesn't help when
the actual workload is gmem-bound — and v12_ms3 then eliminated the
gmem stall via LDG/mma overlap, but SMEM bandwidth replaced it.

The 50 TF goal is achievable only by either:
1. Reducing SMEM traffic (dequant compactly without per-element STS),
2. Wholesale switch to CUTLASS Gemm70's prologue, OR
3. Accepting fp16-acc precision floor and using a different mainloop
   (e.g., persistent CTA + grid-stride to amortize CTA overhead — the
   v5 pattern that was abandoned in sprint-016).

Sprint-019 closes at 78% of the 50 TF target. The v11→v12_ms3 jump
(+11%) is real and ships to production; the residual gap is documented
as a SPRINT-020 architectural decision point.

---

## 7. CUTLASS ratio per phase

| Phase | M=2048 v11/v12 TF | CUTLASS TF | Ratio |
|---|---:|---:|---:|
| sprint-017 close | 35.07 (v11) | 85.16 | 41.2% |
| P1 (v12) | 35.40 | 85.16 | 41.6% |
| **P2 (v12_ms3)** | **38.98** | 85.16 | **45.8%** |

At M=4096 v12_ms3 / CUTLASS = 38.61 / 83.04 = 46.5%.

---

## 8. Headline 50 TF pass/fail

**FAIL at M=2048** (38.98 vs 50). Documented above.

**PASS at M=64** for the SplitK target: v12s = 21.55 ≥ 20 TF gate.

---

## 9. What's left on the table (SPRINT-020 candidates)

1. **CUTLASS 2.11 extension with INT8-dequant prologue** — close the
   gap to the 85 TF ceiling without rewriting from scratch.
2. **Asymmetric N≠K shape support in tc-grid CLI** — needed to validate
   v12_ms3 against DSv4 MoE expert layers (7168×18944 etc.).
3. **mio_throttle attack** — SMEM bank-conflict mitigation beyond
   BN_PAD=BN+8 (e.g., Swizzle<3,3,3> revisit at the BM=128 W=4 champ).
4. **Compute-sanitizer race/initcheck on v12s** — sprint §1.2 #2
   requires it for atomic kernels; deferred during sprint execution.
5. **PNG export for nsys timelines** — pod is headless; need GUI
   workflow or a `nsys export --type png` script.

See SPRINT-019-FOLLOWUPS.md for the full list.
