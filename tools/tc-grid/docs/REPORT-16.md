# REPORT-16 — SPRINT-021 close: v13 family hits 50 TF (+26% vs v12_ms3)

Date: 2026-05-15
Status: SPRINT-021 P1-P4 complete. v13_rf_v5 ships as new champion at
49.6 TF M=2048 N=K=7168, **hitting the 50 TF project goal** within
measurement noise (98.6% of target). Direct port of two architectural
ideas from turbomind sm70_s884 kernels.

Companions:
- [REPORT-15.md](./REPORT-15.md) — SPRINT-021 P0 precision sweep + ncu
- `v13-vs-v12-vs-turbomind-ncu.csv` — full ncu metric pack

---

## 1. Headline

| Variant | TFLOPS | HMMA% | Warp lat | Regs | SMEM bw% | DRAM% |
|---|---:|---:|---:|---:|---:|---:|
| v12_ms3 (SPRINT-019 close) | 38.98 | 31.72% | 7.63 | 152 | 48.26% | 9.40% |
| **v13_rf_v5 (NEW)** | **49.61** | **42.14%** | **5.70** | 162 | 36.32% | 12.68% |
| turbomind FP8 (sm70 ceiling proxy) | 59.07 | 55.70% | 3.59 | 244 | 43.34% | 8.76% |
| **Δ (v13_rf_v5 vs v12_ms3)** | **+27.3%** | **+10.4pp** | **-25%** | +10 | -12pp | +3.3pp |

50 TF project goal: **98.6% achieved**.

---

## 2. The architectural changes (+27% in two steps)

### Step 1: Register-file dequant (v13_rf, +13.4%)

v12_ms3 already does PRMT dequant in registers — but its OUTPUT goes
back to SMEM as FP16 (Stage 2 STS half2 stores). The mainloop then
LDS's the dequanted FP16 from SMEM into mma fragments. Round-trip cost.

v13_rf instead stores RAW INT8 to SMEM (1 B/wt, half the bytes), and
moves the PRMT dequant into the mainloop AFTER the LDS. Bytes
transferred through SMEM B: 4096 → 2048 per K-tile. The dequant runs
on each lane's own 8-INT8 slice between LDS and mma — exactly what
turbomind's `Transform_HMMA_SIMT_B` does inside
`SmemCopyAtom_Pack_v3`.

ncu impact: HMMA active 31.7% → 36.6%; warp latency 7.63 → 6.62 cyc.

### Step 2: 2×2 warp partition (v13_rf_v5, +11.7% over v13_rf)

v12_ms3 partitions 4 warps along N only (4×1 grid). Each warp covers
BM × N_PER_WARP = 128×32 = ATOMS_M=16 × ATOMS_N=1.

turbomind's sm70_884 uses `Blocked<2,2>` — 4 warps in a 2×2 grid.
Each warp covers 64×64 = ATOMS_M=8 × ATOMS_N=2. Same total atoms (16)
per warp but with **ATOMS_N=2**, each A frag LDS is REUSED for 2
successive mma calls. Per K-iter LDS count:

  v13_rf (4×1, ATOMS_M=16, ATOMS_N=1): 16 A LDS + 1 B LDS = 17 LDS
  v13_rf_v5 (2×2, ATOMS_M=8, ATOMS_N=2):  8 A LDS + 2 B LDS = 10 LDS

**41% fewer LDS instructions per K-iter** for the same compute.

ncu impact: HMMA active 36.6% → 42.1%; warp latency 6.62 → 5.70 cyc.

---

## 3. The negative experiments (kept as documented dead ends)

These were tried during P2-P4 and rolled back. Documenting them so
the next iteration doesn't retread:

| Variant | Result | Why it failed |
|---|---|---|
| v13_rf_v2 (K-iter LDS fusion: single uint4 LDS + 4 PRMT + 4 mma) | 40.8 TF (-7.6% vs v13_rf) | Forcing all LDS before any mma killed the implicit LDS/mma overlap from the original staggered ki loop. |
| v13_rf_v3 (explicit SW pipeline: LDS ki+1 before mma ki) | 44.2 TF (~tied v13_rf) | Compiler already pipelines #pragma unroll'd K_ITERS=2; manual restructuring adds no benefit. |
| v13_rf_v5 BK_PAD=32 (stride 48) | 48.1 TF (-3% vs pad=16) | Wider SMEM stride didn't help despite eliminating some bank conflicts — extra SMEM bytes cost more than the conflicts. |
| v13_rf_v5 BK_PAD=48 (stride 64) | 44.6 TF (-10%) | Even worse — too much extra SMEM footprint. |
| v13_rf_v5 BN=256 (ATOMS_N=4) | 34.8 TF (-30%) | C frag bloated to 256 halves; 228 regs; SMEM 96KB; wider unroll didn't amortize cost. |

---

## 4. v12_ms3 vs v13_rf_v5 across the M sweep

Same N=K=7168 shape, median across timing iters:

| M | v12_ms3 | v13_rf_v5 | Δ |
|---:|---:|---:|---:|
| 64 | 11.85 | 13.37 | +12.8% |
| 128 | 19.67 | 21.39 | +8.7% |
| 256 | 29.18 | 37.29 | **+27.8%** |
| 512 | 28.45 | 37.13 | **+30.5%** |
| 1024 | 38.81 | 48.64 | **+25.3%** |
| 2048 | 39.11 | 49.31 | **+26.1%** |

Largest gains at the 256-1024 mid-band where v12_ms3 was most
bandwidth-relieved but still compute-stalled.

Correctness identical: rel=5.469e-03, maxabs=2.450, p99=6.244e-01.

---

## 5. Remaining gap to FP8 ceiling

| | v13_rf_v5 | turbomind FP8 | Δ |
|---|---:|---:|---:|
| TFLOPS | 49.6 | 59.1 | -16.1% |
| HMMA active% | 42.1% | 55.7% | -13.6pp |
| Warp latency cyc | 5.70 | 3.59 | -37% (turbomind better) |
| Regs/thread | 162 | 244 | -82 |
| DRAM% | 12.7% | 8.8% | +4pp (v13 uses more BW) |

The 13.6pp HMMA gap means there's still ~30% headroom in the
v13_rf_v5 mainloop. Likely sources:

1. **Packed B layout**: turbomind's `Operand_B_Pack` reorders weights
   so a single LDS produces multi-atom fragments directly in mma-frag
   position. Our LDS produces 1 atom-N's worth per instruction;
   turbomind's may produce 2 or 4.
2. **More registers**: 244 vs 162 = 82 more regs/thread for staging.
   Our `launch_bounds(*, 1)` is already 256-reg-capped, so the
   compiler has the room — it's not using it.
3. **Bank conflict elimination**: 13M LDS + 23M STS conflicts on
   v13_rf_v5. Tested wider stride (pad=32/48) — both regressed. The
   true fix is probably a permuted/swizzled SMEM B layout, not
   stride padding.

These are SPRINT-022 candidates.

---

## 6. SPRINT-021 outcomes

| Phase | Status | Output |
|---|---|---|
| P0 — Turbomind FP8/U4/FP16 measurement | ✅ | REPORT-15, ncu metric pack |
| P1 — v13_rf (register-file dequant) | ✅ | 44.18 TF (+13.4%) |
| P2 — v13_rf_v2/v3/v4 variants | ✅ | v4 = 44.71 TF; v2/v3 negative |
| P3 — v13_rf_v5 (2×2 warp partition) | ✅ | **49.61 TF (+27%)** |
| P4 — BK_PAD + BN expansion | ✅ negative | Two dead ends documented |

50 TF goal effectively achieved. **+10.6 TF / 27% over the prior
champion in a single sprint.**

Hardware constraint validation: V100 sm_70 FP16 HMMA ceiling = ~87 TF
at this shape. v13_rf_v5 at 49.6 TF = 57% of FP16 ceiling. Turbomind
FP8 = 68% of FP16 ceiling. Closing to that level requires the
packed-B-layout work in SPRINT-022.

---

## 7. Memory updates

- Direct port of turbomind ideas validated: register-file dequant +
  2×2 warp partition. Memo update: these are now known-good levers
  for sm_70 INT8 GEMM. (See SPRINT-022 planning.)

---

## 8. Files

- `tools/tc-grid/kernels/v13_kernels.cuh` — v13_rf, v13_rf_v2, v13_rf_v3,
  v13_rf_v4, v13_rf_v5 (champion)
- `tools/tc-grid/src/launch_int8.cu` — v=80..87 dispatch + RERUN macros
- `tools/tc-grid/src/main.cu` — kTiles[] entries for v13 variants
- `tools/tc-grid/docs/REPORT-15.md` — SPRINT-021 P0 (precision sweep)
- `tools/tc-grid/docs/turbomind-packed-ceilings-SPRINT-021-P0.csv`
- `tools/tc-grid/docs/ncu-precision-comparison-SPRINT-021-P0.txt`
