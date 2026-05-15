# REPORT-9: Tier B retrospective + instrumented grid sweep

Date: 2026-05-13
Branch: sprint-015-tp8-baseline
Hardware: V100 SXM2 32GB (gpu-01, sm_70)

## TL;DR

- Production winner for **INT8** at M ≥ 256: `128x128x32_w4_v4` — bit-correct, ~27.7 TFLOPS at M=2048.
- Production winner for **INT4 / FP4 / FP8**: `128x64x32_w4_v3` / `v2` — bit-correct, 14-20 TFLOPS depending on format.
- Production winner at **M ≤ 64** for all formats: `32x64x32_w4_v3_b1` (BM=32 high-occupancy variant from Tier B1).
- Tier B work (B1 FRAG_M=2, B3 triple-buffer) is closed. **v4 — not v3 — is the actual winning Tier-S baseline** once measured at production shape (prior reports overstated v3).
- v8 (FP16 accumulator) was attempted but **fails bit-correctness contract** (rel=5.47e-3 vs ≤1e-3 target). Yields only 3% speedup. Not viable without mixed-precision rework.

## Method (corrected)

Prior reports leaned on speculation. Report 9 uses:

1. **Full grid sweep** at M ∈ {64, 256, 1024, 2048}, N=K=7168, `uniform_small` dist, all kernel versions (v1-v8) and tile configs (BM × BN × BK × WARPS × FRAG_N).
2. **Bit-correctness gate** applied per-row: `maxabs ≤ 0.1` AND `p99 ≤ 0.05` AND `rel ≤ 1e-3`. v3 baseline (`rel=2.59e-4`) sets the tolerance floor.
3. **Hardware counter profiling** (ncu, separate run in `ncu_M{64,256,2048}_int8_lut.csv`) for top variants.

## Best bit-correct config per (format, M)

| format | M | tile | ms | TFLOPS | rel |
|---|---:|---|---:|---:|---:|
| INT8 | 64 | 32x64x32_w4_v3_b1 | 0.531 | 12.4 | 2.6e-4 |
| INT8 | 256 | 64x128x32_w4_v3 | 0.976 | 26.9 | 2.6e-4 |
| INT8 | 1024 | 128x128x32_w4_v4 | 3.769 | 27.9 | 2.6e-4 |
| INT8 | 2048 | 128x128x32_w4_v4 | 7.588 | 27.7 | 2.6e-4 |
| INT4 | 64 | 32x64x32_w4_v3_b1 | 0.961 | 6.9 | 2.6e-4 |
| INT4 | 256 | 128x64x32_w4_v3 | 1.596 | 16.5 | 2.6e-4 |
| INT4 | 1024 | 128x64x32_w4_v2 | 5.588 | 18.8 | 2.6e-4 |
| INT4 | 2048 | 128x64x32_w4_v2 | 11.188 | 18.8 | 2.6e-4 |
| MXFP4 | 64 | 32x64x32_w4_v3_b1 | 0.625 | 10.5 | 1.9e-4 |
| MXFP4 | 256 | 128x64x32_w4_v3 | 1.345 | 19.6 | 1.9e-4 |
| MXFP4 | 1024 | 128x64x32_w4_v3 | 5.346 | 19.7 | 1.9e-4 |
| MXFP4 | 2048 | 128x64x32_w4_v3 | 10.670 | 19.7 | 1.9e-4 |
| F8 | 64 | 32x64x32_w4_v3_b1 | 1.275 | 5.2 | 1.9e-4 |
| F8 | 256 | 128x64x32_w4_v3 | 1.939 | 13.6 | 1.9e-4 |
| F8 | 1024 | 128x64x32_w4_v3 | 7.604 | 13.8 | 1.9e-4 |
| F8 | 2048 | 64x64x32_w4_v3 | 14.463 | 14.6 | 1.9e-4 |

cuBLAS FP16 ceiling at M=2048: **98.28 TFLOPS** (78% of nominal V100 peak). INT8 best is 28% of cuBLAS = ~32% of practical TC peak.

## v8 (FP16 accumulator) — negative result

| variant | ms | TFLOPS | rel | spills |
|---|---:|---:|---:|---:|
| v3 128x128_w4 | 7.651 | 27.5 | 2.6e-4 | 1540 B |
| v4 128x128_w4 | 7.588 | 27.7 | 2.6e-4 | (no spill) |
| **v8 128x128_w4** | **7.408** | **28.4** | **5.5e-3** | **(no spill)** |

v8 = v3 layout with `FragC = wmma::fragment<accumulator, ..., half>` instead of float. Halves c_frag register footprint (8→4 regs each), eliminates v3's 1540-byte spill, and runs 3.2% faster. **But** the K=7168 reduction loses too many mantissa bits — relative error jumps 20x, well outside the tolerance contract.

To make this viable: chunked promotion (accumulate in half for CHUNK_K K-tiles, promote into a float c_acc, reset). This adds complexity (~50 lines per kernel) and temporarily doubles c_frag registers during the promote, so the win may evaporate. Filed as **v9** candidate; not blocking.

## What ncu actually said (correcting prior reports)

From `ncu_M2048_int8_lut.csv`, v3 best variant (128x128_w4, FRAG_M=8/FN=2):

| metric | value | interpretation |
|---|---:|---|
| TC% | 22.29 | tensor-core utilization vs peak |
| WARP% | 12.46 | active warps/cycle (occupancy proxy) |
| **ShtSb%** | **30.48** | **register-dependency stall — largest single stall** |
| LngSb% | 22.86 | memory-dep stall (includes bank-conflict cost) |
| BarrSt% | 4.06 | `__syncthreads` stall (Reports 6/7 claimed ~25%) |
| Regs | 194 | per thread (vs 256K/SM → ~6% occupancy ceiling) |
| BankConf | 390M | NOT zero (prior reports claimed padded BK_PAD=BK+8 eliminated conflicts; it gives 4-way conflict) |

**Corrections to prior reports:**
- Report 6: claimed `__syncthreads` was 25% of stalls. Actual: 4%. Wrong by 6x.
- Report 7: claimed v3's padding eliminated bank conflicts. Actual: 390M conflicts; padding cannot reach zero on V100 col-major B (any valid `ldm` is a multiple of 8 → `gcd(ldm/2, 32) ≥ 4`).
- Report 8: claimed Tier A1 (BK=64) failed due to "L2 prefetch limits". Actual: BK=64 doubles c_frag spills (v6 had 222 regs and TC=16.8%) — register pressure regression dominates.

## Tier B closure

| sub-task | status | best config | win over v3 baseline |
|---|---|---|---|
| **B1**: FRAG_M=2 / BM=32 high-occupancy | done | `32x64x32_w4_v3_b1` at M=64 | wins at small M only |
| **B2**: split-K | not pursued in this sprint (compile-time gate via `split_k=1`); deferred |
| **B3**: 3-stage triple-buffer (v7) | done | underperforms v3/v4 at production shape; SMEM-constrained to BM≤64 |
| Bit-correctness | done | tolerance gate built into harness; all bit-ok configs above |
| Full grid sweep | done | 385 (variant × path) entries per M × 4 M values |

## Recommended actions before Sprint 016

1. **Promote v4 to default INT8 kernel** in the integration path (currently v3 is the default in some launchers).
2. **B1 dispatch** at M ≤ 64: select `32x64x32_w4_v3_b1` for INT8 (or `32x64x32_w4_v3_b1` for other formats — same tile shape wins across formats at small M).
3. **Defer XOR-swizzle / manual-fragment v9**: bank conflicts are the second-largest stall, not the first. The first is register pressure (Short Scoreboard, 30%). XOR-swizzle is ~400 lines of engineering for an estimated 5-10% speedup; v9 mixed-precision could deliver similar wins with simpler code if precision is recovered.
4. **Re-evaluate INT4 spill warnings**: v3 INT4 at 128x256 spills 1664 B. Bringing INT4 BN=256 back in scope only after spills are resolved.

## Files / artifacts

- Raw sweep: `run_M{64,256,1024,2048}.csv`
- ncu raw: `ncu_M{64,256,2048}_int8_lut.csv`
- ncu parser: `/tmp/parse_ncu.py`
- Best-by-shape extractor: `/tmp/best_by_shape.py`
- v8 source: `kernels/v8_kernels.cuh` (kept in tree, gated off bit-correctness)
- New constraint memory: `~/.claude/projects/-Users-ravi-repos-deepseek/memory/v100_wmma_smem_conflict_constraint.md`
