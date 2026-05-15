# REPORT-11 — v10 ships + full-spectrum bit-correctness validation

Date: 2026-05-13
Branch: sprint-015-tp8-baseline
Hardware: V100 SXM2 32GB (gpu-01, sm_70)

## TL;DR

- **v10 row-major B is the new INT8 production champion**, winning at every M ≥ 256 across full spectrum.
- **Best result this session: 29.18 TFLOPS** (INT8 LUT, 64x128x32_w4_v10, M=4096) — up from baseline v4 (27.73 TF).
- Full-spectrum sweep (M ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} × 4 formats × all kernel variants): 882 OK runs, 783 bit-correct, 99 bit-FAIL — **every bit-FAIL is from v9 (broken-by-design, see REPORT-10)**. All v0–v8 and v10 variants hold the rel ≤ 1e-3 tolerance.
- Gap to goal (50 TFLOPS): need 1.72×. Quick-win pipeline on `wmma::*` API path is exhausted; next move is v11 (m8n8k4 inline PTX + XOR swizzle).

## Method

1. Built tc-grid against `tcg-dev` pod on gpu-01.
2. Swept M values: {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} × N=K=7168 × `uniform_small` dist.
3. Every (format, kernel-version, tile) combo run; tolerance gate applied per-row (`rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`).
4. Champion per (format, M) selected as lowest-ms among bit-correct OKs.

## Bit-correctness audit

| total OK rows | bit-correct | bit-FAIL | bit-FAIL kernels |
|---:|---:|---:|---|
| 882 | 783 | 99 | exclusively `mm_int8_lut_v9a*` (CHUNK_K ∈ {2,4,8} × {128x128, 64x128, 64x64}) at every M |

v9 was implemented in SPRINT-016 as a chunked FP16/FP32 mixed-precision experiment. It builds and runs but has a fragment-layout bug: WMMA half-accumulator and float-accumulator fragments use different lane→element mappings on sm_70, so the register-resident `c_f.x[e] += __half2float(c_h.x[e])` promote scrambles outputs (rel = 7.07e-1 across the board). The tolerance gate correctly caught all 99 cases. Filed for SPRINT-017 follow-up: SMEM round-trip variant.

## Champions per (format, M)

| format | M | best tile | ms | TFLOPS | rel |
|---|---:|---|---:|---:|---:|
| INT8 | 16 | 32x64x32_w4_v3_b1 | 0.458 | 3.59 | 2.61e-4 |
| INT8 | 32 | 32x64x32_w4_v3_b1 | 0.514 | 6.40 | 2.60e-4 |
| INT8 | 64 | 32x64x32_w4_v3_b1 | 0.529 | 12.43 | 2.60e-4 |
| INT8 | 128 | 32x128x32_w4_v3_b1 | 0.750 | 17.54 | 2.59e-4 |
| INT8 | 256 | **64x128x32_w4_v10** | 0.953 | **27.60** | 2.60e-4 |
| INT8 | 512 | **64x128x32_w4_v10** | 1.921 | **27.39** | 2.59e-4 |
| INT8 | 1024 | **128x128x32_w4_v10** | 3.629 | **29.00** | 2.59e-4 |
| INT8 | 2048 | **128x128x32_w4_v10** | 7.309 | **28.79** | 2.59e-4 |
| INT8 | 4096 | **64x128x32_w4_v10** | 14.426 | **29.18** | 2.59e-4 |
| INT4 | 16-128 | 32x64x32_w4_v3_b1 / 64x64x32_w4_v3 | 0.90–1.08 | 1.82–12.15 | 2.56e-4 |
| INT4 | 256-4096 | 128x64x32_w4_v2/v3/v4 | 1.6–22.2 | 16.4–19.0 | 2.55e-4 |
| MXFP4 | 16-64 | 32x64x32_w4_v3_b1 | 0.527–0.624 | 3.12–10.54 | 1.85e-4 |
| MXFP4 | 128-4096 | 64x64 / 128x64 v3 | 0.82–19.95 | 16.0–21.10 | 1.85e-4 |
| F8 | 16-128 | 32x64x32_w4_v3_b1 | 1.22–1.40 | 1.35–9.41 | 1.85e-4 |
| F8 | 256-4096 | 128x64 / 64x64 v3 | 1.94–28.16 | 13.56–14.95 | 1.85e-4 |

Note: INT8 v10 takes over at M=256+ across both 64x128 and 128x128 tiles depending on which has better grid divisibility for that M. For small M (≤128), the BM=32 high-occupancy v3_b1 variants still win — register-pressure / latency-hiding trade favors more CTAs over larger tiles.

## v10 design recap (from REPORT-10)

v10 = v4 base + row-major B SMEM layout:
- `FragB` uses `wmma::row_major` instead of `wmma::col_major`
- SMEM B layout: `sB[k][n]` (k outer, n inner, stride `BN_PAD = BN + 8`)
- B-tile store writes transposed during dequant (row-major sB)
- `wmma::load_matrix_sync(b, ptr, ldm=BN_PAD)` reads with N as inner stride

Why it wins: WMMA's lane→element mapping for `matrix_b<row_major>` has 32 threads accessing N-adjacent halves within the same K row → near-stride-1 bank access. Measured MIO throttle dropped from 27.96% (v4 best) to 10.93% (v10 best), TC% rose 22.54 → 23.59. Bank conflicts on loads stayed nearly flat (~334M vs 308M), but the *pattern* of access serializes far less under the MIO unit.

## Iteration 2 experiments (no wins beyond v10 base)

| change | result | verdict |
|---|---|---|
| `__launch_bounds__(WARPS*32, 2)` | 28.92 → 28.98 TF | noise; compiler can't free regs to enable 2 CTAs/SM at 190 regs/thread |
| New tiles: 64x256_w8, 128x256_w8 | 28.32 TF best; 128x256 spilled 972 B | not competitive |
| Add L2 prefetch (Tier S inherited from v4) | 28.91 vs 28.92 | noise; MIO+sync already hides L2 latency |
| BK=64 tiles (96KB SMEM opt-in) | 13.8 TF (regress 50%) | L1 cache shrink from 96KB SMEM hurt gmem |

## Iteration 3 experiments (small wins)

| change | result | verdict |
|---|---|---|
| **HMUL dequant** (precompute scale as half, use `__hmul(short2half(qs), s_h)` instead of float-mul) | 28.92 → **29.32** TF at M=2048 | **WIN +1.4%** |

The HMUL change: was `__float2half((float)qs[i] * s)`, now `__hmul(__short2half_rn((short)qs[i]), s_h)`. Saves 1 instruction per dequant element (avoids the float intermediate). Bit-correct.

These exhaust the "easy" deltas on top of v10. **The wmma::* API path is at its local optimum: ~29.3 TFLOPS sustained.**

## Stall budget on v10 128x128_w4 (M=2048)

| stall | % | meaning |
|---|---:|---|
| mio_throttle | 10.93 | SMEM pipe back-pressure (was 27.96 on v4) |
| long_scoreboard | 22.70 | memory-dep |
| short_scoreboard | 21.51 | register-dep |
| barrier | 4.42 | `__syncthreads` |
| math_pipe_throttle | <1 | tensor cores not the bottleneck |
| **TC% achieved** | **23.59** | (vs 22.54 v4; we use 23.6% of TC peak) |

Remaining ~40% unaccounted is the WAIT bucket — warps waiting on various dependencies, masked by low occupancy (WARP%=12.23, single CTA per SM bound by Regs=190).

## What blocks 50 TFLOPS

The wmma::* API on V100 INT8 has structural limits we've now exhausted:

1. **Bank conflicts**: 334M load conflicts remain; padding-only fixes cannot reach zero (ldm%8 constraint forces gcd(ldm/2, 32) ≥ 4 — REPORT-9 documents). XOR swizzle is the only path, but it requires manual fragment loads (bypasses `wmma::load_matrix_sync`).
2. **Occupancy**: 1 CTA/SM = 4 warps/SM (of 64 max) = 6% theoretical occupancy ceiling, driven by 190 regs/thread. Reducing regs without precision loss requires either FP16 acc (v8/v9 attempted, broken) or a smaller c_frag set with re-load (more SMEM traffic).
3. **Compute density**: TC% is only 23.59. cuBLAS FP16 achieves 75-85 TFLOPS here; that's our practical upper bound. Closing that gap requires more compute per memory load (deeper K-tile pipeline) OR fewer non-compute stalls.

## Next step: v11

Drop `wmma::*` API. Adopt turbomind's V100 GEMM design:

1. **m8n8k4 HMMA inline PTX** (`mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`) — V100's native instruction; m16n16k16 wmma decomposes to 4 of these but with a fixed lane layout we can't change.
2. **XOR-swizzled SMEM** (`offset ^ ((offset & yyy_mask) >> Shift)`, CUTLASS pattern from turbomind/core/layout.h) — actually eliminates bank conflicts (not just shifts the pattern).
3. **Manual `Lds` fragment loads** (ld.shared.b128) matching the m8n8k4 lane mapping — required because swizzled SMEM addressing breaks load_matrix_sync's linear-stride assumption.

Expected: TC% climb from 23.6% to 40%+ → 45-55 TFLOPS range. Estimated effort: 200-300 LOC, multi-session correctness debugging risk.

References: `/tmp/turbomind/lmdeploy/src/turbomind/kernels/gemm/arch/{mma_sm70.h, operand_sm70_s884.h, smem_copy_sm70.h, config_sm70_s884.h}` + `/tmp/turbomind/lmdeploy/src/turbomind/kernels/core/{mma.h, layout.h}`.

## Artifacts

- Full sweep CSVs: `tools/tc-grid/docs/full_M{16,32,64,128,256,512,1024,2048,4096}.csv` (882 OK rows total)
- v10 source: `tools/tc-grid/kernels/v10_kernels.cuh`
- ncu data (v4): `tools/tc-grid/docs/ncu_v4_full.log`, `ncu_v4_bank.log`
- ncu data (v10): `/tmp/ncu_v10.log` (uncopied)
- Dev pod: `tcg-dev` (kept running for v11 implementation)
