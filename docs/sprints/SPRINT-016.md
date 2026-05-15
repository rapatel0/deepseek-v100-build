# SPRINT-016 — Close the cuBLAS gap on V100 INT8 / FP4 / FP8

**Status: CLOSED 2026-05-13.** Plan-vs-outcome divergence: v9 (mixed-precision) was broken
by WMMA half/float fragment lane-mapping mismatch; v10 (row-major B SMEM) was added
mid-sprint and became the new INT8 production champion at 29.49 TFLOPS (M=2048, +6.4% vs
v4 baseline). Definition-of-Done items below are checked against actual outcomes in the
"Sprint closure" section. Follow-up sprint: SPRINT-017 (v11 m8n8k4 PTX + XOR-swizzle,
target 50 TFLOPS).

Sprint goal: get the bit-correct INT8 kernel from 27.7 → 35+ TFLOPS at M=2048 N=K=7168 on V100, and either confirm or replace the production winners for INT4 / MXFP4 / F8 based on measurement. All work on a single V100 (gpu-01) against the live `tcg-dev` pod.

## Overview

Reports 1-9 closed Tier B. The remaining throughput lever is **register pressure**, not bank conflicts (ncu showed Short Scoreboard 30% > Long Scoreboard 22%; padding cannot zero bank conflicts on V100 col-major B regardless). v8 demonstrated FP16 c_frag eliminates the 1540-byte spill in v3 and yields a 3% speedup — but breaks the rel ≤ 1e-3 tolerance contract (rel=5.5e-3 at K=7168). v9 = chunked promote (FP16 inner, FP32 outer) recovers correctness while preserving most of the register win.

Two secondary investigations: (a) the INT4 LUT-vs-bitshift hypothesis (cheap shift+mask plus a 16-entry LUT vs full LUT) and (b) the spill-blocked INT4 BN=256 tile.

## Use cases

- **DSv4-Flash MoE matmul, M ∈ {64, 256, 1024, 2048}, N=K=7168, all four weight formats.** Bit-correctness against the sprint-015 P2 tolerance contract is non-negotiable.
- **Profiling** with `/usr/local/cuda/bin/ncu` on gpu-01 (DCGM-exporter paused before each run; restored after).

## Architecture

### P0 — v9 chunked mixed-precision accumulator

New file: `tools/tc-grid/kernels/v9_kernels.cuh`. Same structure as v3/v4 except:
- `FragC_half = wmma::fragment<accumulator, 16, 16, 16, half>` — inner-loop accumulator.
- `c_h[FRAG_M][FRAG_N]` of `FragC_half` resets every `CHUNK_K` K-tiles.
- Maintain `c_f[FRAG_M][FRAG_N]` of `FragC_float` (the float accumulator) summed across chunks.
- Promote: for each (fm, fn) pair, convert each of the 8 elements of `c_h[fm][fn]` to float and add into `c_f[fm][fn]`. Do this serially per (fm, fn) to bound peak register usage to one half + one float fragment alive at a time.
- Choose CHUNK_K by sweep (candidates: 2, 4, 8). K=7168 / BK=32 = 224 k-tiles → 56/112/28 promote events respectively.
- Final write: store float c_f to SMEM with FP32 store_matrix_sync, then thread-stride to gmem (same pattern as v3).

Launcher: extend `tools/tc-grid/src/launch_int8.cu` with `LAUNCH_V9` / `RERUN_V9` macros following the existing v7/v8 patterns; `smem_bytes_opt_v9 = smem_bytes_opt_v3`.

Apply v9 to all four formats (INT8, INT4, MXFP4, F8) — same dequant front-end, only the accumulator type and chunked-promote epilogue change. Five files: `v9_kernels.cuh` (INT8 path), `v9_int4_kernels.cuh`, `v9_fp4_kernels.cuh`, `v9_fp8_kernels.cuh`. Or one templated kernel if the dequant lambda can be parameterized — prefer four files for clarity.

### P0 — INT4 LUT-vs-bitshift focused ncu

No new code. Just a focused ncu sweep on existing `kernels::int4_v3::mm_int4_lut_v3` (LUT path) vs `kernels::int4_v3::mm_int4_bitshift_v3` (bitshift). Metrics:
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `launch__registers_per_thread`, `launch__shared_mem_per_block_dynamic`
- `smsp__warp_issue_stalled_short_scoreboard*` (register), `_long_scoreboard*` (memory)
- `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed` (compute pipe usage)

Decision rule: keep both if they win different M ranges; drop the loser otherwise.

### P0 — INT4 BN=256 spill remediation

Investigate `mm_int4_lut_v3<128,256,32,4,8,4>`: 1664-byte spill stores, 1472 spill loads. Hypotheses (test each by isolated build):
1. Too many c_frags alive (8×4 = 32 frags × 8 regs = 256 regs just for accumulator). **Fix candidate:** chunk the c_frag set — process FRAG_N=4 in two passes of FRAG_N=2 each (re-load A inside the outer pass to keep half the c_frags alive). 
2. A/B fragment storage in registers is over-prefetched. **Fix candidate:** load only one A fragment at a time inside the k-loop.
3. Dequant LUT held in register. **Fix candidate:** ensure LUT is in SMEM not registers (verify by inspecting PTX).

If none of these clears the spill, document the architectural blocker and defer BN=256 to SPRINT-017.

### P1 — multi-shape MoE validation

Extend `main.cu` shapes beyond (M ∈ {64,256,1024,2048}, N=K=7168). Add two DSv4-relevant shapes from the v25a/v25b MoE call sites. Verify v4 (or v9) wins generalize.

### P2 — XOR swizzle (deferred unless P0 closes <50% of gap)

If after P0 the INT8 best is still below ~35 TFLOPS (50% of remaining gap to cuBLAS 98 TFLOPS), revisit XOR-swizzle in a new sprint. Out of scope for SPRINT-016.

## Implementation phases

| Phase | Task | Acceptance | ETA |
|---|---|---|---|
| P0.1 | v9 INT8 kernel + launcher wiring | builds; CHUNK_K=4 passes bit-correctness gate at M=2048 | 1 build cycle |
| P0.2 | CHUNK_K sweep (2, 4, 8) at M ∈ {64..2048} | best CHUNK_K identified per (format, M) | 1 build cycle |
| P0.3 | v9 INT4 / MXFP4 / F8 variants | same gate at all four formats | 1 build cycle |
| P0.4 | INT4 LUT-vs-bitshift focused ncu | decision on path retention | 1 ncu pass |
| P0.5 | INT4 BN=256 spill triage | working BN=256 variant OR documented blocker | 2 build cycles |
| P1.1 | multi-shape MoE validation | v9 wins on ≥1 extra shape | 1 build cycle |
| Done | REPORT-10 written; ledger updated | summary table, ncu breakdown, next-sprint pointer | — |

## Files summary

New:
- `tools/tc-grid/kernels/v9_kernels.cuh`
- `tools/tc-grid/kernels/v9_int4_kernels.cuh`
- `tools/tc-grid/kernels/v9_fp4_kernels.cuh`
- `tools/tc-grid/kernels/v9_fp8_kernels.cuh`
- `tools/tc-grid/docs/REPORT-10.md`

Modified:
- `tools/tc-grid/src/launch_int8.cu` (LAUNCH_V9, RERUN_V9, smem calc)
- `tools/tc-grid/src/launch_int4.cu` (v9 dispatch)
- `tools/tc-grid/src/launch_fp4.cu` (v9 dispatch)
- `tools/tc-grid/src/launch_fp8.cu` (v9 dispatch)
- `tools/tc-grid/src/main.cu` (v9 tile entries, multi-shape if P1.1 runs)

## Definition of Done

1. v9 INT8 (CHUNK_K=4 or best from sweep) bit-correct at M=2048: `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`. Must beat v4 by ≥3% TFLOPS to justify retention.
2. v9 for all four formats either ships or is documented as "no win over v3/v4 baseline" with measured numbers.
3. INT4 LUT-vs-bitshift decision: documented winner per (M) with quantitative basis.
4. INT4 BN=256: working bit-correct variant in grid, OR documented blocker with PTX evidence.
5. Updated grid sweep CSVs (`run_M*.csv`) including v9.
6. REPORT-10 written with ncu stall breakdown for v9-best vs v4-best (Short Scoreboard, Long Scoreboard, Barrier, register count).
7. Per-row tolerance gate continues to hold for *every* OK-status row in the sweep (regression guard).

## Risks

- **R1 — Mixed-precision promote burns the win**. Temporary 2x c_frag during conversion may push register count back up. Mitigation: serialize the promote per (fm,fn) — only one half + one float frag alive at a time. Validate by checking `launch__registers_per_thread` in ncu after v9 builds.
- **R2 — Precision still degrades at high K**. If CHUNK_K=2 still fails the contract, the chunked approach can't work and we fall back to v4. Mitigation: CHUNK_K=2 is the floor; if it fails, kill v9.
- **R3 — BN=256 spill is unresolvable without algorithmic restructure**. Acceptable outcome: documented blocker → SPRINT-017.
- **R4 — DCGM-exporter accidentally re-enabled during the sprint** (some manifest applies the gpu-operator default label). Mitigation: re-check `kubectl get nodes gpu-01 -o jsonpath='{.metadata.labels.nvidia\.com/gpu\.deploy\.dcgm-exporter}'` before each ncu run.

## Security

Standard. No network exfil. ncu pods only on gpu-01 (already in cluster). No new credentials. No external dependencies beyond the existing `localhost:32000/rotorquant:tc-grid` registry image, but we're now using the `tcg-dev` pod for builds, which mounts the host source dir — confirm `/srv/dev/dsv4-cuda/deepseek-sprint017` is the intended path before each significant edit.

## Dependencies

- `tcg-dev` pod live (already running).
- DCGM-exporter paused before ncu runs (`kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite`) and restored after (`=true`).
- CUDA 12.2.2 in pod (already installed at session start).
- No external APIs.

## Open questions

1. CHUNK_K sweep — 2 / 4 / 8 likely covers the space; deeper sweep only if results are ambiguous.
2. If v9 wins at high M but loses to v3_b1 at M=64 (no surprise — small M doesn't have enough K-tiles to amortize promote overhead), do we ship a dispatch table or stick with v4 + v3_b1 as today? Likely former; trivial dispatch added in launcher.
3. Multi-shape MoE: which DSv4 shapes are actually called in the production decode path? Need an authoritative list from the v25a build artifacts. If unavailable, default to {1024, 4096, 8192} × N=K=7168 as a stress sweep.

---

## Sprint closure (2026-05-13)

### What actually shipped

| component | planned | actual | TFLOPS |
|---|---|---|---|
| INT8 v9 mixed-precision | ship CHUNK_K-best, +3% over v4 | **broken-by-design**, rel=0.71 across the board (WMMA half/float frag lane-mapping mismatch on sm_70) | n/a |
| INT8 v10 row-major B | not planned | **NEW PRODUCTION CHAMPION**, +6.4% vs v4 baseline | 29.49 @ M=2048 |
| HMUL dequant on v10 | not planned | bit-correct +1.4% on top | 29.32→29.49 |
| INT4 LUT-vs-bitshift | choose winner | LUT kept; bitshift dropped | n/a |
| INT4 BN=256 spill | working variant OR documented blocker | **blocker documented** (256 c_frag regs > 255 sm_70 cap) | n/a |
| Multi-shape MoE validation | P1 stretch | not run | n/a |

### Definition-of-Done audit

1. ✅ INT8 best at M=2048 bit-correct (`rel=2.59e-4`, well inside contract) — but via v10, not v9.
2. ⚠️ v9 for all four formats: shipped to tree gated off; documented as broken in REPORT-10/11. v10 row-major B is the actual replacement.
3. ✅ INT4 LUT-vs-bitshift documented (REPORT-10).
4. ✅ INT4 BN=256 architectural blocker documented (>255 reg cap). Two-pass FRAG_N fix deferred to SPRINT-017.
5. ✅ Full-spectrum CSV sweep (`full_M{16..4096}.csv`) — 882 OK rows, 783 bit-correct, 99 bit-FAIL all attributable to v9.
6. ✅ REPORTs 10 + 11 written.
7. ✅ Per-row tolerance gate held for every non-v9 OK row.

### Sprint-goal verdict

**Did NOT hit "35+ TFLOPS" target.** Peak 29.49 TF (+6.4% vs v4 baseline). The `wmma::*` API
path is now exhausted at this ceiling per ncu measurement (Short Scoreboard 21.5%, MIO 10.9%,
Long Scoreboard 22.7%, TC% 23.6). Path to 50 TFLOPS requires dropping `wmma::load_matrix_sync`
entirely — captured in SPRINT-017 / V11-DESIGN.md.

### Carry-forward

- **SPRINT-017** (active): v11 m8n8k4 inline PTX + XOR-swizzled SMEM. Roadmap in
  `tools/tc-grid/docs/V11-DESIGN.md`. Goal: 50 TFLOPS on V100 INT8.
- **SPRINT-016-FOLLOWUPS.md**: v9 SMEM round-trip variant, INT4 BN=256 two-pass FRAG_N,
  multi-shape MoE validation — all still open.
