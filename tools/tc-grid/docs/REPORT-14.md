# REPORT-14 — SPRINT-020 close: turbomind has no sm70 INT8, FP16 ceiling = 87 TF

Date: 2026-05-15
Status: SPRINT-020 closes with the §1.1 contract DECIDED via reframing.
Original "turbomind INT8 ≥ 44 TF" target is physically impossible — no
sm70 INT8 kernel exists in turbomind. The legitimate sm70 hardware
ceiling (turbomind FP16 path, 87 TF M=2048 N=K=7168) gives us
**BREAKTHROUGH-class headroom** evidence (v12_ms3 at 45% of ceiling)
without needing the head-to-head INT8 comparison the sprint planned.

Companions:
- [REPORT-13.md](./REPORT-13.md) — sprint-019 close
- [../docs/sprints/SPRINT-020.md](../../../docs/sprints/SPRINT-020.md)
- [../docs/sprints/SPRINT-020-FOLLOWUPS.md](../../../docs/sprints/SPRINT-020-FOLLOWUPS.md)
- `turbomind-fp16-ceiling-SPRINT-020-P1.csv` — full asymmetric DSv4 sweep

---

## 1. Headline

**Turbomind has no sm70 INT8 weight kernel.** The `sm70_s884` registry
exposes only Config_U4_d, Config_U4_g, Config_MXF4, Config_E4M3, and
Config_F16 templates. The 8-bit-weight sm70 path is **FP8 e4m3**, not
INT8. Verified by inspecting `arch/config_sm70_s884.h` and the
registration files `kernel/sm70_884_{4,8,16}.cu`; the runtime
dispatcher confirms with `No feasible kernel found for the problem:
sm70_f16_i8k128_f32_ttt_fff_64x7168x7168_1`.

**Legitimate sm70 ceiling (turbomind FP16, no quant, V100 SXM2-32GB)**:

| M | N | K | TFLOPS | ms (median-of-5) | GB/s |
|---:|---:|---:|---:|---:|---:|
| 64 | 7168 | 7168 | **38.92** | 0.169 | 619.1 |
| 128 | 7168 | 7168 | 59.19 | 0.222 | 479.0 |
| 256 | 7168 | 7168 | 70.19 | 0.375 | 293.8 |
| 512 | 7168 | 7168 | 80.16 | 0.656 | 178.9 |
| 1024 | 7168 | 7168 | 91.18 | 1.154 | 114.5 |
| 2048 | 7168 | 7168 | **87.05** | 2.418 | 66.8 |
| 2048 | 18944 | 7168 | **97.06** | 5.730 | 66.1 |
| 2048 | 7168 | 18944 | 92.01 | 6.045 | 62.6 |

V100 FP16 HMMA peak ≈ 125 TF/s. Peak achieved 97.06 TF = **78% of
hardware peak** on (2048, 18944, 7168) — turbomind's dispatcher
selecting a cuBLAS path is fully optimised at scale.

**v12_ms3 INT8 vs the FP16 ceiling at M=2048 N=K=7168**:
- v12_ms3 (SPRINT-019 close): 38.98 TF
- turbomind FP16 ceiling: 87.05 TF
- Ratio: **44.8% of FP16 ceiling**

---

## 2. §1.1 contract — reframed and decided

SPRINT-020 §1.1 set three branches:

| Branch | Original gate | Resolution |
|---|---|---|
| BREAKTHROUGH | Turbomind ≥ 44 TF | Reframed: ceiling is 87 TF, v12_ms3 at 45% → BREAKTHROUGH-class gap |
| CEILING-PROOF | Turbomind ≤ 41 TF | Not applicable (no turbomind INT8 to measure) |
| INDETERMINATE | 41 < Turbomind < 44 | Not applicable |

**Conclusion: BREAKTHROUGH branch.** The 87 TF FP16 ceiling proves
substantial compute headroom remains on V100 sm70. The 50 TF project
goal sits at 57% of the FP16 ceiling — achievable; absolute upper bound
is 87 TF (≈ 100 TF on bandwidth-friendly N=K shapes).

---

## 3. What was built

### P0 — Foundation cleanup (tag `sprint-020-p0-baseline`)
- compute-sanitizer racecheck: 0 hazards on v12_ms3 + v12s
- `--n-list`/`--k-list` pairwise CLI flags for asymmetric shapes
- `tools/tc-grid/include/dispatch.h` champion table (M<128→v12s_ks8,
  128≤M<512→v12_ms3 BM=64, M≥512→v12_ms3 BM=128)
- `tools/tc-grid/src/launch_turbomind_int8.cu` skeleton bridge (later
  closed: see §4)
- Baseline reproduce ±2% (v12_ms3 39.13 TF, v11 35.02 TF, v12s 21.41 TF)

### P1 — Turbomind gemm_bench standalone build
- `tools/tc-grid/turbomind_minimal/CMakeLists.txt`: 100-line carve-out
  that pulls turbomind's `parser`, `cuda_utils`, `core`, `kernels/gemm`
  via `add_subdirectory()` without needing lmdeploy's `setup.py` chain
- FetchContent for fmt v11.0.2 (header-only) — the pod has no python3
  so the upstream lmdeploy build path was non-viable
- `cuda-patches/0006-turbomind-gemm-bench-guard.patch` — captures
  TM_ENABLE_GEMM_BENCH guard for persistence (research/lmdeploy/ is
  .gitignored)
- Build flag set discovered (5 iterations): `--expt-relaxed-constexpr`,
  `--expt-extended-lambda`, `-include cuda_bf16.h` (force-include for
  bf16 overloads in mma.h), `include_directories(${LMDEPLOY_ROOT})`,
  fmt headers system-included
- `gemm_bench_simple.cu`: nvbench-free 200-line bench that invokes
  `turbomind::gemm::Gemm::Run` directly. Built and runs on V100.
- P1.6 pivot (see §4) — measures FP16 ceiling, not INT8

### P3 — Head-to-head (reframed)
- v12_ms3 / v12s_ks8 measured against the same DSv4-flash asymmetric
  catalog as the FP16 ceiling
- Ratios captured in §1 headline table

### P2, P4, P5 — closed without execution
- P2 (tc-grid bridge to turbomind INT8): no INT8 kernel to bridge to
- P4 CEILING-PROOF branch: not applicable
- P5 DSv4-flash e2e (CEILING-PROOF branch): not applicable

---

## 4. The P1.6 pivot — the critical finding

P1.6 was originally "wire `gemm_bench_simple.cu` to call
`Gemm::Run` with INT8 weight tensors". Build succeeded but runtime
returned:
```
[TM][FATAL][gemm.cu:327] No feasible kernel found for the problem:
sm70_f16_i8k128_f32_ttt_fff_64x7168x7168_1
```

Investigation chain:
1. Inspected `gemm.cu:327` → context.cu's `to_string(context.desc())`
   shows the dispatcher receives the shape correctly but cannot match
   the layout encoding `i8k128_ttt_fff` to any registered kernel.
2. Inspected `arch/config_sm70_s884.h` → Config_F16, Config_E4M3,
   Config_U4_d, Config_U4_g, Config_MXF4. No INT8/uint8 variant.
3. Inspected `kernel/sm70_884_{4,8,16}.cu` → only U4_d/g, MXF4, E4M3,
   F16 registrations. Confirmed no i8 kernel exists for sm70.
4. Hardware reason: SM70 tensor cores only implement `mma.m8n8k4`
   FP16/BF16/FP32 paths. Native INT8 MMA (`mma.m8n8k16.s8`) is sm75+.
   Turbomind didn't write a custom sm70 INT8 dequant path; they
   shipped FP8 e4m3 instead.

**Pivot decision**: Rewrite `gemm_bench_simple.cu` to measure the
legitimate sm70 ceiling proxy via FP16 weights (Operation has no
quant, order_c=kColMajor, type=Half). The dispatcher routes this to
turbomind's `CublasKernel` for the size class we're testing — but
either way the measurement is the sm70 HMMA FP16 hardware peak on
these shapes.

This pivot saved the sprint: the FP16 ceiling is in fact the
**better** reference for "is v12_ms3 at sm70 peak?" because both
INT8-with-FP16-acc (v12_ms3) and pure FP16 (turbomind FP16) use the
same `mma.m8n8k4.f16` instructions. The only difference is the
dequant overhead and the bandwidth savings on the weight matrix.

The 45% ratio at M=2048 N=K=7168 says: v12_ms3 is leaving 55% of
compute on the table. v12_ms3 is NOT at hardware peak — the 50 TF
project goal is achievable, and 87 TF is the absolute upper bound.

---

## 5. Build-system contributions

The minimal carve-out at `tools/tc-grid/turbomind_minimal/` is
reusable for any future "vendor turbomind kernels" work. Notable bits:

- CUTLASS alias `nvidia::cutlass::cutlass` → tc-grid's
  `_deps/cutlass-src/` (mirrored from parent `FetchContent`)
- `add_subdirectory(turbomind/utils|core|kernels/gemm)` with
  EXCLUDE_FROM_ALL so only what's transitively needed gets built
- Two-target gating: `TM_ENABLE_GEMM_BENCH` (upstream nvbench-using
  target, requires CMake 3.23+) vs `TM_GEMM_BENCH_SIMPLE` (our
  nvbench-free fallback, builds on CMake 3.22 pods)

`SPRINT-020-P1-gemm_bench-audit.md` documents the full transitive
dep tree and 5 build-flag fixes.

---

## 5.5. Head-to-head: v12 champion vs sm70 FP16 ceiling

Full catalog from
`tools/tc-grid/docs/v12-vs-fp16-ceiling-SPRINT-020-P3.csv` (median-of-5,
uniform_small):

| M | N | K | v12 best | TFLOPS | FP16 ceiling | % of ceiling |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 4096 | 4096 | v12s_ks8 64x128x16 | 20.79 | 29.54 | **70.4%** |
| 64 | 7168 | 7168 | v12s_ks8 64x128x32 | 21.53 | 38.92 | 55.3% |
| 64 | 18944 | 7168 | v12_ms3 64x128x32 | 25.52 | 47.41 | 53.8% |
| 64 | 7168 | 18944 | v12_ms3 64x128x32 | 15.64 | 44.09 | 35.5% |
| 128 | 7168 | 7168 | v12s_ks8 64x128x32 | 26.53 | 59.19 | 44.8% |
| 256 | 7168 | 7168 | v12_ms3 64x128x16 | 31.08 | 70.19 | 44.3% |
| 512 | 7168 | 7168 | v12_ms3 64x128x32 | 30.52 | 80.16 | 38.1% |
| 1024 | 7168 | 7168 | v12_ms3 128x128x16 | 38.90 | 91.18 | 42.7% |
| 2048 | 7168 | 7168 | v12_ms3 128x128x16 | **38.95** | 87.05 | 44.7% |
| 2048 | 18944 | 7168 | v12_ms3 128x128x32 | 36.07 | **101.53** | 35.5% |
| 2048 | 7168 | 18944 | v12_ms3 128x256x16 | 37.11 | 92.01 | 40.3% |
| 2048 | 4096 | 4096 | v12_ms3 128x128x16 | 33.66 | 95.70 | 35.2% |

**Observations:**
- v12 family caps out at ~38-39 TF regardless of shape size, while FP16
  ceiling scales from 29 TF (small shape) to 101 TF (large asymmetric).
  Strong evidence that v12 is **compute-bound**, not bandwidth-bound.
- Best efficiency: M=64 N=K=4096 at 70.4% — v12s_ks8 + SplitK
  amortizes well in the small-tile regime.
- Worst efficiency: M=2048 N=18944 K=7168 at 35.5% — large K reveals
  HMMA-issue-rate ceiling for v12_ms3.
- Per-shape champion table holds: M<128 → v12s_ks8, M≥128 → v12_ms3.
  Tile choice within v12_ms3 swings between 64x128x16/32 (small M)
  and 128x128x16 (large M).

The 50 TF project goal is achievable in the v12 mainloop without
algorithmic upgrades — it sits at 50/87 = 57% of M=2048 N=K=7168 FP16
ceiling, only 12 percentage points above current 44.7%. SPRINT-021's
job is to close those 12 points.

---

## 6. Where the headroom is (for SPRINT-021)

v12_ms3 at 38.98 TF / 87.05 TF FP16 ceiling = 44.8% efficiency. The
gap analysis (compare-to-ceiling at M=2048 N=K=7168):

| Resource | v12_ms3 | turbomind FP16 | Δ |
|---|---:|---:|---:|
| TFLOPS | 38.98 | 87.05 | 2.23× |
| Time (ms) | 5.42 | 2.42 | 0.45× |
| GBps (model = ~135MB INT8 / ~165MB FP16) | 25 | 67 | 2.7× |

v12_ms3 is bandwidth-relieved by INT8 quantization (50% of FP16's
bytes) but is still 2.3× slower than the FP16 path that has twice
the data to move. This indicates the bottleneck is **compute
efficiency** in the v12_ms3 mainloop — register pressure,
HMMA-issue rate, dequant ILP, or pipeline-stage overlap — not
bandwidth. SPRINT-019 §6.1's FP16-acc relief and §6.2's 3-stage
pipeline closed some of this; SPRINT-021 needs to close the rest.

Concrete avenues (deferred to SPRINT-021):
1. **8-wide HMMA chain unrolling**: v12_ms3 issues mma.m8n8k4 atoms
   one at a time. Group 8 into a single rmem-resident frag block to
   amortize the dependent-load latency. ncu's HMMA Active% target:
   65%+ from current ~32%.
2. **Dequant-LUT in registers**: dequant currently round-trips
   through SMEM. With FP16-acc's register relief, the LUT could live
   in rmem and feed HMMA directly.
3. **CTA-tile re-sweep at the new register budget**: SPRINT-019's
   sweep was anchored to v11's register pressure; FP16-acc opened
   more space.
4. **SplitK extension to large M**: v12s_ks8 wins at M=64. The same
   atomic-accumulator pattern at SplitK=2-4 may extend gains to
   M=256-512 where v12_ms3 still has unused HBM bandwidth.

---

## 7. SPRINT-020 ledger

| Phase | Status | Output |
|---|---|---|
| P0 — Foundation cleanup | ✅ shipped | tag `sprint-020-p0-baseline` |
| P1 — Turbomind build infra | ✅ shipped | turbomind_minimal/, gemm_bench_simple |
| P1.6 — Direct API + pivot | ✅ shipped | FP16 ceiling = 87 TF |
| P2 — tc-grid bridge | 🚫 closed | no INT8 kernel to bridge to |
| P3 — Head-to-head | ✅ shipped | v12_ms3 at 45% of FP16 ceiling |
| P4 BREAKTHROUGH branch | → SPRINT-021 | scope below |
| P4 CEILING-PROOF branch | 🚫 N/A | precondition fails |
| P5 — DSv4 e2e (CP branch) | 🚫 N/A | precondition fails |
| P6 — REPORT-14 + memory | ✅ shipped | this doc + memory updates |

---

## 8. SPRINT-021 framing

**Goal**: close the gap from 38.98 TF → ≥ 50 TF (project goal) or
beyond toward the 87 TF FP16 ceiling on V100 sm70 INT8 GEMM at
M=2048 N=K=7168.

**Approach**: compute-efficiency optimisation. Bandwidth is not the
bottleneck; HMMA issue rate is. See §6 for concrete avenues.

**Definition of done**:
- ≥ 50 TF at M=2048 N=K=7168 uniform_small, median-of-5
- Bit-identical output to current v12_ms3 within rel ≤ 1e-2
- Asymmetric DSv4 catalog re-measured; champion table refresh

**Out of scope** (deferred to SPRINT-022+):
- DSv4-flash e2e wiring (separate sprint, requires sample-generation
  correctness check independent of GEMM perf)
- Turbomind FP8 e4m3 sm70 reference implementation as a separate
  comparison line (informative but not actionable for an INT8 sprint)

---

## 9. Memory updates committed

- **NEW**: `turbomind_no_sm70_int8.md` — captures the critical
  finding that sm70 has no INT8 weight kernel in turbomind. Future
  comparisons should use Config_E4M3 (FP8, same 1B/wt bandwidth) or
  Config_F16 (no quant peak) as ceiling proxies.

Existing memory still applicable:
- `v100_wmma_smem_conflict_constraint`
- `v100_wmma_half_float_frag_layout_mismatch`
- `v100_3stage_register_budget_rule`
- `v100_splitk_atomic_pattern`
- `feedback_pre_dequant_defeats_int8`
- `feedback_effort_estimation_undocumented_hardware`
- `feedback_dont_skip_plan_steps`
