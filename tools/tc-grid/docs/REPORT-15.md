# REPORT-15 — Precision regime sweep on V100 (turbomind FP16/FP8/U4 + v12 INT8)

Date: 2026-05-15
Status: SPRINT-021 P0 measurement complete. Triangulates the
"compute-bound" claim from REPORT-14 with ncu counter data across four
precision regimes on the same shape (M=2048 N=K=7168).

Companions:
- [REPORT-14.md](./REPORT-14.md) — SPRINT-020 close (FP16 ceiling only)
- `turbomind-packed-ceilings-SPRINT-021-P0.csv` — TFLOPS table across configs
- `ncu-precision-comparison-SPRINT-021-P0.txt` — raw ncu output

---

## 1. Headline

**v12_ms3 INT8 is HMMA-issue-rate bound, not bandwidth bound.** ncu counters
on the same shape (M=2048 N=K=7168, V100 SXM2-32GB) show:

| Config | TFLOPS | HMMA active% | DRAM% | SMEM bw% | Regs/thread | Warp lat (cyc) |
|---|---:|---:|---:|---:|---:|---:|
| **v12_ms3 INT8** (tc-grid) | 38.98 | **31.72%** | 9.40 | 48.26 | **152** | **7.63** |
| turbomind FP8 (Config_E4M3) | 59.07 | 55.70% | 8.76 | 43.34 | 244 | 3.59 |
| turbomind U4 (Config_U4_g) | 64.67 | 60.94% | 7.87 | 38.63 | 242 | 3.80 |
| turbomind FP16 (cuBLAS s884) | 87.05 | **83.53%** | 18.43 | 55.64 | 254 | 3.62 |

Per-config bandwidth: v12 INT8 = 1 B/wt, FP8 = 1 B/wt, U4 = 0.5 B/wt,
FP16 = 2 B/wt.

**Three decisive observations:**

1. **TFLOPS tracks HMMA active% linearly.** 31% → 39 TF, 56% → 59 TF,
   61% → 65 TF, 84% → 87 TF. The mainloop's ability to keep HMMA pipes
   busy is the single biggest perf lever.

2. **DRAM is NEVER the bottleneck.** Even FP16 at 87 TF uses only 18%
   DRAM. v12 INT8 at 39 TF uses 9%. The bandwidth advantage of
   quantization is *unused*; the compute pipeline can't consume it.

3. **v12_ms3 has 2× more warp stall** (7.63 cyc vs 3.59-3.80 cyc).
   Combined with lower register count (152 vs 244+), this points at
   instruction-issue serialization — the dequant chain is waiting
   on dependent loads.

---

## 2. The DSv4 implication

DSv4-flash is an FP4/FP8 model. The native precision options for V100
sm70 inference now have measured ceilings:

| DSv4 strategy | Ceiling at M=2048 | vs current v12 INT8 |
|---|---:|---:|
| Continue v12 INT8 dequant path | 38.98 TF (today) | baseline |
| Port to turbomind Config_E4M3 (FP8) | 59 TF | **+52%** |
| Port to turbomind Config_U4_g (U4) | 65 TF | **+66%** |
| Pure FP16 (no quant) | 87 TF | +123% but 2× memory |

**Strategic option for DSv4 deployment**: rather than continuing to
push v12 INT8 toward the 50 TF goal, port the V100 inference path to
turbomind's FP8 e4m3 mainloop. The model is already trained in FP8;
no precision conversion needed; immediate +52% throughput.

This wasn't visible during SPRINT-020 because FP8/U4 weren't measured.
The "50 TF goal" framing assumed INT8 was the only quantization
choice. It isn't.

---

## 3. What was measured

### 3.1 Bench infrastructure

`tools/tc-grid/turbomind_minimal/gemm_bench_packed.cu` — nvbench-free
direct invocation of `turbomind::gemm::Gemm::Run`, with weight
packing via `GetConverters()` + `Convert()` matching
`models/linear_weight.cc::prepare()`'s "General quantization format
conversion path". Supports `--configs fp16,fp8,u4,fp4` via CLI flag.

Each config:
- fp16 → no quant, fp16 weights, dispatches to `cutlass_70_tensorop_f16_s884gemm`
- fp8  → Config_E4M3 packed weights + fp16 scales (Operand_V_Pack<uint16_t>)
- u4   → Config_U4_g packed weights + fp16 scales (Operand_V_Pack<uint32_t>)
- fp4  → Config_MXF4; **requires group_size=32**, default 128 fails registry match (FOLLOWUP)

### 3.2 TFLOPS catalog

`turbomind-packed-ceilings-SPRINT-021-P0.csv` — 60 rows covering:
- fp16: full M×{(N,K)} catalog
- fp8: full catalog
- u4: square shapes only (asymmetric N≠K hits illegal memory access — FOLLOWUP)

Headline TFLOPS at M=2048 N=K=7168 (median-of-5):
- fp16: 87.05
- fp8: 59.07
- u4: 64.67
- v12_ms3 INT8 (carried from SPRINT-019): 38.98

### 3.3 ncu metric pack

`ncu-precision-comparison-SPRINT-021-P0.txt` — `ncu --metrics ...`
captures of one mainloop kernel launch per config. Common metrics:
- `sm__pipe_tensor_op_hmma_cycles_active.sum.pct_of_peak_sustained_elapsed`
- `sm__cycles_active.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum.pct_of_peak_sustained_elapsed`
- `sm__warps_active.avg.pct_of_peak_sustained_active`
- `launch__registers_per_thread`
- `smsp__average_warp_latency_per_inst_issued.ratio`

ncu protocol: pause DCGM exporter via node label (`nvidia.com/gpu.deploy.dcgm-exporter=false`)
before each run; restore after. `-k regex:gemm_kernel` for the
turbomind packed kernels, no `-k` filter for the cuBLAS path (different
launch sequencing).

---

## 4. Where the gap to FP8 comes from (v12 → +52%)

v12_ms3 hits 31% HMMA active; turbomind FP8 hits 56%. Both use the
same `mma.m8n8k4.f16` instructions, same V100 HBM, same 1 B/wt
bandwidth budget. The 56/31 = 1.81× gap is purely in how the
mainloop orchestrates instruction issue.

Likely contributors (in order of expected impact):

1. **Dequant chain depth (7.63 vs 3.59 warp latency)**: v12 dequants
   INT8→FP16 via SMEM LUT round-trip. The LUT lookup has latency
   that serializes HMMA issue. Turbomind's `Transform_HMMA_SIMT_B`
   does the dequant in registers between SMEM load and HMMA, with
   no SMEM round-trip.

2. **Register budget underutilization**: v12 uses 152 regs/thread;
   turbomind kernels use 244-254. v12 has 100+ free registers to
   stage more deeply unrolled HMMA chains. This is the unused
   headroom REPORT-14 §6 flagged.

3. **Pack layout**: turbomind packs B into a kernel-friendly layout
   that allows wider HMMA fragments per LDS. v12 reads INT8 in
   row-major with a separate dequant kernel pattern.

The path to closing 39 → 50 TF (project goal) requires lifting v12's
HMMA active from 31% to ~40%. That's a single inner-loop refactor —
move dequant from SMEM round-trip to register-resident.

---

## 5. SPRINT-021 framing — concrete next step

Two parallel paths, both informed by these measurements:

**Path A — Lift v12_ms3 HMMA active to 40%+**
- Move dequant LUT from SMEM to registers (eliminate the round-trip)
- Wider HMMA chain unroll (use the 100 spare registers/thread)
- Target: 38.98 → ≥ 50 TF at M=2048 N=K=7168

**Path B — DSv4-flash FP8 port via turbomind**
- Wire DSv4 inference path through turbomind's FP8 e4m3 kernel
- Skip the INT8 quantization step entirely (model is FP8 native)
- Target: 59 TF baseline; expand to 65+ TF if turbomind U4 wins on
  any DSv4 layer shape

Path B is the higher-leverage move *for DSv4 deployment specifically*.
Path A keeps v12 viable as a general INT8 fallback. SPRINT-021 should
prioritize Path B; Path A continues as Path A2 stretch.

---

## 6. Follow-ups for SPRINT-021 P0+

| Item | Severity | Notes |
|---|---|---|
| U4 illegal memory access at N≠K asymmetric shapes | Important | only square shapes (N=K) work; debug the q_desc swap or partials sizing for N=18944 |
| MXF4 (FP4) needs group_size=32 not 128 | Nice-to-have | registry shows MXF4 only registered with `, 1, 32, ...` in sm70_884_8.cu; add per-config group_size override |
| ncu metric pack save script as committed asset | Nice-to-have | currently in /tmp/run_ncu3.sh; promote to tools/tc-grid/scripts/ |
| Asymmetric DSv4 shapes for U4/FP8 head-to-head vs v12 | Important | once U4 N≠K bug fixed; complete the comparison table |
