# SPRINT-016 INTENT — V100 INT8/FP4/FP8 throughput

## Seed
Cut the gap between our 27.7 TFLOPS bit-correct INT8 best and cuBLAS's 98 TFLOPS FP16 ceiling on V100. Empirical Tier B work (Reports 1-9) shows: v4 (= v3 + Tier S) wins INT8 at M≥256; v3 wins FP4/FP8/INT4; v3_b1 (BM=32) wins at M≤64. ncu identifies register pressure (Short Scoreboard 30%) as the dominant stall — NOT bank conflicts, which are mathematically impossible to zero on V100 col-major B via padding alone (WMMA requires `ldm % 8 == 0` → minimum 4-way conflict). v8 (FP16 acc) gave 3% speedup + zero spills but broke bit-correctness (rel=5.5e-3 vs ≤1e-3 contract).

## Orientation summary
- **Current state**: tc-grid harness mature, 385 (variant × path) per M tested. 8 kernel versions (v0-v8), v4/v3 production winners. cuBLAS at 98 TFLOPS = our ceiling target.
- **Recent direction**: Sprints 013-015 closed compression and tolerance contracts. Sprint 015 P2 spec §7 defines per-direction FP16 tolerance for the broader DSv4-Flash integration. This sprint focuses on raw kernel throughput.
- **Key modules**: `tools/tc-grid/kernels/{v3,v4,v5,v6,v7,v8}_kernels.cuh`, `tools/tc-grid/src/launch_{int8,int4,fp4,fp8}.cu`, `tools/tc-grid/src/main.cu`. ncu CSVs in `tools/tc-grid/docs/ncu_M*.csv`. Reports 1-9 in same dir.
- **Constraints**: single V100 (gpu-01, sm_70); `tcg-dev` pod live with hostpath `/srv/dev/dsv4-cuda/deepseek-sprint017` mounted at `/src` (avoid Job churn); xattr `com.apple.provenance` blocks rsync ownership preservation, mitigate with chown after first transfer; tolerance contract rel ≤ 1e-3 strict; DCGM-exporter must be paused on gpu-01 before ncu (`kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite`).
- **No VISION.md**. No prior SPRINT-NNN.md files in docs/sprints/ — sprint planning was previously informal in tools/tc-grid/docs/REPORT-N.md.

## Candidate work items (from Report 9 + user input)

### P0 — pursue
- **W1: v9 chunked mixed-precision** (FP16 inner / FP32 promote per CHUNK_K=4 K-tiles). Targets the dominant Short Scoreboard 30% stall directly. Expected ~5-10% win over v4 if precision is preserved; net could be 30 TFLOPS INT8. Risk: temporary 2x c_frag regs during promote may eat the win.
- **W2: INT4 LUT-vs-bitshift focused ncu run**. User hypothesis: 16-entry LUT may reduce SMEM pressure and ALU work vs bitshift unpack. Validate or kill. ncu metrics: `dram__throughput`, `launch__registers_per_thread`, `smsp__warp_issue_stalled_short_scoreboard*`.
- **W3: INT4 BN=256 spill remediation**. ncu shows v3 INT4 128x256 spills 1664 B → BN=256 currently dropped from grid. If spills can be reduced (smaller c_frag region, fewer simultaneous frags), BN=256 may unlock 10-15% throughput at large M.

### P1 — conditional
- **W4: multi-shape MoE validation**. Verify v4-wins-INT8 generalizes from N=K=7168 to other DSv4 MoE call shapes. Add 2-3 shape entries to harness.
- **W5: persistent-CTA grid-stride output loop** (v5 was Tier A3, slightly underperformed v3 — re-examine with fresh register-pressure lens).

### P2 — defer to SPRINT-017 unless P0 closes <50% of gap
- **W6: XOR swizzle + manual fragment load** (bypassing `load_matrix_sync`). ~300-400 lines per kernel variant. Only justified if v9 / INT4 LUT / BN=256 combined fall short.
- **W7: int8_v9 alt — store partial c to SMEM, free regs mid-K-loop**.

## Success criteria
1. v9 implemented; bit-correct (rel ≤ 1e-3); benchmarked at M ∈ {64, 256, 1024, 2048}.
2. INT4 LUT vs bitshift comparison delivers a quantitative answer (drop, keep, or replace).
3. BN=256 spill investigation produces either a working BN=256 variant or a documented "blocked by register architecture" note with concrete blocker.
4. Updated full-grid sweep CSVs (`run_M*.csv` v2) with v9 included.
5. REPORT-10 with: ms/TFLOPS table at all M values, ncu stall breakdown for v9-best vs v4-best, generalization checks for at least one extra DSv4 shape.

## Verification strategy
- Per-row tolerance gate in tc-grid harness (already implemented): `maxabs ≤ 0.1 ∧ p99 ≤ 0.05 ∧ rel ≤ 1e-3`.
- ncu instrumented profiling for top configs after each kernel change (single GPU on gpu-01, DCGM paused).
- All work iterates against `tcg-dev` pod; no Jobs.

## Uncertainty
- **Correctness — Medium**: v9 mixed-precision design is conceptually clear but precision-preserving promote schedule (CHUNK_K) is a tuning parameter; choose by measurement.
- **Scope — Medium**: W1 alone is bounded; if W1 underperforms we widen to W6/W7.
- **Architecture — Low**: Stays within existing kernel/launcher structure. New file `kernels/v9_kernels.cuh` + launcher gating.

## Open questions for interview
1. Confirm sprint-015 P2 tolerance rel ≤ 1e-3 is the binding constraint for v9 promotion to production (vs a looser "good enough for DSv4-Flash" threshold).
2. If v9 closes <50% of the cuBLAS gap and W6 (XOR swizzle) is the only remaining lever, escalate to a SPRINT-017 dedicated to manual fragment loads, or accept the current ceiling?
3. INT4 LUT result presentation: keep both LUT and bitshift paths in production, or pick one based on benchmarks?
