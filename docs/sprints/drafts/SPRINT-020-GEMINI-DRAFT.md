# SPRINT-020 — V100 INT8 GEMM: The Architectural Break

## 1. Overview

SPRINT-019 successfully lifted the V100 INT8 GEMM baseline from 35 TF to **38.98 TF (M=2048)** and reached **21.55 TF (M=64)**. However, it fell short of the 50 TF headline goal. REPORT-13 identifies a new bottleneck: **mio_throttle (SMEM bandwidth)**. The v12 kernel family, despite its 3-stage pipeline, is hitting a ceiling at ~46% of the CUTLASS Gemm70 reference.

SPRINT-020 is the **architectural decision point**. We must choose between incrementalism and a major break. This plan commits to the **Turbomind Port Path** as the primary vehicle to cross 50 TF, while simultaneously clearing the SPRINT-019 technical debt (sanitizers) and preparing for DSv4 MoE deployment.

### 1.1 Headline Targets

- **M=2048**: Reach **≥ 50 TF** via Turbomind port OR provide a definitive "ceiling proof" (e.g., Turbomind also hits ≤ 41 TF).
- **M=64**: Reach **≥ 25 TF** via Turbomind or optimized v12s.
- **Asymmetric MoE Validation**: Validate the champion against the 3 critical DSv4 shapes: 7168×18944, 18944×7168, and 2048×7168.
- **Production Wiring**: Integrate the per-(M, shape) dispatch rules into the DSv4 inference path.

---

## 2. Use Cases

### 2.1 DSv4-Flash MoE Inference
- **Asymmetric Shapes**: Validation of kernels against DeepSeek-V4 MoE expert dimensions (N≠K).
- **Variable M**: Ensuring optimal kernel selection for both decode (M=1) and prefill (M=2048+).

### 2.2 Performance Engineering
- **Ceiling Comparison**: Using Turbomind's highly optimized sm_70 GEMM as the absolute performance reference for the V100 architecture.

---

## 3. Architecture

### 3.1 Current State (v12_ms3)
- 3-stage pipeline (LDG → rmem → STS → SMEM → mma).
- FP16 accumulator.
- Bottleneck: SMEM bank conflicts and bandwidth (mio_throttle).

### 3.2 Proposed Architecture (Turbomind Integration)
- **Turbomind Kernels**: Leverage `research/lmdeploy/src/turbomind/kernels/gemm/` sm_70 specializations.
- **Key Components**:
  - `mainloop_sm70.h`: Specialized V100 mainloop.
  - `iterator_sm70.h`: Memory access patterns optimized for Volta.
  - `scheduler_sm70.cuh`: CTA scheduling logic.
- **Dispatcher**: Version 70 reserved for Turbomind-based kernels.

---

## 4. Implementation

### Phase P0 — Sanitizer Debt & CLI Extension (Rigor & Readiness)
**Goal**: Clear SPRINT-019 debt and prepare `tc-grid` for asymmetric MoE validation.

1. **v12s Sanitizer Pass**: Run `compute-sanitizer --tool {racecheck, initcheck}` on v12s with adversarial KSPLIT factors. Fix any discovered race conditions in the atomic accumulation path.
2. **tc-grid CLI Extension**: Modify `tools/tc-grid/src/main.cu` to support `--n-list` and `--k-list`.
3. **MoE Profiling**: Run the v12_ms3 champion against the 6 DSv4 MoE shapes. Establish the new baseline for asymmetric performance.

### Phase P1 — Turbomind gemm_bench (The Ceiling Proof)
**Goal**: Determine if 50 TF is physically possible with Turbomind's kernels.

1. **Standalone Build**: Build `gemm_bench` from the Turbomind source tree (`research/lmdeploy/src/turbomind/kernels/gemm/test/`).
2. **Benchmark**: Measure Turbomind performance at M=2048 and M=64 on V100.
3. **Decision Gate**:
   - If Turbomind hits ≥ 45 TF → Proceed to P2 (Wholesale Port).
   - If Turbomind hits ≤ 41 TF → Accept v12 ceiling; shift focus to Deployment Integration (P4).

### Phase P2 — Turbomind Port Integration (The Major Break)
**Goal**: Wire Turbomind's sm_70 kernels into `tc-grid`.

1. **Wrapper Development**: Create `tools/tc-grid/kernels/turbomind_wrapper.cuh` to map `tc-grid` layouts to Turbomind template parameters.
2. **Dispatcher Wiring**: Add `LAUNCH_TURBOMIND` macros to `tools/tc-grid/src/launch_int8.cu` (Version 70).
3. **Verification**: Tier-1 CPU-reference tests + `compute-sanitizer`.

### Phase P3 — Asymmetric Optimization & Dispatch
**Goal**: Finalize dispatch rules for all production shapes.

1. **N≠K Tuning**: Optimize tile selection for asymmetric MoE shapes.
2. **Dispatch Rule Encoding**: Implement the per-(M, N, K) dispatch table in `launch_int8.cu`.
3. **Validation**: Full sweep across DSv4 catalog.

### Phase P4 — DSv4 Inference Integration
**Goal**: Move the champion kernels into the actual model serving code.

1. **Code Migration**: Extract the winning kernels and dispatcher from `tc-grid` into the DSv4-flash inference backend.
2. **End-to-End Test**: Verify token generation correctness and latency improvement in a real model run.

---

## 5. Files Summary

### New Files
- `tools/tc-grid/kernels/turbomind_wrapper.cuh`: Integration layer for Turbomind templates.
- `tools/tc-grid/docs/REPORT-14.md`: SPRINT-020 close report.

### Modified Files
- `tools/tc-grid/src/main.cu`: CLI extension for `--n-list`, `--k-list`.
- `tools/tc-grid/src/launch_int8.cu`: Version 70 dispatcher; per-(M, N, K) logic.
- `research/lmdeploy/src/turbomind/kernels/gemm/CMakeLists.txt`: Build updates for standalone bench.
- `common/reasoning-budget.cpp`: (Placeholder) Wiring into inference path.

---

## 6. Definition of Done

- **M=2048**: ≥ 50 TF reached OR Turbomind ceiling documented at < 45 TF.
- **M=64**: ≥ 25 TF reached.
- **Correctness**: All kernels pass `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0` and `compute-sanitizer`.
- **Integration**: Per-(M, shape) dispatch wired into DSv4-flash.
- **Documentation**: REPORT-14 identifies the final architectural winner for V100 INT8.

---

## 7. Risks

1. **Turbomind Build Complexity (HIGH)**: The Turbomind source tree has deep dependencies. Standalone build might take significant time.
   - *Mitigation*: Time-box P1 to 2 sessions; fallback to CUTLASS extension if blocked.
2. **Correctness of External Kernels (MEDIUM)**: Integrating Turbomind might surface layout mismatches.
   - *Mitigation*: Mandatory Tier-1 CPU tests before any benchmark.
3. **v12s Race Conditions (LOW)**: Sanitizer might reveal deep bugs in SPRINT-019's SplitK.
   - *Mitigation*: Prioritize P0 to clear debt before adding new complexity.

---

## 8. Security

- **Sanitizer Enforcement**: Mandatory `racecheck` for all atomic kernels (v12s and Turbomind).
- **No Third-Party Blobs**: All code built from source (`_deps` or `research/`).

---

## 9. Dependencies

- **Hardware**: V100 sm_70 (gpu-01).
- **Source**: `research/lmdeploy/src/turbomind/kernels/gemm/`.
- **Tools**: `ncu`, `compute-sanitizer`, `cmake`.

---

## 10. Open Questions

**Q1: Major direction?**
**Preferred Path**: **Turbomind Port (Hybrid Approach)**.
The v12 family is SMEM-bandwidth bound. Turbomind's sm_70 implementation is the most promising "architectural break" to hit 50 TF. We will use `gemm_bench` as a ceiling proof first (P1), then proceed to a wholesale port if the delta justifies the effort. Parallelly, we ensure deployment readiness for DSv4 MoE shapes.

**Alternatives Considered**:
- **CUTLASS Extension**: Rejected as primary path due to complexity of custom dequant in CUTLASS 2.x on V100.
- **Pure Deployment Integration**: Rejected as primary path because it abandons the 50 TF headline goal.

**Q2: gemm_bench as reference vs port?**
It must be both. `gemm_bench` serves as the "ceiling proof" to validate the investment in a wholesale port.

**Q3: Asymmetric MoE shapes?**
The 6 shapes from SPRINT-019 §6.6 remain the priority, with focus on 7168×18944 and 18944×7168.
