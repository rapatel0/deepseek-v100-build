# Gemini Critique: SPRINT-020-CODEX-DRAFT.md

## Overview
The Codex draft for SPRINT-020 is a well-structured, evidence-driven plan that correctly identifies the transition from local tile-tuning to architectural validation. By focusing on the **Turbomind port path**, it leverages existing high-quality code in the repo (`research/lmdeploy/src/turbomind/`) to test a different mainloop architecture. The "Ceiling Proof vs. Breakthrough" framework is excellent for ensuring the sprint provides value even if performance targets aren't met.

## Strengths
- **Clear Decision Framework:** The use of "Breakthrough" (>= 44.0 TF) and "Ceiling Proof" (<= 41.0 TF) thresholds provides unambiguous success/failure criteria.
- **Leverages Existing Assets:** Utilizing the Turbomind SM70 kernels instead of starting from scratch or doing heavy CUTLASS surgery is a high-leverage move.
- **Hybrid Benchmark Approach:** Bringing `gemm_bench` up first (P1) and then bridging to `tc-grid` (P2) allows for rapid early signal before investing in full harness integration.
- **Asymmetric Shape Focus:** Correctly identifies that `N == K` is a bottleneck in current tooling and promotes MoE-relevant shapes (18944/7168) to first-class targets.

## Weaknesses
- **Bridge Complexity Underestimated:** Adapting `tc_grid::LaunchSpec` to `turbomind::gemm::Operation` in P2.1 might be more involved than "one launcher" suggests, especially regarding quantization layouts (Turbomind often expects specific weight interleaving for INT8).
- **Toolchain Fragmentation:** Adding `TM_ENABLE_GEMM_BENCH` and `TCGRID_ENABLE_TURBOMIND_GEMM` flags adds to the build system's surface area.
- **Sanitizer Scope:** P0.2 clears debt on `v12s` but doesn't explicitly mandate sanitizing the *new* Turbomind bridge code, which is arguably higher risk.

## Gaps in Risk Analysis
- **Quantization Compatibility:** The draft assumes `tc-grid` and Turbomind use compatible INT8 quantization schemes (scales, zero-points, and memory layouts). If Turbomind's SM70 kernels require a specific pre-pack or layout not supported by `tc-grid`'s current generator, P2 will stall.
- **SMEM/L1 Resource Contention:** The draft notes `v12_ms3` is SMEM-bandwidth bound. It lacks a specific risk for Turbomind hitting the same wall or having higher SMEM pressure that limits occupancy on V100.
- **Binary Size/Build Time:** Including Turbomind kernels in `tc-grid` might significantly increase build times for a tool that currently builds quickly.

## Missing Edge Cases
- **M=1 and small M < 64:** While MoE shapes are covered, the ultra-small batch cases typical of highly concurrent serving (where Split-K might actually hurt) are secondary.
- **Non-multiple-of-8 Shapes:** The draft focuses on 7168 and 18944. Performance at "odd" shapes that might occur in different models should be at least mentioned as a potential pitfall for the Turbomind registry.
- **VRAM Pressure:** No mention of the memory overhead for Turbomind's `DispatchCache`. On a V100, this is rarely an issue, but the footprint should be quantified.

## Definition of Done (DoD) Completeness
- **Missing ncu comparison criteria:** DoD mentions "backed by CSV + ncu evidence" but doesn't specify *which* metrics (e.g., SM efficiency, SMEM throughput, or pipeline stalls) must be improved/analyzed for the "Ceiling Proof."
- **Missing Tolerance Bound:** "Within tolerance" is mentioned, but a concrete value (e.g., `max_abs_diff < 1e-2`) should be in the DoD to prevent performance wins that sacrifice correctness.
- **CI/Build Integration:** DoD should include a check that the new `CMake` flags don't break existing `tc-grid` or `turbomind` builds on non-V100 targets.

## Concrete Recommendations
- **Phase P2.5:** Add a "Data Layout Validation" step to ensure Turbomind kernels can consume `tc-grid`'s standard INT8 tensors without an expensive re-layout that masks GEMM performance.
- **Target Metrics:** Explicitly target **>= 40% SM Efficiency** for the breakthrough case.
- **Sanitizer:** Add `compute-sanitizer` validation for `launch_turbomind_int8.cu` to the DoD.
- **File Path Clarification:** Ensure `research/lmdeploy/src/turbomind/kernels/gemm/registry.cu` is updated to allow external registration if the bridge requires it.
