# SPRINT-018 — CUTLASS V100 INT8 GEMM (ceiling proof, NOT production)

**Status: P0+P1 SHIPPED — P2+P3+P4 SCOPED OUT.** Ceiling proven at **85.93 TFLOPS**
at M=2048; production work pivots back to SPRINT-017 V11 to close the gap.

**Key architectural finding (committed to memory as `feedback_pre_dequant_defeats_int8`):**
CUTLASS sm_70 has no fused INT8→FP16 dequant template (mixed-input GEMM is sm_80+ in
CUTLASS). To use CUTLASS on V100, we must pre-dequant W to FP16 in gmem, which doubles
VRAM (28 GB vs 14 GB for DSv4) and doubles HBM read bandwidth — defeating the purpose
of INT8 quantization. **The 85 TF P1 number is a CEILING, not a production path.**

**Pivot:** the fused on-GPU dequant path on sm_70 is hand-rolled (turbomind does it this
way too). That's SPRINT-017 V11. P1's measurement now serves as the target V11 must
close to.

Sprint goal: integrate NVIDIA CUTLASS V100 INT8 GEMM into tc-grid as a candidate
kernel, **hit 50+ TFLOPS at M=2048**, N=K=7168, bit-correct. Published CUTLASS V100
INT8 numbers suggest 60–75 TF ceiling, so 50 TF is a realistic target.

Pairs with: [SPRINT-017.md](./SPRINT-017.md) (V11 hand-rolled),
[V11-EXECUTION-PLAN.md](../../tools/tc-grid/docs/V11-EXECUTION-PLAN.md),
[TURBOMIND-INSIGHTS.md](../../tools/tc-grid/docs/TURBOMIND-INSIGHTS.md).

## Why this sprint exists

SPRINT-017 / V11 (hand-rolled m8n8k4 + XOR swizzle) targets 35–45 TF at M=2048
based on TURBOMIND-INSIGHTS analysis. That likely misses the 50 TF goal even
with full Steps 1–5 completed. CUTLASS's pre-tuned V100 INT8 GEMM is the
production-grade fallback. NVIDIA engineers spent months tuning lane mappings,
register pressure, SMEM swizzles — we get all that for free.

This sprint **does not** replace V11. V11 produces learning + infrastructure
(m8n8k4 PTX wrapper, manual Lds patterns, XOR swizzle) that transfers to future
hardware / future formats. CUTLASS produces a production champion *now* for the
specific goal of 50 TF on V100 INT8.

## Phases — completed scope

| Phase | Task | Result | Status |
|---|---|---|---|
| P0 | CUTLASS 2.11.0 FetchContent + sm_70 build verification | Test PASS on V100 (16384/16384 outputs match) | ✅ SHIPPED |
| P1 | Minimal CUTLASS FP16-in GEMM wired into tc-grid as version=40 candidate; pre-dequant W INT8→FP16 outside the timing loop | **85.93 TF at M=2048, bit-correct** (50 TF goal SMASHED) | ✅ SHIPPED (as ceiling proof) |
| P2 | ~~Wire INT8 dequant front-end into CUTLASS MmaCore~~ | **SCOPED OUT.** CUTLASS sm_70 has no mixed-input template (sm_80+ only). Custom prologue would be CUTLASS-internals work for less win than V11 hand-roll. | ❌ NOT VIABLE |
| P3 | ~~CUTLASS tile sweep~~ | **SCOPED OUT.** Without P2 the CUTLASS path is pre-dequant-only; not a production target. | ❌ NOT VIABLE |
| P4 | ~~REPORT-13~~ | Roll findings into SPRINT-017 V11 close report instead. | — |

## Decision gates

- **After P0**: does CUTLASS compile in our env (CUDA 12.2.2, sm_70-only build)?
  - ✅ Yes → proceed.
  - ❌ No (CUTLASS dep conflict, header issue) → blocker. Investigate `cutlass-3.x` vs `cutlass-2.x` (sm_70 may require older branch). 1 hr buffer for version triage.

- **After P1**: does FP16-in CUTLASS hit > 50 TF on its own?
  - ✅ Yes → ceiling confirmed; proceed to P2 dequant integration.
  - ⚠️ Close to 50 but not over → CUTLASS V100 INT8 ceiling is lower than published numbers suggest on our shape. Decision: continue to P2/P3 anyway (smaller win still useful); pre-commit to **option B fallback** (pre-dequant + cuBLAS FP16) if total system can't hit 50 TF.
  - ❌ < 30 TF → CUTLASS misconfigured. Debug.

- **After P3**: best CUTLASS + dequant at M=2048?
  - ≥ 50 TF → **goal hit**. Ship.
  - 35–50 TF → ship as production champion (still beats v10 / v11). Document 50 TF as "not achievable on V100 INT8 dequant-to-FP16 path; pre-dequant + cuBLAS reserved for prefill-only bench."
  - < 35 TF → unexpected. Likely dequant integration sub-optimal. Debug or fall back to option B.

## Definition of Done

1. CUTLASS submodule / FetchContent pinned to a specific commit/tag (reproducible build).
2. `cutlass_int8_kernels.cuh` builds clean against sm_70 in tcg-dev pod.
3. Bit-correct: `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1` at every M ∈ {64, 256, 1024, 2048, 4096}.
4. Goal verdict written: explicit pass/fail on **50 TFLOPS at M=2048**.
5. Per-M champion table in REPORT-13: which kernel wins each M (small-M v10s SplitK / mid-M v10 / large-M CUTLASS).
6. Production dispatch rule documented (which kernel to call for which M range).
7. SPRINT-018-FOLLOWUPS.md captures deferred work (e.g., CUTLASS-INT4 if INT8 lands).

## Risks

- **R1 — CUTLASS V100 INT8 templates may assume specific weight layouts** (e.g., padded, interleaved, pre-permuted) that don't match our DSv4 INT8 weight format. **Mitigation**: P2's dequant front-end goes through SMEM, so we choose the layout entering CUTLASS. No constraint from CUTLASS's gmem iterator.
- **R2 — Template-instantiation compile-time explosion**. CUTLASS templates can take 5–10 minutes per instantiation to compile. **Mitigation**: keep our shipped tile set small (4–6 configs, not the full V100 sweep grid).
- **R3 — Binary size**. CUTLASS produces large `.so` files. **Mitigation**: only include sm_70 instantiations, prune unused dtypes.
- **R4 — sm_70 specific CUTLASS support depth**. CUTLASS 3.x focuses on sm_80+. V100 path may need CUTLASS 2.x branch. **Mitigation**: P0 decision gate verifies version compatibility before deeper investment.
- **R5 — Numerical divergence**. CUTLASS's accumulator order may differ from our reference, producing rel > 1e-3. **Mitigation**: tolerance contract is per-row in the harness; if it fails, investigate via maxabs and p99 to confirm it's accumulator-order noise vs a real bug.

## Fallbacks (if CUTLASS itself can't hit 50 TF)

In order of preference:

1. **Pre-dequant W → FP16 + cuBLAS FP16 GEMM**: ~2 hr. Ceiling 85 TF. **Bench-only** for full DSv4 (28 GB FP16 weights blow memory budget); fine for prefill / single-layer comparison. Use this to *prove* the 50 TF ceiling exists and is reachable.
2. **Accept ~35–42 TF ceiling**: document V100 INT8 dequant-to-FP16 hand-tuned engineering limit. Ship v10s + (CUTLASS if it improved over v10) + V11 (if its infrastructure was kept).

## Sequencing relative to SPRINT-017 / V11

SPRINT-018 is a **parallel track**, not a fallback that waits for V11 to fail:
- **Can start immediately** in any session — CUTLASS work touches different files than v11.
- **No shared dependencies**: V11 uses `mma_sm70.cuh` + `v11_kernels.cuh`; CUTLASS uses `cutlass_int8_kernels.cuh` + a `_deps/cutlass-src` checkout.
- **Decision sequence**:
  - If V11 Step 3 (XOR swizzle, SPRINT-017 Wave 3) lands ≥ 50 TF: keep V11, defer SPRINT-018.
  - If V11 lands < 35 TF or stalls > 6 hr beyond plan: pivot fully to SPRINT-018.
  - If both work: ship CUTLASS at M ≥ 1024 and V11 at smaller M (decision rule from per-M champion table).

## Files summary

New:
- `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` — namespace `int8_cutlass`, template wrapping CUTLASS `device::Gemm`.
- `tools/tc-grid/docs/REPORT-13.md` — sprint close report.
- `_deps/cutlass-src/` (FetchContent target, not checked in).

Modified:
- `tools/tc-grid/CMakeLists.txt` — `FetchContent_Declare(cutlass GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git GIT_TAG <pinned>)`. Include CUTLASS headers in `target_include_directories(tc-grid PRIVATE ${cutlass_SOURCE_DIR}/include)`.
- `tools/tc-grid/src/launch_int8.cu` — `LAUNCH_CUTLASS(b_m, b_n, b_k, w)` + `RERUN_CUTLASS`. `s.version == 40` dispatch.
- `tools/tc-grid/src/main.cu` — 4–6 CUTLASS Tile entries with `version=40`.

## Dependencies

- `tcg-dev` pod on gpu-01 (already running).
- CUDA 12.2.2 toolchain.
- Network access from pod for `FetchContent` to clone CUTLASS at build time (verify with `kubectl exec ... curl -I https://github.com`).
- No new credentials.

## Open questions

1. **CUTLASS version**: pin to `v3.x` or `v2.11.0`? sm_70 support quality differs. Default plan: start with `v2.11.0` (last 2.x release, mature sm_70), revisit if there's a compelling 3.x feature.
2. **Compile-time cost**: CUTLASS templates may push our build from 30s to 5+ min. Acceptable in dev pod; need to validate that incremental rebuilds don't trigger full CUTLASS recompile.
3. **Weight-layout interop with DSv4 production**: tc-grid uses our quantize+pack routine. DSv4 in production uses GGUF / sharded weights. The dispatcher integration (which we already need for v10s) is separate work — SPRINT-018 ships the kernel, not the dispatcher.
4. **CUTLASS 3.x ECT** (Extended Compute Tile) — does it offer any sm_70 path that beats 2.x? Likely no, but worth a 15-min check.

## Estimated total effort

| Phase | Estimated | With memory-feedback multiplier (3×) |
|---|---:|---:|
| P0 | 1 hr | 3 hr (CUTLASS version triage is unknown) |
| P1 | 1.5 hr | 4 hr |
| P2 | 1.5 hr | 4 hr |
| P3 | 1 hr | 3 hr |
| P4 | 1 hr | 2 hr |
| **Total** | **6 hr** | **~16 hr (2 sessions)** |

Per `feedback_effort_estimation_undocumented_hardware.md`: CUTLASS templates have build-system gotchas, sm_70 path is less-trodden than sm_80+, so the 3× multiplier is appropriate.

## Success criteria recap

**Sprint succeeds if**: CUTLASS INT8 ships at ≥ 50 TF on M=2048, bit-correct, production-dispatcher rule documented.
**Sprint partially succeeds if**: CUTLASS INT8 ships at 35–50 TF — beats v10/v11 but doesn't hit 50; fallback option B (pre-dequant + cuBLAS for prefill bench) becomes the documented path to 50.
**Sprint fails if**: CUTLASS V100 INT8 dequant-to-FP16 path can't beat v10 (29 TF). In that case the V100 hand-tuned ceiling is the hardware limit and we ship v10/v10s/v11 as final, with explicit engineering tradeoff written up.
