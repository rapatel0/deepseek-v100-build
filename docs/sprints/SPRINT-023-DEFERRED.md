# SPRINT-023 — Deferred items

Items raised in drafts/critiques/interview but explicitly scoped OUT of
SPRINT-023. Most carry forward to SPRINT-024 with the perf landing.

---

## 1. Decode TPS perf gate (≥20 t/s target)

**What**: The intent and Claude's draft set a 20 t/s decode gate. Codex
proposed 8 t/s floor / 12 t/s stretch. All three critiques agreed the target
needs grouped MoE or batched amortization to be physically reachable on V100.

**Why deferred**: User chose "no perf gate — ship infrastructure, measure
outcome." SPRINT-023 = infra + survey + measurement. SPRINT-024 lands perf.

**Target sprint**: SPRINT-024

**Prerequisites**:
- SPRINT-023 P5 TPS measurement against the infra path
- P0.3 survey result on grouped MoE availability
- Decision on M-regime target (single-slot M=1 vs batched M>1 amortization)

**Files**: `docs/sprints/SPRINT-024.md` will set the gate based on what
SPRINT-023 P5 measures.

---

## 2. Grouped MoE dispatch design

**What**: Lift launch overhead by issuing one kernel per layer covering all
top-k active experts, instead of one kernel per active expert per layer
(464 launches/token vs 58).

**Why deferred**: SPRINT-023 P0.3 surveys whether turbomind has an existing
grouped primitive. If yes, integration happens in SPRINT-024's first phase.
If no, SPRINT-024 designs one — that's a multi-week kernel-design sprint.

**Target sprint**: SPRINT-024

**Prerequisites**: SPRINT-023 P0.3 survey output

**Files**: `research/lmdeploy/src/turbomind/kernels/gemm/` (survey scope);
new `ggml/src/ggml-cuda/grouped_moe_*.cu` if design path taken.

---

## 3. F8_E4M3_B128 dense layers via turbomind

**What**: Codex's draft scoped FP8 dense only. SPRINT-023 includes the
conversion utility (P2) but doesn't necessarily wire dense dispatch.

**Why deferred**: Dense layers are already on GPU and contribute ~7 GB of
the model. The MoE expert path is the bottleneck. Wiring FP8 dense through
turbomind adds 0-30% on the small fraction of compute that's not the
bottleneck. Best done as a follow-on once the MoE path is producing real
numbers.

**Target sprint**: SPRINT-024 or SPRINT-025 depending on prio

**Prerequisites**: SPRINT-023 P2 (conversion utility) supports F8_E4M3_B128
already; P3+P4 only need to add dense tensor patterns to the `-ot` regex

**Files**: same dispatch surface as MXFP4 path; trivial extension if utility
already works.

---

## 4. Hot-expert JSON profile generation

**What**: A profile pipeline that runs DSv4-Flash on representative traffic,
captures expert routing frequencies per layer, emits a JSON of top-K hot
experts per layer (the format that `llama-deepseek4-hot.h` consumed before
nisparks refactored it away).

**Why deferred**: SPRINT-023 doesn't need a real profile — the `-ot` regex
mechanism (already supported) lets the user pin tensors manually. Profile
pipeline can be its own sprint, gated on whether SPRINT-024 perf data shows
hot-only deployment matters.

**Target sprint**: SPRINT-025+ (when hot-only is clearly the right
deployment mode)

**Prerequisites**: SPRINT-023 P5 data showing the perf gap between
"all experts on GPU" (if VRAM permits) vs "first-N layers on GPU"

**Files**: new `tools/profile-dsv4-experts/` directory, GGUF metadata
schema, dispatcher to read the JSON.

---

## 5. Dynamic hot-expert promotion/demotion

**What**: Runtime monitoring of which experts are hot in the live workload,
streaming experts on/off GPU to track changing distribution.

**Why deferred**: Way too speculative. Need static-profile evidence first
that hot deployment beats uniform. Risk of complexity explosion (LRU policy,
prefetch decisions, KV cache interactions).

**Target sprint**: Future (SPRINT-026+)

**Prerequisites**: Static profile shows clear hot/cold expert clustering;
workload visibly drifts over time in a measurable way.

**Files**: would touch `src/llama-memory-deepseek4.cpp`,
`src/llama-context.cpp`, new `src/llama-expert-streamer.cpp`.

---

## 6. WMMA-MMVQ MoE port from sprint-017 P2+P3

**What**: Per SPRINT-022-DEFERRED-PORTS.md, our sprint-017 work on V100
tensor-core WMMA dispatch in `mmvq.cu` (commits 61f9ebebe + 5ec3a9b41)
hasn't been forward-ported to the nisparks FP4/FP8 baseline. New files
`wmma-mmvq.{cu,cuh}` don't conflict; the dispatcher hook needs manual 3-way
merge.

**Why deferred (still)**: SPRINT-023 uses a different dispatch mechanism
(buffer-type + C ABI to turbomind). WMMA-MMVQ would be a third path. Adding
two paths in one sprint = scope creep. Decision: if SPRINT-023 P0 finds
turbomind beats our WMMA-MMVQ at M=1, WMMA-MMVQ stays archived. Otherwise
SPRINT-024 considers porting it as a competitor to the turbomind path.

**Target sprint**: SPRINT-024 (decision) or never (if turbomind wins
decisively)

**Prerequisites**: SPRINT-023 P0.1 microbench numbers

**Files**: `ggml/src/ggml-cuda/wmma-mmvq.{cu,cuh}` + `mmvq.cu` 3-way merge.

---

## 7. PCIe expert streaming (Gemini's Path B)

**What**: Keep experts in CPU RAM, prefetch active experts to GPU via PCIe
just-in-time per layer. With Gen3 x16 (~14 GB/s effective) and ~3% of
weight per token at MXFP4, theoretical 10-15 t/s without resident allocation.

**Why deferred**: Architecturally different from the SPRINT-023 plan. Worth
measuring as a Plan B if SPRINT-023's resident-GPU path doesn't outperform.

**Target sprint**: SPRINT-024 (if SPRINT-023 P5 numbers disappoint)

**Prerequisites**: SPRINT-023 baseline data + PCIe streaming feasibility
microbench

**Files**: new `src/llama-expert-streamer.cpp`; would interact with
`llama-memory-deepseek4.cpp`.

---

## 8. NVFP4 support

**What**: `GGML_TYPE_NVFP4` (4 fp4 values + e4m3 scale) — nisparks's
`Bring up native FP4 FP8 quant support` added the type, but no production
DSv4 model uses it.

**Why deferred**: Dead code path for our model. Not worth investing.

**Target sprint**: Never unless a future DSv4 variant uses NVFP4.

---

## 9. Multi-GPU tensor-parallel

**What**: Split DSv4 across multiple V100s via TP (the sprint-016 P0
`abde07824` seq_rm bypass was for this).

**Why deferred**: Single-GPU isn't at ceiling yet. Multi-GPU only worth it
when single-GPU optimization saturates.

**Target sprint**: SPRINT-026+ (after single-GPU peak established)

**Prerequisites**: SPRINT-024/025 single-GPU perf landed

**Files**: cherry-pick `abde07824` from sprint-016-tensor-unlock branch
when relevant.

---

## 10. M=1 amortization via multi-slot batching / speculative decoding

**What** (user-raised during interview): M=1 decode is a worst-case for
compute-bound kernels. Two paths lift M:
- **Continuous batching**: N parallel sequences = effective M=N (already
  supported by llama-server `--parallel N`)
- **Speculative decoding**: draft model proposes K candidates, target model
  verifies in batch = effective M=K

Both move the operating point closer to where turbomind's prefill ceilings
apply.

**Why deferred from SPRINT-023**: SPRINT-023 is infrastructure. The choice
of which M-regime to optimize for changes the production target shape.

**Target sprint**: SPRINT-024 design phase should decide M-regime target
(single-slot M=1 vs batched M>1) based on SPRINT-023 measurement data.

**Prerequisites**: SPRINT-023 P5 data; profiling of expected production
parallel-slot usage; speculative-decoding draft-model availability

**Files**: `tools/server/` for parallel handling already exists; speculative
decoding has existing infra in llama.cpp.

---

## 11. Custom v13_rf_v6 grouped-MoE INT8 kernel (Gemini's Path A)

**What**: Skip turbomind entirely; extend our v13_rf_v6 (49 TF measured) into
a grouped-MoE kernel that runs the existing INT8 path with batched expert
calls. Requires offline MXFP4→INT8 conversion (changes model recipe).

**Why deferred**: Violates "pre-dequant defeats INT8" memory rule. Changes
the model's published numerical recipe. SPRINT-023 explicitly chose the
turbomind direction.

**Target sprint**: Considered in SPRINT-024 only if turbomind path fails
decisively at P0.1.

**Files**: would extend `tools/tc-grid/kernels/v13_kernels.cuh`; new
conversion tool to re-quantize MXFP4 → INT8 with bit-equivalent accuracy
preservation (research-grade open question).

---

## 12. ABI changes (new `GGML_OP_MUL_MAT_ID_MOE_DSV4`)

**What**: Gemini's draft proposed adding a new GGML op enum for the
MoE-specific dispatch. SPRINT-023 explicitly does NOT do this.

**Why deferred**: ABI changes to ggml are forever-cost. Buffer-type
abstraction achieves the same dispatch routing without touching the
public enum.

**Target sprint**: Never (rejected by design)

---

## Summary table

| # | Item | Target sprint | Blocker |
|---|---|---|---|
| 1 | 20 t/s decode perf gate | SPRINT-024 | SPRINT-023 P5 data |
| 2 | Grouped MoE dispatch design | SPRINT-024 | SPRINT-023 P0.3 survey |
| 3 | F8_E4M3_B128 dense via turbomind | SPRINT-024/25 | none (utility ready in 023) |
| 4 | Hot-expert JSON profile pipeline | SPRINT-025+ | SPRINT-024 hot-only perf data |
| 5 | Dynamic hot-expert promotion | SPRINT-026+ | Static profile evidence |
| 6 | WMMA-MMVQ MoE port (sprint-017) | SPRINT-024 (decision) | SPRINT-023 P0 numbers |
| 7 | PCIe expert streaming | SPRINT-024 (Plan B) | SPRINT-023 P5 disappoints |
| 8 | NVFP4 support | Never | DSv4 doesn't use NVFP4 |
| 9 | Multi-GPU TP | SPRINT-026+ | Single-GPU peak |
| 10 | M=1 amortization (parallel + spec decode) | SPRINT-024 design | SPRINT-023 data |
| 11 | Custom v13_rf_v6 grouped-MoE INT8 | SPRINT-024 (fallback) | Turbomind P0 fails |
| 12 | New GGML_OP enum | Rejected by design | — |
