# SPRINT-023 — Gemini Critique (kernel-perf bias)

**Date**: 2026-05-15
**Reviewer perspective**: performance-kernel engineer. Bias toward (a) per-launch
overhead math, (b) HBM bandwidth ceilings as the *real* DoD gate, (c) treating
the MoE expert call as a *batched* operation rather than 256 independent GEMMs.

This critique reads SPRINT-023-INTENT.md, SPRINT-023-CLAUDE-DRAFT.md, and
SPRINT-023-CODEX-DRAFT.md. The TL;DR: **both drafts adopted the intent's shape
of solution ("wire turbomind into ggml-cuda mul_mat dispatch") without
questioning whether per-expert GEMM dispatch is even the right primitive at
this scale.** It probably isn't. See §3 for the SYNTHESIS.

---

## 0. Hard numbers reviewers should agree on before going further

These set the ceiling for any DoD this sprint adopts. Both drafts gesture
at the math; neither writes it down explicitly.

### 0.1 DSv4-Flash routing arithmetic

- Layers with MoE: ~58 (60 total minus a handful of dense lead-in layers
  per DSv4 family convention; both drafts use 60, accept that for an upper
  bound).
- Experts per MoE layer: 256
- Top-K activation: 8 per token
- Therefore **per-token active GEMMs = 58 × 8 ≈ 464** (not "256 × 60", as
  Claude P4 implies at line 561 "256 launches/forward").
- Claude's risk math (`256 × 8 × 32 × 60 = 491k launches/decode`) is also
  wrong — it multiplies "256 experts × 8 active" which double-counts.
  Real launch count: **~464 launches/token × 32 tokens ≈ 14.8k launches
  for tg32**, *if* we dispatch one Gemm::Run per active expert.

That's still a lot, but it's not 491k. The actual launch budget is the
correct denominator for the host-side overhead analysis.

### 0.2 HBM bytes-per-token (the actual decode floor)

At MXFP4 (0.5 B/wt) + per-32-group E8M0 scale (1 B per 32 weights → 1/32
overhead), bytes/wt ≈ **0.531 B**.

Per active expert FFN (gate + up + down), the weight footprint is:
- gate: 7168 × 2048 × 0.531 = 7.79 MiB
- up:   7168 × 2048 × 0.531 = 7.79 MiB
- down: 2048 × 7168 × 0.531 = 7.79 MiB
- Total per active expert: **~23.4 MiB**

Per-token bytes touched (just the routed-expert weights):
- 464 active experts × 23.4 MiB = **~10.6 GiB / token**

On V100 SXM2 with ~750 GB/s achievable HBM bandwidth:
- Decode floor = 10.6 GiB / 750 GB/s = **~14.5 ms/token** just for MoE
  weight reads
- That's a **~69 t/s ceiling** for the MoE portion alone if the routing,
  attention, KV, and dense layers were free.

Add the dense (~5% of params, but every-layer-every-token, so it's the
attention QKV + dense MLP for ~58 layers): another ~3–5 ms/token.
KV cache attention at 2K ctx: ~5 ms/token. Routing softmax + scatter: ~1
ms/token.

**Honest decode ceiling = ~25–28 ms/token = ~36–40 t/s** if the kernels
ran perfectly. The intent's **20 t/s is plausible**, Codex's **8 t/s
floor is too conservative**, Claude's **20 t/s gate is at the ragged
edge** — every host-side inefficiency eats directly into the 12–13 ms
of slack between ceiling and target.

### 0.3 At top-8 routing, **~3% of the model is touched per token**

- Active params per token: 8/256 × ~242B MoE params + ~14B dense ≈
  7.6B + 14B = 21.6B activated of 284B → **7.6% of params** touched.
  (Active param fraction is higher than active-expert fraction because
  every layer's dense path runs every token.)
- The MoE-only active fraction: 8/256 × 60 layers = 0.0234 × 60 = **2.34%
  of MoE params per token, repeated 60 times across layers**. So the
  *per-layer* active MoE working set is 8/256 = 3.1%, hit 60 times.

This is the lever neither draft pulls on: **the per-layer working set
is 8 experts × 23.4 MiB = 187 MiB**, which fits in 4× V100's L2 budget
(L2 is 6 MiB on V100, so no — but it fits comfortably in the 22 GiB
free VRAM). **The full per-token working set is 10.6 GiB, fits in VRAM
trivially.** This is what makes GPU-resident experts a good plan in
principle. *Both drafts hand-wave the "16 hot tensors" and "first 16
layers" partial-residency stories without acknowledging that full
residency at MXFP4 is only ~75 GiB and 2-GPU TP makes it free.*

### 0.4 Per-launch overhead reality check

CUDA kernel launch on V100 with non-trivial kernel args: ~5–10 µs
*best case*, 15–30 µs with stream synchronization. Claude quotes
"5–10 µs/launch × ~200 active experts/token = 1–2 ms/token" — that's
right for the average case but **misses the worst case**:

- At 464 launches/token × **20 µs** = 9.3 ms/token of pure launch
  overhead. That's **~33% of the 28 ms/token budget**.
- At **30 µs** (turbomind's `Gemm::Run` has descriptor build + cudaMalloc
  for workspace + barrier setup — see `gemm_bench_packed.cu:474–520`):
  13.9 ms/token = **50% of budget**. Game over for 20 t/s.

**This is the critical missing experiment.** Neither draft schedules a
single-kernel launch-overhead microbench against turbomind's `Gemm::Run`
at M=1 N=K=7168 *before* P0. Codex flags it in §7.1 but doesn't make
it a phase. Claude flags it in §10 Open Q2 but doesn't action it.
**Both should have made this the first 30 minutes of P0.**

### 0.5 M=1 mat-vec performance of `Config_MXF4` — UNKNOWN AND UNMEASURED

Neither draft has data on this. REPORT-15 measured M ∈ {64, 2048}.
Claude's draft §10-Q3 admits "we haven't measured" M=1. Codex's §7.9
acknowledges turbomind tiles are BM ≥ 8 so M=1 wastes 87.5% of compute.

**The likely reality**: at M=1 N=K=7168, the GEMM is 100% HBM-bound.
A tile-padded `Config_MXF4` BM=8 instance reading 8× the activation but
the same weight will give roughly the same wall-clock as a hypothetical
true mat-vec — because the dominant cost is the weight read, not the
activation read or the math. Codex's framing of "wastes 87.5% of compute"
is misleading: it wastes compute but compute isn't the bottleneck.
**Effective throughput at M=1 is HBM-bound regardless of tile padding.**

But this also means: **turbomind's Config_MXF4 has no advantage over a
well-written CUDA mat-vec at M=1.** The 50+ TF Claude tabulates is an
M=2048 number. At M=1 we're at ~50 TF × (compute-bound-fraction). If
HBM-bound, effective TFLOPS ≈ (2 × N × K × bytes/wt^-1 × bandwidth_GBs)
≈ 2×7168×7168 / (7168×7168×0.531 / 750e9) ≈ **2.82 TFLOPS effective**
at M=1.

So the intent's TFLOPS-ceiling justification is **measuring the wrong
ceiling for the decode hot path**. M=2048 is the prefill ceiling.
For decode, the right ceiling is "HBM bandwidth × 2 / bytes/wt" =
~2.8 TFLOPS, and what matters is whether turbomind's mainloop **achieves
the HBM bandwidth peak**, not whether it achieves 50 TF.

---

## 1. Critique of SPRINT-023-CLAUDE-DRAFT.md

### 1.1 Strengths (where the perf reasoning is sound)

- **P0 is correctly the first thing**: re-running `Config_MXF4` at
  `group_size=32` before committing to MXFP4 is the right gate. If it
  flunks at production shapes, the sprint pivots.
- **Buffer-type architecture (§3.2)** is genuinely the right call. The
  alternative (touching `convert.cu`) mixes per-row dequant with
  tile-format packing; those are different abstractions. Buffer-type
  per-tensor opt-in via `-ot` is a clean UX.
- **Build flag `GGML_TURBOMIND_GEMM` + runtime `LLAMA_DISABLE_TURBOMIND`
  kill-switch** (§P1) is good defense-in-depth. Compile-time guard for
  binary size, runtime kill-switch for production rollback.
- **DoD-12 (kill-switch test)** is the right belt-and-suspenders gate.
- **P3 correctness gate at rel ≤ 2e-2** aligns with SPRINT-015 P2
  contract — not invented from thin air.
- The risk table (§7) **correctly identifies per-expert launch overhead
  as the likely cap** (Risk: "Per-expert kernel launch overhead saturates
  host CPU"). Good honest flagging.

### 1.2 Weaknesses

- **The 20 t/s DoD-9 / 20 t/s DoD-10 targets are presented as "the sprint
  ships when DoD-1 through DoD-8 + DoD-12 + DoD-13 are met"** with perf
  as soft gates. Read literally: a sprint that lands the dispatch shim but
  ships 6 t/s passes. That's *fine for engineering hygiene* but it means
  the headline number is decoupled from the integration cost. Reviewer
  should ask: **if perf can miss and we still ship, what's the actual
  point of this sprint?** Claude's draft doesn't make this question
  uncomfortable enough.

- **Architectural gap: each expert call IS treated as independent.**
  §3.3 "MoE-ids path: iterate the `expert_bounds` array... for each
  expert with `bounds[e+1] - bounds[e] > 0`, launch one `tmg::Gemm::Run`
  with the per-expert slice." That's 256 launches per layer worst case,
  or 64 best case (when only top-8 experts have rows). No grouped-GEMM
  or batched-GEMM API is invoked. **This is the central kernel-perf
  defect of both drafts** — see §3 below.

- **HBM bandwidth math is implicit, never spelled out.** Claude trusts
  REPORT-15's M=2048 TFLOPS number to predict decode TPS. As §0.5 above
  shows, this is the wrong ceiling. The right ceiling for decode is
  bandwidth-bound; the M=2048 number tells us nothing useful about
  M=1 / M=8 small-batch decode.

- **"First 16 layers" heuristic in P5 is unjustified by data.** Claude
  cites "Empirically MoE routing has more entropy in early layers; this
  is a defensible default." That's a community folklore claim that
  hasn't been measured for DSv4-Flash specifically. The math in
  §0.3 above shows full residency at MXFP4 is ~75 GiB — too big for
  one V100, but the per-layer hot working set is small. The right
  partial-residency strategy is **per-layer**, not "first N layers."

- **P4 launch-overhead estimate** is wrong. "256 launches/forward × 32
  tokens × 60 layers = 491k launches" double-counts (already noted in
  §0.1 above). Real number is ~14.8k launches for tg32. The 5 ms
  overhead estimate is roughly right by coincidence (overestimate of
  launches × underestimate of µs/launch).

- **P5 / P6 plan no measurement of M=1 turbomind performance.**
  This should be a P0 sub-experiment, NOT a P6 ncu run. If M=1 sucks,
  the entire dispatch architecture needs rethinking.

- **Activation cast F32 → F16 → F32 cost is dismissed (§4-P3, Risk row
  4) but never measured.** At per-expert M=1 (decode), activation cast
  is `K=7168` floats per cast = 28 KiB. 464 casts × 28 KiB × 2 (in + out)
  = 26 MiB/token of cast traffic. Trivial bandwidth-wise (~35 µs at
  750 GB/s) but each cast is **a separate kernel launch** unless batched.
  If each cast = 5 µs launch overhead × 464 × 2 = 4.6 ms/token. **Same
  order of magnitude as the GEMM launch overhead.** Not free.

### 1.3 Risk gaps

- **No per-launch overhead microbench scheduled in P0.** This is the
  single most important measurement for this sprint. It determines
  whether the architecture is viable at all.
- **No grouped/batched-GEMM exploration.** Turbomind has a grouped GEMM
  API path (look in `research/lmdeploy/src/turbomind/kernels/gemm/`);
  it's not mentioned anywhere in either draft.
- **No M=1 vs M=128 dispatch differentiation in the predicate.** Both
  drafts use the same code path for both. M=1 and M=128 have completely
  different bottleneck profiles and probably want different kernels.
- **Workspace per-stream partitioning is deferred (P6 stretch).** If
  stream-batched is the only way to hit 20 t/s, deferring its design to
  the end of the sprint is high risk.

### 1.4 DoD completeness

- DoD-7 ("byte-identical to baseline first 32 generated tokens, greedy +
  seed-pinned"): **this is too strict.** Quantized MoE dispatch will
  produce numerically different (but semantically equivalent) tokens
  because of the 2e-2 tolerance on the GEMM. The right DoD is
  "perplexity within 1% of CPU baseline on a fixed eval set" or
  "log-prob KL ≤ 0.01 over first 256 tokens." Byte-identity for
  greedy sampling **assumes the entire argmax is stable under 2e-2
  perturbations**, which it isn't for a 256-expert top-K router where
  the K-th score might be 5e-3 above the K+1-th.
- DoD-11 (HMMA active% ≥ 45% on MXF4 mainloop): **measured at what M?**
  Not specified. At M=2048 (prefill) sure, ≥45% is achievable. At M=8
  (decode), HMMA active% will be much lower because the kernel is
  bandwidth-bound. Need to disambiguate: "≥45% at M=128, ≥10% at M=8
  with HBM utilization ≥ 60%."
- **Missing DoD**: HBM bandwidth utilization at decode. This is the
  metric that matters for tg32 and it's never named.
- **Missing DoD**: per-launch overhead measurement. Should be a numbered
  gate, not an open question.

---

## 2. Critique of SPRINT-023-CODEX-DRAFT.md

### 2.1 Strengths

- **§7.1 honest pushback on 20 t/s** is the most valuable single
  contribution across both drafts. The math is roughly right (modulo
  §0.4 nit on launches/token) and the conclusion — 8–12 t/s realistic,
  20 t/s is multi-sprint — is plausible.
- **dlopen-loaded carve-out (§3, §7.4, §7.5)** is the better build
  architecture. Pulling fmt + turbomind templates into the ggml-cuda
  static archive is a real cost; isolating behind a tiny C ABI is
  hygienic. Claude's "FetchContent into ggml-cuda CMakeLists" is
  simpler but couples too tightly.
- **§7.6 honest VRAM accounting** catches things Claude misses:
  weight duplication (original GGUF block stays for fallback) is a real
  cost. Claude's "free 22.5 GiB" number is the pre-duplication number.
- **§7.7 "Convert rc != 0 = ABORT, never silently fall back"** is the
  right correctness posture.
- **§7.10 reminding everyone llama.cpp upstream doesn't take AI-generated
  PRs** is the kind of operational hygiene the intent should have flagged
  but didn't.
- **Smaller scope (FP8 only, not MXFP4)** is defensible: FP8 has REPORT-15
  measured numbers; MXFP4 at gs=32 is unmeasured.
- **Mandatory pre-P1 questions in §10** are the right pattern. "Read
  `convert.cu :: dequantize_f8_e4m3_b128` before writing the packer" —
  excellent forcing function.

### 2.2 Weaknesses

- **The FP8-only decision is the wrong fork.** The intent says DSv4-Flash's
  *experts* are MXFP4; the *dense* layers are F8_E4M3_B128. So Codex's
  Phase 1–5 plumbs FP8 dispatch... for tensors that don't exist in the
  MoE path. The intent §"Open questions" #5 already noted that
  F8_E4M3_B128 is the dense path. Codex appears to have **conflated
  "dense FP8" with "expert FP8"**. Re-read intent doc §"Orientation
  summary" first bullet: "MXFP4 experts + F8_E4M3_B128 dense". MXFP4
  is the expert path. Doing FP8-only here optimizes the *not*-bottleneck.

- **8 t/s floor is too conservative IF the launch-overhead bug isn't
  the bottleneck.** Codex's §7.1 implicitly assumes per-launch overhead
  saturates the host. If we group/batch the expert GEMMs (see §3 below),
  20 t/s is achievable. The right framing isn't "20 is unrealistic" but
  "20 is unrealistic with the proposed per-expert-launch architecture."

- **No HBM bandwidth math.** Same gap as Claude. §7.1 says "FP8 at
  ~700 GB/s effective HBM2 is `7168*7168 / 700e9 ≈ 73 µs per gemv per
  expert`" — that's the per-expert wall-clock at M=1, but Codex doesn't
  carry the math through to the per-token total or HBM bandwidth
  utilization fraction.

- **Hot-list scope ("16 tensors in 1–2 layers" §10-Q2)** is too small
  to demonstrate the architecture. At 2 layers × 8 hot experts of 256,
  most routing decisions still hit cold (CPU) experts and the per-token
  TPS lift is barely above noise. The mock JSON needs to cover **at
  least one full top-8 routing decision per layer for at least 8 layers**
  to demonstrate real lift.

- **dlopen approach has a subtle correctness risk** that §7.5 misses:
  CUDA contexts are per-process, but **CUDA streams are not safely
  shareable across dlopen boundaries** if the carve-out and ggml-cuda
  link against different CUDA runtime versions. Codex says "CUDA 12.2 on
  gpu-01" — same version in both, fine for now. If anyone ever builds
  the carve-out with a different CUDA, silent corruption. Worth a
  CUDA-version-handshake check in `tm_version()`.

- **Phase 3's `M ≤ 256` gate is unmotivated.** Codex says "skip prefill
  for now; FP8 sm70 at small M is the lowest-risk window." But REPORT-15
  shows FP8 sm70 is **best** at M=2048 (59 TF). Small-M is *worse*, not
  better. The right gate is probably `M ≥ 8` (skip M=1 to scalar), not
  `M ≤ 256`.

### 2.3 Risk gaps

- **§7.1 launch-overhead estimate misses the grouping option.** Same
  blind spot as Claude.
- **§7.9 dismisses M=1 as "still bandwidth-bound, fine"** but doesn't
  follow up with "therefore, what bandwidth fraction does
  `Config_E4M3` achieve at M=1?" The whole sprint hinges on this.
- **No fallback if hot-list-hit-rate is low.** If the 16 hand-authored
  experts don't actually get routed often, the user sees zero TPS
  improvement. There's no instrumentation in the plan to measure
  hot-list hit rate at runtime.

### 2.4 DoD completeness

- D4 (`tg32 ≥ 8 t/s`) — measurable, achievable, defensible.
- D6 (`HMMA active% ≥ 45%`) — same defect as Claude DoD-11: at what M?
- **Missing**: HBM bandwidth utilization gate.
- **Missing**: hot-list hit rate. If the JSON lists 16 experts and at
  runtime only 2 get routed, no perf moves. Need: ">=50% of routed
  experts hit the turbomind path."
- **Missing**: per-launch overhead measurement.
- D8 ("Sprint report with TFLOPS table, ncu numbers, explicit
  deferred list") is good hygiene, not a perf gate.

---

## 3. SYNTHESIS — Is "wire turbomind into ggml-cuda mul_mat dispatch" the right shape?

**My answer: No, not exactly. It's the right ingredient but the wrong
primitive.** Both drafts accepted the intent's framing without questioning
it. The intent says "dispatch through `Config_MXF4`/`Config_E4M3`
per-mul_mat." The drafts dutifully plan how to dispatch one `Gemm::Run`
per active expert per layer per token.

**That's the wrong granularity for an MoE decode hot path.** Here's why:

### 3.1 The MoE expert call is a grouped/batched-GEMM operation, not 256 independent GEMMs

DSv4-Flash routing at decode-time produces, per token:
- One activation vector A of size K=7168 (the per-token output of the
  router scatter)
- A list of 8 active expert IDs out of 256
- For each active expert e_i, we need: `Y_i = A · W_{e_i}` where W is
  the expert-specific weight matrix

This is **a grouped GEMM (or "GEMM with variable batch")** in cuBLAS /
cuBLASLt / CUTLASS terminology. NOT 8 separate GEMMs. NOT 256 separate
GEMMs. The right primitive is one of:

- **CUTLASS Grouped GEMM** — same kernel, multiple problem shapes
  (different N, K, or weight pointer), one launch.
- **CUTLASS Batched GEMM with strided weights** — if all 8 experts'
  weight matrices live in a contiguous buffer indexed by expert ID.
- **Turbomind's own grouped/batched API** — `research/lmdeploy/src/turbomind/kernels/gemm/`
  almost certainly has a moe_gemm.cu or equivalent for sm70/sm80;
  neither draft checked.
- **A custom fused kernel** — read the gather-routed activations,
  multiply by per-expert tile-loaded weights, scatter outputs. One
  launch per layer instead of 8.

At 58 layers × 1 launch/layer × 32 tokens = **~1856 launches** for tg32.
Compare to per-expert 14.8k launches. **8× reduction in launch overhead**,
and that's the difference between 20 t/s feasible and 20 t/s not.

### 3.2 What both drafts miss in the codebase

Neither draft mentions:
- Whether turbomind has an MoE grouped-GEMM kernel for sm70 (it almost
  certainly does — `lmdeploy` is *built* for MoE inference)
- Whether ggml's `mul_mat_id` already has a "fused MoE" code path in
  some backends (it does — see cuBLAS path in `ggml-cuda/mmid.cu` if it
  exists, or the CPU `ggml_compute_forward_mul_mat_id` reference)
- Whether the `expert_bounds` array can drive a single grouped launch
  instead of N separate launches

**Action item for either draft**: before P0, spend 30 minutes
searching `research/lmdeploy/src/turbomind/kernels/gemm/` for "moe",
"grouped", "batched", "expert". If turbomind already has an MoE-aware
GEMM, the entire dispatch architecture changes.

### 3.3 The right shape of solution (my recommendation)

**Sprint 023 should be: "Build a turbomind-backed MoE grouped-GEMM
operator and plumb it into ggml's `mul_mat_id` for DSv4-Flash experts,
single dispatch per layer."**

Concretely:
1. **P0**: measure (a) `Config_MXF4` at gs=32 ceiling [as Claude says],
   AND (b) per-launch overhead microbench at M=1, M=8, M=128 [as
   neither says], AND (c) survey lmdeploy for an existing grouped-GEMM
   kernel.
2. **P1**: build the carve-out per Codex's §7.4/§7.5 hygiene (dlopen
   shared lib, tiny C ABI) — Codex's architecture wins here.
3. **P2**: GGUF MXFP4 → turbomind packed converter (per-tensor packing
   at upload, per Claude's §3.2 buffer-type) — Claude's UX wins here.
4. **P3**: **grouped-GEMM dispatch**, not per-expert dispatch. One
   `Gemm::Run` (or `MoeGemm::Run`) per layer per token, taking the
   expert-bounds array as an input. If lmdeploy has the kernel, we
   wrap it. If not, this becomes a SPRINT-024 task and SPRINT-023 ships
   the per-expert version with a known 8× launch-overhead tax —
   honestly documented at ~10 t/s.
5. **P4**: correctness gates (perplexity ≤ 1% drift, NOT byte-identity).
6. **P5**: VRAM probe + selective placement.
7. **P6**: end-to-end measurement with HBM-utilization metric as the
   *real* perf gate.

### 3.4 Recommended DoD revisions for whichever draft wins

- **Replace TFLOPS-at-M=2048 ceiling thinking with HBM-bandwidth-at-M=1
  ceiling.** Decode is bandwidth-bound; the M=2048 number is a red
  herring for tg32.
- **Add: per-launch overhead measured ≤ 15 µs at M=1.** If turbomind's
  `Gemm::Run` is heavier than that, dispatch architecture must change
  before P3.
- **Add: HBM bandwidth utilization ≥ 60% on the MoE GEMM at decode.**
  This is the real perf gate. A kernel hitting 60% of 900 GB/s peak
  = 540 GB/s = ~20 ms/token of MoE weight reads = ~45 t/s ceiling
  before other overheads.
- **Replace "tg32 ≥ 20 t/s" with tiered gates**:
  - Floor (must-have to ship): `tg32 ≥ 10 t/s` AND HBM utilization
    ≥ 50% on the MoE GEMM (proves the architecture works even if
    cold experts cap us)
  - Target (sprint succeeds): `tg32 ≥ 16 t/s` AND HBM utilization ≥ 60%
  - Stretch: `tg32 ≥ 20 t/s` (requires grouped-GEMM)
- **Replace "byte-identical tokens"** with "**perplexity within 1% of
  CPU baseline on 256-token WikiText-2 sample**" — quantization tolerance
  legitimately changes argmax in routing, so byte-identity is the wrong
  contract.
- **Add: hot-list routing hit rate ≥ 50%.** Without this, a poorly-chosen
  hot list silently makes the feature a no-op.

### 3.5 Are EITHER of these drafts the right plan?

**Neither, as-is.** Pick from each:

| From Claude | From Codex |
|---|---|
| Buffer-type architecture (§3.2) | dlopen + C ABI carve-out (§3, §7.5) |
| `-ot exps=CUDA0_TURBOMIND` UX | Per-phase mandatory open-question answers (§10) |
| P0 ceiling re-measurement | §7.1 honest perf math |
| Kill-switch env var (DoD-12) | §7.6 honest VRAM accounting |
| Buffer round-trip test (DoD-4) | §7.7 abort-not-silent-fallback |
| | FP8 dlopen build hygiene |

Add (from neither):
- **Grouped-GEMM as the primitive, not per-expert dispatch.**
- **Per-launch overhead microbench in P0.**
- **HBM bandwidth utilization as the headline perf gate.**
- **Perplexity drift gate replacing byte-identity gate.**
- **Hot-list hit-rate instrumentation.**

### 3.6 If forced to pick one draft as-is

**Claude wins on architecture and UX. Codex wins on engineering
discipline and perf realism.** If forced to pick one starting point,
**Claude's draft + Codex's §7 risk analysis** is the merge target.
Codex's FP8-only scope decision is a bug (it optimizes the dense path
when experts are the bottleneck) and Codex's 8 t/s floor is too low if
we do grouped GEMM. Claude's 20 t/s gate is too ambitious if we don't.

**Best path forward**: a SPRINT-023-MERGE-NOTES.md document that takes
Claude's architecture and Codex's risk discipline, **and adds an explicit
"investigate grouped-GEMM in lmdeploy" task as P0.b**. If the kernel
exists, the sprint plan changes shape. If it doesn't, we still know
the ceiling.

---

## 4. Specific factual issues to fix in each draft

**Claude (SPRINT-023-CLAUDE-DRAFT.md):**

- §3.3 "256 launches per forward pass" — should be ~464 (top-K, not all
  experts).
- Risk row 5: "256 launches/forward × 32 tokens × 60 layers = 491k
  launches/decode" — wrong by factor of ~32×. Should be ~14.8k for tg32.
- §3.3 "8-active-experts-per-token design means at M=2048 prefill we
  average ~64 tokens/expert" — only if routing is perfectly uniform.
  Real MoE routing is heavily skewed; some experts will have M=0, others
  M=200+. Need actual distribution data.
- DoD-7 byte-identity gate is too strict (per §1.4 above).
- DoD-11 needs M disambiguation.

**Codex (SPRINT-023-CODEX-DRAFT.md):**

- §1 / §3 scope conflation: FP8-only means F8_E4M3_B128 dense path is
  what gets dispatched, but the *expert* path (the actual bottleneck) is
  MXFP4. Re-read intent doc to confirm which tensor type is the experts
  in DSv4-Flash. **If the conclusion is "do MXFP4 experts, not FP8
  dense," the sprint plan inverts.**
- §3 "8–16 *named* experts, not 256×60" — at 8 hot experts per layer
  with 60 layers = 480 hot expert tensors, not 8–16. The "16 named"
  framing is for a mock test, not production residency. Clarify which
  is intended.
- §7.1 "~30–80 µs / launch" — high end is plausible for first launch
  with descriptor build; steady-state is 5–15 µs. Tighten the estimate.
- §7.1 "8 active experts × 60 layers × per-token = ~30 ms/token just
  in launch overhead = 33 t/s ceiling" — math is roughly right (8×60 =
  480 launches × 60 µs = 28.8 ms) but uses the conservative end of µs/launch.
  At 15 µs typical: 480 × 15 = 7.2 ms/token, **70 t/s ceiling**. So the
  "20 t/s unreachable" conclusion is too pessimistic.
- §7.6 "60 layers × 8 hot-experts = 48 GiB — physically impossible" —
  recheck at MXFP4 bytes-per-weight (0.531), not FP8 (1.0). At MXFP4
  the full-residency cost is closer to 75 GiB, also impossible on one
  V100 but a different number.
- Appendix A: same M-disambiguation issue on HMMA active%.

---

## 5. Recommendation for the planner

1. **Don't merge either draft as-is.** Both have valuable structure;
   neither is correct on the kernel-perf math.

2. **Spend a half-day on P0.b (pre-planning research) before scheduling
   anything**: read lmdeploy/turbomind sources for grouped/batched MoE
   GEMM. Run a 30-minute launch-overhead microbench using the existing
   `gemm_bench_packed.cu` harness with M=1 N=K=7168 and time *just the
   kernel launch + descriptor build*, not the GEMM. This is the single
   most decision-relevant data point.

3. **Use Claude's buffer-type + Codex's dlopen-carveout** as the
   architecture. Best of both.

4. **Set DoD on HBM bandwidth utilization, not TFLOPS.** TFLOPS at M=2048
   is the prefill ceiling and not the decode bottleneck.

5. **Accept 12–15 t/s as the realistic decode target this sprint**,
   with 20 t/s achievable in SPRINT-024 only if grouped-GEMM lands
   AND prefill path moves to turbomind AND KV-cache attention is
   optimized.

6. **Make per-launch overhead and grouped-GEMM survey explicit P0
   subtasks.** They're the decision-pivots for the whole sprint.

7. **Replace the byte-identity correctness gate with a perplexity
   drift gate.** The 2e-2 GEMM tolerance can change argmax-tied
   routing decisions; byte-identity is the wrong contract.

---

*End of critique.*
