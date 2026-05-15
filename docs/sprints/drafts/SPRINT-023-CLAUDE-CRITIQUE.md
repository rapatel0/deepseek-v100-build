# SPRINT-023 — Claude Critique of CODEX and GEMINI drafts

**Reviewer**: Claude (Opus 4.7, 1M context), playing senior systems engineer
hat with V100/MoE/GGML scars from sprints 015–022.
**Date**: 2026-05-15
**Inputs reviewed**: SPRINT-023-INTENT.md (shared brief), SPRINT-023-CODEX-DRAFT.md,
SPRINT-023-GEMINI-DRAFT.md.

Bottom line up front: CODEX is the safer, executable plan; GEMINI raises the
single most important design objection (single-matmul vs grouped-MoE) but
trips over its own bookkeeping and bundles three week-long uncertainties
into one sprint. The right SPRINT-023 is CODEX's structure with GEMINI's
**P0 measurement gate** bolted on as the entrance criterion.

---

## 1. CODEX draft critique

### 1.1 Strengths

- **The framing is honest about what the sprint actually is.** "Smallest
  correct slice" + "single named expert tensor at a time" + "FP8 only" is
  exactly the discipline that has burned sprint 017 / 019 / 021 when we
  tried to do too much. §1's table "Why this scope, not the intent's scope"
  is the strongest piece of argumentation in either draft — it identifies
  the three independent integration risks (MXFP4, FP8, hot-expert
  selection) and explicitly refuses to bundle them.
- **The 20 t/s pushback is technically correct.** §7.1's launch-tax math
  (8 experts × 60 layers × ~50 µs ≈ 24 ms/token ⇒ ~40 t/s ceiling
  before any compute) is the right back-of-envelope. And the
  bandwidth-bound argument for M=1 (7168×7168 / 700 GB/s ≈ 73 µs/gemv per
  expert) closes the loop. The intent's "4.2× over baseline" was a target
  arrived at by working backwards from the desired headline number, not
  forwards from kernel physics. CODEX is right to refuse it as a gate.
- **The dlopen / separate `.so` decision is the correct ABI call.** §7.4
  and §7.5 are the cleanest pieces of engineering in either draft. The
  intent silently assumed "link gemm2 into ggml-cuda" — which would drag
  fmt, parser, and ~40+ template-heavy TUs into a backend that strives for
  C ABI cleanliness. The carve-out-as-`.so` + four C functions
  (`tm_pack_fp8_b128`, `tm_gemm_fp8_mmv`, `tm_handle_free`, `tm_version`)
  is the right surface. The `nm -D` exit criterion in Phase 1 is a real
  gate.
- **§7.6 corrects the intent's VRAM math.** The intent claimed 22.5 GiB
  free. CODEX correctly points out that (a) every hot tensor is
  **duplicated** during this sprint (original GGUF block stays resident
  for fallback, packed copy is new), (b) compute buffers grow with
  prefill batch, and (c) "60 layers × 8 hot-experts = 48 GiB" is
  physically impossible on a 32 GiB card. The honest budget of ~19.5 GiB
  used / ~12.5 GiB free is the number to plan with.
- **§7.2 names the unsexy real cost of the sprint correctly.** The intent
  treats `block_f8_e4m3_b128` → turbomind packed layout as a footnote.
  CODEX correctly identifies it as "the sprint's true cost": opaque
  `Convert` call, undocumented error codes, scale-type mismatch (E8M0 vs
  kFloat), K-alignment per registered tile size, scale-stride conversion
  from inline-block to separate `(N, K/G)` matrix. This is the failure mode
  that eats sprints silently — it smoke-tests fine, produces junk at
  token 2.
- **Phase gates are independently mergeable and verifiable.** D1's `nm -D`,
  D2's tolerance gate, D3's `=OFF` bit-equivalence, D6's ncu metric pack
  are all things a reviewer can actually run. The "do not start phase N+1
  until phase N's DoD ticks green" discipline matches SPRINT-019 lessons
  in the user's MEMORY.md.

### 1.2 Weaknesses

- **No grouped-GEMM consideration at all.** This is the single biggest
  hole. GEMINI's §1 is correct that DSv4 MoE is not a single matmul — it's
  top-k × per-expert batched. CODEX wires `turbomind::gemm::Gemm::Run`
  into the `case GGML_TYPE_F8_E4M3_B128:` branch in `mmvq.cu` and hopes
  that calling it 8 times per token per layer is good enough. §7.1's own
  launch-tax math admits this: "8 active experts × 60 layers × per-token
  = ~30 ms/token just in launch overhead = 33 t/s ceiling **before any
  compute**". CODEX *uses this number to push back on 20 t/s*, but then
  proposes an architecture that **inherits that ceiling**. If the
  argument against the intent's 20 t/s is "we can't beat ~33 t/s on
  launches", then the plan must address launches, not accept them.
- **"Smallest slice" is too small to learn anything.** A 16-tensor
  hand-authored hot list with FP8 only, no MXFP4, no MoE routing
  awareness, gives an integration-cleanliness win but a marginal
  perf win. The 8 t/s floor (D4) is 1.7×. That's plausibly achievable
  by leaving the experts on CPU and just unblocking the host loop in
  `src/models/deepseek4-family.cpp`. We won't know from this sprint
  whether GPU experts are worth the integration cost.
- **The "FP8 only" choice misses that DSv4-Flash's experts are MXFP4,
  not FP8.** The shared brief is explicit: "MXFP4 experts + F8_E4M3_B128
  dense" (intent line 23). The 145 GiB production model has FP8 only on
  dense layers; the MoE experts — which is the thing this whole sprint
  is supposed to move to GPU — are MXFP4. CODEX's Phase 1–5 ship FP8
  dense path. That helps dense, but **dense is not the bottleneck**:
  pp ≈ tg in SPRINT-022 (4.28 vs 4.73) proves we're expert-bound, not
  dense-bound. CODEX's §7.6 budget table also implicitly assumes "16
  hot tensors × ~100 MiB" — that geometry is expert tensors, not dense.
  The narrative says "move FP8 dense to GPU"; the budget says "move 16
  hot experts to GPU". These are inconsistent.
- **The fallback "with env unset and JSON absent, behavior is
  bit-identical to SPRINT-022" claim in Phase 3 is overconfident.** The
  shim still adds a runtime check in the hot path of `mmvq.cu`. Even a
  predicated `if (!tm_dispatch_try(...))` introduces an indirect call
  and a hot-tensor-pointer cache lookup per dispatch. The
  bit-equivalence is obviously true; the perf-equivalence within ±1%
  (D3 gate) is plausible but should be verified, not asserted.
- **Phase boundaries leak.** Phase 4 ("Upload-time packing hook") edits
  `ggml-cuda.cu`'s `ggml_backend_cuda_buffer_set_tensor` to call
  `tm_pack_tensor_if_hot`. That edit lives in the shim TU
  (`turbomind_dispatch.cu`), but it has to be **called from** the
  backend buffer code. The Phase 3 description does not provision that
  call site, and the Phase 4 description doesn't acknowledge it modifies
  a Phase 3 file. In practice these are one phase, not two.
- **Phase time estimates are optimistic in line with MEMORY.md's
  "multiply by 3" rule.** Stated: 1.5–2 + 1–2 + 1 + 1 + 1 = 5.5–7
  days. With the unknowns CODEX itself enumerates (§7.2's opaque
  `Convert`, §7.7's undocumented `rc`, §7.8's cuBLAS symbol-collision
  possibility, §7.9's tile-padding for M=1), 12–18 days is the realistic
  range. The "phased smallest-slice" framing should have led to fewer,
  smaller phases or a longer total.

### 1.3 Risk gaps

CODEX has the strongest risks section of the three documents, but it
still misses or under-weights:

- **MoE dispatch loop in `src/models/deepseek4-family.cpp` itself.**
  §7.1 mentions "host-side gather/scatter for top-K routing per token
  (verify, but this is the GGML norm)" and then moves on. If host-side
  routing is what makes pp ≈ tg, then GPU-resident experts may not
  unblock anything until the routing itself moves into the graph. This
  needs P0-style measurement, not a parenthetical.
- **Validation of the carve-out's `.so` runtime loader path.** §7.10
  mentions `RTLD_NOW | RTLD_LOCAL`, but this is buried in the security
  section. The real risk is that `libggml-turbomind.so` ends up linked
  against a CUDA runtime / CUTLASS variant that doesn't match
  ggml-cuda's. Two CUDA runtimes in one process = silent corruption.
  Needs a check in Phase 1 exit criteria.
- **GGUF block scale interpretation is listed as an open question (#1)
  but **not** as a risk.** §7.2 mentions it; §10 Q1 mentions it; but
  there's no risk row capturing "if we get the E8M0 ↔ kFloat conversion
  wrong, the smoke test passes and we produce garbage downstream." This
  is the SPRINT-019 P1.1 lane-mapping ghost — opaque ISA / opaque
  format, no inverse, no error code. Belongs in §7 with a high impact
  rating.
- **No risk for the upstream nisparks code path drifting.** The intent
  notes the integration depends on the scalar `vec_dot_*_q8_1` path that
  nisparks's a8062c0a9 introduced. If we rebase against upstream during
  the sprint, that path may change. The `if (!tm_dispatch_try(...)) {
  existing_path; }` pattern depends on `existing_path` being stable.
- **Power/thermal on shared `gpu-01`.** REPORT-15 §3 noted DCGM exporter
  interference; §9 promises to "promote `/tmp/run_ncu3.sh` to repo as
  part of D6" but doesn't budget time for it or treat shared-node
  contention as a measurement risk.

### 1.4 DoD completeness

| Gate | Verifiable? | Meaningful? | Notes |
|---|---|---|---|
| D1 `nm -D` clean | Yes | Yes | Strongest gate; trivial to verify, hard to fake. |
| D2 tolerance test | Yes | Yes | But the gate values (5e-3 M=1, 2e-2 M=128) are quoted from §Phase 2 — need to confirm they match the SPRINT-015 P2 contract for the actual model dtype. |
| D3 `=OFF` ±1% | Yes | Partially | ±1% on `tg32` from a 32-token decode is well within run-to-run variance. Needs median-of-N specification (D4 has median-of-3, D3 doesn't). |
| D4 `tg32 ≥ 8.0 t/s` median-of-3 | Yes | Partial | 8 t/s is achievable but doesn't validate that the **plumbing was worth it**. A more meaningful gate: "tg32 with `=ON` is statistically distinguishable from `=OFF` by ≥3σ over N=10 runs." |
| D5 VRAM < 28 GiB | Yes | Yes | Solid. |
| D6 ncu HMMA active% ≥ 45% | Yes | Partial | At M=1, padding to BM=8 means 7/8 of the MMA work is waste. The "active%" metric won't distinguish waste from real work. Needs sm__pipe_tensor_op_hmma instead, or a metric that captures effective TF. |
| D7 stretch 12 t/s | Yes | Stretch | OK. |
| D8 report committed | Yes | Bookkeeping | OK. |

Missing DoD gates:
- **No correctness gate against the actual model end-to-end.** D2 tests
  4 random shapes against the dequant reference. There's no gate that
  says "logits at token 32 match within X% between `=ON` and `=OFF`".
  Without that, D4's 8 t/s could ship with subtly wrong outputs.
- **No "fallback works under fault" gate.** If `Gemm::Run` returns
  rc != 0, §7.7 recommends abort. There's no D-row testing that abort
  fires and the diagnostic is useful.

### 1.5 Implementation realism

- **File paths are correct and grounded in real code.** `mmvq.cu` line
  959–964 (F8_E4M3_B128 case) is real; `gemm_bench_packed.cu`
  build_packed_weight() lines 188–348 are real; the
  `tools/tc-grid/turbomind_minimal/CMakeLists.txt` pattern is the right
  lift-and-shift source. CODEX clearly read the code.
- **The `ggml/src/ggml-cuda/CMakeLists.txt` plus `ggml/CMakeLists.txt`
  edits are concrete and minimal.** Two-line `option()` + one-line
  source-list add. Easy to review, easy to revert.
- **Phase boundaries are mostly actionable but phase 3 + 4 overlap (see
  weakness above).** Combine.
- **The "16 tensors in layers 5–8 (4 layers × 4 experts)" suggestion in
  §10 Q2 is a real, runnable hot list.** Better than the intent's
  "16-entry mock JSON" hand-wave.
- **One actionable gap**: there's no spec for how `tm_handle` is
  associated with a ggml tensor. The map is "`tm_weight_cache:
  map<tensor_ptr, PackedWeight>`" — but tensor pointers are reused as
  ggml allocators recycle backing memory. Should be keyed off
  `(ggml_buffer*, tensor_name_hash)` or similar. Phase 4 will hit this
  on first cache invalidation.

---

## 2. GEMINI draft critique

### 2.1 Strengths

- **Best single design insight in either draft**: §1's identification
  that "Turbomind's `Gemm::Run` is a single-matmul interface; DSv4 MoE
  inference does 256 expert tensors, top-8 routed per token, batched
  per-expert." This is correct and CODEX missed it. If we plug a
  non-grouped API into a grouped problem, launch tax dominates. Both the
  intent doc and CODEX implicitly assume "wrap the existing scalar case
  with a turbomind call" — GEMINI is the only voice arguing that's the
  wrong abstraction.
- **The HBM-bandwidth ceiling derivation in "Theoretical decode ceiling"
  is the strongest numerical analysis in any of the three documents.**
  15.5 GiB/token / 900 GB/s = ~58 tok/s ceiling. This sets the right
  upper bound and frames the design problem as "minimize HBM bytes per
  token and kernel launches per token" — which is the correct framing.
  CODEX gestures at this but doesn't put a number on it; GEMINI does.
- **P0 measurement-first phase is the right discipline.** §4 P0's three
  measurements — per-layer breakdown of where SPRINT-022 time goes,
  streaming overlap with `cudaMemcpyAsync`, MXFP4→INT8 fidelity check —
  would directly invalidate or validate the entire premise of the
  sprint. This is the SPRINT-021 "measure ceilings before optimizing"
  lesson generalized. CODEX does not have an equivalent.
- **The "pipelined CPU experts" alternative (P0 path B) is a genuinely
  contrarian insight.** Top-8 of 256 = ~3% of MoE weight per token. At
  PCIe Gen3 = 15.75 GB/s sustained, with prefetch overlap, the math
  "experts stream from host at ~10–15 t/s ceiling" is at least
  worth measuring before we commit ~18 GiB VRAM to a hot pool. If P0.2
  shows ≥70% overlap, the answer to the sprint may be "don't move
  experts to GPU at all; fix the dispatch overlap." That's a million-
  dollar question and CODEX never asks it.
- **Owning the kernel end-to-end is a real long-term value.** §3 point 1:
  "v13_rf is 49 TF measured. MXFP4 sm70 is theoretically faster… but
  requires turbomind's `Config_MXF4` which is `group_size=32` only and
  currently fails registry match. We own v13 end-to-end; we do not own
  turbomind." This is consistent with the MEMORY.md
  "effort estimation for undocumented hardware work — multiply by 3"
  rule.

### 2.2 Weaknesses

- **The MXFP4→INT8 re-quantization premise contradicts a load-bearing
  user memory.** MEMORY.md "Pre-dequant defeats INT8 quantization":
  > gmem must stay at INT8; dequant happens in SMEM where bandwidth is
  > free. Any TFLOPS number that pre-dequants outside the kernel is a
  > ceiling, not a production number.
  GEMINI proposes precisely this: take MXFP4 weights, **pre-dequant**
  them at load time, **re-quantize** to INT8, and store them on GPU. The
  draft frames this as a fidelity question (§P0.3, "≤ 0.5% mean rel
  error"). It is also a **semantic question**: the production model is
  MXFP4 because the model authors chose MXFP4 — converting to INT8
  changes the model's published quantization recipe. The 0.5% rel error
  gate is far too loose; a serious perplexity check needs full eval
  (HellaSwag / WikiText) not 100 random activations. §10 Q7 admits this:
  "If P0.3 shows MXFP4→INT8 loses > 1 pp perplexity, OR…" — but the gate
  in §P0.3 is **not** perplexity, it's activation-level rel error. The
  proposed gate doesn't catch the case the author worries about.
- **The grouped-MoE kernel scope is too large for one sprint.** §4 P2 is
  "5 days" for: a new templated header (`v14_grouped_moe.cuh`), a new
  outer dispatcher, two specializations (decode + prefill), a cold-expert
  streamer with ping-pong buffers + per-layer weight-pointer-table
  patching, and a new bench harness. That's a 2–3 sprint deliverable on
  its own. The user's MEMORY.md says "execute every step of a written
  plan; skip only with explicit user approval". A plan that prescribes
  P2 in 5 days will trigger that violation.
- **Adding a new `GGML_OP_MUL_MAT_ID_MOE_DSV4` op is a bigger commitment
  than the draft acknowledges.** §4 P3.1 says "Add `GGML_OP_MUL_MAT_ID_MOE_DSV4`
  to `ggml/include/ggml.h` and `ggml/src/ggml.c`." That touches the core
  ggml op enum, which has cross-backend implications: every other backend
  (CPU, Metal, Vulkan, SYCL) needs at least a default-fall-through case.
  And §10 Q6 admits "this is a private-fork performance hack… we
  shouldn't try" to upstream it — so we're forking the core op enum. That
  has follow-on cost forever.
- **The "1 launch per layer" claim doesn't survive the routing pattern.**
  §3 point 3's signature `(W[E][N][K], scales[E][N][K/QK], A, ids[T][topk],
  expert_offsets[E+1])` is fine, but §1's claim of "one kernel launch per
  layer" requires every active expert in the layer to be in the same
  launch. If the hot pool is partial (we only have top-32 of 256 in
  VRAM), and cold experts are streamed via `cudaMemcpyAsync` on a
  side stream, then we either wait for cold streams to land (defeats
  the "one launch" claim) or fire a second launch for cold experts
  (defeats it too). The architecture diagram in §3 hand-waves this.
- **The PCIe overlap math is optimistic.** §"Pipelined-CPU-experts
  alternative": "PCIe Gen3 x16 = **15.75 GB/s** sustained". Real
  sustained Gen3 x16 through `cudaMemcpyAsync` from pinned RAM is
  typically 11–12 GB/s, and only if there's no other PCIe traffic. With
  KV cache writes on the same bus this drops further. The "10–15 tok/s
  with experts streaming" number is the upper end of the realistic
  range, not the median.
- **"v13_rf at 49 TF" is for **dense GEMM** at M=N=K=2048, not for
  MoE-pattern small-M, sparse-routed workloads.** GEMINI inherits the
  v13 mainloop and assumes the TF carries over. SPRINT-019/021's
  champion tables were all measured on packed dense GEMM. The grouped-
  MoE kernel will have lower MMA utilization due to (a) per-expert M is
  small (avg ~5 tokens/expert at pp128, 1 at decode), (b) atomic
  accumulation across split experts adds memory traffic, (c) the
  routing-table lookup is per-CTA. §7's risk row "v14 grouped kernel
  HMMA active% lower than v13's 42%" is honest, but the headline
  ceiling number in §1 should be more conservative.
- **Decode variant signature is inconsistent.** §3 point 6 says decode
  variant is "M=1..8, BN=128, K_split=8" and "reuses v12s SplitK
  champion". v12s is INT8 single-GEMM, not grouped. Reusing its
  SplitK pattern across a grouped kernel means the SplitK atomics
  (MEMORY.md "V100 SplitK atomic pattern") have to be coordinated
  across experts, which is new mechanism not in v12.

### 2.3 Risk gaps

- **No risk for the new `GGML_OP_*` enum slot conflicting with upstream
  llama.cpp drift.** §10 Q6 acknowledges it as "private-fork hack" but
  the risks table doesn't list "next upstream merge breaks our op enum."
- **No risk for perplexity regression from MXFP4→INT8 re-quant.** §7
  row 1 lists "tolerance" risk; §10 Q7 lists "1 pp perplexity" as a
  kill condition. These two aren't reconciled. The risks table should
  carry the perplexity row with mitigation.
- **No risk for the cold-expert streamer producing race conditions when
  the same expert is hot in one layer and cold in another.** The
  streaming staging buffer is ping-pong (2 buffers, §4 P2.4), so
  cross-layer expert reuse may stomp on a still-in-flight kernel's
  weight read. Stream ordering / events / semaphores not specified.
- **No risk for `expert_offsets[E+1]` construction overhead per token.**
  The current `mmid_helper` builds this once per `mul_mat_id` op call.
  At decode, that's per-token. The cost is small but not free. Belongs
  in the risks table because it eats into the launch-tax savings.
- **No risk for v14 cuBLAS-vs-handwritten regression**: if the v14
  kernel ends up below 30 TF on decode, the right answer might be
  cuBLASLt grouped FP16, which §10 Q3 mentions as a stretch. The risk
  isn't formalized.
- **The 5e-3 rel error gate (§P4.1) is **looser** than the SPRINT-015
  P2 tolerance contract** (1e-2 / 5e-3 row-parallel). The text says
  "rel ≤ 2e-2 col-parallel, ≤ 1e-2 row-parallel" but cites SPRINT-015
  — the actual MXFP4-on-sm70 contract from that sprint is row-parallel
  ≤ 5e-3 per the intent doc line 65. This is a wrong-tolerance hazard.

### 2.4 DoD completeness

GEMINI's §6 is **conditional** on the P0 exit memo path choice. That's a
clever structure but creates several issues:

- **Path A's DoD restates the intent's 20 t/s target.** The whole
  contrarian framing of the draft is "20 t/s is hard but the right bar".
  CODEX argued 20 is wrong. GEMINI accepts 20. That's a real disagreement
  the synthesis must resolve.
- **No correctness gate beyond "≤ tolerance rel error vs CPU scalar on
  8 expert mul_mats × 2 shapes."** Same gap as CODEX, plus the tolerance
  number is questionable (see above).
- **No build-cleanliness gate.** No equivalent of CODEX's D1 `nm -D` for
  the v14 kernel's library footprint. The new op enum / new
  `mmid_moe_dsv4.cu` will pull in v13 templates. Should be size-budgeted.
- **No ncu metric specification for grouped MMA workload.** §P4.4 says
  "HMMA active% ≥ 40% in decode, ≥ 50% in prefill." HMMA active% is
  measured per kernel; for a grouped kernel where each CTA may handle
  a different expert with different K-padding, the metric averages over
  CTAs and hides the worst-case expert. Need `sm__pipe_tensor_op_hmma`
  per-CTA or a TF-effective metric.
- **The "if Path C wins P0, we lose this debate" line is honest but
  imprecise.** Falling back to the intent doc's plan is not "the same
  as CODEX's plan" — the intent doc is MXFP4 + FP8 + hot-expert in one
  sprint. CODEX is FP8-only sub-scope. GEMINI should be explicit which
  fallback it accepts.

### 2.5 Implementation realism

- **`tools/tc-grid/kernels/v14_grouped_moe.cuh` lifting v13_rf_v6's body
  is plausible** because the inner-K loop is reusable. But the outer
  dispatcher (top-k routing, expert_offsets iteration, per-CTA expert
  assignment) is new code, not lifted.
- **`tools/quant/mxfp4_to_int8_repack.cpp` doesn't exist and would be a
  ~1500-line tool by itself.** It needs to: parse GGUF, identify expert
  tensors by name pattern, dequant MXFP4 to FP16, requant FP16 to INT8
  per-group, emit a side-car file with the right alignment. This is
  understated as "lifted to production quality" in §P1.1.
- **`src/llama-model-loader.cpp` edits for "side-car mmap + hot pool
  allocation" are non-trivial.** That file is shared with all backends;
  adding env-gated branches is a review surface that hits every backend
  maintainer.
- **Per-layer timing instrumentation in `src/llama-context.cpp` (§P0.1)
  is realistic and useful** — and is the kind of thing CODEX should
  have included too.
- **The streaming dispatcher needs a thread, a stream, two CUDA events,
  and a per-layer pointer-table patch.** §4 P2.4 calls this "owns a
  non-default stream + 2 ping-pong staging buffers" — that's the
  surface, but the implementation is closer to 500–800 lines of carefully
  ordered CUDA event code. Three-day estimate (part of P2's "5 days") is
  optimistic by the MEMORY.md 3× rule.

---

## 3. Cross-cutting issues both drafts share

- **Neither draft acknowledges that DSv4-Flash experts are MXFP4 in the
  GGUF.** CODEX's "FP8 only" plan moves dense, not experts. GEMINI
  re-quants experts to INT8 (loses fidelity, changes the model recipe).
  The shared brief is explicit (intent §"SPRINT-022 baseline"). Whichever
  plan wins must answer "what dtype lives on GPU for the experts" with a
  number that's grounded in the model's actual tensors.
- **Neither draft has an end-to-end correctness gate against the actual
  model.** Both have per-kernel tolerance tests; neither has "32-token
  decode logits within X% of `=OFF` build". Without this, both can ship
  with subtly broken outputs.
- **Neither draft fully addresses host-side MoE routing in
  `src/models/deepseek4-family.cpp`.** SPRINT-022's pp ≈ tg is the
  signal that host routing is on the critical path. CODEX waves at it;
  GEMINI proposes a new op enum that would let routing live in the
  kernel but doesn't fully unwind the model graph. The real question —
  "what fraction of per-token time is in `top_k_select` on the CPU
  versus the experts themselves" — is unanswered. GEMINI's P0.1 is
  the closest, but P0.1 is one bullet, not a phase.
- **Neither draft budgets time to promote `/tmp/run_ncu3.sh` to repo or
  to wire it into a regression harness.** Both cite REPORT-15 §6 as a
  follow-up; both treat it as background work.

---

## 4. SYNTHESIS — which plan to ship

**CODEX wins on execution discipline. GEMINI wins on design insight.
The right SPRINT-023 is CODEX-structured with a GEMINI P0 prefix.**

### What to take from CODEX

1. **Carve-out as separate `.so` with C ABI shim** (CODEX §3, §7.4, §7.5).
   This is the correct ABI decision and not negotiable. Four functions
   only (`tm_pack_*`, `tm_gemm_*`, `tm_handle_free`, `tm_version`).
2. **Dlopen-loaded, env-flag-gated default-off integration** (CODEX §7.4,
   §7.11). Wrap, don't replace, the existing scalar dispatch.
3. **Honest perf floor / stretch instead of 20 t/s as a gate** (CODEX
   §7.1). Headline aspiration stays in the report; gate is the floor.
4. **Real VRAM budget** (CODEX §7.6 numbers — ~19.5 GiB used / ~12.5
   GiB free for 16 hot tensors). Not 22.5 GiB free as the intent claims.
5. **D1 `nm -D` build-cleanliness gate, D3 `=OFF` bit-equivalence, D5
   VRAM ceiling, per-phase exit criteria** (CODEX §6). Verifiable and
   meaningful — keep all of these.
6. **The "GGUF block → turbomind packed conversion is the sprint's real
   cost" risk framing** (CODEX §7.2). Add a corresponding hard gate:
   read `convert.cu :: dequantize_f8_e4m3_b128` *before* writing the
   packer; emit a written one-page spec of the scale conversion
   contract before any code lands.
7. **Force the hot-expert selection decision** (CODEX §7.3): static
   JSON, 16 tensors hand-authored, layers 5–8.

### What to take from GEMINI

1. **P0 measurement-first phase** (GEMINI §4 P0). Three days, three
   measurements:
   - P0.1: per-layer time breakdown in SPRINT-022 baseline. This tells
     us how much of per-token time is host MoE routing vs CPU expert
     compute vs GPU dense vs attention. **Without this, we don't know
     if moving experts to GPU helps at all.**
   - P0.2: synthetic `cudaMemcpyAsync` streaming overlap. If overlap
     ≥ 70%, the right move may be cold-expert streaming rather than
     hot-residency.
   - **Do NOT include P0.3 (MXFP4→INT8 re-quant fidelity)** — the
     re-quant premise conflicts with MEMORY.md and the model's
     published recipe. Replace P0.3 with: **"P0.3: measure single
     `block_f8_e4m3_b128` → turbomind packed FP8 round-trip end-to-end
     correctness on a 7168×7168 tensor."** Same time budget, but
     measures the real engineering risk.
2. **The grouped-MoE design objection is correct, but the response is
   to acknowledge it as a SPRINT-024 deliverable, not to ship a
   v14_grouped_moe.cuh in this sprint.** Add a §Deferred Followups
   row: "Grouped MoE op (`GGML_OP_MUL_MAT_ID_MOE_DSV4`) deferred to
   SPRINT-024 pending P0.1 evidence that launch tax is the dominant
   cost." If P0.1 shows the host loop is the bottleneck, redirect
   SPRINT-024 to a host-side fix instead.
3. **HBM-bandwidth ceiling derivation in the report's introduction.**
   The number (15.5 GiB / 900 GB/s ≈ 58 t/s) is the right upper bound
   and frames the design discussion correctly. CODEX should adopt it.
4. **End-to-end model correctness gate**: take GEMINI's P4.1 spirit
   ("bit-equivalence vs CPU scalar") but tighten to "32-token decode
   logits-L1 within X% of `=OFF` build" as a top-level DoD row.
5. **The "what kills this sprint" question** (GEMINI §10 Q7). Adopt
   the discipline: write three explicit kill-conditions in the sprint
   doc, not nine open questions.

### What to drop from both

- **The "FP8 only, dense only" framing in CODEX** — replace with
  "MXFP4 expert FP8 path" or accept that this sprint integrates
  *only* the FP8 dense layers and explicitly defer MoE-expert
  integration to SPRINT-024. Either decision is fine; the current
  CODEX draft sits between the two.
- **GEMINI's MXFP4→INT8 re-quant.** Violates MEMORY.md pre-dequant
  rule and changes the model's recipe. Out.
- **GEMINI's new `GGML_OP_*` enum slot.** Carries forever-cost. Out.
- **The intent's bundled three-objective scope** (MXFP4 + FP8 +
  hot-expert in one sprint). Both drafts agree on this implicitly.
- **Both drafts' "20 t/s as a gate" treatment** — CODEX rejects it
  (correct), GEMINI accepts it as Path A (wrong). Floor is 8, stretch
  is 12, aspiration is 20. Document the math in the report; gate on
  the floor.

### Recommended synthesized sprint shape

| Phase | Owner draft | Days | Deliverable |
|---|---|---|---|
| P0 — Measurement | GEMINI | 3 | REPORT-17.md picks Path A (FP8 dense), Path B (stream cold experts), or Path C (stay on CPU, fix host loop). Includes per-layer time breakdown, streaming overlap measurement, and FP8 packed round-trip correctness. |
| P1 — Carve-out `.so` | CODEX §Phase 1 | 2–3 | `libggml-turbomind.so` with 4-function C ABI, `nm -D` clean, smoke test green. |
| P2 — Scale-conversion spec + packer | CODEX §Phase 2 expanded | 3–4 | Written one-page contract for E8M0 ↔ kFloat conversion (read `convert.cu` first); packer matches; 4-shape tolerance test passes. |
| P3 — Dlopen + dispatch shim (Phase 3+4 merged) | CODEX §Phase 3 + 4 | 3 | Hot list at load time, runtime hook in `mmvq.cu`, ggml-cuda symbol cleanliness intact. |
| P4 — End-to-end correctness + perf | CODEX §Phase 5 + GEMINI P4.1 | 2 | Logits match `=OFF` within X%, `tg32` floor 8, stretch 12, VRAM < 28 GiB. |

Total: 13–15 days. Honest. Phased. Each phase independently mergeable.
Each phase has a kill criterion that pulls the plug before sunk cost
compounds.

The single biggest synthesis-time decision the user needs to make:
**should SPRINT-023 ship FP8-dense integration (a smaller, lower-risk
sprint that doesn't move the MoE bottleneck) or should it ship
something that actually moves expert compute to GPU (in which case
GEMINI's grouped-MoE objection is right and the timeline is two
sprints, not one)?** Both drafts dodge this. CODEX implicitly picks the
former; GEMINI implicitly picks the latter. The intent doc wants both.

My recommendation: CODEX's FP8-dense plan in SPRINT-023, with GEMINI's
P0 as the entrance gate; if P0 says "host loop is the real
bottleneck", pivot SPRINT-024 to the host loop, not the kernel.
