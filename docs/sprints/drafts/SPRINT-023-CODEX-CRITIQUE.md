# SPRINT-023 — Codex critique of Claude + Gemini drafts

**Date**: 2026-05-15
**Reviewer**: Codex (implementation-focused, skeptical)
**Drafts reviewed**:
- `SPRINT-023-CLAUDE-DRAFT.md` — turbomind plug-in, MXFP4-first, 6 phases
- `SPRINT-023-GEMINI-DRAFT.md` — contrarian custom grouped-MoE INT8 kernel + cold-streaming

Bias note up front: I am biased toward whatever ships in <2 weeks with a clean rollback. I have intentionally pushed back hard on both drafts. Neither draft is naive; both are wrong in different, important ways.

---

## 1. Claude draft critique

### 1.1 Strengths

- **Phase ordering is buildable as written.** P0 (recovery measurement) → P1 (CMake) → P2 (buffer type) → P3 (no-ids dispatch) → P4 (MoE ids) → P5 (VRAM) → P6 (perf) is a defensible sequence. Each phase has a ship gate that a CI bot could enforce (`cmake -DGGML_TURBOMIND_GEMM=ON` builds; bit-compare passes; `nvidia-smi` reports <30 GiB).
- **File paths are grounded.** `mmvq.cu:947`, `mmq.cu:23`, `common/arg.cpp:2279`, `mmq.cu:180` for `ggml_cuda_launch_mm_ids_helper`, the existing `case GGML_TYPE_MXFP4` rows — I spot-checked these and they line up with the tree. This is not LLM dream-code.
- **The buffer-type design avoids `convert.cu` bloat.** Putting layout conversion in a new `CUDA0_TURBOMIND` buffer type instead of folding it into `convert.cu` is the right call. `convert.cu` is per-row dequant; turbomind packing is a whole-tensor one-time conversion. Keeping them separate keeps the blast radius small.
- **`-ot` integration is the cleanest UX.** Reusing `--override-tensor` regex (`exps=CUDA0_TURBOMIND`) means no new CLI surface and the operator's mental model is unchanged. This is a real strength that the Gemini draft misses entirely (Gemini adds a new `GGML_OP_*` op enum which is the *opposite* of a clean UX).
- **Explicit MXFP4-only scope.** Defers F8_E4M3 dense to SPRINT-024. Honest and correct — the intent doc lists FP8 as a success criterion but the Claude draft argues (correctly) that scope-discipline beats one-sprint-doubles-the-converter-work.
- **Runtime kill-switch (`LLAMA_DISABLE_TURBOMIND=1`) is mature.** Compile-time flag for binary size + symbol cleanliness; runtime env var for production rollback. DoD-12 is the kind of DoD a SRE actually cares about.

### 1.2 Weaknesses

- **20 tok/s is asserted, not derived.** The draft inherits the intent target ("tg32 ≥ 20") and never builds a roofline. Gemini's §"Theoretical decode ceiling" does the math (15.5 GiB/token / 900 GB/s ≈ 58 t/s HBM ceiling, ~45–50 realistic). Claude's "we'll just be 4.2× the baseline" is a wish, not a model. **If the underlying bottleneck is launch overhead (which Risk #5 acknowledges at 4.9 s/decode worst case — that *alone* caps us at <0.2 t/s), then no amount of Config_MXF4 mainloop tuning saves us.** The draft notes this in Open Question #2 and Risk #5 but does not turn either into a P0 sub-experiment. It should.
- **Per-expert kernel-launch arithmetic is alarming and ignored.** Risk #5 literally says "491k launches/decode × 10 µs = 4.9 s overhead." That number kills the perf target on contact. The draft then says "P4 ships sequential; P6 must measure this." That is backwards. **Launch-overhead measurement must be P0** (a 30-min synthetic experiment) — if it's pathological, you redesign before P1 instead of building 5 phases of dead infrastructure.
- **MoE-ids path completely glosses over what `tmg::Gemm::Run` expects.** §3.3 says "iterate the `expert_bounds` array … launch one `tmg::Gemm::Run` with the per-expert slice." Has anyone confirmed turbomind's `Gemm::Run` accepts an arbitrary M *per call* without recompiling the registry? gemm2's launcher caches dispatch decisions in `dispatch_cache.h`. 491k calls/decode through a cache miss path is a separate disaster from launch latency. Needs P0 validation.
- **"WORKSPACE reuse across CUDA streams" risk is hand-waved.** Risk row says "P6 stream-batched would need per-stream workspace partitioning — explicitly flagged in P6 plan, not done blindly." Translation: "we don't know how to do this and will figure it out." If P6 is the perf path, this is the load-bearing risk and there is no mitigation in the plan.
- **No mention of the F8_E4M3 dense path's contribution.** Per-token decode hits dense FP8 every layer (~7.5 GiB) AND experts. Even if MoE experts go from CPU (slow) to GPU-MXFP4 (fast), the dense layers still run on nisparks's scalar path. Gemini explicitly counts dense in the roofline; Claude treats dense as "5% of params, not the bottleneck" without numerical support. That handwave needs validation.
- **`init_tensor` packing leaves no path for partial-load recovery.** §3.2 says "the canonical GGUF bytes are freed after packing." What happens when `init_tensor` succeeds for layers 0–10 then OOMs at layer 11? The intermediate state is a mix of packed+unpacked tensors with no rollback. Spec needs an explicit two-phase load (allocate all VRAM first, pack second, free staging third) or an atomic rollback on OOM.
- **Open Question #5 is a P1 blocker disguised as a curiosity.** "Does `-ot exps=…` regex support `CUDA0_TURBOMIND`?" If the answer is no, the whole UX premise of the draft falls over and P2's ship gate is unreachable. This must be answered in the first hour of P1, not deferred to "P2 ship gate must explicitly test."
- **P0 ship gate at 45 TF MXF4 is set without prior measurement.** The intent doc says MXF4 sm70 only registers at `group_size=32` and was *never measured*. SPRINT-021 P0 ran gs=128 (no match → skipped). 45 TF is a hopeful interpolation. The draft acknowledges this but doesn't say what happens to schedule if MXF4 hits, say, 35 TF: "fall back: pivot SPRINT-023 to F8_E4M3_B128 dense port" is a whole different sprint. That's a P0 fork in the road, not a mitigation.

### 1.3 Risk gaps (engineering risk — ABI, build, deploy, rollback)

- **Build system: link chain of `gemm2 + core + parser + cuda_utils + fmt + cutlass` into `ggml-cuda` is treated as "the same as tc-grid."** It is not. `tc-grid` builds a *standalone* binary. ggml-cuda is a shared library consumed by `libllama` and bundled in distro packages. Adding a vendored CUTLASS + FetchContent fmt to a library that downstreams (`llama-cli`, `llama-server`, `llama-quantize`) link against changes their link line. Symbol collisions with downstream forks that pull their own fmt are likely. **Not addressed.**
- **ABI: turbomind's `tmg::Gemm` is a C++ class with constructors that touch CUDA globals.** If `libggml-cuda.so` is dlopen'd (which llama.cpp's backend registry does), `tmg::Gemm` ctors run on every load. If a downstream consumer links two backend libraries that both pull turbomind, you get duplicate symbols. The draft says "compile-time guard keeps unrelated builds clean" but doesn't address dlopen / dual-backend cases.
- **Deployment: binary size growth not quantified.** Vendored CUTLASS + 30 turbomind `.cu` files compiled for sm70 only adds 30–100 MB to `libggml-cuda.so`. The draft says "the flag lets distros opt in" but doesn't measure the actual delta. Pre-built binaries shipping with `=ON` will surprise downstream packagers.
- **Rollback: the draft has TWO rollback paths and one is half-built.**
  - Compile-time `=OFF`: clean. Whole turbomind branch disappears.
  - Runtime `LLAMA_DISABLE_TURBOMIND=1`: needs the *predicate* to honor it but the *buffer type* still ate the tensor at load time. If you flip the env var mid-process (or load a model then flip), you have packed weights in a "turbomind buffer" with no dispatcher. Either the buffer falls back to a scalar path operating on packed bytes (silent corruption!) or it errors. The draft doesn't say which.
- **CUDA 12.2 + gcc 11.4 + cmake 3.22.1 lock-in.** `--expt-relaxed-constexpr` + cuda_bf16.h forced-include + CUTLASS interactions are known-fragile across toolchain versions. The draft pins to one combo without saying what happens if a user is on CUDA 12.6 or gcc 12. Acceptable for V100 only target, but the failure mode (build break at unrelated symbol) needs to be in a "supported toolchain" matrix.
- **`cuda-patches/` directory is required but untracked.** Open Question / Dependency calls out that turbomind sources live in `research/lmdeploy/` (untracked) and patches live in `cuda-patches/`. Anyone who clones the repo without those won't build with `=ON`. The draft doesn't say whether `cuda-patches/` becomes a tracked submodule or stays a local-only hack.

### 1.4 DoD completeness

- **DoD-1 through DoD-8 + DoD-12 + DoD-13 are CI-enforceable.** Good.
- **DoD-9 / DoD-10 / DoD-11 are perf-bar DoDs that the draft *explicitly* says are non-blocking** ("Sprint ships when DoD-1 through DoD-8 + DoD-12 + DoD-13 are met"). This is honest but it means the sprint can "ship" without hitting the 20 t/s target. The intent doc treats 20 t/s as the **North Star**. Either the intent doc relaxes the perf bar to "infrastructure landed, perf measured" or the DoD list is wrong.
- **DoD-7 byte-identical first-32-tokens** is the right correctness gate — measurable, deterministic, CI-friendly. Best DoD in either draft.
- **Missing DoD on launch overhead.** Given that Risk #5 is the dominant risk to DoD-9, there should be a DoD like "measured per-expert launch overhead at M=8 is <X µs" gated in P0.
- **Missing DoD on toolchain compatibility.** "Builds on V100 + CUDA 12.2" is one config. No DoD says what happens on CUDA 12.6 or sm75 with flag ON. Probably should fail-fast or skip-with-warning; nothing in the DoD spells it out.

### 1.5 Implementation realism

- **Phase ordering is mostly executable** but P0 is mis-scoped. P0 should be MXF4-ceiling-measurement AND per-expert-launch-overhead. Right now P0 only does the former, kicking the launch question to P6.
- **File paths are real.** I checked `mmvq.cu` (947, 953, 959 are the MXFP4/NVFP4/F8 dispatch rows) and `mmq.cu` (23, 26, 276 are the corresponding case rows). The draft is grounded.
- **Activation cast strategy is hand-wavy.** "Cast at the dispatch boundary into a pool-allocated F16 buffer" — pool allocations have non-trivial cost when you do 491k per decode. Either pre-allocate per layer at model load (which the draft considers in Open Question #4 then waves off) or measure it. Probably fine, but unmeasured.
- **The "first 16 layers heuristic" for hot-expert placement** (P5) is a reasonable MVP but the rationale "MoE routing has more entropy in early layers" is a thumb-suck. Should be cited or measured. No real cost to measure (single CPU profile run produces hit counts per layer in <10 min).
- **Schedule realism**: 6 phases. P0 is days. P1 is days (FetchContent + CMake + symbol audit). P2 is a week (buffer-type registration is fiddly — `init_tensor` ordering, ALIGN handling, multi-device peer access). P3 is a week. P4 is a week (MoE ids edge cases). P5+P6 are days. **Realistic calendar: 3-4 weeks**, not the implied 2. Per the user's calibration note ("multiply gut estimates by 3× when signals like opaque ISA mappings, commented-out upstream targets, or sticky build-system caches are present"), 6 weeks is the prudent number.

---

## 2. Gemini draft critique

### 2.1 Strengths

- **The roofline math is the most valuable single contribution in either draft.** §"Theoretical decode ceiling" establishes 15.5 GiB/token HBM read at MXFP4 → 58 t/s HBM ceiling → ~45–50 t/s realistic. 20 t/s = 44% of ceiling. This is the kind of derivation the Claude draft owes the intent and doesn't deliver. **Either sprint plan should start with this.**
- **Calling out the grouped-GEMM mismatch is correct.** Turbomind's `Gemm::Run` is a single-matmul API; MoE inference is grouped-GEMM-with-routing. The draft is right that wrapping the single-matmul API in a per-expert loop pays the launch tax 256 × 8 × 60 = 122,880 times per token at worst case. The Claude draft *also* identifies this risk (Risk #5) but doesn't act on it; Gemini makes it the central architectural point.
- **Cold-streaming dark horse is the most useful framing in either draft.** "What if experts on GPU is the wrong move and prefetched-from-host wins?" is a question that costs 3 days to answer and could save 4 weeks of mis-built infrastructure. Even if you ultimately build the Claude path, having the streaming-overlap experiment in your back pocket is high-value information for SPRINT-024+.
- **P0 is a real decision gate**, not a measurement. Three paths (A: hot INT8 grouped, B: pipelined CPU experts, C: turbomind plug-in) with explicit exit criteria is the right structure. Claude's P0 is a tactical recovery; Gemini's P0 is a strategic fork.
- **Reuses owned infrastructure (v13_rf at 49 TF measured).** The MEMORY.md note about v13_rf reaching 88% of FP8 ceiling at INT8 is the strongest argument for "we own this kernel, we can extend it." When something breaks at 2am, you can debug your own code; you cannot debug turbomind's `Gemm::Run` dispatch cache.

### 2.2 Weaknesses

- **MXFP4→INT8 re-quant is asserted as ~0.5% perplexity loss with no evidence.** §3 says "loses ~0.5% perplexity (well within nisparks tolerance contract)." Where does 0.5% come from? MXFP4 → INT8 is a *lossy precision change*, not a layout change. DSv4 was *trained* with MXFP4 experts. Re-quantizing the weights post-training is a model-modification action with unknown downstream effect on long-context generation, refusal behavior, and tool-use. P0.3's "≤ 0.5% mean rel error on 100 random activations" is a per-tensor numerical-error check, NOT a perplexity check. **This is a model-fidelity issue masquerading as a numerical-precision issue.** A real validation requires wikitext-2 perplexity measurement before/after — a multi-hour run, not 100 random activations.
- **Custom kernel timeline is fantasy.** P2 says "5 days" for a new templated grouped-MoE kernel with two specializations (decode + prefill), cold-streaming integration, and a bench harness. The v13 line took **multiple sprints** to land at 49 TF. v14 grouped is a new dispatch shape, new register allocation, new SMEM staging layout, new cold-streaming co-design. **Realistic: 3–4 sprints, not 5 days.** This is the calibration-rule violation the MEMORY.md warns about ("multiply gut estimates by 3× when…").
- **"One launch per layer" is the goal but isn't shown to be achievable on sm70.** Grouped GEMM with per-expert top-k indexing requires either CUDA Dynamic Parallelism (slow on sm70), CUDA Graphs (works but a non-trivial migration), or a single mega-kernel that resolves `ids` internally. The draft picks option 3 (the mega-kernel) but doesn't explain how a single CTA finds the (token, top-k-position) pairs that map to its expert without a host-side prefix-sum + barrier. The kernel sketch in P2.2 is plausible but not pseudocode; a careful read shows the CTA needs `expert_offsets` precomputed (which exists, fine) plus a per-expert iteration over tokens, which means register/SMEM pressure scales with `tokens_per_expert` — unbounded at long prefill.
- **Cold-expert streaming bandwidth math is too optimistic.** §"Pipelined-CPU-experts alternative" claims "PCIe Gen3 x16 = 15.75 GB/s sustained → ~280 ms per token = ~3.6 tok/s naive." That's correct for the *sustained* number but PCIe doesn't hit sustained throughput at the granularity of one expert (single transfer ~145 MiB ≈ 9 ms ideal but the per-transfer setup overhead is real). At 256 transfers/layer × 60 layers, the per-call cost is what dominates, not the bandwidth ceiling. The draft never derives the per-call number.
- **`GGML_OP_MUL_MAT_ID_MOE_DSV4` as a new ggml op is the wrong abstraction.** Adding op enum values to `ggml/include/ggml.h` for a downstream-specific op makes the fork un-upstreamable AND introduces enum-value conflicts the day upstream adds a new op at the same numeric slot. The draft acknowledges this in Open Question #6 ("No, and we shouldn't try [to upstream]") but doesn't appreciate that *this also affects forward-compat with upstream pulls*. Every `git merge upstream/main` becomes a hand-resolution exercise.
- **Hot/cold split assumes "hot-experts.json" exists or doesn't matter.** The draft says synthetic-uniform-top-32 ships first, real profile is SPRINT-024 owned. But §3 lists "hot-experts.json (offline profile)" in the architecture diagram as if it's input. If the synthetic profile gets you to 20 t/s, fine; if not, the sprint silently depends on SPRINT-024 work to hit its own DoD. The Claude draft handles this more honestly with the "first 16 layers" heuristic (which is also a synthetic profile, but explicitly bounded).
- **Removes turbomind from the runtime entirely.** This is a strength (smaller surface) but also a weakness: we lose the proven `Config_E4M3` 59 TF and `Config_U4_g` 65 TF data points as runtime options. If v13 grouped delivers, say, 35 TF in production-grouped configuration, we have no fallback to turbomind without re-litigating the whole sprint.
- **§"Theoretical decode ceiling" arithmetic has a probable error.** "256 experts × 145 MiB expert × 8 routed = ~8 GiB of weight per token." But not every layer activates the same 8 experts; the routing changes per layer. If 8 experts × 60 layers are activated per token at MXFP4, that's `8 × 60 × ~38 MiB/expert = ~18 GiB/token` of expert weight read, not 8 GiB. Even if the per-expert size estimate is off, the per-token weight-read should be derived from `(active_experts/token) × layers × bytes_per_expert`. The draft conflates per-token and per-layer. This matters because the realistic ceiling moves from 58 t/s to ~35 t/s, putting 20 t/s at 57% of ceiling instead of 44% — still achievable but tighter.

### 2.3 Risk gaps (engineering risk)

- **No build flag fallback story.** `GGML_MOE_INT8_REPACK=OFF` is mentioned (P3.3) but the rollback semantics are unclear. The side-car `.int8-experts.bin` file is generated by an offline tool — what happens if it's stale? If the user runs `=ON` build with a model that was repacked at an older sprint, do we detect the mismatch? No version check is in the spec.
- **Side-car file format spec is missing.** "side-car file `MODEL.int8-experts.bin` with `(E, N, K)` per layer" is one line of design. Endianness, layer-order, magic number, version, scale layout, dimension order — none specified. This is the kind of detail that bites in P4 verification when bit-compare disagrees.
- **No ABI discussion.** Adding `GGML_OP_MUL_MAT_ID_MOE_DSV4` to the public `ggml.h` enum changes the ABI for *every consumer* of libggml.so. Other backends (CPU, Metal, Vulkan) will see an unknown op enum and either crash or fall back to scalar (does ggml have a default fallback for unknown ops? answer: no, it asserts). This single line breaks the multi-backend story.
- **Rollback to SPRINT-022 baseline isn't tested.** Claude has DoD-2 (`=OFF build runs identical to baseline`). Gemini has it as a footnote in DoD #2: "Build with `GGML_MOE_INT8_REPACK=OFF` produces SPRINT-022 baseline identical TPS within ± 2%." Same gate, less prominent. But the ggml.h op-enum change persists at compile time regardless of the build flag — so the "OFF" build still has the new op enum which is a public ABI change. **Cannot cleanly roll back without reverting the header.**
- **Deployment story for the side-car is zero.** The side-car file is generated by a new tool (`mxfp4_to_int8_repack.cpp`) but nothing in the plan covers distribution, checksums, or how downstream packagers ship it. Operationally this is a "run a tool, get a file, hope it's right" workflow with no version-pinning to the GGUF source.
- **Custom kernel correctness is harder to verify than turbomind plug-in.** The Claude draft can bit-compare against nisparks's scalar MXFP4 path (10 cases, ±2e-2). The Gemini draft has to verify both the grouped routing AND the INT8 re-quant fidelity. The correctness surface is roughly 2× larger.

### 2.4 DoD completeness

- **DoD lists are noticeably looser than Claude's.** Gemini's §6 has 4 numbered bullets vs Claude's 13. The detail is missing:
  - No CI-enforceable check on "≤ 0.5% perplexity delta" — and as noted, the proxy "≤ 0.5% mean rel error on activations" is not the same thing.
  - No DoD on the side-car file format or version.
  - No explicit measurement of grouped-kernel launch count vs the turbomind-per-expert baseline; the central architectural argument (fewer launches = faster) is asserted but not gated.
  - "If Path A wins P0 then …" branches are not separately enumerated as DoDs per branch.
- **The "If Path B wins P0, rewrite §4 P1-P3" escape hatch** is honest but means the sprint plan is conditional on a measurement that hasn't happened. A CI bot cannot enforce conditional DoDs. Either commit to a path before sprint start (with the P0 measurement as a separate hardening sprint), or accept the plan has indeterminate scope.
- **HMMA active% gate (≥40% decode, ≥50% prefill) is good and measurable.** Comparable to Claude's DoD-11.
- **Missing DoD on cold-rate.** The cold-streaming path is in P4.4's bench config but there's no gate like "cold expert rate on standard prompt set <X%" — which is the *whole premise* of the hot/cold split working.

### 2.5 Implementation realism

- **File paths are mostly grounded** but skinnier than Claude's: `mmq.cu:180` for `ggml_cuda_launch_mm_ids_helper` is cited (real); `src/llama-model-loader.cpp` for the load hook (real); v13_rf_v6 reference is from owned code so it's verifiable.
- **The biggest realism failure is the kernel timeline.** v14 grouped MoE in 5 days is not credible given:
  - Prior sprints needed 2–3 sprints to add a single new register-allocation strategy to v10/v11/v12 line
  - Grouped routing changes the SMEM staging pattern (per-expert weight base pointers, dynamic offsets) which is the bank-conflict-sensitive part per MEMORY.md
  - The "decode variant with SplitK=8" needs reduction across split groups — already known fiddly from v12s
- **§4 ordering has a subtle problem**: P1 (host-side repack tool + load hook) is sequenced before P2 (the kernel) but P1 emits a file format that P2 has to consume. Without the kernel side spec'd, P1 risks producing the wrong format. Either P1 and P2 are co-spec'd or P1 ships and P2 finds a layout mismatch.
- **Implicitly demands a v13 → v14 codegen extension that's never been done.** v13 is single-matmul. v14 grouped is a *different kernel family*. The draft says "Body is v13_rf_v6's K-fused inner loop unchanged" but the K-loop body is one piece — the SMEM tiling, A-fragment iteration, and epilogue reduction all change for grouped. "Unchanged" is generous.

---

## 3. Cross-cutting issues both drafts dodge

1. **Neither draft addresses what happens when DSv4-Flash routes to a cold expert (or unmodeled) at decode time.**
   - Claude: scalar fallback path always exists, but the operator just *moved* expert weights into the turbomind buffer. If routing hits an expert that's on CPU (because `-ot` regex put only some on GPU), do we fall back to nisparks scalar on the CPU copy? The draft assumes yes but never shows the path.
   - Gemini: cold-streamer fires `cudaMemcpyAsync + wait`. Synchronous wait on the host stalls every other expert. Open Question #5 admits this; nothing in the plan bounds the worst-case latency.

2. **Activation precision (F32 → F16 → F32) is dismissed in both drafts.**
   - Claude: "DSv4 is already FP8/FP4 trained so activations are noise-tolerant."
   - Gemini: doesn't even mention it.
   - This is a model-correctness claim. Validation = perplexity on wikitext, not "first 32 tokens match." The first-32-tokens DoD (Claude's DoD-7) catches gross corruption but NOT a slow drift in long-context generation. If either sprint ships and wikitext perplexity rises 2%, we have a quality regression nobody flagged.

3. **Neither draft addresses concurrent multi-stream correctness in the existing ggml-cuda graph.**
   - ggml-cuda may run multiple ops on multiple streams. A turbomind kernel that allocates a workspace via the ggml pool needs to live on the right stream. Claude flags this in S-2 but doesn't say where stream ownership comes from. Gemini doesn't mention streams at all in the integration layer.

4. **The intent doc demands `F8_E4M3_B128 GPU path correctness` as a success criterion.**
   - Claude defers it to SPRINT-024. Gemini doesn't even acknowledge it (proposes an INT8 path that doesn't address dense F8 layers at all).
   - **Both drafts are non-compliant with the intent.** The planner needs to either relax the intent or one of these drafts needs to add F8 dense.

5. **Hot-expert profile JSON.**
   - Intent doc lists "Hot-expert profile source" as Open Question #2.
   - Claude defers to SPRINT-024 with the "first 16 layers" heuristic.
   - Gemini ships a synthetic-uniform `.tsv` and defers real profiling to SPRINT-024.
   - **Same answer, different wrapper.** Neither solves it.

6. **Neither draft says where REPORT-17 lives or how it relates to REPORT-15/16.**
   - Both drafts assume a REPORT-17. Claude calls it `REPORT-17-SPRINT-023.md` (P6). Gemini calls it `REPORT-17.md` (P0 exit memo). These are different things. Naming collision needs resolution before either lands.

---

## 4. On the specific contested questions

### Is 20 tok/s decode realistic?

- **Gemini's roofline is correct in spirit, sloppy in arithmetic.** Re-derive: DSv4-Flash activates ~37B params per token. Top-8 of 256 experts → on the order of 8 experts × 60 layers ≈ 480 expert invocations × ~38 MiB MXFP4 per expert ≈ **18 GiB of expert weight read per token**, plus ~7.5 GiB dense FP8 = **25 GiB/token**. At V100 HBM 900 GB/s, the pure-HBM ceiling is **36 tok/s**. Realistic at 50% achieved → **~18 tok/s**. **20 tok/s sits AT or SLIGHTLY ABOVE the realistic HBM ceiling.**
- This means **the target is tight but not impossible** — and critically, it means *kernel efficiency matters less than not reading the weights twice*. Both drafts' bottleneck story (launch overhead, MMA utilization) is secondary to "are we hitting HBM at peak rate."
- Claude's draft does not derive this; Gemini's does but mis-counts.
- **Conclusion: 20 tok/s is plausible but only with HBM-bound execution. If either draft introduces redundant memory traffic (e.g. F32↔F16 casts that double-touch activations, or scale buffers read multiple times), the target is missed regardless of kernel choice.**

### Does GROUPED GEMM matter?

- **Gemini's framing is correct: turbomind's `Gemm::Run` is single-matmul.**
- **But Gemini overstates the consequence.** Modern CUDA launch overhead on a warm context is ~3–5 µs, not 10 µs. At 480 expert-invocations/token × 5 µs = 2.4 ms/token = **416 tok/s ceiling from launch alone**, comfortably above 20 tok/s.
- The *real* problem with single-matmul-per-expert isn't launch overhead — it's **HBM amplification**. Every per-expert call re-reads its weight slice, and if the kernel doesn't fuse top-k routing, the activation tensor gets read once per (token, expert) pair instead of once per token. Gemini gestures at this; Claude misses it.
- **Conclusion: grouped GEMM matters not for launch-tax-elimination but for activation-traffic-reduction. Both drafts could improve here. Claude could fuse multiple per-expert calls into a single graph capture; Gemini could write the grouped kernel. The cost/benefit favors graph capture (low effort) over a new kernel (high effort).**

### Is the build complexity being handled honestly?

- **No, in both drafts.**
- Claude says "the same FetchContent dance as tc-grid" — but tc-grid builds a standalone binary, not a backend library that gets dlopen'd. The complexity delta is real and unacknowledged.
- Gemini hand-waves the build entirely ("no new third-party, no CUTLASS pull-in") — but adding a new GGML op enum is a *public-ABI* change with downstream effects across all backends. Different category of complexity, equally underplayed.
- **Claude is more honest about the cost (lists fmt, CUTLASS, FetchContent, the `--expt-relaxed-constexpr` flag, etc.) but underestimates the symbol/ABI risk for shared-library consumers.**

### Is the rollback path clean?

- **Claude's rollback is** *partially* **clean.** Compile-time `=OFF` is clean. Runtime `LLAMA_DISABLE_TURBOMIND=1` post-load is fundamentally broken (buffer ate the tensor; predicate disable produces orphaned packed weights). The draft owes a spec for "buffer-type fallback when dispatcher is disabled."
- **Gemini's rollback is broken at the ABI level.** Adding `GGML_OP_MUL_MAT_ID_MOE_DSV4` to the public enum is not rollback-able at runtime. The build flag toggles whether the op fires, but the enum persists. Downstream consumers see the new value forever.
- **Neither draft has a "we shipped this and now production broke and we need to revert in 1 hour" plan.**

---

## 5. Synthesis: which draft to ship and minimum safe scope

### Direct answer

**Ship a hybrid that takes the Claude draft as the structural skeleton and grafts Gemini's P0 measurement gate onto the front.** Specifically:

1. **Use Claude's phase ordering and file layout.** The `CUDA0_TURBOMIND` buffer-type integration is the lowest-friction way to add a new dispatch path to `ggml-cuda` without breaking ABI or upstream merges. File paths are grounded; CMake glue follows established tc-grid pattern.

2. **Replace Claude's P0 with Gemini's P0.** Gemini's "three-path decision gate" is structurally correct. Run it as 3–5 days of measurement before committing to any phase of implementation:
   - **P0.1**: MXF4 ceiling at `group_size=32`, all production shapes. Claude already proposes this.
   - **P0.2**: Per-expert kernel launch overhead synthetic at M ∈ {1, 8, 64}. Single-matmul vs streamed. Gemini's launch-tax argument either is or isn't real — measure it.
   - **P0.3**: PCIe-streaming overlap synthetic, copying expert weights from pinned host while a v13_rf kernel runs. Gemini's "Path B" dark horse — confirms or rules out the cheaper option.
   - **P0.4**: HBM-roofline reality check: run one-layer-of-DSv4-Flash with experts resident, count actual HBM bytes/token via ncu's `dram__bytes.sum`. Confirms whether 20 tok/s is HBM-bound at all.
   - **P0 exit memo** picks ONE of: turbomind plug-in (Claude path), streaming experts (Gemini Path B), or **defer the sprint** if no path clears 15 tok/s in measurement.

3. **Cut F8_E4M3 dense from scope.** Both drafts agree on this. The intent doc must be relaxed — or this becomes SPRINT-024.

4. **Drop the custom v14 grouped kernel.** Gemini's signature contribution is the *insight* (grouped matters), not the implementation. The implementation is a 3-sprint risk against a 1-sprint reward. If P0 shows launch-tax is the bottleneck, mitigate via CUDA Graphs capture in the Claude path (much cheaper than a new kernel).

5. **Add explicit DoDs missing from both drafts:**
   - **Perplexity check**: wikitext-2 perplexity within 0.5% of CPU baseline (the only real correctness test for a precision-changing path).
   - **Launch-overhead measurement**: P0.2 produces a number; DoD requires it stays below the modeled budget.
   - **HBM achieved fraction**: ncu `dram__bytes_read.sum.pct_of_peak` ≥ 50% on decode (the real bottleneck per the roofline).
   - **Rollback drill**: a script that flips `=OFF` and re-runs the SPRINT-022 baseline, gated to identical (±1%) TPS. CI bot can run this.

### Minimum SAFE scope (what to actually ship)

If forced to a 2-week box, ship **only this**:

- **Week 1 (P0)**: Run Gemini's 3-path measurement gate. Produce REPORT-17 with a single recommended path AND launch-overhead numbers AND HBM-achieved baseline. **No code in `ggml/src/ggml-cuda/` yet.**
- **Week 2 (P1+P2 of Claude path, IF P0 picks Claude)**:
  - CMake flag `GGML_TURBOMIND_GEMM=ON` (P1).
  - `CUDA0_TURBOMIND` buffer type registered + `init_tensor` packs MXFP4 (P2).
  - **No dispatcher yet.** No perf gain yet. Just infrastructure + correctness gate + rollback-cleanness gate.
- **Outcome**: At end of 2 weeks, we either (a) have a clean foundation with measured perf ceilings and a credible plan for SPRINT-024 to land the dispatch, or (b) have a P0 memo saying "actually, cold-streaming wins, here's the redesign."

This explicitly **defers the 20 tok/s target** to SPRINT-024. The Claude DoD-9/10/11 perf bars become SPRINT-024 deliverables. The infrastructure (buffer type, build flag, kill-switch) lands clean in SPRINT-023.

**Why this scope?**

- It does not depend on speculative numbers (MXF4 gs=32 ≥ 45 TF; F32↔F16 cast is free; 5-day grouped kernel).
- The rollback path is genuinely clean: revert one CMake flag and the new buffer-type file; nothing in the existing ggml-cuda dispatch changes.
- It pays the build-system / ABI cost ONCE in SPRINT-023, then SPRINT-024 lands the dispatcher against a known-good foundation.
- Per the user's calibration rule ("multiply gut estimates by 3× for opaque ISA work"), the 6-phase Claude plan is realistically 6 weeks. Splitting it 2-and-4 across sprints respects that.

### What I would NOT ship in SPRINT-023

- A new v14 grouped MoE kernel from scratch (Gemini's P2).
- MXFP4 → INT8 re-quant of production weights without wikitext-2 perplexity gating.
- A new `GGML_OP_*` enum value (Gemini's P3.1).
- Stream-batched per-expert dispatch (Claude's P6 stretch) — defer to SPRINT-024 once launch-overhead is measured.
- The "first 16 layers" or "synthetic uniform top-32" hot-expert heuristic shipped as production. Ship as test scaffolding only; real profile in SPRINT-024.

### Final position

**Take Claude's structure, take Gemini's diagnostics, throw away both drafts' performance promises.** Neither plan as written should ship a "20 tok/s" DoD in SPRINT-023; both should treat SPRINT-023 as infrastructure + measurement, and SPRINT-024 as the perf landing. The Claude draft is the safer bet for the structural work because it minimizes ABI surface; the Gemini draft is the better diagnostic input because it surfaces the questions Claude is missing.

The single highest-leverage action right now is the P0 measurement pack. Three days of `ncu` and `cudaMemcpyAsync` synthetics will tell us whether we are building the right sprint at all.

---

*End critique.*
