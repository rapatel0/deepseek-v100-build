# SPRINT-024 — Critique of Codex and Gemini drafts

**Reviewer:** Claude
**Date:** 2026-05-15
**Scope:** SPRINT-024-CODEX-DRAFT.md and SPRINT-024-GEMINI-DRAFT.md, measured against SPRINT-024-INTENT.md and the actual code at `ggml/vendor/turbomind/api.cc`, `ggml/vendor/turbomind/include/ggml-turbomind-api.h`, and `research/lmdeploy/.../matrix_ptr.h`.

---

## Cross-cutting technical reality check

Before grading the drafts, the ground truth their claims need to line up with:

1. **C ABI surface (`ggml-turbomind-api.h` lines 173-186).** Grouped takes:
   ```c
   const void* const* weights_packed,
   const void* const* scales_packed,
   ```
   i.e. a *device pointer* whose pointee is `num_experts` slots wide.

2. **What the kernel actually reads (`matrix_ptr.h` lines 61-97, `api.cc` lines 645/664/677).** `Bdesc.ld = 0` triggers the `kBlocked`/`kIndexed` resolve path:
   ```cpp
   if (ptr.stride == 0) {
       (uint4&)ptr = __ldg((const uint4*)param.ptr + g);
   }
   ```
   That's a **16-byte load per expert** into a `StridedPtr{void* ptr; int stride;}` (declared `__align__(16)` so the trailing 4 bytes are pad). The ABI declares `void* const*` but the kernel reads `StridedPtr*`. The "hidden contract" is real and binding for *both* `weights_packed` and `scales_packed`.

3. **`packed_ld` rule (`api.cc` lines 499-514 single-expert path).**
   ```cpp
   b_packed_ld = (conv_w->order == kRowMajor) ? packed_cols : packed_rows;
   ```
   For sm70 HMMA_884 OPERAND_B Pack_M=1 row-major resolution, that resolves to `K*32` (confirmed by SPRINT-023 P2 — `Bdesc.ld is K*32, not K` — see memory `turbomind_packed_b_ld_factor`). After the operand-tag swap (`if (get_operand_tag(conv_w->pack) != OPERAND_A) std::swap(rows, cols); order = ~order;`), the column-major branch is the live path for the FP8/MXFP4 configs we actually use; "K*32" is the answer in *every* case we have shipped, but the helper must derive it from the converter, not hardcode it.

4. **Synchronous host-device copy in the existing `api.cc` grouped body (lines 606-608).** Every grouped call does a *blocking* `cudaMemcpy` to read `expert_offsets[num_experts]` for `total_tokens`. This serialises against any in-flight stream work and is a latent perf footgun that neither draft notices.

5. **Layer GEMM count.** A MoE FFN layer is `w1`, `w2`, `w3` (with `w1w3` often fused). Grouped dispatch is **per-MoE-linear**, not per-layer: 2 grouped launches/layer with `w1w3` fused, 3 unfused. Not 1.

6. **`num_experts > 1` in the registry.** SPRINT-023 P2.3 only exercised `num=1`. The current sm70 entries (`Config_E4M3<kColMajor, 0>`, `Config_MXF4<kColMajor, 0>`) must be confirmed to support `num > 1` for grouped ragged-batch on sm70 before this lever is real.

---

## Codex draft (SPRINT-024-CODEX-DRAFT.md)

### Strengths

- **Decision-completeness framing** (ship / extend-with-followup / abort-with-diagnosis) — directly addresses the INTENT requirement that "sprint is successful only if it is decision-complete". Gemini does not have this.
- **P0 baseline + instrumentation gate before changes** — sound. Locks the comparison point and makes grouped-path execution observable before any correctness/perf gating, so a regression has a known anchor.
- **StridedPtr layout is correct** in §3.3: separate arrays for weights and scales, each entry `{ void* ptr; int stride; }`, `static_assert(sizeof == 16)`. Matches `matrix_ptr.h` exactly.
- **Hidden ABI contract called out as a first-class risk** (§4 and Risk #2): "the grouped C ABI signature currently looks like raw pointer arrays, but the sm70 grouped iterator effectively expects StridedPtr records when ld == 0." This is the right framing — that is the actual bug shape for anyone reusing this ABI.
- **Fused/unfused topology recognised** (§Implementation P2 step 4): "fused w1w3 → one grouped launch; w2 → one grouped launch; unfused → three grouped launches total." Gemini misses this.
- **Stop-loss criteria explicit on the secondary phase** (P4): "If P4 measures below a 3% gain and complicates the graph materially, stop and defer it." Prevents scope creep into FP16 boundary work if the cast pair is not the bottleneck.
- **Interpretation band for perf gate** (P5): `≥24` ship, `21-24` profile & decide, `<21` do not declare success. Better than a binary gate because launch amortisation has a known finite floor.
- **Test coverage for empty experts** (P3 step 2): "at least one empty expert" in the focused fixture.
- **Correct ordering of math gates** (P3 step 4): "compare grouped TURBOMIND against the legacy TURBOMIND per-expert path first. CPU-MoE remains a secondary reference for overall quality, not the first math gate." Matches INTENT.

### Weaknesses

- **`weight_ptrs_host[e] = src0->data + e * src0->nb[2]`** (§3.3) assumes the packed buffer is laid out with `nb[2]` = per-expert stride in bytes. That is true for the *current* upload path because `ggml_cuda_turbomind_init_tensor` packs per-expert contiguously, but the contract is not enforced and not asserted. If the upload changes (e.g. F-06 per-expert scales alignment), this silently breaks. Worth an `assert(src0->nb[2] == ggml_turbomind_packed_bytes(...).weight)` at the call site.
- **`packed_ld` helper "reuse the same formula api.cc does"** (§4) is hand-wavy. `api.cc` builds the descriptor inline (`b_packed_ld = (order == kRowMajor) ? packed_cols : packed_rows`), it is not a callable helper. The grouped path either (a) duplicates the math, (b) factors out `packed_ld_from_converter()` in api.cc and exports it as a side-effect, or (c) lets api.cc compute it once during pack and store it back through a new out-param on `pack_weight_expert`. The draft should pick one.
- **No effort/calendar estimate per phase.** Gemini's day estimates are coarse but at least force a thought about scope. Without them, P3 (real-model end-to-end with non-MIN weights) and P0 (baseline) get the same weight in a reader's eye when they are 2 hours vs 1+ days.
- **"Temporary debug switch" for legacy path** (P2 step 5) — good intent but no removal criterion. Should be tied to closeout in P6.
- **P5 profiling focus mentions `cudaMemcpyAsync` overhead for pointer tables** but does not call out the existing *synchronous* `cudaMemcpy` at api.cc:607. That memcpy is the most likely candidate for "host-side metadata overhead erodes the launch win" (Risk #3) and would not even appear in a kernel-launch trace.

### Gaps in risk analysis

- **Synchronous host-device sync at api.cc:607.** Not in the risk table. If grouped dispatch is called ~58 times/sec (estimated from the bottleneck math in INTENT), 58 sync points/sec is enough to cap throughput on its own; it is the dominant suspect if grouped lands at 18-21 t/s.
- **`num > 1` on sm70 registry not verified.** SPRINT-023 only exercised `num=1`. The single-expert proof does not prove ragged-batch num>1 dispatch on sm70 `Config_E4M3<kColMajor, 0>` / `Config_MXF4<kColMajor, 0>` — this needs a P0 sub-step. Codex implicitly assumes it works.
- **Pre-baked vs per-launch pointer tables.** Per-launch host-build + H2D upload of `weight_ptrs_dev` and `scale_ptrs_dev` is `num_experts * 16 bytes * 2 arrays` per grouped call — small in bytes, but a host kernel-prep cost that hits every launch. Could be built once at model load (experts are static). Not discussed.
- **k_pack_value uniformity across experts.** All experts share dims, so all packings should emit the same `k_pack_value`, but the grouped helper takes a single `k_pack_value` scalar — so this assumption *must* hold. Not asserted.

### Missing edge cases

- All experts unused on a token (no routes for a layer). Behaviour of `Bdesc.num` when `total_tokens == 0`?
- An expert receiving exactly 1 token vs many (M=1 sub-tile vs M=k). Probably fine but it is the regime where launch amortisation matters least.
- Mixed precision across experts within a single MoE linear — not a real case today, but if ever introduced (e.g. some experts FP8, some MXFP4) the single `ggml_type` parameter would break. Mention as a non-goal.

### Definition of Done completeness

7 items, each verifiable. Notably strong on item 3 (explicit `packed_ld = K*32` for the common path), item 6 (numeric gate on both MIN-16e and MIN-32e), and item 7 (ship-decision artifact). Notably weak on:

- Quality verification (item 5) — "coherent output" is looser than INTENT's "≥75% token match" or P3's "exact match for 32 generated tokens on 3 prompts with `temp=0`". Either tighten item 5 to mirror the P3 gate, or accept that the gate is soft on the ship line.
- No DoD item explicitly demands the closeout artifact (P6 deliverable). Item 7 covers the wording; an explicit "follow-up doc filed if perf misses" would lock that in.

### Specific technical errors

- **None that block correctness.** §3.3 layout, §4 packed_ld rationale, and the kernel-ABI interpretation are all consistent with the source.

---

## Gemini draft (SPRINT-024-GEMINI-DRAFT.md)

### Strengths

- **Compact and readable.** A reviewer can hold the whole plan in their head.
- **Per-phase day estimates** (1, 4-5, 2, 2-3, 2) — concrete enough to be useful as a sanity check on scope.
- **Use-case table with "useful output even if sprint stops here"** for each phase — good for resumability.
- **Risk table format** with likelihood/impact/mitigation columns — easier to scan than Codex's prose.
- **Quantitative quality gate** (P2.2): ≥75% token ID match on 5 prompts. Mirrors INTENT directly.

### Weaknesses

- **§3.1 step 2 describes StridedPtr incorrectly:**
  > "Build a device array of `StridedPtr` structs, one per expert. Each struct contains: `weight`: pointer to packed expert weight. `scales`: pointer to per-expert scale buffer. `stride`: packed leading dimension."

  This is **wrong**. The actual struct (`matrix_ptr.h:9-13`) is `__align__(16) StridedPtr { void* ptr; int stride; }` — one pointer per struct, one stride per struct. Weights and scales need **two separate StridedPtr arrays**, each with `num_experts` entries; you do not combine them into one record. If implemented as described, the 16-byte `__ldg(uint4*)` load in the kernel resolve path would read `{ ptr=weight, stride=lower-32-of-scales }` and the upper half of the scales pointer would land in the next expert's `ptr` field — silent memory corruption.

- **§3.1 step 3 launch-count claim is wrong:**
  > "Single Launch: Call `ggml_turbomind_mul_mat_grouped` once per layer. This replaces ~6 serial calls."

  Grouped is per-*MoE-linear*, not per-layer. With `w1w3` fused there are 2 grouped launches per layer; unfused there are 3. The "~6 serial calls" figure also conflates active-experts-per-linear with total launches: `~6 active experts × 3 linears = ~18 serial launches per layer`, replaced by 2-3 grouped launches per layer. INTENT's "~58 launches/token instead of 464" math (line 56) makes this clear; Gemini's compression of the numbers obscures it.

- **§3.1 step 2 `packed_ld = K * 32`** asserted unconditionally as "derived from `Packing_v2`". True for the row-major path before operand-tag swap; once `get_operand_tag(conv_w->pack) != OPERAND_A` triggers `swap(rows, cols); order = ~order` in api.cc, the helper has to source `b_packed_ld` from the post-swap state. The right answer happens to still be `K*32` for the FP8/MXFP4 sm70 configs in flight, but the formulation hides where the constant comes from and would break silently if the converter resolution moves.

- **P1 over-packed:** "Implement helper + wire into mul_mat_id + correctness + perf gate, 4-5 days." Codex correctly splits this into P0 (baseline + instrumentation), P1 (grouped helper + ABI plumbing), P2 (mul_mat_id integration), P3 (correctness). Folding all four into one phase removes the rollback points — if perf misses, you have nothing to bisect against because correctness and integration landed in the same phase.

- **Quality gate compares grouped TURBOMIND against CPU MoE (P2.2)** but INTENT says: "Compare grouped TURBOMIND against the legacy TURBOMIND per-expert path *first*. CPU-MoE remains a secondary reference for overall quality, not the first math gate." Codex gets this right; Gemini does not.

- **DoD item 5 ("FP16 activations maintained across the FFN boundary") is a ship-blocker** in Gemini's framing, but INTENT explicitly classifies F-02 as "Nice-to-have" (intent line 47) and a "small follow-on win" (line 33). Making FP16-boundary cleanup a hard gate is scope creep on a phase the intent flagged as optional. Codex correctly marks its P4 as "secondary, only if P1-P3 pass" with a stop-loss.

- **No instrumentation phase.** Debug counters are mentioned only in P4 (final measurement), so during P1's 4-5 day build there is no visibility into whether the grouped path is being taken or how many experts are active per token. Codex's P0 step 2 fills this gap.

- **No legacy/grouped switch.** If grouped goes in cleanly and then a quality bug appears in P2, Gemini has no fast path back to per-expert dispatch for bisection. Codex's "temporary debug switch" addresses this.

### Gaps in risk analysis

Three risks total, vs. Codex's five. Specifically missing:

- **Hidden ABI contract** (the `void* const*` → `StridedPtr*` reinterpretation). This is the most likely correctness bug and Gemini does not mention it at all.
- **Synchronous `cudaMemcpy` at api.cc:607** as a per-launch sync point. Risk #1 ("`mul_mat_grouped` has hidden overhead") is rated *Low* with mitigation "P0 microbench already showed launch-bound nature; amortization is mathematically sound" — but the sync memcpy is exactly the kind of hidden overhead that survives a launch-count-only analysis. Should be Medium with explicit profiling in P0.
- **`num_experts > 1` not exercised on sm70 in SPRINT-023.** Same gap as Codex but Gemini is more vulnerable because P1 wraps everything into one phase.
- **Per-launch pointer-table upload cost** vs pre-baked at load time.
- **Real-model availability / fit on 32 GiB.** INTENT flags this; Gemini lists `DSv4-Flash-AVG-16e` under Dependencies but does not surface "what if it doesn't fit / doesn't exist locally" as a risk.

### Missing edge cases

- Empty expert (zero tokens routed). Not in any fixture.
- Fused vs unfused `w1w3`. Not addressed in dispatch logic.
- Stream-ordering interaction with the existing sync memcpy. Not addressed.
- `num_experts = 1` degenerate case (single active expert) — should the helper still call grouped, or fall back? Not specified.

### Definition of Done completeness

6 items. Specific problems:

- Item 2 perf gate covers only MIN-16e (≥24 t/s); INTENT and Codex include MIN-32e (≥23.5 t/s). The bottleneck is launch overhead, which is largely model-size-independent, so MIN-32e ought to clear the bar too — but verifying it is the point.
- Item 5 (FP16 boundary as ship-blocker) — scope creep, as above.
- No DoD item demanding a follow-up doc if perf misses (Codex item 7 covers this).
- No DoD item for instrumentation/visibility of the grouped path being exercised. Without that, post-merge regressions are silent.
- DoD has no abort/decision branch — implies the only allowed outcome is "ship", which makes it a poor gate for a sprint whose INTENT says "either ship, profile-and-extend, or stop and explain."

### Specific technical errors

1. **§3.1 step 2 — StridedPtr layout combining weight+scales+stride in one struct.** Wrong. Two separate `StridedPtr` arrays needed; each entry is `{ void* ptr; int stride; }`. Implementing as described would silently corrupt scales reads.
2. **§3.1 step 3 — "one grouped call per layer ... replaces ~6 serial calls."** Off by a factor of 2-3× on launches-per-layer. Numerics still work out to a meaningful win, but the launch budget for P5's profile claim is wrong as written.
3. **§3.1 step 2 — `packed_ld = K * 32` asserted unconditionally.** Right answer, wrong derivation. Should be sourced from converter resolution + operand-tag swap, not stated as a constant.
4. **§5 Files Summary entry on `api.cc`: "Ensure `ggml_turbomind_mul_mat_grouped` uses `packed_ld` correctly."** Misdirected. `api.cc` already uses `packed_ld` correctly *inside* `Bdesc.ld` for the single-expert path; what's missing is that the grouped path sets `Bdesc.ld = 0` and the *caller-side* buffer pointed to by `weights_packed` must be a `StridedPtr` array whose `.stride` field carries `packed_ld`. The work is in `ggml-cuda-turbomind.cu`, not in `api.cc`. Edits to `api.cc` should be limited to (a) documenting the hidden contract in the header, (b) optionally removing the sync memcpy at line 607.

---

## Comparative summary

| Dimension | Codex | Gemini |
|---|---|---|
| Phase granularity | P0-P6, separate baseline & instrumentation | P0-P4, P1 over-packed |
| Decision-completeness | Explicit ship / extend / stop bands | Implicit ship-only |
| StridedPtr layout correctness | Correct | **Wrong (combined struct)** |
| Launch-count math | Per-MoE-linear, fused/unfused split | **Per-layer, off by 2-3×** |
| Hidden ABI contract risk | Called out as Risk #2 | Missing |
| Sync `cudaMemcpy` at api.cc:607 | Not noticed | Not noticed |
| Quality gate ordering (legacy TURBOMIND first vs CPU first) | Correct per INTENT | **Reversed** |
| FP16 boundary status | Secondary with stop-loss | **Ship-blocker (scope creep)** |
| Empty-expert fixture | Yes | No |
| Risk count / depth | 5, deeper | 3, thinner |
| Day estimates per phase | No | Yes |
| Effort estimation present | No | Yes (1-5 days/phase) |
| `num > 1` sm70 registry verification | Not addressed | Not addressed |
| Legacy/grouped debug switch for bisection | Yes | No |

**Recommendation for merge notes:** Use Codex's phase structure and risk framing as the spine. Borrow Gemini's day estimates, per-phase use-case table, and quantitative quality gate (≥75% token match on 5 prompts). Fix the three gaps both drafts share: the sync `cudaMemcpy` at api.cc:607, `num > 1` registry verification in P0, and pre-baked vs per-launch decision for pointer-table uploads. Drop Gemini's combined-StridedPtr layout and launch-count framing entirely.
