# SPRINT-025 — Deferred items

Items raised in the three drafts or cross-critiques but explicitly scoped OUT of SPRINT-025.

---

## 1. Speculative decoding / multi-slot decode (carry from SPRINT-024-DEFERRED #1)

**What:** Run multiple decode slots in parallel through a single dispatch, amortizing per-token launch cost across M > 1. Or, integrate speculative decoding to multiply effective decode TPS by acceptance rate.

**Why deferred:** SPRINT-025 stays single-slot. Multi-slot adds scheduler-level changes orthogonal to multi-GPU model fit. Better to land scale first, then perf optimization.

**Target sprint:** SPRINT-026 (if SPRINT-024 has shipped grouped MoE) or SPRINT-027 (if 024 still pending)

**Prerequisites:** SPRINT-025 baseline measured; ideally SPRINT-024 grouped path landed.

**Files:** Dispatch surface; graph builder for spec decode.

---

## 2. Option B per-device opaque-context turbomind API

**What:** Bump `ggml_turbomind_api_version`. New C ABI surface: `ggml_turbomind_init` returns an opaque handle; all calls take the handle. Replaces global state with caller-owned context.

**Why deferred:** Option A (per-device state inside the .so, indexed by cuda_device) is sufficient and avoids an ABI bump. Option B is the architecturally cleaner long-term answer but more invasive. Promote only if Option A exposes a library-level singleton P2 can't fix in-place.

**Target sprint:** SPRINT-026 if Option A hits a wall; never if Option A is sufficient.

**Files:** `ggml/vendor/turbomind/include/ggml-turbomind-api.h`, `api.cc`, `ggml-cuda-turbomind.{cu,cuh}`.

---

## 3. `LLAMA_SPLIT_MODE_ROW` for deepseek4 (P6 deferred outcome)

**What:** Lift the `LLAMA_SPLIT_MODE_ROW not implemented for architecture 'deepseek4'` guard in `src/llama-model.cpp:770-771`; marry row-split semantics with the CUDA_TURBOMIND buffer type for expert tensors.

**Why deferred:** P6 conditional in SPRINT-025 with a 1-day kill criterion. Almost certainly punts because:
- Row-split for MoE expert weights needs split-buffer-type semantics that CUDA_TURBOMIND doesn't natively support.
- Communication overhead per layer (`ncclAllReduce`) on PCIe-without-full-NVLink probably outweighs benefit on V100.

**Target sprint:** SPRINT-026 if P6 punts and LAYER underperforms; otherwise indefinitely.

**Files:** `src/llama-model.cpp` (guard), `ggml/src/ggml-cuda/ggml-cuda-turbomind.{cu,cuh}` (split-buffer cousin), `ggml/src/ggml-cuda/ggml-cuda.cu` (split-buffer registry).

---

## 4. F8_E4M3_B128 dense layers on multi-GPU

**What:** Extend the family-alias buft to handle dense linears (`attn_q.weight`, `attn_kv_*.weight`, `attn_o.weight`). Same `-ot 'attn_(q|kv_a|kv_b|o).weight=CUDA_TURBOMIND'` pattern.

**Why deferred:** P3 family-alias mechanism naturally supports it, but verifying + benchmarking on multi-GPU adds scope. Better to ship 256e first, dense FP8 as a follow-on.

**Target sprint:** SPRINT-026 stretch.

**Files:** No code changes — only `-ot` regex extension and a single-line bench.

---

## 5. Generated per-layer override mechanism (Gemini-implied alternative)

**What:** At model load, programmatically generate per-layer `-ot` patterns: `blk.0.ffn_*_exps.weight=CUDA_TURBOMIND0`, `blk.1.ffn_*_exps.weight=CUDA_TURBOMIND0`, etc.

**Why deferred:** User chose the family-alias approach in the interview because it preserves pipeline parallel (`model.has_tensor_overrides() == false`). Generated overrides would set the flag and disable pipeline parallel — measurable cost.

**Target sprint:** Never (rejected by design).

---

## 6. NCCL inter-node networking

**What:** Multi-node TP across separate physical hosts via NCCL TCP/IP bootstrap.

**Why deferred:** Single node (`gpu-01` with 8 V100) has enough VRAM. Inter-node networking adds NCCL config + k8s network policy + cross-node bandwidth concerns.

**Target sprint:** SPRINT-028+ if a model larger than 256 GiB ever lands.

**Files:** k8s network policy; NCCL_SOCKET_IFNAME; new pod manifests.

---

## 7. Auto-balancing layer-split distribution

**What:** Replace the default uniform `-sm layer` distribution with a model-aware policy that accounts for embedding-layer asymmetry (the first GPU gets the token embeddings; the last gets the LM head and the unembedding) and per-expert variance.

**Why deferred:** Manual `-ts` rebalance is the SPRINT-025 fallback (P4 stop-the-line); auto is a SPRINT-026+ ergonomics item.

**Target sprint:** SPRINT-026+ stretch.

**Files:** `src/llama-model-loader.cpp` placement policy.

---

## 8. ggml CUDA pool for turbomind scratch

**What:** `ggml_turbomind_init` currently `cudaMalloc`s 256 MiB of barriers/partials/flags per device (= 2 GiB across 8 GPUs). Should use the ggml CUDA pool for shared allocator efficiency.

**Why deferred:** Carry-over from SPRINT-023 F-05 (`Use ggml CUDA pool`). Same buffer, just bigger when multiplied by 8 devices. Not blocking SPRINT-025; only matters if VRAM gets tight.

**Target sprint:** SPRINT-026+ when convenient.

**Files:** `ggml/vendor/turbomind/api.cc` `ggml_turbomind_init`.

---

## 9. Perplexity sweep on 256e

**What:** Full `perplexity` run on WikiText-2 or HellaSwag on the 8-GPU 256e config to verify quality vs a known reference.

**Why deferred:** SPRINT-025 verifies output coherence via greedy 32-token decode on fixed prompts. Perplexity is a more rigorous check but compute-expensive and not blocking.

**Target sprint:** Ad-hoc / SPRINT-026 if a quality regression is suspected.

---

## 10. CUDA graph capture for the multi-GPU path

**What:** Verify and fix `GGML_CUDA_USE_GRAPHS` compatibility with the multi-GPU + CUDA_TURBOMIND configuration.

**Why deferred:** SPRINT-025 P5.5 logs whether graphs engage; if they don't, SPRINT-026 picks up the diagnosis. Also gated on SPRINT-024 P1.4 (sync `cudaMemcpy` removal).

**Target sprint:** SPRINT-026.

**Files:** `ggml/src/ggml-cuda/ggml-cuda.cu` graph capture path.

---

## 11. Cross-GPU activation tensor handling for non-FFN layers

**What:** Layer-split moves activations between GPUs at layer boundaries. The default ggml-cuda scheduler handles this, but the FFN-routing-then-CUDA_TURBOMIND-dispatch path is bespoke; verify activations don't get stuck on wrong device during MoE routing.

**Why deferred:** Implicitly handled by ggml-cuda's existing scheduler with layer-split + pipeline-parallel. Only becomes its own work if SPRINT-025 P3.6 decode crashes with "wrong device" errors.

**Target sprint:** Conditional — only if SPRINT-025 P3/P4 surfaces an issue.

---

## 12. Expert parallelism (256 experts sharded across GPUs)

**What:** Instead of (or in addition to) layer-split, shard experts WITHIN a layer across GPUs. Each GPU owns 32 of the 256 experts; top-k routing scatters tokens across GPUs.

**Why deferred:** Different architecture from tensor parallelism. Would need a custom MoE dispatch that's aware of expert-to-GPU mapping and gathers results across GPUs. Layer-split is simpler and fits comfortably on 8× V100 32GB; expert-parallel only becomes interesting if a model has more experts than fit on one GPU per layer (DSv4-Flash-256e has 256 experts/layer × 0.6 GiB each ≈ 150 GiB — comfortably fits on one GPU if the rest of the layer is small enough; not the case here).

**Target sprint:** SPRINT-028+ if a real expert-parallel-needing model surfaces.

---

## Summary table

| # | Item | Target sprint | Blocker |
|---|---|---|---|
| 1 | Multi-slot decode / speculative | SPRINT-026/027 | SPRINT-025 baseline |
| 2 | Option B opaque-context API | SPRINT-026 conditional | Option A hits a wall |
| 3 | LLAMA_SPLIT_MODE_ROW for deepseek4 | SPRINT-026 conditional | P6 punts + LAYER underperforms |
| 4 | FP8 dense layers on multi-GPU | SPRINT-026 stretch | Family-alias works for experts first |
| 5 | Generated per-layer overrides | Never | Rejected per interview |
| 6 | NCCL inter-node | SPRINT-028+ | Model >256 GiB |
| 7 | Auto-balancing layer split | SPRINT-026+ | Ergonomics |
| 8 | ggml CUDA pool for TM scratch | SPRINT-026+ | Convenience |
| 9 | Perplexity sweep | Ad-hoc | Quality regression suspected |
| 10 | CUDA graph capture compat | SPRINT-026 | SPRINT-024 P1.4 |
| 11 | Cross-GPU activation hand-off bugs | Conditional | P3/P4 surfaces issue |
| 12 | Expert parallelism | SPRINT-028+ | Larger model |
