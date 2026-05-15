# SPRINT-026 — Speculative Decoding for DSv4-Flash (V100 sm70)

**Status:** DRAFT 2026-05-15
**Predecessor:** SPRINT-025 (multi-GPU 256e landing)
**Successor:** SPRINT-027 (multi-slot / continuous batching)

---

## 1. Overview

SPRINT-026 lands speculative decoding for DSv4-Flash on the V100/TURBOMIND path. Speculative decoding multiplies effective decode throughput by drafting $K$ tokens via a cheap strategy (small model or n-gram cache) and verifying them in a single target-model batch pass. For memory-bandwidth-bound decode at $M=1$ (the "launch-bound" regime), this amortizes target model costs over multiple accepted tokens.

This sprint integrates the existing `llama.cpp` speculative plumbing with the `CUDA_TURBOMIND` dispatch path. We evaluate two strategies:
1. **N-gram lookup**: Draft-free, zero additional VRAM, effective for repetitive content (code, logs).
2. **Small Draft Model**: Uses `DSv4-Flash-AVG-16e` (18 GiB) as a draft for the `DSv4-Flash-256e` (156 GiB) target on 8x V100 hardware.

**Success Gate:** $\ge 1.3\times$ decode TPS uplift on the 256e target with $\ge 0.50$ median acceptance rate.

---

## 2. Use Cases

| Case | Benefit |
|---|---|
| **Chat / Instruction Follow** | Faster time-to-last-token; improved perceived latency for long responses. |
| **Code Completion** | N-gram cache excels at local repetition (variable names, syntax patterns), providing high acceptance with zero VRAM cost. |
| **Batch Inference** | Speculative decoding lifts the "effective $M$" per target pass, moving the operating point toward compute-bound ceilings. |

---

## 3. Architecture

### 3.1 Speculative Integration

We utilize the established `common_speculative` machinery in `common/speculative.{h,cpp}`. This infrastructure handles:
- **Drafting**: Token generation via `ngram` or a separate `llama_context` for a draft model.
- **Verification**: Single-pass batch verification on the target `llama_context`.
- **Correction**: Branching back to the last accepted token on mismatch.

The `llama-server` already plumbs `--model-draft` and `--speculative-type` into `server-context.cpp`. This sprint ensures these hooks correctly initialize the `CUDA_TURBOMIND` backends for both models.

### 3.2 Tokenizer Constraint

Speculative decoding requires the draft and target models to share an identical vocabulary (same token IDs). All `DSv4-Flash` variants share the `deepseek4` architecture and vocabulary. 
- **Valid pairs**: (16e draft, 256e target), (8e draft, 16e target), etc.
- **Invalid pairs**: (Qwen draft, DSv4 target).

### 3.3 VRAM Management (Two Models)

On the **8-GPU (gpu-01)** target:
- **Target (256e)**: ~156 GiB distributed across 8 GPUs (~19.5 GiB/GPU).
- **Draft (16e)**: ~18 GiB. 
- **Placement**: The draft model should be pinned to a single GPU (likely GPU 0) via `-od 'CUDA_TURBOMIND0'`. 
- **Total VRAM (GPU 0)**: 19.5 (shard) + 18 (draft) + KV/Scratch > 32 GiB.
- **Mitigation**: Offload some 256e layers from GPU 0 to GPUs 1-7 using `-ts` (tensor split) or rebalance with `-sm layer` weights to keep GPU 0 under the 32 GiB ceiling.

### 3.4 Sampling & Determinism

Under `temp=0` (greedy), speculative decoding is **mathematically exact**.
- **Same-Output Gate**: For any prompt, output tokens from `target-only` and `target+spec` must be identical.
- **Divergence = Bug**: Any token mismatch at `temp=0` indicates a flaw in the verification or KV-cache rewind logic.

---

## 4. Implementation

### P0 — Baseline & N-gram Smoke (2 days)

**Goal:** Establish single-GPU baselines and verify the draft-free path.

1. **P0.1 — Baseline TPS**: Measure `AVG-16e` (single GPU) decode TPS using `llama-bench`. This is the reference for single-GPU spec-decode lift.
2. **P0.2 — N-gram Functional**: Run `llama-server --model AVG-16e --speculative-type ngram_cache --lookup-cache-dynamic`.
3. **P0.3 — Correctness**: Verify `temp=0` output matches the baseline exactly for 10 chat/code prompts.
4. **P0.4 — N-gram Lift**: Measure TPS lift on the server via `predicted_per_second`. Expect 1.1-1.2x on repetitive code.

### P1 — Dual-Model Load Infrastructure (3 days)

**Goal:** Load two `CUDA_TURBOMIND` models in one process.

1. **P1.1 — Registry Verification**: Ensure `ggml-cuda-turbomind.cu` can handle multiple model instances (vetted in SPRINT-025 P2).
2. **P1.2 — Draft Model Load**: Test `llama-server -m AVG-16e -md AVG-16e -od 'CUDA_TURBOMIND'`. 
   - *Note: Self-speculation (draft=target) is only for loading verification; it will fit on 2x V100 but not 1x.*
3. **P1.3 — Resource Isolation**: Verify the draft model uses its own `llama_context` and doesn't stomp on target-model scratch/barriers.

### P2 — Single-GPU Acceptance Sweep (2 days)

**Goal:** Quantify the "Draft Quality" of small DSv4 models.

1. **P2.1 — Draft Setup**: Load a (small, non-real) `MIN-8e` draft and `AVG-16e` target.
2. **P2.2 — Acceptance Floor**: Measure acceptance rate on mixed chat. If median < 0.30, document that random weights are insufficient drafts (expected).
3. **P2.3 — Metrics**: Aggregated from server logs: `acceptance_rate`, `tokens_per_draft`, `ms_per_draft_pass`, `ms_per_verify_pass`.

### P3 — Multi-GPU 256e Target Integration (3 days)

**Goal:** Land the production speculative configuration.

1. **P3.1 — Rebalance for Draft**: Adjust `-sm layer` or `-ts` on the 8-GPU pod to carve out 18 GiB on GPU 0 for the `AVG-16e` draft model.
2. **P3.2 — Full Load**: 
   ```bash
   llama-server -m /models/256e.gguf -md /models/AVG-16e.gguf \
     -sm layer -ngl 999 -ot 'exps=CUDA_TURBOMIND' \
     -od 'CUDA_TURBOMIND0' --n-draft 4
   ```
3. **P3.3 — Functional Gate**: One completion request succeeds with `AVG-16e` drafting for `256e`.

### P4 — Performance Measurement & Tuning (2 days)

**Goal:** Reach the 1.3x TPS target.

1. **P4.1 — Acceptance Sweep**: Measure acceptance across `n-draft` ∈ {2, 4, 8, 16}. Identify the "knee" where verification overhead outweighs drafting gain.
2. **P4.2 — TPS Lift**: Calculate `(spec_tps / baseline_tps)` for the 256e model.
3. **P4.3 — Acceptance Floor**: Ensure median acceptance $\ge 0.50$ on high-value prompts.

### P5 — REPORT-20 Closeout (2 days)

**Goal:** Document findings and provide reproducible artifacts.

1. **P5.1 — REPORT-20**: Tables for:
   - Baseline vs N-gram vs Draft-Model TPS.
   - Acceptance rates per prompt category.
   - VRAM usage per GPU (showing the draft model on GPU 0).
2. **P5.2 — Reproducibility**: Log exact command lines used for the 8-GPU speedup.

---

## 5. Files Summary

### Modified

| Path | Change |
|---|---|
| `tools/server/server-context.cpp` | Enhanced logging for speculative stats (acceptance per slot). |
| `common/arg.cpp` | Ensure `-od` / `--override-tensor-draft` correctly identifies `CUDA_TURBOMIND` family. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Safety checks for concurrent model instances. |

### New

| Path | Purpose |
|---|---|
| `docs/sprints/drafts/SPRINT-026-GEMINI-DRAFT.md` | This document. |
| `tools/server/bench/spec_bench.py` | New script to drive server-side speculative benchmarking. |
| `docs/reports/REPORT-20.md` | Final performance report. |

---

## 6. Definition of Done

1. ✅ `llama-server` boots with both `target` and `draft` models on `CUDA_TURBOMIND` without crash.
2. ✅ **Same-Output Gate**: `temp=0` output is bit-identical between spec and non-spec runs.
3. ✅ **Performance Uplift**: Measured decode TPS for 256e + 16e-draft $\ge 1.3\times$ baseline.
4. ✅ **Acceptance Rate**: Median acceptance $\ge 0.50$ for mixed chat prompts.
5. ✅ **Memory Accounting**: Successful 8-GPU load documented with rebalance parameters.
6. ✅ **N-gram Comparison**: TPS uplift for `--speculative-type ngram_cache` quantified.
7. ✅ REPORT-20 written and committed.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | VRAM OOM on GPU 0 with Draft + Target | High | High | Use `-sm layer` weights or `-ts` to shift target layers off GPU 0. |
| 2 | Low acceptance rate (< 0.30) | Medium | High | Revert to N-gram or investigate draft model quality; confirm same-tokenizer. |
| 3 | Spec-decode overhead (verify pass) > draft win | Medium | Medium | Limit `n-draft` to 4; ensure `CUDA_TURBOMIND` verify pass is efficient. |
| 4 | SPRINT-025 multi-GPU hasn't landed | Medium | High | Iterate on N-gram / single-GPU first; delay P3-P4. |

---

## 8. Security

- **No New Surface**: Reuses existing `llama-server` endpoints.
- **Model Isolation**: Draft and target models run in separate contexts; no data leakage across slots unless shared by the server.

---

## 9. Dependencies

1. **SPRINT-023/024/025**: Requires the `CUDA_TURBOMIND` path and multi-GPU loading.
2. **Models**: `DSv4-Flash-256e` (target) and `DSv4-Flash-AVG-16e` (draft).
3. **Hardware**: 8x V100 pod (`gpu-01`).

---

## 10. Open Questions

1. **Optimal `n-draft`**: Is 4 the sweet spot for DSv4, or should we push to 8+?
2. **Draft Re-balancing**: Should we automate the shifting of layers off GPU 0 when a draft is detected, or keep it as a manual `-ts` requirement?
3. **N-gram vs Draft**: Which is better for the average user? N-gram is "free," while Draft takes 18 GiB.
