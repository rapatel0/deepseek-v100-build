# SPRINT-022 deferred ports — 4 core-path commits from sprint-016-tensor-unlock

Status: All 4 deferred until baseline is operational. Each evaluated below.

## 1. `abde07824` — tools/server/server-context.cpp split-mode tensor seq_rm bypass

**Original**: SPRINT-016 P0 fast-bypass for ggml-alloc abort during seq_rm probe under `--split-mode row` (tensor-parallel TP=8). Single V100 use-case (single-device) does not hit this bug.

**Relevance to DSv4-Flash on single V100**: **NONE**. We're not running TP=8.

**Verdict**: SKIP. Resurrect only if we ever go multi-GPU TP.

---

## 2. `2d4ad4367` — convert_hf_to_gguf.py --n-experts pruning expert-averaging

**Original**: SPRINT-016 loop-A. Adds `--n-experts N` flag that prunes a 256-expert DSv4 to N experts by averaging the dropped experts into the kept ones. Used to create the "MIN-16e" model variants for performance testing on a single 32GB V100.

**Relevance**: The real DSv4-Flash-FP4-FP8 GGUF from HuggingFace is the production target. We don't need pruning. The MIN-16e variants were stand-ins.

**Verdict**: SKIP. The patch is preserved in the original sprint-016-tensor-unlock branch if ever needed for fresh measurement-only model variants.

---

## 3. `61f9ebebe` — ggml-cuda WMMA-MMVQ MoE kernels (+ dispatcher hook)

**Original**: SPRINT-017 P2. NEW files `ggml/src/ggml-cuda/wmma-mmvq.{cu,cuh}` containing FP4/FP8-unpack-in-registers → FP16 WMMA → FP32 accumulate kernels for V100 sm70. Modifies `mmvq.cu` (244 insertions) to dispatch routed-MoE through the WMMA path.

**Relevance**: **HIGH** — same precision regime as nisparks's FP4/FP8 path. This is the exact "tensor-core acceleration for the FP4/FP8 GGUF mul_mat" lever we want.

**Conflict with nisparks**: `wmma-mmvq.{cu,cuh}` are NEW files (no conflict). The `mmvq.cu` modifications DO conflict — nisparks made +49/-14 line changes in `mmvq.cu` for FP4/FP8 dispatcher support. Our change made +244/-50 lines in the SAME dispatch logic. **Manual three-way merge required.**

**Verdict**: PORT FORWARD AFTER BASELINE. 
1. Get nisparks baseline TPS first.
2. Then introduce `wmma-mmvq.{cu,cuh}` standalone (no conflict).
3. Then manually port the dispatcher hook in `mmvq.cu` on top of nisparks's existing FP4/FP8 dispatch.
4. Measure delta. Keep only if it beats nisparks stock at the relevant M values.

---

## 4. `5ec3a9b41` — flip WMMA-MoE dispatch to opt-in (M=1 decode finding)

**Original**: SPRINT-017 P3. After P2, M=1 decode regressed with WMMA path active. Patch makes the WMMA-MoE dispatch opt-in via an env var (default OFF). Lets prefill (large M) use WMMA while decode (M=1) uses the stock path.

**Relevance**: Pairs with #3. Decode-vs-prefill dispatch tuning. Once #3 is ported, this is the next layer.

**Verdict**: PORT AFTER #3. Will need re-tuning against nisparks's FP4/FP8 dispatch path; the M=1 threshold may shift on the new code.

---

## Summary

| # | Commit | Action | When |
|---|---|---|---|
| 1 | abde07824 split-mode bypass | SKIP | Multi-GPU TP only |
| 2 | 2d4ad4367 n-experts pruning | SKIP | Production model is real, no pruning |
| 3 | 61f9ebebe WMMA-MMVQ MoE | PORT after baseline | High-value, manual 3-way merge in mmvq.cu |
| 4 | 5ec3a9b41 WMMA opt-in flag | PORT after #3 | Decode-vs-prefill dispatch |

Next: pull DSv4-Flash-FP4-FP8 GGUF, build llama-server, measure baseline TPS. Then revisit #3 and #4.
