# SPRINT-022 — DSv4-Flash operational on V100 (baseline captured)

Date: 2026-05-15
Status: **Tier 0 complete.** Build works; the 256-expert FP4/FP8 native
GGUF loads and serves; baseline TPS captured. Ready to triage and
incrementally introduce the deferred core-path optimizations.

---

## 1. What we did

1. Created `sprint-022-dsv4-integration` branch from upstream
   `nisparks/experiment/deepseek-v4-dynamic-graph` (the production
   integration branch with `Bring up native FP4 FP8 quant support`).
2. Imported sprint-016+ `tools/tc-grid/` + `docs/sprints/` lab harness
   (71 SAFE commits collapsed to one import commit `9f902e4f7`).
3. Wrote triage doc `SPRINT-022-DEFERRED-PORTS.md` for the 4 core-path
   commits from sprint-016+. Two SKIP (TP-only + model-pruning), two
   PORT-AFTER-BASELINE (WMMA-MMVQ MoE kernels).
4. Built `llama-cli` + `llama-server` + `llama-bench` for V100 sm_70
   in a fresh `llamacpp-build` pod on gpu-01 (CUDA 12.2, cmake 3.22.1,
   gcc 11.4). Build flags:
   ```
   cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON
     -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_F16=ON
     -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
     -DCMAKE_BUILD_TYPE=Release
   ```
5. Smoke-tested the FP4/FP8 native quant load with the small
   `DSv4-Flash-MIN-8e-fixed.gguf` (12.5GB) — model loads, GPU memory
   layout works, generation runs (output is gibberish as expected for
   MIN-* perf-test variants).
6. Loaded the full `DSv4-Flash-256e-fixed.gguf` (145GB, 284.33B params,
   256 experts) with `-ngl 99 -ot "exps=CPU"`. Dense on V100, MoE on
   host RAM.
7. Ran `llama-bench -p 128 -n 32` to capture baseline TPS.

---

## 2. Baseline measurements

V100-SXM2-32GB + 251GB host RAM, gpu-01, fresh build off
`sprint-022-dsv4-integration`:

| Model | Quant | Test | TPS |
|---|---|---|---:|
| DSv4-Flash-256e (284.33B) | F8_E4M3 + MXFP4 | **pp128** (prefill) | **4.28 ± 0.00** |
| DSv4-Flash-256e (284.33B) | F8_E4M3 + MXFP4 | **tg32** (decode) | **4.73 ± 0.00** |

Memory layout: 7.5 GiB on V100 (dense layers, attention) + 147.9 GiB
on host (MoE expert weights).

Raw log: `SPRINT-022-baseline-V100-cpu-moe.log`

Why pp ≤ tg (unusual for MoE): with `exps=CPU` every token's expert
forward pass is bottlenecked by host RAM bandwidth + CPU compute.
Prefill = 128 forward passes of the full MoE; decode = 32 forward
passes. Per-token cost is roughly identical because GPU dense is fast
relative to CPU expert compute. The ratio TG/PP ≈ 1 confirms we're
CPU-expert-bound, not GPU-bound.

---

## 3. What's next

### Tier 1 — Port the WMMA-MMVQ MoE kernels (sprint-017 P2+P3)

Two commits deferred:
- `61f9ebebe` — new files `wmma-mmvq.{cu,cuh}` (won't conflict) + 244-
  line dispatcher hook in `mmvq.cu`
- `5ec3a9b41` — opt-in flag for M=1 decode (prefill uses WMMA, decode
  uses scalar)

Manual three-way merge of `mmvq.cu` required against nisparks's +49/-14
FP4/FP8 dispatch changes.

Expected upside: faster prefill at the M values where WMMA tensor cores
beat scalar dispatch. Won't help current bottleneck (CPU expert
compute) but raises the dense layer ceiling when more layers fit on GPU.

### Tier 2 — Move MoE experts to GPU (turbomind Config_E4M3 / MXF4)

The 59 TF FP8 sm70 ceiling from SPRINT-021 P0 (REPORT-15) applies
DIRECTLY to MXFP4-weighted MoE experts. If we can fit even a few hot
experts on GPU and call turbomind's `Config_MXF4` for them, we get
~15× speedup vs CPU compute for those experts.

Required work:
- Add a CUDA dispatch path for `GGML_TYPE_F8_E4M3_B128` and
  `GGML_TYPE_MXFP4` that calls turbomind's `Gemm::Run`
- Hot-expert selection logic to decide which experts move to GPU
- VRAM budget: 32 GiB - 8 GiB dense - 1 GiB context = ~23 GiB for hot
  experts. Each MXFP4 expert at ~8.4B/256 = 33M params × 0.5 byte/wt
  = ~17 MB. We can fit ~1300 experts × layers — actually limited by
  layer count (60-ish layers × top-256 experts each), but we only
  need the HOT ones.

### Tier 3 — Multi-GPU TP

Out of scope until single-GPU is fully optimized. The deferred
`abde07824` split-mode bypass would matter here.

---

## 4. Operational readiness

This is the answer to "can we get operational with what we have":

| Question | Answer |
|---|---|
| Does the build work end-to-end on V100? | ✅ Yes |
| Does the FP4/FP8 GGUF load? | ✅ Yes (F8_E4M3_B128 + MXFP4 types both supported) |
| Does generation run? | ✅ Yes (numbers in §2) |
| Is it fast enough for production? | ❌ Not at 4.7 tok/s decode — bottleneck is CPU expert compute |
| What unlocks production speed? | Tier 2 (GPU expert kernels via turbomind FP8/MXF4) |

---

## 5. Files

- `docs/sprints/SPRINT-022-DEFERRED-PORTS.md` — triage of 4 deferred commits
- `docs/sprints/SPRINT-022-baseline-V100-cpu-moe.log` — raw bench output
- This file — operational summary

Git state:
- Branch: `sprint-022-dsv4-integration`
- Tip: this commit
- Upstream: `nisparks/experiment/deepseek-v4-dynamic-graph`

Build artifacts on pod `llamacpp-build` (namespace `llm`, node `gpu-01`):
- `/workspace/llamacpp/build/bin/llama-{cli,server,bench,completion}`
- Built with sm_70, GGML_CUDA=ON, GGML_CUDA_F16=ON
