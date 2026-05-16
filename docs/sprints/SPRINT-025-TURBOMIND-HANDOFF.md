---
title: SPRINT-025 TurboMind Handoff
date: 2026-05-16
status: fixed, pending cleanup/commit
---

# TurboMind DSv4 Handoff

## Executive Summary

The DeepSeek v4flash / DSv4-Flash-256e TurboMind failure on 8x V100 was fixed.

Root cause: the GGML MXFP4 block layout was unpacked incorrectly before TurboMind packing. The old deinterleave path treated each MXFP4 byte as two adjacent K values. GGML actually stores the low nibble for `k = j` and the high nibble for `k = j + 16` within each 32-value block.

This permuted every MXFP4 expert weight block in the TurboMind pack path, so the all-layer model produced token loops like `# # #`. It was not a MoE routing/alignment bug, not grouped scatter, and not FP16 output saturation.

## Current State

Repo: `/Users/ravi/repos/deepseek`

Relevant local changes:

- `ggml/vendor/turbomind/ggml-turbomind-deinterleave.cu`
  - Fixes MXFP4 deinterleave mapping.
- `ggml/vendor/turbomind/test_correctness.cpp`
  - Fixes the host MXFP4 reference so the test catches this class of bug.
- `ggml/vendor/turbomind/CMakeLists.txt`
  - Adds `test_ggml_turbomind_grouped_compare`.
- `ggml/vendor/turbomind/test_grouped_compare.cpp`
  - New DSv4-shape grouped-vs-single compare harness.
- `docs/sprints/SPRINT-025-PATCH-REPORT.md`
  - Updated with root cause, patch, and verification.
- `docs/sprints/SPRINT-025-TURBOMIND-HANDOFF.md`
  - This handoff.

There are many unrelated untracked files in the worktree. Do not clean or revert them unless the user explicitly asks.

## Exact Bug

GGML reference dequantization for `block_mxfp4` maps:

```c
low nibble  -> k = j
high nibble -> k = j + 16
```

The previous TurboMind deinterleave mapped:

```c
low nibble  -> k = 2*j
high nibble -> k = 2*j + 1
```

Corrected code in `ggml/vendor/turbomind/ggml-turbomind-deinterleave.cu`:

```cpp
const int k0 = bk * QK_MXFP4 + t;
weight_u16_out[k0                  * N + row] = lo;
weight_u16_out[(k0 + QK_MXFP4 / 2) * N + row] = hi;
```

Corrected host test reference in `ggml/vendor/turbomind/test_correctness.cpp`:

```cpp
int k0 = b * 32 + j;
int k1 = k0 + 16;
acc += __half2float(A[m * K + k0]) * fp4_e2m1_to_f32(lo) * scale;
acc += __half2float(A[m * K + k1]) * fp4_e2m1_to_f32(hi) * scale;
```

## Verification Already Run

Inside pod `llamacpp-build-8gpu`, namespace `llm`:

```bash
cd /workspace/llamacpp/ggml/vendor/turbomind/build_so
export LD_LIBRARY_PATH=$PWD:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
./test_ggml_turbomind_correctness ./libggml-turbomind.so
./test_ggml_turbomind_grouped_compare ./libggml-turbomind.so
```

Fresh results:

- `test_ggml_turbomind_correctness`: PASS
  - MXFP4: `max_abs=1.3622e-02`, `p99=7.2098e-03`, `rel=1.8053e-04`
- `test_ggml_turbomind_grouped_compare`: PASS
  - `down_decode`: `N=4096 K=2048 active=6 tpa=1`, PASS
  - `gate_up_decode`: `N=2048 K=4096 active=6 tpa=1`, PASS
  - `down_prompt`: `N=4096 K=2048 active=6 tpa=4`, PASS
  - `gate_up_prompt`: `N=2048 K=4096 active=6 tpa=4`, PASS

Full model verification:

- Full 43-layer `CUDA_TURBOMIND0..7` DSv4-Flash-256e on 8x V100 now decodes coherent Fibonacci text.
- 96-token probe measured `13.09 tok/s`.
- Previous symptom was `#` token loops; that no longer appears.

Saved successful output in pod:

```bash
/tmp/all43-fixed-nocache-long.json
```

## Current Running Server

Pod:

```bash
kubectl exec -n llm llamacpp-build-8gpu -- bash
```

Server health:

```bash
curl -fsS http://127.0.0.1:12500/health
```

Current process was started with:

```bash
./llama-server \
  -m /models/DSv4-Flash-256e-fixed.gguf \
  -ngl 99 \
  -sm layer \
  -ot 'blk\.([0-5])\..*exps.*=CUDA_TURBOMIND0,blk\.([6-9]|10)\..*exps.*=CUDA_TURBOMIND1,blk\.(1[1-6])\..*exps.*=CUDA_TURBOMIND2,blk\.(1[7-9]|2[0-2])\..*exps.*=CUDA_TURBOMIND3,blk\.(2[3-7])\..*exps.*=CUDA_TURBOMIND4,blk\.(2[8-9]|3[0-2])\..*exps.*=CUDA_TURBOMIND5,blk\.(3[3-8])\..*exps.*=CUDA_TURBOMIND6,blk\.(39|4[0-2])\..*exps.*=CUDA_TURBOMIND7' \
  -t 8 \
  --port 12500 \
  --no-warmup \
  --cache-ram 0 \
  --slot-prompt-similarity 0.0 \
  -c 4096
```

As of the final audit, health was OK and all four slots were idle.

## Do Not Re-Chase

The following were tested and are not the root cause:

- MoE grouped alignment or scatter.
- Empty expert handling.
- Multi-device TurboMind state or workspace reuse.
- CUDA graph capture.
- Pool allocator reuse.
- FP16 final `D` output saturation.
- Gate/up orientation only.
- `ffn_down_exps.weight` only.

The decisive clue was that both grouped and per-expert paths were wrong because both consumed the same incorrectly packed MXFP4 weights.

## Remaining Separate Issue

There is still a DeepSeek4 server slot/KV reuse bug:

- Repeated identical `/completion` requests can return `Invalid input batch`.
- Log mentions stale sequence positions and DeepSeek4 runtime state export not implemented.
- This is independent of TurboMind packing. The first full all-layer TurboMind request after server start is correct.

Current server uses `--cache-ram 0 --slot-prompt-similarity 0.0` to reduce prompt-cache reuse, but this does not fully fix the underlying DeepSeek4 slot/recurrent-state issue. If another agent investigates it, treat it as a separate server/memory lifecycle bug.

## Suggested Next Steps

1. Review the local diff for scope.
2. Keep the MXFP4 deinterleave and host reference fixes.
3. Keep `test_grouped_compare.cpp` or fold equivalent DSv4-shape coverage into an existing TurboMind test.
4. Rebuild locally/pod-side after any cleanup:

```bash
cd /workspace/llamacpp/ggml/vendor/turbomind/build_so
cmake --build . -j 8
```

5. Re-run:

```bash
./test_ggml_turbomind_correctness ./libggml-turbomind.so
./test_ggml_turbomind_grouped_compare ./libggml-turbomind.so
```

6. If testing the full server, restart it before each deterministic prompt unless the slot/KV bug has been fixed.
