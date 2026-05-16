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

- `src/llama-memory-deepseek4.cpp`
  - Fixes DeepSeek4 slot/KV full-sequence removal for nonzero server slots.
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

As of the latest audit, health was OK. The patched restart used:

```bash
export LD_LIBRARY_PATH=/workspace/llamacpp/build/bin:/workspace/llamacpp/ggml/vendor/turbomind/build_so:${LD_LIBRARY_PATH:-}
```

This matters: without the TurboMind build output on `LD_LIBRARY_PATH`, the server can log `dlopen(libggml-turbomind.so) failed` and fall back during tensor upload.

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

## Slot/KV Follow-Up

The DeepSeek4 server slot/KV reuse bug has also been fixed.

Root cause: `llama_memory_deepseek4::seq_rm()` only accepted `seq_id == 0` or `seq_id < 0`. Server slot ids are sequence ids, so slots 1, 2, and 3 could not be fully cleared. The server then submitted the next prompt for that slot at position 0 while DeepSeek4 memory still reported the previous final position, producing `Invalid input batch` with an `inconsistent sequence positions` diagnostic.

Patch: `src/llama-memory-deepseek4.cpp::seq_rm()` now normalizes `[p0,p1)`, supports empty/no-op removals, supports full-sequence removal for any valid nonnegative `seq_id`, and still rejects true partial removals.

Verification after rebuilding `llama-server`:

- Explicit `id_slot:3`, same prompt twice with `cache_prompt:false`: `HTTP 200` twice.
- Explicit slots `0,1,2,3`, two passes each: all eight requests returned `HTTP 200`.
- Default slot selection, four repeated requests: slots `0,1,2,3` returned `HTTP 200`.
- Log scan: no `Invalid input batch`, no `inconsistent sequence positions`, no `failed to initialize batch`, no `failed to truncate tokens`.

## Suggested Next Steps

1. Review the local diff for scope.
2. Keep the DeepSeek4 `seq_rm()` fix.
3. Keep the MXFP4 deinterleave and host reference fixes.
4. Keep `test_grouped_compare.cpp` or fold equivalent DSv4-shape coverage into an existing TurboMind test.
5. Rebuild locally/pod-side after any cleanup:

```bash
cd /workspace/llamacpp/ggml/vendor/turbomind/build_so
cmake --build . -j 8
cd /workspace/llamacpp
cmake --build build --target llama-server -j 8
```

6. Re-run:

```bash
./test_ggml_turbomind_correctness ./libggml-turbomind.so
./test_ggml_turbomind_grouped_compare ./libggml-turbomind.so
```

7. If testing the full server, include the TurboMind build output in `LD_LIBRARY_PATH`, wait for `/health` to return `{"status":"ok"}`, and run repeated-slot probes before trusting multi-request behavior.
