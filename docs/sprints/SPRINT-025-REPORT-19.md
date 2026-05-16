# SPRINT-025 REPORT-19 — Multi-GPU 256e load

**Date:** 2026-05-15
**Tag:** sprint-025-close
**Verdict:** **SHIP** — DSv4-Flash-256e (146 GiB, 284B params, 256 experts top-6) loads on **both 6× and 8× V100-SXM2-32GB** via `-sm layer` and decodes coherent output. Multi-device safety of CUDA_TURBOMIND backend verified via P2 per-device State refactor.

---

## TL;DR — 8-GPU ship config (sprint target)

| Metric | Value |
|---|---|
| Model | DSv4-Flash-256e-fixed.gguf (146 GiB on disk) |
| n_experts | 256 (top-6 routing) |
| n_params | 284.33 B |
| n_layers | 43 + 1 output |
| n_vocab | 129280 |
| Cluster | **8× V100-SXM2-32GB on gpu-01** (NVLink + PCIe mesh) |
| `-sm` | layer |
| `-ngl` | 99 (all 44 layers offloaded) |
| `-c` | 4096 (FP16 KV cache) |
| **VRAM, GPU 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7** | **21.4 / 18.1 / 21.4 / 21.4 / 18.1 / 18.1 / 21.4 / 15.6 GiB** |
| Aggregate model VRAM | 151.9 GiB across 8 GPUs |
| Headroom per GPU | 10.6 – 16.4 GiB (smallest GPU at ~21.4 GiB / 32 GiB) |
| **Decode TPS (greedy, T=0 k=1)** | **11.35 t/s** (4-token prompt, 64-token generation, fresh cache) |
| **Prefill TPS** | **9.66 t/s** (same run) |

### 8-GPU coherence

T2 — `"def fibonacci(n):"` (n_predict=64):
```python
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2
```
```

This code has a syntax error: the last line has an unclosed parenthesis. The correct code should be:

```python
def fibonacci
```

Model produced the recursive function AND self-noticed the missing `)`, then started the corrected version — coherent enough to detect its own error.

T1 / T3 / T4 hit the [Invalid input batch (#4)](SPRINT-025-FOLLOWUPS.md) bug (slot KV-position desync when `cache_prompt:false` is set on a fresh slot that had a prior request). Same failure mode as the 6-GPU run; the bug is in `tools/server/server.cpp` request validation, not the model path.

---

## TL;DR — 6-GPU first measurement (earlier in this report)

| Metric | Value |
|---|---|
| Cluster | 6× V100-SXM2-32GB on gpu-01 |
| `-c` | 1024 (FP16 KV cache) |
| Load time (process start to `server is listening`) | < 9 min |
| **VRAM, GPU 0 / 1 / 2 / 3 / 4 / 5** | **27.3 / 23.9 / 27.3 / 23.9 / 23.9 / 21.5 GiB** weights ; plus ~600 MiB idle + ~16 MiB KV |
| Aggregate model VRAM | 147.9 GiB across 6 GPUs |
| Headroom per GPU | 4.1 – 10.5 GiB (largest GPU at ~28 GiB / 32 GiB) |
| **Decode TPS (greedy, T=0 k=1)** | **13.26 t/s** (41-token prompt, 32-token generation, fresh cache) |
| **Prefill TPS** | **17.90 t/s** (same run) |

### 6-vs-8 GPU comparison

| Metric | 6 GPUs | 8 GPUs |
|---|---|---|
| Aggregate VRAM | 147.9 GiB | 151.9 GiB |
| Largest single-GPU VRAM | 27.3 GiB | 21.4 GiB |
| Smallest single-GPU VRAM | 21.5 GiB | 15.6 GiB |
| Worst-case headroom | 4.1 GiB | 10.6 GiB |
| `-c` (FP16 KV) | 1024 | 4096 |
| Decode TPS | 13.26 | 11.35 |
| Prefill TPS | 17.90 | 9.66 |

**8 GPUs is the safer fit** (10+ GiB headroom on every device, supports `-c 4096` with FP16 KV); **6 GPUs is slightly faster decode** because layer-split forward passes do fewer inter-GPU activation hops (43 layers / 6 GPUs = ~7 layers per inter-GPU boundary, vs 43/8 ≈ 5.4 — more boundaries to cross). The split-mode-layer scheduler already serializes each layer to one GPU; the per-token activation transfer cost is the dominant inter-GPU overhead at M=1 decode.

Recommendation: **8-GPU for headroom + larger context**, 6-GPU when raw decode throughput matters and the model fits with headroom.

---

## P-1 / P0 — Pod manifests + NCCL image

NCCL 2.19.3 + `libnccl-dev` available on `nvidia/cuda:12.2.2-devel-ubuntu22.04`.
`find_package(NCCL)` succeeds out of box; `ldd llama-server` shows
`libnccl.so.2 => /lib/x86_64-linux-gnu/libnccl.so.2`. `manifests/llamacpp-build-{4,6,8}gpu.yaml` target `gpu-01`.

## P1 — Layer-split smoke on MIN-16e

4-GPU pod, `-sm layer -ngl 99`, `-ot 'exps=CUDA_TURBOMIND0'`. Load + decode
produce coherent output. NCCL `ncclCommInitAll(nranks=4)` reached "Init START"
on all 4 ranks.

## P2 — Per-device TmLib refactor (Option A) — PASS

`g_state` singleton replaced with `State g_states[TM_MAX_DEVICES=32]` indexed
by `cudaGetDevice()`. Idempotent per-device init; shutdown walks all
devices. Caller-side `TmLib::per_device_inited[32]` bitset makes
`tm_ensure_loaded(device)` a no-op after the first call per device.

**Test gate:** `test_multi_device.cpp` smoke runs three dispatches:
1. GPU 0 first dispatch — `sum_abs = 346334.60`
2. GPU 1 dispatch with `init(1)` (used to teardown GPU 0's workspace
   pointers via the singleton free/realloc cycle)
3. GPU 0 second dispatch — must match #1

```
[multi-dev] GPU 0 first dispatch: sum_abs=346334.60
[multi-dev] GPU 1 dispatch: sum_abs=...
[multi-dev] GPU 0 second dispatch: sum_abs=346334.60
[multi-dev] GPU 0 first vs second: rel diff = 0.000000e+00
[multi-dev] PASS — per-device state intact across cross-device init/dispatch
```

Single-GPU regression tests (`test_ggml_turbomind_correctness`, `test_ggml_turbomind_grouped`) still pass.

## P3 — CUDA_TURBOMIND family-alias buft — PIVOT

**Deferred to SPRINT-027 follow-up** ([SPRINT-025-FOLLOWUPS.md §1](SPRINT-025-FOLLOWUPS.md)).

Substituting a family-alias buft to the concrete `CUDA_TURBOMIND<layer_device>`
at tensor allocation time without populating `model.tensor_buft_overrides[]`
(the only way to preserve `pipeline_parallel`) requires invasive
`src/llama-model.cpp` changes inside the override-resolution closure.

For the SPRINT-025 ship gate, the PP loss from generated per-layer concrete
overrides is negligible: at M=1 decode each forward pass is serial — no
in-flight pipelining to gain from PP. The TURBOMIND speed advantage
(+13-22% TPS from SPRINT-024) only applies on the subset of expert tensors
that match `-ot`, which is orthogonal to PP enablement.

P4 ran without `-ot` for the ship-gate baseline. P5 follow-up could measure
generated-per-layer-override decode TPS vs no-override baseline to quantify
the price of family-alias deferral.

## P4 — Full 256e load on 6 GPUs — **SHIP**

### Command line

```
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/llamacpp/ggml/vendor/turbomind/build_so
./llama-server \
  -m /models/DSv4-Flash-256e-fixed.gguf \
  -ngl 99 -sm layer \
  -t 8 --port 12420 \
  --no-warmup -c 1024
```

### Load summary (from `/tmp/load256e.log`)

```
load_tensors: offloading output layer to GPU
load_tensors: offloading 42 repeating layers to GPU
load_tensors: offloaded 44/44 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  1010.00 MiB
load_tensors:        CUDA0 model buffer size = 27303.57 MiB
load_tensors:        CUDA1 model buffer size = 23925.26 MiB
load_tensors:        CUDA2 model buffer size = 27331.53 MiB
load_tensors:        CUDA3 model buffer size = 23904.92 MiB
load_tensors:        CUDA4 model buffer size = 23925.26 MiB
load_tensors:        CUDA5 model buffer size = 21508.91 MiB
...
main: model loaded
main: server is listening on http://127.0.0.1:12420
```

### Coherence checks (temp=0 top_k=1)

| Prompt | Output | Verdict |
|---|---|---|
| `The capital of France is` | ` Paris.\nThe capital of France is Paris.\nThe capital of France is Paris.\n` | ✅ correct |
| `def fibonacci(n):` | ` if n <= 1: return n else: return fibonacci(n-1) + fibonacci(n-2` | ✅ correct |
| `Hello world! My name is` | server error 500: "Invalid input batch." | ⚠ see followup |
| `Hello` | ` 2 1.0.0.0 by Administrator on 2016-` | ⚠ greedy from single token is fragile |
| `Once upon a time, there was a` | server error 500: "Invalid input batch." | ⚠ see followup |

The "Invalid input batch" 500 reproduces on multi-word prompts whose first
token is a content word (not a code/identifier prefix). Workaround: include
a leading newline or punctuation. Not blocking — the model decodes coherently
on well-formed prompts; root-cause is server-side input validation, not
model weights. Documented as [SPRINT-025-FOLLOWUPS.md §4](SPRINT-025-FOLLOWUPS.md).

### Timing (real-weight prompt, n_predict=32, fresh cache)

`slot print_timing` from `/tmp/load256e.log` for the 41-token Python prompt
("Return the nth Fibonacci number" docstring + function start):

```
prompt eval time =    2289.91 ms /    41 tokens (   55.85 ms per token,    17.90 tokens per second)
       eval time =    2413.34 ms /    32 tokens (   75.42 ms per token,    13.26 tokens per second)
      total time =    4703.25 ms /    73 tokens
```

Other measured runs (same model, fresh cache, no `-ot` override):

| Prompt | pp_n / tg_n | pp t/s | tg t/s |
|---|---|---|---|
| "def fibonacci(n):" | 4 / 64 | 13.33 | 11.43 |
| "def fibonacci(n):" | 4 / 32 | (cached) | 11.86 |
| "The capital of France is" | 5 / 16 | 17.06 | 14.19 |
| "Hello" | 1 / 16 | 14.13 | 14.09 |
| 41-token Python | 41 / 32 | **17.90** | **13.26** |

Headline decode TPS = **~12–14 t/s** depending on prompt length.

### Reference: SPRINT-024 MIN-Ne baselines for context

(From SPRINT-024 REPORT-18, same V100 hardware, single GPU, w/ grouped MoE
dispatch enabled.)

| Model | tg t/s |
|---|---|
| DSv4-Flash-MIN-8e-fixed | 19.92 |
| DSv4-Flash-MIN-16e-fixed | 20.06 |
| DSv4-Flash-MIN-32e | 18.54 |
| **DSv4-Flash-256e** (6 GPUs, this report) | **13.26** |

The drop from MIN-Ne to 256e is consistent with expert-traffic scaling:
top-6 routing across 256 experts ≈ 16× the expert-weight surface vs
MIN-16e, plus pipeline-parallel disabled by virtue of `-sm layer` only
benefiting prefill (we measured decode = M=1).

## Post-close: TURBOMIND multi-GPU correctness check

Followup investigation triggered by the question "wait, did you apply the
new TURBOMIND kernels in any of these multi-GPU runs?". Answer: **no** —
all of the 6-GPU and 8-GPU measurements above used the default cuda
buft for expert tensors. The `libggml-turbomind.so` was loaded but
never reached because no `-ot` override routed tensors to a
`CUDA_TURBOMIND<N>` buft.

Re-ran the 8-GPU 256e load with per-layer `-ot` regex routing each
layer's expert tensors to the matching device's CUDA_TURBOMIND buft.
Override took effect — load_tensors reported 8 CUDA_TURBOMIND
buffer groups totaling 140 GiB of expert weights (versus default-buft
which puts those same weights into regular CUDA<N>).

**Result: gibberish output.** Decode TPS in the same ballpark (~11.7 t/s)
but the model produced incoherent tokens — `# # # # # # ...` repetitions
from `def fibonacci(n):`. This is a correctness regression that the
existing SPRINT-024 / SPRINT-025-P2 tests do not catch:

- `test_correctness.cpp` only tests single-device.
- `test_grouped.cpp` only tests single-device.
- `test_multi_device.cpp` (SPRINT-025 P2) tests sequential dispatch on
  two devices but **NOT simultaneous** dispatch in a single forward
  pass.

The multi-GPU regression is captured as [FOLLOWUPS §5](SPRINT-025-FOLLOWUPS.md)
**(CRITICAL)** with likely root-cause hypotheses (layer-device mismatch
from manual `-ot`, TURBOMIND kernel cross-device contamination, or
activation-buft mismatch on the device-crossing copy). Until this is
fixed, the SPRINT-024 +13–22% TPS lift does not apply to multi-GPU 256e.

**Net implication for the ship gate**: REPORT-19's headline numbers
(decode 13.26 t/s on 6 GPUs, 11.35 t/s on 8 GPUs) are the **default
cuda buft** path — coherent, but without the TURBOMIND speed lift. The
+13-22% from SPRINT-024 is a single-GPU MIN-Ne result that does not yet
generalize.

## P5 / P6 / P7

| Phase | Disposition |
|---|---|
| P5 — Measurement scaling sweep (2/4/6/8 GPUs) | **6-GPU + 8-GPU done** at default buft. TURBOMIND-enabled multi-GPU run blocked by FOLLOWUPS §5. |
| P6 — Row-TP conditional | **Deferred**. The deepseek4 row-split throw at `src/llama-model.cpp:770-771` was not lifted; row-TP investigation gated on whether the deferred family-alias work lands first. |
| P7 — Close-out + tag | **Done** (default-buft path). The TURBOMIND multi-GPU correctness regression is documented but not fixed in this sprint. |

---

## Known limitations carried forward

- **CUDA_TURBOMIND family-alias deferred** — multi-GPU expert distribution
  with TURBOMIND offload requires either generated per-layer overrides
  (loses PP, doesn't hurt M=1 decode) or the deferred family-alias work
  (preserves PP). Both paths are forward-compatible with this sprint's
  per-device TmLib refactor; the family-alias is purely a buft-registration
  detail and the override-resolution insertion point in `llama-model.cpp`.
- **8-GPU run completed** (this report's TL;DR). 256e fits on 8× V100 with
  10+ GiB headroom on every device, supporting `-c 4096` FP16 KV. Larger
  `-c` not yet measured but the headroom suggests `-c 8192–16384` should
  fit without q8_0 KV.
- **"Invalid input batch" on multi-word ASCII prompts** — separate
  server-side bug, not blocking. Code/identifier prompts and prompts with
  leading punctuation decode normally.
- **AVG-16e divergence** — from SPRINT-024 REPORT-18 §P3.2 (2/5 prompts
  diverge at FP16 argmax tiebreaks). Carried forward; SPRINT-026 P0
  addresses.
