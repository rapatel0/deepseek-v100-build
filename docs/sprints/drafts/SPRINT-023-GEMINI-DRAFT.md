# SPRINT-023 — Custom grouped-MoE sm70 kernel: stop pretending MoE is `mul_mat`

**Sprint number**: 023 (alternative draft)
**Date**: 2026-05-15
**Predecessor**: SPRINT-022 (operational baseline, 4.28 pp / 4.73 tg t/s)
**Position**: Contrarian to SPRINT-023-INTENT.md

---

## 1. Overview

The intent doc wants to wire `turbomind::gemm::Gemm::Run` into
`ggml-cuda/mmvq.cu` + `mmq.cu` and call it a day. **This is the wrong
abstraction.** Turbomind's `Gemm::Run` is a single-matmul interface; DSv4
MoE inference does **256 expert tensors, top-8 routed per token, batched
per-expert**. Plugging a single-matmul API into `ggml_mul_mat_id` (which
already exists and already batches per expert via `expert_bounds`) means
we get one of two failures:

1. **Decode (tg)**: top-8 tiny per-expert matmuls (M=1 each, sometimes
   bucketed to M=k≤8 per expert when shared experts collapse). Turbomind
   is shaped around M≥128 tiles; its smallest registered `BM` for
   `Config_E4M3` / `Config_U4_g` is 8, but the launcher cost per-call
   (kernel dispatch, scale-buffer wiring, the `gemm2` Params struct
   marshalling) dominates the math. We will pay a launch tax 8× per
   token per layer × 60 layers = 480 launches per token. At V100's ~5 µs
   minimum launch latency that is **2.4 ms of pure launch overhead per
   token**, i.e. a hard ceiling of ~420 t/s before any FLOPs are done —
   but with batched-per-expert sparsity, the effective ceiling is far
   lower.

2. **Prefill (pp)**: better, because M scales with token count, but
   turbomind's launcher still picks tiles per-call. With 256 experts ×
   60 layers and an average ~5 tokens per expert per layer at pp128, we
   spend most of the kernel time in launch/wind-down, not the
   mainloop.

What we actually want is **one kernel launch per layer** that fuses
top-k routing + grouped per-expert MXFP4 dequant + MMA. Grouped GEMM,
not single GEMM. Turbomind doesn't have a sm70 grouped path. CUTLASS
does (`GroupedGemm`), but not at MXFP4 weight precision on sm70 (the
`Config_MXF4` family is turbomind-private). So we either reuse our own
v13 family (which already does INT8 weight at 49 TF) or write the
grouped wrapper ourselves.

**This sprint proposes:**

1. **Reject** the turbomind-plug-in path as the primary deliverable.
2. **Ship** a custom grouped-MoE sm70 kernel (`mmid_int8_v13_group`)
   that takes `(W[E][N][K], scales[E][N][K/QK], A, ids, expert_offsets)`
   and emits `C = W[ids[tok]] @ A[tok]` in a single launch, reusing the
   v13_rf mainloop body already shipped in SPRINT-021.
3. **Keep MXFP4 weights on CPU** for cold experts. Move only the
   top-N hot experts to GPU as **INT8 re-quant** (since v13 already
   wins at 49 TF for INT8 on V100 and we control that kernel
   end-to-end). MXFP4 stays as the canonical disk format.
4. Use the freed VRAM (~22.5 GiB) for a **prefetch ring buffer** that
   streams cold experts from pinned host RAM 1-2 layers ahead of the
   token's forward pass, hiding the PCIe transfer behind compute.

This is a smaller-surface, higher-leverage move than the intent. The
intent ships in 3+ weeks; this draft ships measurable uplift in 1.

---

### Theoretical decode ceiling (this matters before any code)

DSv4-Flash activates **37B params per token** (256 experts × 145 MiB
expert × 8 routed = ~8 GiB of weight per token forward at MXFP4
0.5 B/wt; plus ~7.5 GiB of dense FP8 hit every token).

At V100 HBM bandwidth **900 GB/s** and weights touched per token T:
- Pure GPU resident, MXFP4: T ≈ 8 GiB experts + 7.5 GiB dense = **15.5
  GiB/token → 58 tok/s HBM ceiling** (decode).
- Plus KV cache reads (small per token, <50 MiB) and overhead → realistic
  ceiling **~45-50 tok/s decode** if every weight read hits HBM and
  HMMA active% stays near 50%.
- Intent target of 20 tok/s = 44% of this ceiling. **Achievable but not
  trivial** — turbomind's single-matmul dispatch will not deliver this
  because launch tax + low M occupancy will dominate.

If we accept that 20 tok/s is the bar, then the design lever is **not**
"how fast can the mainloop go" but **"how few HBM bytes do we move per
token"** and **"how few kernel launches per token"**. Both argue for
grouped MoE, not single-matmul-per-expert.

---

### Pipelined-CPU-experts alternative (the deeper contrarian take)

A real production V100 deployment may not need experts on GPU at all.
At pp128 = 4.28 t/s we are CPU-bound on per-token MoE compute (not
PCIe bandwidth — the intent doc misreads this signal). Concretely:

- Top-8 of 256 experts → only ~3% of MoE weight touched per token
- 145 GiB model × 3% × FP8 = ~4.4 GiB / token effective
- PCIe Gen3 x16 = **15.75 GB/s** sustained → ~280 ms to transfer per
  token's experts = ~3.6 tok/s if naive
- With **prefetching 1-2 layers ahead** (overlap PCIe with compute on
  the current layer's GPU dense FFN + attention) we hide most of it
  → effective ceiling ~10-15 tok/s with experts streaming, no
  resident allocation
- **Hot-resident top-N experts** (the intent's plan) combined with
  cold-expert streaming for the rest gets us to 20-30 tok/s without
  needing to fit the whole MoE in 22.5 GiB

We should at least measure this before committing to "all experts
resident." See P0 below.

---

## 2. Use Cases

| Use case | Driver | Why it changes design |
|---|---|---|
| Single-user interactive decode (chat) | tg ≥ 20 t/s, latency < 1 s first-token | Optimize for kernel-launch tax + HBM bytes/token; tiny M, tight loop |
| Long prefill (RAG, code) | pp ≥ 50 t/s | Optimize for M ≥ 128, where the grouped kernel actually gets MMA-bound |
| Memory-pressured deployment (no room for hot-experts) | Run on operator nodes that share GPU with other services | Pipelined-CPU-experts path must work as a build-time option |
| Validation against canonical CPU path | Bit-equivalence with nisparks's scalar fallback | Same correctness contract as the intent doc — non-negotiable |
| Profile-guided hot-expert loadout | Per-prompt-type expert distribution | JSON loadout, but with **dynamic top-up** if a prompt activates cold experts > threshold (see Open Questions) |

---

## 3. Architecture

```
        +------------------+        +-------------------+
        | GGUF MXFP4 mmap  |        | hot-experts.json  |
        | (canonical disk) |        | (offline profile) |
        +--------+---------+        +---------+---------+
                 |                            |
                 | one-time repack on load    |
                 v                            v
  +---------------------------+   +------------------------+
  | Pinned host RAM           |   | VRAM hot pool          |
  | [E_cold][N][K] INT8 + sc  |   | [E_hot][N][K] INT8+sc  |
  |  ~120 GiB                 |   |  ~18 GiB (top-32 exp)  |
  +-------------+-------------+   +-----------+------------+
                |                             |
                |  PCIe stream, prefetched    |
                |  L+1 while running L        |
                v                             v
        +-----------------------------------------+
        |  Custom kernel: mmid_int8_v13_group     |
        |  inputs: W*, scales*, A, ids,            |
        |          expert_offsets, hot_mask        |
        |  output: C                               |
        |  one launch per layer per pp/tg step    |
        +-----------------------------------------+
                          |
                          v
                 ggml-cuda graph (existing)
```

Key design decisions:

1. **INT8 not MXFP4 on GPU.** v13_rf is 49 TF measured. MXFP4 sm70 is
   theoretically faster (~58 TF projected, see REPORT-15 §1) but
   requires turbomind's `Config_MXF4` which is `group_size=32` only
   (REPORT-15 §3.1 footnote) and currently fails registry match. We
   own v13 end-to-end; we do not own turbomind. Pick the kernel we can
   debug.

2. **One-time MXFP4 → INT8 re-quant at load.** MXFP4 has E2M1 mantissa
   + per-32 E8M0 scale. Re-quant to INT8+fp16-scale loses ~0.5%
   perplexity (well within nisparks tolerance contract). Repack happens
   once per model load, in a llama-load hook (NOT in
   `ggml-cuda/convert.cu` — see §4 P1).

3. **Grouped kernel signature**: `mmid_int8_v13_group<BM,BN,BK,WARPS>(
   W_qs[E][N][K], W_scales[E][N][K/QK], A[T][K], ids[T][topk],
   expert_offsets[E+1], C[T][N])`. CTA grid is `(N/BN, E_active)`
   with each block iterating its own slice of `ids` to compute the
   per-token contributions for its assigned expert.

4. **Cold-expert streaming**: a small dispatcher thread issues
   `cudaMemcpyAsync` of cold experts for layer L+1 on a non-default
   stream while the kernel for layer L runs. If `hot_mask[expert] = 0`
   the kernel reads weights from a transient streaming buffer instead
   of the hot pool.

5. **No `ggml_mul_mat` integration at the dispatch.cu level.** We add a
   new op `GGML_OP_MUL_MAT_ID_MOE_DSV4` (or a backend-specific path
   keyed off `src0->buffer == hot_expert_pool`). This avoids touching
   the 1000+-line switch statements in `mmvq.cu`/`mmq.cu` for every
   ggml type. The op fires only when the tensor is in our hot pool;
   everything else falls through to existing dispatch.

6. **Two configs of the same kernel**:
   - Decode variant (M=1..8, BN=128, K_split=8) — reuses v12s SplitK
     champion from `dispatch.h` per REPORT-16 §"Final champion table"
   - Prefill variant (M≥128, BM=128 BN=128 BK=16 W=4) — reuses
     v13_rf_v6 champion

The kernel is **MoE-aware end-to-end**, not a single GEMM wrapped in a
loop.

---

## 4. Implementation

### P0 — Measurement (3 days). NO CODE that locks in architecture.

**Goal: validate or kill the "experts on GPU" premise before we
build for it.**

- P0.1 — Instrument current SPRINT-022 baseline: per-layer time
  breakdown of CPU expert compute, PCIe time for `ggml_get_rows`-ish
  expert fetch (if any), and GPU dense+attention time. Add timers in
  `src/llama-context.cpp` graph-execute path; output to
  `logs/sprint-023-p0-baseline-breakdown.csv`.
- P0.2 — Synthetic test: implement a 1-layer prototype that streams
  expert weights from pinned host RAM via `cudaMemcpyAsync` on a
  separate stream while a v13_rf kernel runs. Measure the overlap
  ratio. File: `tools/tc-grid/src/moe_stream_overlap.cu` (new). If
  overlap covers ≥70% of transfer time, the pipelined-CPU path is
  competitive and we should keep it as a fallback build.
- P0.3 — Re-quant fidelity check: write
  `tools/quant/mxfp4_to_int8_repack.cpp` (new). Re-quant 16 random
  expert tensors from the production GGUF. Compute relative error on
  100 random activations per tensor. Gate: ≤ 0.5% mean rel error,
  ≤ 2% p99. If we fail, fall back to MXFP4-on-GPU (and bite the
  turbomind `Config_MXF4` bullet — but only if we must).

**P0 exit gate**: a 1-page memo in `tools/tc-grid/docs/REPORT-17.md`
that picks one of three paths:
- (A) Hot-resident INT8 grouped kernel (the proposal here)
- (B) Pipelined CPU experts, keep MXFP4 on host (the dark-horse)
- (C) Original intent: turbomind plug-in (the safe-default fallback)

### P1 — Repack and host-side hot-expert plumbing (2 days)

- P1.1 — `tools/quant/mxfp4_to_int8_repack.cpp` lifted to production
  quality. Reads GGUF, identifies tensors matching
  `blk.*.ffn_(gate|up|down)_exps`, emits a side-car file
  `MODEL.int8-experts.bin` with `(E, N, K)` per layer.
- P1.2 — Load hook in `src/llama-model-loader.cpp` (new branch keyed
  off `GGML_MOE_INT8_REPACK=ON` env). On model load, mmap the side-car
  and allocate the **hot pool** in VRAM, sized per `hot-experts.json`
  loadout file.
- P1.3 — Loadout file format: simple TSV
  `layer expert weight_offset_in_pool`. Synthetic profile (uniform top-32
  per layer) ships as `tools/data/hot-experts-uniform.tsv`. The
  realistic profile generator stays out of scope for this sprint (it
  needs an unrelated profiling pass on production traces).

### P2 — Grouped MoE kernel (5 days)

- P2.1 — New header
  `tools/tc-grid/kernels/v14_grouped_moe.cuh`. Templated on
  `(BM, BN, BK, WARPS, ATOMS_M, ATOMS_N, TOPK)`. Body is v13_rf_v6's
  K-fused inner loop unchanged.
- P2.2 — Outer dispatcher logic:
  - Grid `(ceil(N/BN), num_active_experts)` where active = experts
    appearing in any token's top-k for this layer
  - Each CTA loads its slice of `expert_offsets[e]..expert_offsets[e+1]`
    into SMEM; iterates over the (token, top-k-position) pairs that
    map to its expert; accumulates into the correct row of C
  - Reuses `ggml_cuda_launch_mm_ids_helper` upstream (see
    `mmq.cu:180`) for the `expert_bounds` array — no new helper needed
- P2.3 — Two specializations:
  - `v14_grouped_decode` (BM=8, BN=128, BK=32, splitK=8) for tg path
  - `v14_grouped_prefill` (BM=128, BN=128, BK=16, W=4) for pp path
- P2.4 — Cold-expert streaming path:
  `tools/tc-grid/src/cold_expert_streamer.cu` (new). Owns a non-default
  stream + 2 ping-pong staging buffers in VRAM (~1 GiB each, sized for
  one layer's worst-case cold expert footprint). Kernel reads from
  `weight_ptr_table[expert_id]` which gets patched per layer.
- P2.5 — Bench harness:
  `tools/tc-grid/src/grouped_moe_bench.cu` (new). Generates a
  realistic top-k routing pattern (Zipf over 256 experts, alpha=1.2)
  and measures: kernel time, launch count, achieved TF, HMMA active%
  via ncu hook.

### P3 — ggml-cuda integration (2 days)

- P3.1 — Add `GGML_OP_MUL_MAT_ID_MOE_DSV4` to `ggml/include/ggml.h`
  and `ggml/src/ggml.c`. Op detection in
  `ggml/src/ggml-cuda/ggml-cuda.cu`: keyed off
  `src0->buffer == g_hot_expert_pool` AND `op->op == GGML_OP_MUL_MAT_ID`.
- P3.2 — Backend dispatch in
  `ggml/src/ggml-cuda/mmid_moe_dsv4.cu` (new). Calls the v14 kernel
  family. No edits to `mmvq.cu` or `mmq.cu` — both fall through if the
  buffer isn't our pool.
- P3.3 — Fallback path: `GGML_MOE_INT8_REPACK=OFF` builds out the hot
  pool, the `mmid_moe_dsv4` op never fires, behavior is identical to
  SPRINT-022.

### P4 — Verification and benchmarks (2 days)

- P4.1 — Correctness: bit-equivalence vs CPU scalar (nisparks) path.
  Same SPRINT-015 P2 tolerance contract (rel ≤ 2e-2 col-parallel,
  ≤ 1e-2 row-parallel) on 8 random expert mul_mats per shape.
- P4.2 — End-to-end `llama-bench` with three configs:
  - SPRINT-022 baseline (`-ot exps=CPU`)
  - This sprint hot-resident (`GGML_MOE_INT8_REPACK=ON`, no streaming)
  - This sprint hot+streaming (full pipelined path)
- P4.3 — VRAM budget check (stay under 30 GiB total).
- P4.4 — ncu pack on the v14 kernel: confirm HMMA active% ≥ 40% in
  decode, ≥ 50% in prefill. If we miss, the kernel needs more work
  before we ship.

---

## 5. Files Summary

**New:**
- `tools/quant/mxfp4_to_int8_repack.cpp`
- `tools/data/hot-experts-uniform.tsv`
- `tools/tc-grid/kernels/v14_grouped_moe.cuh`
- `tools/tc-grid/src/cold_expert_streamer.cu`
- `tools/tc-grid/src/grouped_moe_bench.cu`
- `tools/tc-grid/src/moe_stream_overlap.cu` (P0)
- `tools/tc-grid/docs/REPORT-17.md` (P0 exit memo)
- `ggml/src/ggml-cuda/mmid_moe_dsv4.cu`
- `ggml/src/ggml-cuda/mmid_moe_dsv4.cuh`

**Modified:**
- `ggml/include/ggml.h` (new op enum value)
- `ggml/src/ggml.c` (op metadata)
- `ggml/src/ggml-cuda/ggml-cuda.cu` (op detection + dispatch hook)
- `src/llama-model-loader.cpp` (side-car mmap + hot pool allocation)
- `src/llama-context.cpp` (per-layer timing instrumentation for P0)

**Untouched (deliberately):**
- `ggml/src/ggml-cuda/mmvq.cu` — fall-through, no changes
- `ggml/src/ggml-cuda/mmq.cu` — fall-through, no changes
- `ggml/src/ggml-cuda/convert.cu` — no per-tensor MXFP4→INT8 here;
  repack is a one-time host-side pass
- All existing turbomind carve-out at `tools/tc-grid/turbomind_minimal/`
  — kept as a measurement asset, not on the runtime path

---

## 6. Definition of Done

1. P0 exit memo (`REPORT-17.md`) commits to one path with evidence
2. If Path A (proposed):
   - `GGML_MOE_INT8_REPACK=ON` build flag works on V100 sm70
   - `tg32` ≥ **20 t/s** on DSv4-Flash with synthetic-uniform hot loadout
   - `pp128` ≥ **20 t/s**, stretch ≥ 50 if HMMA active% hits 50%
   - VRAM ≤ 30 GiB
   - Correctness: ≤ tolerance rel error vs CPU scalar on 8 expert
     mul_mats × 2 shapes
   - ncu HMMA active% logged for both decode and prefill variants
   - Build with `GGML_MOE_INT8_REPACK=OFF` produces SPRINT-022 baseline
     identical TPS within ± 2%
3. If Path B (pipelined CPU experts) wins P0:
   - rewrite §4 P1-P3 around the streaming dispatcher, keep the
     grouped kernel work but parameterize it for cold-only inputs
4. If Path C (turbomind) wins P0:
   - we lose this debate; fall back to the intent doc's plan

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MXFP4→INT8 re-quant blows tolerance | Med | High | P0.3 measures this before we build the kernel. Fallback = MXFP4-on-GPU via turbomind path |
| v14 grouped kernel HMMA active% lower than v13's 42% | Med | Med | Per-expert small slices (avg ~5 tokens) may underutilize MMA pipes. Compensate via wider BN tiles when expert is hot |
| Cold-expert streamer can't hide PCIe latency at long context | Med | High | Pre-warm hot pool aggressively; if cold-rate > 30%, regress to all-hot allocation |
| Hot-experts.json synthetic profile doesn't match real routing | High | Med | Document this is provisional; SPRINT-024 owns the real profiling pass |
| Op type registration upstream conflict | Low | High | Use a non-upstream op enum reserved range; gated by `GGML_DSV4` macro |
| INT8 quant ceilings out at v13's 49 TF, MXFP4 plug-in could have done 58 TF | Med | Med | Acknowledged: this is a deliberate complexity trade. v13 we own and can extend (v13_rf_v6 is at 88% of FP8 ceiling already). Turbomind we don't, and integration is open-ended |
| 1-week prototype slips when grouped-routing edge cases bite | High | Med | Build P0 first; if P0 takes > 1 week, descope to P3 only (hot-only, no streaming) |

---

## 8. Security

- No new network surface; runs local on V100.
- Side-car file `MODEL.int8-experts.bin` is generated from a trusted
  GGUF; treat as same trust level as the source GGUF.
- `hot-experts.json` (or .tsv) is user-supplied config that selects
  which tensors live in VRAM. Validate against the model's actual
  expert count to prevent OOB allocation. Reject malformed TSV early.
- No SUID, no privileged ops. Build flag `GGML_MOE_INT8_REPACK=OFF`
  defaults are unchanged behavior.

---

## 9. Dependencies

- **V100 sm_70**, CUDA 12.2, gcc 11.4 — same as SPRINT-021/022
- **gpu-01 build pod** (`llamacpp-build`) — same
- **Existing**:
  - `v13_kernels.cuh` (SPRINT-021 P1) — body lifted into v14
  - `mma_sm70.cuh` — unchanged
  - `tools/tc-grid/` harness — bench reused, no new infra
  - ggml's `mul_mat_id` + `ggml_cuda_launch_mm_ids_helper`
    (`mmq.cu:180`) — reused as-is
- **Removed dependency**: turbomind carve-out is **not** linked into
  llama.cpp build under Path A. We keep `turbomind_minimal/` as a
  measurement asset only.
- **No new third-party**: no CUTLASS pull-in, no fresh CUDA lib.

---

## 10. Open Questions

1. **Dynamic hot-expert promotion at runtime?** The intent doc
   explicitly defers this. We agree, but only if P4 shows ≤ 5% cold-rate
   on typical prompts. If cold-rate is higher, we need either a richer
   loadout or runtime promotion. Defer the decision to the end of P4.
2. **Per-layer vs per-model hot loadout?** Some layers may have flatter
   expert usage than others. The TSV format allows per-layer; the
   SPRINT-024 profiling pass should produce per-layer numbers.
3. **Should we just use cuBLAS FP16 with on-GPU dequant?** REPORT-15 §1
   shows cuBLAS FP16 hits **87 TF** but at 2 B/wt = doubles the HBM
   footprint. The 22.5 GiB hot pool budget gets us 8-12 hot experts in
   FP16 vs 32 in INT8. Worth a P0.4 measurement? **Probably yes** —
   add as a P0.4 stretch.
4. **NVFP4 — same answer as intent doc**: model doesn't use it, defer.
5. **What happens when an expert isn't in hot pool AND isn't yet
   streamed?** Fallback: a synchronous `cudaMemcpyAsync` + wait. This
   regresses to ~CPU speed for that expert. Acceptable if cold-rate
   is low; if not, see Open Question 1.
6. **Is the `GGML_OP_MUL_MAT_ID_MOE_DSV4` op type upstreamable?** No,
   and we shouldn't try. This is a private-fork performance hack. The
   intent doc's path is more upstreamable but slower for our specific
   use case. We're optimizing for this V100 deployment, not for
   contributor-friendliness.
7. **What kills this sprint?** If P0.3 shows MXFP4→INT8 loses > 1 pp
   perplexity, OR if P0.2 shows PCIe streaming can already deliver
   15 t/s without any kernel work, we should pivot to Path B (pipelined
   CPU experts) and treat the grouped kernel as a SPRINT-024
   follow-up.

---

### TL;DR for reviewers

The intent doc proposes wiring turbomind's single-matmul API into ggml-cuda
for MoE inference. MoE inference isn't single-matmul — it's grouped GEMM
with top-k routing. Plugging a non-grouped API into a grouped problem
leaves perf on the table via launch tax and poor MMA utilization at low
per-expert M. The contrarian proposal: spend P0 (3 days) measuring whether
GPU-resident experts are even the right move; if they are, ship a custom
v13-derived grouped kernel that fuses routing + dequant + MMA in one
launch per layer, using INT8 quant we already understand at 49 TF instead
of MXFP4 we don't. Smaller surface, kernel we own end-to-end, fallback to
existing dispatch is one env var.
