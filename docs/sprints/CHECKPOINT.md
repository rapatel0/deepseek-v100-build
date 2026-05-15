# Session Checkpoint — V100 INT8 GEMM 50 TF goal

**Last updated:** 2026-05-14
**Branch:** `sprint-016-tensor-unlock`
**Latest commit:** `76efa14ba` sprint-018 close (P2-P4 scoped out)

This file is the **single entry point for a fresh-context session**. Read it
top-to-bottom and you have everything needed to resume work without re-deriving state.

---

## 1. Where we are (TL;DR)

**Goal:** exceed **50 TFLOPS at M=2048** on V100 INT8 tensor-core GEMM (N=K=7168, bit-correct).

**Current production state (all bit-correct, rel ≤ 1e-3):**

| Operating point | Production champion | TFLOPS | Notes |
|---|---|---:|---|
| M=64 (decode-style) | `v10s_64x128_ks8` (SplitK) | 20.31 | +84% over v10 |
| M=256 | `v10_64x128` | 27.94 | (v11 28.72 small win) |
| M=1024 | **`v11_128x128x16_w4`** | **34.35** | **+19.1% over v10** |
| M=2048 (headline) | **`v11_128x128x16_w4`** | **34.65** | **+17.5% over v10** |
| M=4096 | **`v11_128x128x16_w4`** | **34.18** | **+17.1% over v10** |

**Ceiling proof (NOT production):** CUTLASS 2.11.0 with pre-dequant W INT8→FP16 lands
**85.93 TFLOPS at M=2048**. This proves the FP16 mma path on V100 can deliver 85+ TF on
this shape. It is **not a production path** — pre-dequant doubles VRAM (28 GB FP16 vs
14 GB INT8 for DSv4) and doubles HBM read bandwidth. See
[`feedback_pre_dequant_defeats_int8.md`](../../.claude/projects/-Users-ravi-repos-deepseek/memory/feedback_pre_dequant_defeats_int8.md).

**Per-format SplitK wins at M=64 (this session):**

| Format | v3 baseline | v3s SplitK | Δ |
|---|---:|---:|---:|
| INT8 | 11.02 | 20.31 (ks=8) | **+84%** |
| INT4 | 6.21 | 11.05 (ks=8) | **+78%** |
| MXFP4 | 9.77 | 16.06 (ks=8) | **+64%** |
| F8 | 4.60 | 10.01 (ks=8) | **+118%** |

---

## 2. What's next — V11 Step 2

**The production path on V100 INT8 is fused on-GPU dequant**, exactly what V11 targets.
P1's 85 TF result is the new target ceiling.

**Implementation entry point:**
[`tools/tc-grid/docs/V11-STEP2-HANDOFF.md`](../../tools/tc-grid/docs/V11-STEP2-HANDOFF.md)
— full lane-mapping math, 4 subtasks (~~2a SMEM-roundtrip~~ ✅ → 2b single-tile → 2c full
kernel → 2d sweep), 2 of 3 open questions remain.

**V11 Sprint 017 COMPLETE (2026-05-14):**

Steps 2a–2d (foundation), 2.5 (BK=16), 2.6 (stream cache, negative), and
3 (XOR swizzle, partial) all shipped. Headline outcome:

| Shape | v11 champion | TF | v10 baseline | Δ |
|---|---|---:|---:|---:|
| M=1024 | v11 128x128x16_w4 | **34.35** | 28.85 | **+19.1%** |
| M=2048 | v11 128x128x16_w4 | **34.65** | 29.50 | **+17.5%** |
| M=4096 | v11 128x128x16_w4 | **34.18** | 29.18 | **+17.1%** |

All bit-correct (rel = v10 reference exactly: 2.594e-04 at M≥1024).

Decision-rule placement: 34.65 TF lands just below the "35–50 TF" tier
(beats v10, ship + investigate gap). Gap to 50 TF goal: 1.44×; gap to
P1 CUTLASS ceiling (85 TF): 2.45×.

Negative results documented in v11_kernels.cuh:
- Stream cache (`__ldcs` on B-side) regresses -30% on this shape — W_scales
  reuse is high enough that L1 caching beats Stream's capacity savings.
- XOR Swizzle<3,3,3> on sB helps BK=32 (+14%) but slightly hurts BK=16 (-1%);
  the BK=16 champ's BK_PAD=24 already gives only 4-way conflict.
- 2-uint4 store batching: helps BK=32 (+12%), hurts BK=16 (-1%).

**Next step:** Investigate path past 35 TF — candidates are (a) larger CTA
tile (BM=256? requires register-pressure work to avoid spill), (b) A-side
XOR swizzle (smaller expected win), (c) reduce non-tensor instruction
overhead (44% tensor pipe usage today, headroom to ~70%).

**Decision rule for V11 (updated post-P1):**

| V11 final at M=2048 | Action |
|---|---|
| ≥ 60 TF | Production-quality. Ship. |
| 50–60 TF | Sprint goal met. Ship with documented gap to 85 TF ceiling. |
| 35–50 TF | Beats v10. Ship + investigate gap vs turbomind reference. |
| < 35 TF | Hardware-limit signal. Revisit fundamentals. |

**Foundation already validated in this session:**
- `tools/tc-grid/kernels/mma_sm70.cuh` — m8n8k4 inline PTX wrapper (4/4 sanity tests pass)
- `tools/tc-grid/tests/test_mma_sm70.cu` — the validation test

**V11 Step 2 effort estimate (per `feedback_effort_estimation_undocumented_hardware`):**
- Nominal: 3 hr (lane-mapping derivation)
- Realistic with multiplier: 8–10 hr (multi-session)
- Hardest single piece in V11 — fresh context recommended for the debug loop

---

## 3. Critical memory items — READ BEFORE WORKING

All are in `~/.claude/projects/-Users-ravi-repos-deepseek/memory/` and indexed in `MEMORY.md`:

| Memory | Why it matters |
|---|---|
| `v100_wmma_smem_conflict_constraint.md` | Padding cannot zero bank conflicts on V100; real lever is SMEM access pattern. v10's row-major B was a different mechanism than XOR swizzle |
| `v100_wmma_half_float_frag_layout_mismatch.md` | sm_70 half/float wmma accumulators use different lane→element mappings — broke v9 mixed-precision. Don't mix frag types in registers |
| `feedback_dont_skip_plan_steps.md` | Execute every planned step. SplitK A0 was almost skipped — would have lost the +84% M=64 win. Skip only with explicit auth or unresolvable bug |
| `feedback_effort_estimation_undocumented_hardware.md` | Multiply gut estimates 3× when work involves opaque ISA mappings, commented-out upstream code, or sticky build caches |
| `feedback_pre_dequant_defeats_int8.md` | Pre-dequant doubles VRAM + HBM bandwidth; any TFLOPS that excludes dequant cost is a CEILING, not a production number |

---

## 4. Key source files (production state)

| File | Purpose |
|---|---|
| `tools/tc-grid/kernels/v10_kernels.cuh` | Current INT8 large-M production champion (29.42 TF M=2048) |
| `tools/tc-grid/kernels/v10splitk_kernels.cuh` | INT8 SplitK (small-M champion, version=20) |
| `tools/tc-grid/kernels/v3splitk_kernels.cuh` | INT4/MXFP4/F8 SplitK (version=30) |
| `tools/tc-grid/kernels/mma_sm70.cuh` | V11 m8n8k4 PTX wrappers — validated this session |
| `tools/tc-grid/kernels/cutlass_int8_kernels.cuh` | SPRINT-018 P1 CUTLASS ceiling (version=40, NOT production) |
| `tools/tc-grid/src/launch_int8.cu` | INT8 launcher with v0..v10, v10s, v40 (cutlass) dispatch |
| `tools/tc-grid/src/launch_int4.cu` / `launch_fp4.cu` / `launch_fp8.cu` | Per-format launchers with v3s dispatch |
| `tools/tc-grid/src/main.cu` | Tile registry (kTiles[]); sweep harness |

---

## 5. Sprint state docs

| Doc | Status |
|---|---|
| [SPRINT-016.md](./SPRINT-016.md) | CLOSED — v9 broken, v10 shipped at 29.49 TF (sprint goal of 35 TF missed) |
| [SPRINT-016-DEFERRED.md](./SPRINT-016-DEFERRED.md) | XOR-swizzle promoted to active (SPRINT-017) |
| [SPRINT-016-FOLLOWUPS.md](./SPRINT-016-FOLLOWUPS.md) | v9 SMEM round-trip, INT4 BN=256, multi-shape MoE |
| [SPRINT-017.md](./SPRINT-017.md) | ACTIVE — V11 fused-dequant; decision rule updated for new 85 TF ceiling |
| [SPRINT-018-CUTLASS.md](./SPRINT-018-CUTLASS.md) | P0+P1 SHIPPED as ceiling proof; P2–P4 scoped out |

| Tech doc | Purpose |
|---|---|
| [tools/tc-grid/docs/REPORT-9.md](../../tools/tc-grid/docs/REPORT-9.md) | Tier B retrospective with ncu data |
| [tools/tc-grid/docs/REPORT-10.md](../../tools/tc-grid/docs/REPORT-10.md) | v9 broken-by-design (sm_70 frag layout mismatch) |
| [tools/tc-grid/docs/REPORT-11.md](../../tools/tc-grid/docs/REPORT-11.md) | v10 ships at 29.49 TF + full-spectrum audit |
| [tools/tc-grid/docs/REPORT-12.md](../../tools/tc-grid/docs/REPORT-12.md) | **v11 ships at 35.08 TF + ncu-driven lever log + 6 next-step candidates** |
| [tools/tc-grid/docs/V11-DESIGN.md](../../tools/tc-grid/docs/V11-DESIGN.md) | Original V11 5-step design |
| [tools/tc-grid/docs/V11-EXECUTION-PLAN.md](../../tools/tc-grid/docs/V11-EXECUTION-PLAN.md) | Wave-by-wave decision-gated execution plan |
| [tools/tc-grid/docs/V11-STEP2-HANDOFF.md](../../tools/tc-grid/docs/V11-STEP2-HANDOFF.md) | **NEXT-STEP entry point** — lane-mapping math + subtasks |
| [tools/tc-grid/docs/TURBOMIND-INSIGHTS.md](../../tools/tc-grid/docs/TURBOMIND-INSIGHTS.md) | Source-walk of turbomind sm_70 + counterfactual §L |

---

## 6. Build + run (gpu-01 dev pod)

```bash
# Verify pod is up
kubectl exec -n llm tcg-dev -- bash -c "ls /src/tools/tc-grid && nvidia-smi -L"

# Sync laptop edits to gpu-01 (pod mounts /srv/dev/dsv4-cuda/deepseek-sprint017)
rsync -av tools/tc-grid/kernels/<file> ubuntu@192.168.102.5:/srv/dev/dsv4-cuda/deepseek-sprint017/tools/tc-grid/kernels/

# Pre-rsync chown if needed (only when creating new dirs):
ssh ubuntu@192.168.102.5 'sudo -n chown -R ubuntu:ubuntu /srv/dev/.../<dir>'

# Configure + build
kubectl exec -n llm tcg-dev -- bash -c "cd /src/tools/tc-grid && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build build -j 8"

# Run bench (filter to format + version of interest)
kubectl exec -n llm tcg-dev -- bash -c "cd /src/tools/tc-grid && ./build/tc-grid --m-list 64,256,1024,2048,4096 --nk 7168 --dist uniform_small" 2>&1 | grep -E 'INT8.*v10|cutlass'

# Pause DCGM-exporter before ncu runs
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite
```

**Gotcha**: `CMAKE_CUDA_ARCHITECTURES` as a CACHE variable doesn't survive partial
reconfigures cleanly. If build silently targets sm_52 (wmma errors), `rm -rf build`
and re-cmake with explicit `-DCMAKE_CUDA_ARCHITECTURES=70`.

---

## 7. Session commits (newest first)

```
76efa14ba sprint-018 close (P2-P4 scoped out): CUTLASS sm_70 is ceiling proof only
957a8019c sprint-018 P1: CUTLASS V100 INT8 path -- 50 TF GOAL SMASHED
ff3dfde43 sprint-018 P0: CUTLASS 2.11.0 dependency + sm_70 build verification
a29a06b43 sprint-016/017 state docs: backfill SPRINT-016 closure + REPORT-9/10/11
2deb379d9 sprint-017/018: pre-commit CUTLASS as 50 TF hedge, SPRINT-018 planned
25994610e sprint-017 P1: SplitK port to INT4 / MXFP4 / F8 (v3s pattern)
35753eb7c sprint-017 P0: Step 2 handoff doc -- manual Lds lane-mapping prep
114cc00c1 sprint-017 A0: v10s -- SplitK launcher + kernel (small-M +84% at M=64)
897847fd5 sprint-017 P0: v11 Step 1 -- m8n8k4 PTX wrapper validated on V100
```

---

## 8. Resume protocol for a fresh context

When starting the next session, before any tool calls:

1. Read this CHECKPOINT.md fully.
2. Read `V11-STEP2-HANDOFF.md` for the implementation entry point.
3. Read the 5 memory items above (or skim the index in `MEMORY.md`).
4. Verify pod state: `kubectl exec -n llm tcg-dev -- nvidia-smi -L`.
5. Verify build works (incremental): `cd build && cmake --build . -j 8`.
6. Confirm v10 baseline reproduces (~29.4 TF M=2048).
7. Begin V11 Step 2a: SMEM-roundtrip unit test.

**Do NOT:**
- Skip V11 Step 2 to chase other ideas — the don't-skip rule applies.
- Pre-dequant W to FP16 in gmem as a production proposal — see memory item.
- Use the 85 TF P1 number without the "ceiling, not production" caveat.
- Trust WMMA fragment internals to map cleanly to m8n8k4 — they don't on sm_70.
- Quote effort estimates without the 3× multiplier for lane-mapping / undocumented work.

---

## 9. Open items (not blockers, but tracked)

- **INT4 BN=256 spill** (from SPRINT-016 followups): two-pass FRAG_N approach,
  ~5–10% INT4 large-M uplift if fixed.
- **Multi-shape MoE validation**: only N=K=7168 measured so far. DSv4 production
  shapes may need separate champion identification.
- **v9 SMEM round-trip variant**: alternative mixed-precision approach if register
  pressure becomes the next bottleneck post-V11.
- **MoE-aware dispatcher integration**: production gateway from tc-grid winners
  into actual DSv4 inference path. Not a kernel-tuning task; separate work.
- **gpu-02-4090rtx**: 4090 occupied by `qwen3-moe-rotorquant` — DO NOT EVICT for
  parallel V100 work.
