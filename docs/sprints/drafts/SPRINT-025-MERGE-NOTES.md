# SPRINT-025 — Merge notes

**Date:** 2026-05-15

## Draft comparison

### Claude draft (34 KB / 426 lines)
- Detailed phase structure, granular gates, specific code references.
- Strong TmLib refactor proposal (per-device state).
- Strong VRAM accounting (with per-GPU thresholds).
- **Wrong**: Treats `LLAMA_SPLIT_MODE_TENSOR` as a legacy synonym for `LAYER` (Codex critique caught — `TENSOR` is separate plumbing, requires FA, rejects quantized KV).
- **Wrong**: `-ot 'exps=CUDA_TURBOMIND[0-7]'` is invalid regex syntax — RHS of `-ot` is an exact buft name, not a regex.
- **Wrong**: NCCL `cmake/Modules/FindNCCL.cmake` — actual path is `ggml/cmake/FindNCCL.cmake`, already wired.
- **Wrong**: q8_0 KV ≈ 4× reduction over FP16 — actually ~2× (Q8_0 is 1 byte/value + 1 fp16 scale per 32 values; FP16 is 2 bytes/value).

### Codex draft (19 KB / 458 lines)
- Spine of the merge. Codex correctly identifies the three blockers:
  1. `LLAMA_SPLIT_MODE_ROW` blocked for deepseek4 in `src/llama-model.cpp:770-771`.
  2. `tm_ensure_loaded` + `g_state` are single-device singletons (real bug; not just a "verify" item).
  3. `-ot` regex × per-device buft seam — pins to one GPU; needs either generated overrides or a device-local family alias.
- Explicitly raises `model.has_tensor_overrides()` → pipeline parallel disabled.
- Honest VRAM math (60-76 GiB cluster-wide for KV after allocator + scratch + workspace).
- Conditional P6 row-split investigation with defer story.
- **Weak**: No day estimates per phase, no kill criterion on P6, NCCL `find_package` is slightly over-described as new work.

### Gemini draft (11 KB / 221 lines)
- Cleaner structure, day estimates per phase, K8s manifest naming.
- P4 (q8_0 KV) carved as its own phase, P5.2 scaling-efficiency sweep — both worth borrowing.
- DoD includes single-GPU non-regression — borrow.
- **Wrong**: Treats `LLAMA_SPLIT_MODE_ROW` as a flag-flip available for DSv4 — it throws.
- **Wrong**: NCCL `find_package` listed in Files Summary as new work — already wired.
- **Wrong**: TmLib singleton classified as "Likelihood: Low — Mitigation: mutex". Actually deterministic; mutex doesn't fix it.
- **Wrong**: `-ot 'exps.*=CUDA_TURBOMIND[0-3]'` as a single override — RHS doesn't take regex.
- **Wrong**: 100 GiB cluster slack treated as headroom — repeats the rejected optimistic math.
- **Wrong**: NCCL port-exposure security risk — irrelevant for single-node.

## Critiques: accepted vs rejected

| Critique | Source | Verdict |
|---|---|---|
| Codex names the three real blockers (ROW guard, singleton, -ot seam) | Codex | **Accepted** — basis for spine |
| Per-device TmLib refactor is mandatory P2 work, not "verify" | Codex+Claude on Gemini | **Accepted** — final sprint makes it a hard P2 gate |
| `-ot` × per-device family-alias problem | Codex+Claude on Gemini | **Accepted** — per-user interview, family alias is the chosen approach |
| VRAM cluster-slack math is wrong; budget per-GPU | Codex on Gemini, Claude self | **Accepted** — final uses per-GPU 28/30 GiB thresholds |
| q8_0 KV ≈ 2×, not 4× | Codex on Claude | **Accepted** — final clarifies |
| NCCL `find_package(NCCL)` is already wired | Claude on Codex+Gemini | **Accepted** — P1 reframes as image-packaging + link-verification, not CMake authoring |
| `NCCL_DEBUG=INFO` log artifact in P1 gate, not just `ldd` | Claude on Codex | **Accepted** |
| `ncclCommInitAll` enumerates all visible devices — pod GPU masking via CUDA_VISIBLE_DEVICES | Claude on Codex | **Accepted** — added as P0 visibility gate |
| Pipeline parallel disabled by tensor overrides | Codex (Claude verified at llama-context.cpp:316-321) | **Accepted** — per-user, family alias chosen specifically to avoid this |
| `rm -rf build/` to avoid CMake cache poisoning | Claude on Codex | **Accepted** as P1 prerequisite |
| Single-GPU non-regression check | Gemini DoD item 9 | **Accepted** — added to final DoD |
| 2/4/6/8 GPU scaling sweep | Gemini P5.2 | **Accepted** — added to P5 |
| q8_0 KV as its own phase | Gemini P4 | **Accepted** — added as P4 sub-phase |
| K8s manifest naming convention | Gemini | **Accepted** |
| `nvidia-smi topo -m` in P5 report | Claude | **Accepted** |
| `NCCL_LAUNCH_MODE` / `NCCL_P2P_DISABLE` fallback documented | Claude | **Accepted** as P0.4 risk capture |
| Build/runtime NCCL ABI mismatch risk | Claude | **Accepted** in risks |
| Per-device cold-start `cudaMalloc` of 256 MiB scratch × 8 = 2 GiB budget line | Claude | **Accepted** in VRAM accounting |
| Reproducible command-line block in REPORT-19 | Claude DoD item 10 | **Accepted** |
| Gemini P3.3 transition to `-sm row` inside P3 success path | Gemini | **Rejected** — per-user, ROW is P6 conditional |
| Gemini DoD item 3 (ROW + NCCL allreduce verified) | Gemini | **Rejected** — ROW is P6 conditional |
| Gemini risk #3 (race conditions, Likelihood Low) | Gemini | **Rejected and rewritten** — deterministic behavior; per-device state is the fix |
| Gemini NCCL port exposure as security risk | Gemini | **Rejected** — single-node, no inter-node nets |
| Claude `LLAMA_SPLIT_MODE_TENSOR` as legacy alias | Claude | **Rejected** — separate plumbing per Codex critique |
| Claude `-ot 'exps=CUDA_TURBOMIND[0-7]'` syntax | Claude | **Rejected** — both critiques caught the regex error |

## Interview refinements applied

1. **TM refactor → Option A**: per-device state inside libggml-turbomind.so. No C ABI change. `g_state` becomes `g_states[MAX_DEVICES]` keyed by cuda_device.
2. **Expert placement → family alias**: introduce `CUDA_TURBOMIND` (no device suffix) as a sentinel buft that resolves to the layer's assigned device buft at allocation time. Preserves pipeline parallelism (`model.has_tensor_overrides()` returns false because no tensor-specific override is applied). More work than generated overrides, but the right architectural choice.
3. **Row split → P6 conditional with kill criterion**: investigate after LAYER ships; abandon if more than 1 day of buffer-type-marriage work emerges.
4. **SPRINT-024 sequencing → 025 first**: capability-first sequencing. SPRINT-025 lands per-expert turbomind on multi-GPU; SPRINT-024 ships later as a perf optimization.

## Final phase structure

- **P0** — Hardware + pod variants + NCCL build image (1 day): 4-GPU + 8-GPU pod manifests; NCCL libs in image; topology capture (`nvidia-smi -L`, `nvidia-smi topo -m`); CUDA_VISIBLE_DEVICES = pod request count check.
- **P1** — NCCL build + link verification (1 day): `rm -rf build/`; `-DGGML_CUDA_NCCL=ON`; `ldd | grep libnccl`; smoke harness that calls `ggml_backend_cuda_allreduce_tensor` and grep `NCCL_DEBUG=INFO` for `Bootstrap`.
- **P2** — TmLib per-device refactor (2 days): per-device state inside `libggml-turbomind.so`; idempotent init per cuda_device; explicit "no global current device" gate; 2-device pack-and-dispatch smoke without re-init churn.
- **P3** — CUDA_TURBOMIND family-alias buft (2 days): new sentinel buft that resolves to layer's device buft at allocation; preserves pipeline parallelism. Smoke on MIN-16e with 4-GPU layer-split.
- **P4** — Full 256e layer-split landing + q8_0 KV decision (2 days): 8-GPU pod; `-sm layer -ngl 999 -ot exps=CUDA_TURBOMIND`; measure FP16 vs q8_0 KV per-GPU memory; pick whichever fits with ≥ 2 GiB headroom on the heaviest-shard GPU; greedy 32-token decode.
- **P5** — Measurement + REPORT-19 (1-2 days): `llama-bench` on the 8-GPU layer path; 2/4/6/8 GPU scaling sweep; per-GPU VRAM at load + decode peak; `nvidia-smi topo -m` capture; NCCL_DEBUG log archived.
- **P6** — Row-split investigation (conditional, kill criterion 1 day): lift the deepseek4 row guard; investigate row-split + CUDA_TURBOMIND buffer-type marriage; abandon if > 1 day of structural work needed. Document blocker.
- **P7** — Close-out (0.5 day): tag, follow-ups, memory updates.

Total: 9.5-11 days plus stretch on P6.

## Outcome contract

This is the FIRST run of 256e on the V100 stack. No comparable baseline → no hard perf gate. Sprint ships if:
- 8-GPU layer-split loads and decodes 256e coherently
- Per-device TURBOMIND verified actually executing on more than one GPU
- REPORT-19 captures TPS + VRAM + scaling sweep numbers + reproducible command line

Stop-loss: if Option A per-device refactor exposes a deeper turbomind library-level singleton, escalate as a Plan B re-design (Option B opaque-context) rather than continuing in a degraded state.
