# SPRINT-025 — Claude critique of Codex and Gemini drafts

**Date:** 2026-05-15
**Reviewer:** Claude (Opus 4.7)
**Inputs:** `SPRINT-025-INTENT.md`, `SPRINT-025-CODEX-DRAFT.md`, `SPRINT-025-GEMINI-DRAFT.md`
**Method:** read both drafts against intent, then verified the load-bearing technical claims against the codebase before scoring.

## TL;DR

**Codex** is the substantially stronger draft. It correctly identifies the three issues that will actually decide whether this sprint ships: the `deepseek4` `LLAMA_SPLIT_MODE_ROW` guard, the single-global `TmLib`/`g_state` device hop in turbomind, and the `-ot` regex × per-device buft seam. Its risk register and phase gates are written to surface those issues early.

**Gemini** is structurally cleaner and easier to read, but it gets several load-bearing technical facts wrong and treats this sprint as a flag-flip exercise. In particular it puts row-split TP on the P1 critical path (which is currently blocked by an arch guard), repeats the optimistic "256 − 156 = 100 GiB free" VRAM math the intent explicitly warned about, and dismisses the turbomind global-state issue as "Likelihood: Low" when it is in fact the deterministic behavior of the current code.

Codex is what I would execute against. Gemini supplies useful framing (esp. P4 q8_0 KV, scaling-efficiency sweep, K8s manifest naming) that can be borrowed back into the Codex skeleton.

---

## 1. Verified facts I used as a yardstick

Before grading either draft I checked each load-bearing claim:

| Claim | Verified at | Reality |
|---|---|---|
| `deepseek4` throws on `LLAMA_SPLIT_MODE_ROW` | `src/llama-model.cpp:770-771` | **TRUE** — hard `throw std::runtime_error("LLAMA_SPLIT_MODE_ROW not implemented for architecture 'deepseek4'")`. |
| `find_package(NCCL)` already wired | `ggml/src/ggml-cuda/CMakeLists.txt:184-190` | **TRUE** — already calls `find_package(NCCL)`, gates `GGML_USE_NCCL` define and link on `NCCL_FOUND`. Has a "Warning: NCCL not found" fallback path. |
| `FindNCCL.cmake` exists | `ggml/cmake/FindNCCL.cmake` | **TRUE** — already in repo. |
| `ncclCommInitAll` initializes for **all visible** devices | `ggml/src/ggml-cuda/ggml-cuda.cu:461-467` | **TRUE** — loops `0..info.device_count`, no per-pod awareness. Pod-level GPU masking must come from `CUDA_VISIBLE_DEVICES` / k8s, not the build. |
| `tm_ensure_loaded` holds one global `TmLib` with single `init_device` and **tears down on device hop** | `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu:60-101` | **TRUE** — `static TmLib t` singleton with `init_device` field. The "different device" branch calls `t.shutdown()` then `t.init(device)`. This is exactly the corruption surface Codex flags. |
| `ggml_turbomind_init(int cuda_device)` in `api.cc` holds a global `g_state` and frees `d_barriers / d_partials / d_flags` on device change | `ggml/vendor/turbomind/api.cc:121-167` | **TRUE** — single `g_state.initialized / .device` pair; hop frees workspace buffers globally. Any concurrent dispatch on the previously initialized device would be using freed pointers. |
| `model.has_tensor_overrides()` disables pipeline parallel | `src/llama-context.cpp:316-321` | **TRUE** — `pipeline_parallel = … && !model.has_tensor_overrides()`. Generated per-layer overrides therefore have a measurable cost. |

With those facts pinned, here's the head-to-head.

---

## 2. CODEX-DRAFT review

### 2.1 Strengths

1. **Names the three real blockers up-front, in section 2 / "Overview".** The Codex draft opens by stating that 4 GPUs cannot hold the model, that ROW currently throws for `deepseek4`, and that `CUDA_TURBOMIND` is not multi-device-safe. Each of those is verifiably true and each is decisive for the sprint outcome.
2. **Phase-gate ordering matches risk.** P2 ("multi-device-safe CUDA_TURBOMIND") sits *before* P3 (4-GPU smaller-model harness) and P4 (full 256e), so the most likely correctness hazard is forced to land before the model that would exercise it. This is the right ordering.
3. **Architecture §4 ("Per-GPU `CUDA_TURBOMIND` init is a required refactor") is precise.** It cites `tm_ensure_loaded` and `ggml_turbomind_init` by name, gives two refactor options (A: per-device state inside the API; B: per-device opaque context), and frames Option B as the cleaner long-term choice. This is exactly the surface that broke; the diagnosis is correct.
4. **Architecture §5 raises the `-ot` × multi-device problem nobody else would have caught.** `-ot 'exps=CUDA_TURBOMIND0'` pins to *one* GPU; it does not say "use the TURBOMIND buft belonging to the layer's device." Codex names two ways out and warns that the "generated per-layer overrides" approach disables pipeline parallel because `model.has_tensor_overrides()` flips the cparams flag in `llama-context.cpp`. That's the right level of detail and the citation is real.
5. **VRAM budget §6 is honest math.** It refuses the 256 − 156 = 100 GiB argument and budgets allocator/scratch and decode workspace as separate line items, leaving 60–76 GiB cluster-wide for KV. Per-GPU thresholds (28 GiB / 30 GiB) are testable and stop-the-line.
6. **DoD item 4 ("no global mutable `current device`") is a real, measurable acceptance criterion**, not "experts sharded across all GPUs" hand-waving.
7. **Risk table actually names the failure modes** ("exact-name `-ot` overrides pin experts to one GPU", "tensor overrides disable pipeline parallel"). These are not generic risks; they are mapped to code.

### 2.2 Weaknesses

1. **NCCL build wiring is slightly over-described as new work.** P1 reads as if `find_package(NCCL)` is something this sprint provisions. It's already in `ggml/src/ggml-cuda/CMakeLists.txt:184-190` and `ggml/cmake/FindNCCL.cmake:1` exists. The real work is *image packaging* (libnccl2 + dev headers in the build image) and *link verification*, which Codex says in prose but does not separate cleanly from the CMake side. A future executor reading P1 may waste a half day adding `find_package` that's already there.
2. **No timeline / no day estimates per phase.** Gemini at least puts (2 days / 3 days / …) on each phase. For a sprint that has six phases and an explicit conditional P6, having no rough estimate makes it hard to decide whether P6 stays in the sprint or moves out.
3. **No explicit `NCCL_CHECK` failure-mode discussion under GPU subset.** `ncclCommInitAll` enumerates all `info.device_count` devices visible to the process. If a 4-GPU pod is requested but `CUDA_VISIBLE_DEVICES` is misconfigured to expose more, NCCL will init for the wrong fleet and the run will hang. Codex hints at this via "the container view aligned with the pod request count" but doesn't make it a P0 gate item with a one-line `cudaGetDeviceCount == NCCL comm count` check.
4. **P6 conditional row-split work has no kill criterion.** "Choose one of: row split only for dense weights / new split-capable turbomind buffer semantics / explicit defer to SPRINT-026" is fine, but it doesn't say *under what TPS gap from LAYER* to abandon. Without a number, P6 will overrun.
5. **Doesn't address NCCL inter-GPU topology measurement.** Both drafts mention NVLink-2; neither asks for `nvidia-smi topo -m` to be captured into the report so we know whether `gpu-01` exposes the SXM2 NVLink mesh or a degraded subset. Codex includes `nvidia-smi topo -m` in P0 step 2 but doesn't ask the report to include it.
6. **Missing edge case: `ldd` will succeed on dlopen-only loaders.** A binary that `dlopen`s NCCL at runtime won't show in `ldd` and won't fail the gate. llama.cpp uses direct linkage so this is unlikely to bite, but a P1 gate that *only* relies on `ldd` is brittle; running with `NCCL_DEBUG=INFO` and grepping for `NCCL INFO Bootstrap` is the more reliable check. Codex partly says this; I'd promote it.
7. **No mention of the SPRINT-023 turbomind dlopen path being a second order issue.** `libggml-turbomind.so` is `dlopen`'d once via the global `TmLib`. Even after Option A ("per-device runtime state inside the turbomind API"), there is still only one `dlopen` handle — that is fine, but a future reader could read §4 as implying multiple dlopen sites. A one-line clarification ("one process-wide dlopen handle, per-device state behind it") would close that gap.
8. **No explicit decision about SPRINT-024 prerequisite.** Open Question #4 raises it but the body doesn't commit. The intent already said SPRINT-025 does not block on 024 shipping; Codex should restate that as a constraint instead of leaving it open.

### 2.3 Gaps in risk analysis

- **`NCCL_LAUNCH_MODE` and `NCCL_P2P_DISABLE` are absent.** On a node where the V100 SXM2 NVLink mesh is partial, `ncclAllReduce` can hang or silently fall back to PCIe and be slower than LAYER. The risk table should include "NCCL chooses a degraded transport" with `NCCL_DEBUG=INFO` capture as the mitigation, and `NCCL_P2P_DISABLE=1` as the fallback.
- **Multi-process build / runtime mismatch.** If `libnccl.so` ABI in the build image differs from the runtime pod's NCCL, init can fail with cryptic version errors. Mitigation: ship NCCL in the same image used at runtime, not an external mount.
- **Per-device memory pool contention.** ggml-cuda's pool allocator is per-device; with 8 GPUs and a large model, the worst-case headroom is dominated by the GPU that gets the largest single tensor. Risk table doesn't mention uneven layer-split skew (some layers — embeddings, MoE shared experts — are much larger than others).
- **Cold-start `cudaMalloc` failure mode.** `g_state.partials_size = 4096*4096*sizeof(float)*4` = 256 MiB scratch per device in `ggml_turbomind_init`. Codex's per-device refactor needs to budget 8 × 256 MiB = 2 GiB of scratch just for turbomind. That should be a line item in §6.

### 2.4 Missing edge cases

- What happens on **mixed device hop during prefill** if the per-device refactor lands but a single thread sets `cudaSetDevice` to GPU 7 and then dispatches a tensor whose buffer is on GPU 0? `tm_supports_type` and the buft context disagreeing about device is a real failure mode.
- **`-mg` (main GPU) flag interaction.** If `-mg 3` is set on an 8-GPU pod and pipeline parallel is on, the main GPU's allocator footprint differs from the others. Codex's per-GPU memory threshold should be "no non-main GPU above 28 GiB; main GPU above 28 GiB allowed by N GiB".
- **Build-cache poisoning.** SPRINT-019's effort-estimation memory flags sticky build-system caches as a risk. If the CMake cache from the single-GPU build is reused, `GGML_CUDA_NCCL` may not actually rebuild affected TUs. Recommend `rm -rf build/` in P1.

### 2.5 DoD completeness

The Codex DoD list (9 items) is complete in spirit but is missing:

- **A reproducible command line block** captured in REPORT-19. Item 9 is qualitative ("LAYER vs ROW decision"); a `# How to reproduce` shell block belongs as item 10.
- **An explicit "single-GPU smoke still passes" item.** Gemini DoD item 9 ("No regression on single-GPU 16e/32e paths") is correct and Codex omits it. This is a real risk: the per-device turbomind refactor touches the single-GPU hot path.
- **NCCL `NCCL_DEBUG=INFO` log artifact** archived as part of the report.

---

## 3. GEMINI-DRAFT review

### 3.1 Strengths

1. **Cleaner structure.** Numbered sections, time estimates per phase, K8s manifest names called out. Easier to brief and scan.
2. **P4 (Q8_0 KV) carved out as its own phase.** Codex folds this into "policy for the first landing"; Gemini's split lets you measure the q8_0 vs FP16 VRAM delta cleanly. Good idea, worth borrowing.
3. **P5.2 scaling-efficiency sweep (2 / 4 / 6 / 8 GPUs).** Codex doesn't ask for this; it's a useful diagnostic for whether NCCL is paying its way.
4. **Security §8 includes `IPC_LOCK` for NCCL pinned memory.** That's a real Kubernetes-level concern that Codex doesn't flag.
5. **Open Question 1 (NVLink mesh).** Genuinely worth verifying before depending on row-split TP.

### 3.2 Weaknesses (and where it's wrong)

#### Technical errors

1. **P1.2 "Row-Split Verification" is on the critical path but is currently impossible for DSv4-Flash.** The intent and `src/llama-model.cpp:770-771` both state that `LLAMA_SPLIT_MODE_ROW` throws for `deepseek4`. Gemini's "Load same model with `-sm row`" smoke step will throw immediately. The draft does try to thread the needle by saying "Load a smaller model (e.g., DSv4-Flash-AVG-16e)" in P1.1 — but all DSv4-Flash variants share `arch=deepseek4`, so the row guard fires on the small variant too. The verification path described in P1.2 is not runnable as written.
2. **NCCL build wiring is described as new work** ("Add `find_package(NCCL)`, link NCCL if `GGML_CUDA_NCCL=ON`" in the Files Summary). It is already there at `ggml/src/ggml-cuda/CMakeLists.txt:184-190`. This is the same gap Codex partly has, but Gemini commits to it in the Files table, which is a stronger error.
3. **Per-device CUDA_TURBOMIND init is described as already-working** ("SPRINT-023 implemented `ggml_backend_cuda_turbomind_buffer_type(int device)`. SPRINT-025 ensures … Expert packing/upload happens per-device"). The buft *enumeration* is per-device, but the underlying `TmLib` and `g_state` are single-device singletons that tear down on device hop. Gemini's P2 work plan does not include the refactor; it just verifies the *symptoms* of the bug it has not noticed. This is the largest single technical hole in the draft.
4. **Risk #3 "Per-device TURBOMIND init race conditions — Likelihood: Low — Mitigation: Sequential init or mutex protection in `ggml_turbomind_init`".** Wrong on three axes:
   - It's not a race. It's deterministic behavior of `tm_ensure_loaded`: device-hop unconditionally calls `t.shutdown()` followed by `t.init(device)`.
   - Likelihood is not "Low"; with 8 GPUs and any thread that calls `cudaSetDevice` between dispatches, it is the default behavior.
   - A mutex does not fix it — `g_state.mtx` is already there. The fix is per-device state, not serialization.
5. **VRAM accounting (§7 Risk #1) repeats the rejected math.** "100 GiB slack is plenty" is exactly the optimistic budget the intent told both drafts to avoid. The 156 GiB number is the *weights only*; allocator, turbomind scratch (per-device 256 MiB × 8 = 2 GiB), pipeline-parallel split buffers, KV at full context, and `ggml-cuda` pool fragmentation each take a percentage. 100 GiB drops to 60-ish quickly. Gemini's draft never carries that arithmetic.
6. **P3.3 "Transition to `-sm row`" inside P3** wires row-split into the success path of the full-model landing. If the `deepseek4` row guard isn't lifted first (which itself depends on the buffer-type / split-buffer-type marriage with `CUDA_TURBOMIND`), P3.3 cannot run. P3 will then look like a failure when in fact the upstream work was never staged.
7. **`-ot 'exps.*=CUDA_TURBOMIND[0-3]'` is treated as a single override that does the right thing.** It is not. The `-ot` flag pattern is `regex=BUFT_TYPE_NAME` — the RHS is one exact buffer-type-name string. `CUDA_TURBOMIND[0-3]` is not a regex on the RHS; it is the literal string `CUDA_TURBOMIND[0-3]` and won't match the buft names `CUDA_TURBOMIND0`, `CUDA_TURBOMIND1`, `CUDA_TURBOMIND2`, `CUDA_TURBOMIND3`. The placement mechanism Gemini relies on does not exist. Codex's §5 calls this out explicitly.
8. **`NCCL Port Exposure`** (Security §8.1). NCCL on a single node with no inter-node networking does not "use TCP for coordination" in any way that warrants K8s network isolation — the bootstrap is `SOCK_DGRAM`/`SOCK_STREAM` only when multi-host, and in this sprint everything is intra-node (intent §35: "Single node, no inter-node networking"). The mitigation is a non-issue. Not a critical error but it signals over-generic risk authoring.

#### Structural weaknesses

9. **No P0/P2 turbomind multi-device gate.** Codex makes this a hard phase with a gate; Gemini's P2 gate is "Experts distributed across 4+ GPUs on TURBOMIND buffers; coherent 32-token decode." That is a *consequence*, not a verification of the underlying refactor. A draft could pass this gate on a single thread that never crosses device boundaries between dispatches, and then fail at decode time once `ggml-cuda` actually interleaves devices.
10. **No P6 / no LAYER-vs-ROW decision artifact.** Gemini collapses row-split into P3.3 and P5.1 ("compare `-sm layer` vs `-sm row`"). The intent asked for "an explicit verdict" (Codex §Overview), with a defer path if blocked. Gemini's plan has no defer story — if `-sm row` cannot run, the sprint just has a missing measurement, not a documented blocker.
11. **DoD item 3 ("`LLAMA_SPLIT_MODE_ROW` verified working with NCCL allreduce") is a sprint gate that depends on lifting the `deepseek4` row guard, which is itself unscoped work.** Right now this item cannot be ticked off without code changes Gemini did not plan for. Codex correctly demotes this to conditional P6.

### 3.3 Gaps in risk analysis

- **Turbomind global-state risk is mis-graded** (covered above).
- **No risk for "`-ot` regex does not have the device-aware semantics we assumed"** — this is the single largest correctness landmine and is absent.
- **No risk for `model.has_tensor_overrides()` disabling pipeline parallel** — even if Gemini's P2 routing somehow worked, the consequence on the layer path is uncosted.
- **VRAM risk is misgraded "Likelihood: Low"** when the budget headroom is in the 20-40 GiB band after honest accounting, not the 100 GiB band.

### 3.4 Missing edge cases

- All of Codex's edge cases (above) apply to Gemini too — they are simply not addressed.
- Plus: Gemini does not mention SPRINT-024 sequencing at all. The intent has an explicit open question (#3) on whether 024 must land first; Gemini's draft would silently assume the current per-expert dispatch path on each GPU.

### 3.5 DoD completeness

- DoD item 5 ("`CUDA_TURBOMIND` experts sharded across all GPUs") cannot be ticked without the missing refactor.
- DoD item 9 ("No regression on single-GPU 16e/32e paths") is good and should be borrowed back into Codex.
- DoD has no operational reproducibility item (same gap as Codex).
- DoD has no NCCL log artifact (same gap).

---

## 4. Side-by-side

| Dimension | Codex | Gemini |
|---|---|---|
| Diagnoses the `deepseek4` row guard | yes, P6-conditional | wired into P1/P3 success path; will break |
| Diagnoses turbomind single-global state | yes, P2 refactor | flags only as a mutex/race risk; misses root cause |
| `-ot` × per-device buft seam | flagged with two solution paths | assumed to "just work" |
| VRAM accounting | honest, 60-76 GiB KV budget, per-GPU thresholds | repeats rejected 100 GiB slack |
| NCCL CMake wiring described as new work | partly | yes (Files Summary table) |
| NCCL runtime gate beyond `ldd` | `NCCL_DEBUG=INFO` smoke mentioned | `NCCL_DEBUG=INFO` mentioned in P1.2 |
| Time estimates per phase | no | yes (good) |
| Q8_0 KV as its own phase | folded into policy | yes (good) |
| 2/4/6/8 GPU scaling sweep | no | yes (good) |
| Pipeline-parallel disabled by overrides risk | yes | absent |
| SPRINT-024 sequencing position | open question, not committed | absent |
| DoD includes single-GPU non-regression | absent | yes |
| Risk register accuracy | high | low — three of five rows are mis-graded or wrong |

---

## 5. Recommendation

Use **Codex as the base** and merge in these specific pieces from Gemini:

1. **Per-phase day estimates** (Gemini §4 headers) so P6 has a kill criterion.
2. **P4 "Q8_0 KV as its own phase"** with explicit VRAM-delta measurement, instead of Codex's "set it as policy" framing.
3. **P5.2 "scaling efficiency sweep across 2 / 4 / 6 / 8 GPUs"** — useful diagnostic, low marginal cost.
4. **DoD item "No regression on single-GPU 16e/32e paths"** — the per-device turbomind refactor *will* touch the single-GPU hot path.
5. **K8s manifest naming `ds-v4-256e-{4,8}gpu.yaml`** — fine convention, adopt.

Reject these from Gemini:

1. Row-split TP on the critical path before the `deepseek4` row guard is dealt with.
2. The 100 GiB-slack VRAM framing.
3. Risk #3's mutex-fix-for-turbomind-init mitigation.
4. The `NCCL Port Exposure` security framing.
5. The `-ot 'exps.*=CUDA_TURBOMIND[0-3]'` placement assumption.

Additionally, add to Codex:

6. **P0 gate item: `nvidia-smi -L` count == `cudaGetDeviceCount` == `ncclCommInitAll` comm count.** All three must agree, with `CUDA_VISIBLE_DEVICES` set deliberately by the manifest. This prevents the "NCCL init for the wrong fleet" hang.
7. **Per-device turbomind scratch budget line item** (8 × 256 MiB = 2 GiB) in the VRAM accounting.
8. **`rm -rf build/`** before the first NCCL build to defeat the CMake-cache sticky-cache surface from SPRINT-019.
9. **REPORT-19 artifact list**: reproducible command-line block, `NCCL_DEBUG=INFO` log, `nvidia-smi topo -m` capture, per-GPU memory tables.
10. **P6 kill criterion**: if row-mode requires >2 days of new buffer-type plumbing or shows <1.2× over LAYER, defer.

---

## 6. One-paragraph executive answer

If we ship Codex's draft with the five Gemini borrowings above, this sprint has a credible path to landing the 256e model on 8 V100s. If we ship Gemini's draft as written, P1 will hit the `deepseek4` row guard on day one, P2 will pass without fixing the actual turbomind device-hop bug, and P3 will surface that bug under the heading of "model load fails" with no plan to recover.
