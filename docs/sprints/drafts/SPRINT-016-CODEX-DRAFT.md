# SPRINT-016 CODEX DRAFT

## Overview

Sprint 016 targets the next measurable step after Report 9: improve the V100 INT8 path without breaking the existing bit-correctness contract. The main line of work is a new `v9` INT8 kernel that keeps the `v4`/`v8` tile structure but replaces full-length FP32 accumulation with chunked FP16 accumulation plus periodic FP32 promotion. The sprint also closes two smaller decisions that Report 9 left open: whether INT4 should stay on LUT unpack or move back toward bitshift for some shapes, and whether INT4 `BN=256` can be made viable without unacceptable register spills.

The expected production outcome is conservative: `v4` remains the baseline winner until `v9` proves both faster and bit-correct. Sprint 016 succeeds if it produces a clear answer, not only if `v9` wins. That answer must be backed by fresh sweep CSVs, focused `ncu` captures, and a new `REPORT-10.md`.

## Use Cases

1. INT8 inference at `M in {256, 1024, 2048}`, `N=K=7168` should have a candidate path faster than `128x128x32_w4_v4` while staying within `rel <= 1e-3`.
2. INT8 inference at `M <= 64` should continue to use `32x64x32_w4_v3_b1` unless `v9` unexpectedly wins at small `M`; Sprint 016 should not regress the small-batch path.
3. INT4 kernels should end the sprint with an evidence-backed unpack decision for the large-shape path: keep LUT, revive bitshift for some tiles, or document that bitshift is still inferior.
4. The harness should be able to regenerate reproducible artifacts for the next report from one build tree and one dev pod, without ad hoc local edits.
5. If multi-shape validation is taken on, the harness should prove whether the current winners generalize beyond the square `7168x7168` case used in Reports 1-9.

## Architecture

The control plane stays in `tools/tc-grid/src/main.cu`. It owns the sweep loop, tile catalog, CLI parsing, CSV output, and cuBLAS ceiling rows. Sprint 016 should keep that structure and add only the minimum new surface needed to test `v9` and, if required, a small shape-expansion path.

The main datapath change is a new header, `tools/tc-grid/kernels/v9_kernels.cuh`, with an `int8_v9` namespace parallel to `int8_v4` and `int8_v8`. The initial `v9` family should match the current winning INT8 tiles:

- `128x128x32_w4_v9`
- `128x64x32_w4_v9`
- `64x128x32_w4_v9`
- `64x64x32_w4_v9`

`v9` should start from the `v8` idea, because `v8` already proved that shrinking `FragC` reduces register pressure and removes spills. The difference is that `v9` cannot leave the accumulator in FP16 for the entire `K=7168` reduction. The kernel should instead:

1. Use `wmma::fragment<accumulator, 16, 16, 16, half>` for the inner MMA loop.
2. Run a short chunk of K-tiles, with `CHUNK_K=4` as the default first attempt.
3. Promote the FP16 fragment contents into a separate FP32 accumulator after each chunk.
4. Reset the FP16 fragment and continue.

This keeps the register-pressure improvement from `v8` for most of the loop while restoring a bounded FP32 accumulation path often enough to satisfy the existing tolerance gate.

`tools/tc-grid/src/launch_int8.cu` remains the dispatch layer. It should gain a `#include "v9_kernels.cuh"`, a `version == 9` branch for shared-memory sizing, and `LAUNCH_V9` / `RERUN_V9` macros parallel to the existing `v4` and `v8` blocks. The first sprint cut should support only `UnpackPath::LUT` for `v9`, with explicit `SKIP` notes for unsupported paths, matching the pattern already used for `v4`-`v8`.

The INT4 work should stay inside the existing split between `tools/tc-grid/src/launch_int4.cu`, `tools/tc-grid/kernels/int4_kernels.cuh`, `tools/tc-grid/kernels/int4_bitshift_kernels.cuh`, and `tools/tc-grid/kernels/v3_kernels.cuh`. The goal is not a new INT4 architecture. The goal is to determine whether the present LUT path is still the right default and whether the `128x256` v3 instantiations can be made practical by reducing live ranges or fragment pressure.

If multi-shape validation is included, prefer a small CLI extension in `tools/tc-grid/src/main.cu` over a second standalone tool. The cleanest cut is either:

- add `--n-list` and `--k-list`, or
- add `--shape-list MxNxK,...`

The current `--nk` interface only covers square shapes.

## Implementation

1. Rebuild the current baseline inside the existing `tcg-dev` pod and regenerate a known-good reference set:

```bash
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite
cmake -S /src/tools/tc-grid -B /src/tools/tc-grid/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build /src/tools/tc-grid/build -j
/src/tools/tc-grid/build/tc-grid --m-list 64 --nk 7168 --dist uniform_small > /src/tools/tc-grid/docs/run_M64_v2.csv
/src/tools/tc-grid/build/tc-grid --m-list 256 --nk 7168 --dist uniform_small > /src/tools/tc-grid/docs/run_M256_v2.csv
/src/tools/tc-grid/build/tc-grid --m-list 1024 --nk 7168 --dist uniform_small > /src/tools/tc-grid/docs/run_M1024_v2.csv
/src/tools/tc-grid/build/tc-grid --m-list 2048 --nk 7168 --dist uniform_small > /src/tools/tc-grid/docs/run_M2048_v2.csv
```

2. Add `tools/tc-grid/kernels/v9_kernels.cuh` by copying the `int8_v8` structure, then changing the accumulator scheme to chunked FP16 plus FP32 promotion. Keep the same CTA geometry as `v4` and `v8`, the same `BK=32`, and the same `BK_PAD = BK + 8` shared-memory layout for the first cut. Do not mix in XOR-swizzle or manual fragment loads in this sprint.

3. Wire `v9` into `tools/tc-grid/src/launch_int8.cu`:
   - include the new header,
   - add `smem_bytes_opt_v9` alongside the `v8` case,
   - add `LAUNCH_V9` and `RERUN_V9`,
   - keep `path == LUT` only,
   - return `R.note = "v9 path-B not implemented"` for bitshift.

4. Register the new tiles in `tools/tc-grid/src/main.cu`. Mirror the current `v4` set first:
   - `128x64x32_w4_v9`
   - `128x128x32_w4_v9`
   - `64x64x32_w4_v9`
   - `64x128x32_w4_v9`

5. Run the full INT8 sweep again and compare `v9` against `v4`, `v3`, and `v3_b1`. The acceptance bar is:
   - `maxabs <= 0.1`
   - `p99 <= 0.05`
   - `rel <= 1e-3`
   - faster than `128x128x32_w4_v4` at `M=1024` or `M=2048`

6. Profile only the relevant INT8 kernels with `ncu`, using kernel-name filtering so the harness can stay unchanged:

```bash
ncu --csv --page raw \
  --kernel-name-base demangled \
  --kernel-name regex:mm_int8_lut_v(4|9) \
  --metrics launch__registers_per_thread,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg.pct \
  --log-file /src/tools/tc-grid/docs/ncu_M2048_int8_v4_v9.csv \
  /src/tools/tc-grid/build/tc-grid --m-list 2048 --nk 7168 --dist uniform_small
```

Repeat at `M=256` if the `M=2048` result is promising, and record whether the `v9` gain comes from lower register count, fewer spills, or better scoreboard behavior.

7. Run the focused INT4 decision pass. First compare current LUT and bitshift baselines at a shape where INT4 matters:

```bash
/src/tools/tc-grid/build/tc-grid --m-list 256,2048 --nk 7168 --dist uniform_small > /src/tools/tc-grid/docs/run_int4_focus_v2.csv
ncu --csv --page raw \
  --kernel-name-base demangled \
  --kernel-name regex:mm_int4_.* \
  --metrics launch__registers_per_thread,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg.pct \
  --log-file /src/tools/tc-grid/docs/ncu_int4_focus_v2.csv \
  /src/tools/tc-grid/build/tc-grid --m-list 256,2048 --nk 7168 --dist uniform_small
```

If the data shows the `128x256` v3 path is still spill-bound, reduce scope and document the block rather than forcing a fragile fix. If a fix is attempted, keep it local to the existing `int4_v3` path and re-measure immediately.

8. If P0 finishes early, add multi-shape validation by extending `tools/tc-grid/src/main.cu` to accept non-square shapes and run at least two additional DSv4-relevant tuples. The sprint draft should assume this is conditional work, not a gate for the `v9` kernel itself.

9. Write `tools/tc-grid/docs/REPORT-10.md` with:
   - best config per `(format, M)` after Sprint 016,
   - `v4` vs `v9` table for INT8,
   - `ncu` stall breakdown for the winning INT8 shape,
   - INT4 LUT-vs-bitshift conclusion,
   - a clear negative-result section if `v9` fails.

10. Re-enable the node exporter once profiling is complete:

```bash
kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=true --overwrite
```

## Files Summary

| File | Change |
| --- | --- |
| `docs/sprints/drafts/SPRINT-016-CODEX-DRAFT.md` | Planning artifact for this sprint. |
| `tools/tc-grid/kernels/v9_kernels.cuh` | New INT8 mixed-precision chunked-accumulation kernel family. |
| `tools/tc-grid/src/launch_int8.cu` | `v9` include, dispatch, rerun timing path, and `smem` sizing. |
| `tools/tc-grid/src/main.cu` | Add `v9` tiles; optionally add non-square shape CLI support. |
| `tools/tc-grid/include/tc_grid.h` | Update comments if `LaunchSpec.version` or CLI behavior needs clarification. |
| `tools/tc-grid/src/launch_int4.cu` | INT4 profiling and any local spill-mitigation follow-up. |
| `tools/tc-grid/kernels/v3_kernels.cuh` | Only touched if INT4 `128x256` needs a live-range or fragment-pressure fix. |
| `tools/tc-grid/docs/run_M64_v2.csv` | Fresh baseline plus `v9` results for `M=64`. |
| `tools/tc-grid/docs/run_M256_v2.csv` | Fresh baseline plus `v9` results for `M=256`. |
| `tools/tc-grid/docs/run_M1024_v2.csv` | Fresh baseline plus `v9` results for `M=1024`. |
| `tools/tc-grid/docs/run_M2048_v2.csv` | Fresh baseline plus `v9` results for `M=2048`. |
| `tools/tc-grid/docs/ncu_M2048_int8_v4_v9.csv` | Focused `ncu` capture for the large-shape INT8 comparison. |
| `tools/tc-grid/docs/ncu_int4_focus_v2.csv` | Focused `ncu` capture for the INT4 unpack decision. |
| `tools/tc-grid/docs/REPORT-10.md` | Sprint 016 retrospective and recommendation document. |

## Definition of Done

1. `v9` exists in `tools/tc-grid/kernels/v9_kernels.cuh` and is wired through `launch_int8.cu` and `main.cu`.
2. `v9` passes the existing tolerance contract at `M in {64, 256, 1024, 2048}`, `N=K=7168`.
3. At least one `v9` tile is faster than `128x128x32_w4_v4` at a production-relevant shape, or the sprint documents that `v9` is a measured negative result.
4. `run_M64_v2.csv`, `run_M256_v2.csv`, `run_M1024_v2.csv`, and `run_M2048_v2.csv` are produced from the updated binary.
5. Focused `ncu` evidence exists for `v4` vs `v9` at `M=2048`.
6. The INT4 LUT-vs-bitshift question ends with a documented keep/drop decision.
7. The INT4 `BN=256` investigation ends with either a working measured variant or a concrete blocker note tied to register pressure or spills.
8. `REPORT-10.md` summarizes results, including negative findings and next-sprint recommendations.

## Risks

- `v9` may recover correctness but lose the `v8` speedup once FP32 promotion state is added back, leaving no win over `v4`.
- The best `CHUNK_K` may vary by tile, which can expand scope if the team tries to tune every shape instead of one production winner.
- `ncu` on the shared GPU can skew timings or fail if DCGM is not disabled first.
- INT4 `BN=256` may still be structurally register-bound on V100, producing only a documentation outcome.
- Extending the CLI for non-square shapes can become a side quest; keep it conditional unless the INT8 work lands early.

## Security

This sprint does not add any network service, model loader, RPC surface, or credential path. All work stays inside the CUDA benchmark harness under `tools/tc-grid/` and runs on the existing `tcg-dev` pod against a single V100.

The operational security concern is cluster hygiene, not application attack surface. Profiling requires temporarily disabling the GPU node exporter on `gpu-01`; the sprint must explicitly restore it after `ncu` runs. Output artifacts should remain in the repo workspace under `tools/tc-grid/docs/` and should not include secrets or kube credentials.

## Dependencies

- V100 `sm_70` access on `gpu-01`.
- Existing `tcg-dev` pod with `/srv/dev/dsv4-cuda/deepseek-sprint017` mounted at `/src`.
- `cmake >= 3.18`, CUDA toolkit, and cuBLAS, matching `tools/tc-grid/CMakeLists.txt`.
- `ncu` available in the pod or node environment.
- `kubectl` access to label `gpu-01` before and after profiling.
- Existing harness files: `tools/tc-grid/src/main.cu`, `tools/tc-grid/src/launch_int8.cu`, `tools/tc-grid/src/launch_int4.cu`, `tools/tc-grid/kernels/v3_kernels.cuh`, `tools/tc-grid/kernels/v4_kernels.cuh`, `tools/tc-grid/kernels/v8_kernels.cuh`.

## Open Questions

1. Is `rel <= 1e-3` still the only correctness gate for `v9`, or does Sprint 015 introduce any stricter per-direction tolerance requirement that should be mirrored here?
2. Should `v9` be limited to the four `v4`-mirror tiles in Sprint 016, or should the BM=32 `v3_b1` shapes also get a `v9` cut if the first implementation works?
3. For artifact naming, should Sprint 016 write `run_M*_v2.csv` siblings or overwrite `run_M*.csv` after preserving the Report 9 versions elsewhere?
4. Is non-square DSv4 validation required for sprint acceptance, or only a conditional stretch goal once the INT8 winner is known?
5. If `v9` closes less than half of the remaining gap to cuBLAS, does the next sprint move to XOR-swizzle/manual fragment loads, or do we stop after documenting the practical V100 ceiling?
