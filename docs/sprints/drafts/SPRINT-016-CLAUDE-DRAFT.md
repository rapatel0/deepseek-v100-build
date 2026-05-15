# SPRINT-016 — V100 INT8/FP4/FP8 throughput (v9 mixed-precision + INT4 unblocks)

Status: DRAFT
Date drafted: 2026-05-13
Hardware: V100 SXM2 32GB (gpu-01, sm_70)
Pod: `tcg-dev` (hostpath `/srv/dev/dsv4-cuda/deepseek-sprint017` mounted at `/src`)
Working branch: `sprint-016-tensor-unlock`
Tolerance contract: per-row `maxabs ≤ 0.1 ∧ p99 ≤ 0.05 ∧ rel ≤ 1e-3` (v3 baseline rel ≈ 2.6e-4)

## Overview

Report 9 closed Tier B and put calibrated numbers on the gap: the best bit-correct INT8 kernel `128x128x32_w4_v4` reaches **27.7 TFLOPS at M=2048** vs a cuBLAS FP16 ceiling of **98.28 TFLOPS** on N=K=7168. That is 28% of cuBLAS / ~32% of practical TC peak. ncu on v3/v4 best attributes the dominant stall to register-dependency pressure (Short Scoreboard 30.5%) — not bank conflicts (Long Scoreboard 22.9%) and not barrier wait (Bar 4.1%, vs the 25% prior reports claimed). v8 (FP16 accumulator) eliminated v3's 1540 B spill and ran 3.2% faster, but lost mantissa over the K=7168 reduction (`rel=5.5e-3`) and broke the tolerance contract.

This sprint takes three concrete swings at that register-pressure stall while keeping bit-correctness:

- **W1 (P0) — v9 chunked mixed-precision.** FP16 `c_frag` for inner WMMA accumulation; promote into an FP32 `c_acc` register file every `CHUNK_K` K-tiles. Targets the Short Scoreboard stall directly.
- **W2 (P0) — INT4 LUT vs bitshift A/B.** ncu-instrumented head-to-head; produce a drop/keep/replace decision for the production path.
- **W3 (P0) — INT4 BN=256 spill remediation.** Re-introduce the BN=256 tile (currently dropped from the grid for a 1664 B spill) by shrinking simultaneous `c_frag` residency.

P1 add-ons (W4 MoE-shape generalization, W5 persistent-CTA revisit) are pursued only if Phase 1-3 finish early. P2 items (W6 XOR swizzle, W7 partial-c SMEM stash) are deferred to SPRINT-017 unless W1+W2+W3 close less than 50% of the cuBLAS gap.

Sprint exit is a `REPORT-10.md` quantifying the closure and an updated per-(format, M) dispatch table.

## Use Cases

- **DSv4-Flash INT8 inference path.** v4 is the winning INT8 kernel at M ≥ 256 (Report 9). This sprint either lifts that ceiling with v9 or documents the floor with hard numbers.
- **Small-M MoE expert dispatch.** `32x64x32_w4_v3_b1` wins at M ≤ 64 across all formats. W4 verifies this generalizes to non-(N=K=7168) shapes.
- **INT4 weight-only quantization.** W2 is a precondition to picking the production INT4 path; W3 is the only remaining tile that could meaningfully lift INT4 throughput at large M.
- **FP4/FP8 scale-aware paths.** No new kernel work in scope. If v9 lands cleanly on INT8, the pattern is portable to `fp4_kernels.cuh` / `fp8_kernels.cuh` as a follow-up sprint.

## Architecture

### Kernel layering (unchanged)

```
tools/tc-grid/
├── kernels/
│   ├── v3_kernels.cuh             BK=32, padded B SMEM, FP32 c_frag (winner: INT4/FP4/FP8)
│   ├── v4_kernels.cuh             v3 + Tier S (L2 prefetch + __launch_bounds__ + __ldg) (winner: INT8)
│   ├── v5..v8_kernels.cuh         Closed exploratory variants, kept gated
│   ├── v9_kernels.cuh             NEW — v4 layout, FP16 c_frag, FP32 c_acc, CHUNK_K promote
│   ├── int4_kernels.cuh           Path-A: 16-entry LUT nibble unpack
│   └── int4_bitshift_kernels.cuh  Path-B: bitshift + sign-extend nibble unpack
└── src/
    ├── launch_int8.cu             dispatch v3/v4/.../v9; per-version smem_bytes calc
    ├── launch_int4.cu             dispatch path-A vs path-B
    ├── launch_fp4.cu / launch_fp8.cu
    └── main.cu                    shape list, dist sweep, per-row tolerance gate
```

### v9 design (chunked mixed-precision)

Start from v4 (not v3): v4 is the winning baseline at production shape and brings Tier-S goodies (`__launch_bounds__`, L2 prefetch, `__ldg`) for free. Conceptual diff:

```
using FragC = wmma::fragment<accumulator, 16, 16, 16, half>;   // was float
FragC c_frag[FRAG_M][FRAG_N];           // FP16 accum, 4 regs/elem
float c_acc [FRAG_M][FRAG_N][8];        // FP32 promoted, lives across K-tiles

for (k_tile = 0; k_tile < K_TILES; ++k_tile) {
    // existing v4 load/swizzle/mma_sync
    wmma::mma_sync(c_frag[m][n], a_frag, b_frag, c_frag[m][n]);

    if ((k_tile + 1) % CHUNK_K == 0) {
        // Promote: c_acc += float(c_frag); zero c_frag.
        #pragma unroll
        for (i = 0; i < c_frag[m][n].num_elements; ++i)
            c_acc[m][n][i] += __half2float(c_frag[m][n].x[i]);
        wmma::fill_fragment(c_frag[m][n], __float2half(0.f));
    }
}
// Tail-promote any residual c_frag if K_TILES % CHUNK_K != 0.
// Epilogue writes c_acc → SMEM → gmem (identical to v3/v4 FP32 epilogue).
```

Tuning parameters, all settled by measurement:

- **`CHUNK_K`** — template arg. Default 4. Sweep ∈ {2, 4, 8, 16}; smallest CHUNK_K that holds `rel ≤ 1e-3` at K=7168 across `uniform_small` / `uniform_wide` / `adversarial` is the winner.
- **Register budget** — v8 (half c_frag only, no c_acc) had zero spill at 194 regs. v9 adds c_acc with the same shape as v3's FP32 c_frag (8 regs/elem). Steady-state pressure ≈ `half c_frag (4) + float c_acc (8) = 12 regs/elem`, vs v3's 8. The bet: scoreboard stalls drop more than enough to pay. If `c_acc` itself spills, fall back to W7-light (stash `c_acc` to SMEM between chunks).
- **`FragA / FragB`** unchanged from v4 (`half` row-major / `half` col-major).

### INT4 LUT vs bitshift (W2)

Both paths exist (`int4_kernels.cuh` LUT, `int4_bitshift_kernels.cuh` bitshift); dispatch is selected by `s.path == UnpackPath::LUT|BITSHIFT` in `launch_int4.cu`. Sprint adds:

1. `scripts/ncu_int4_paths.sh` — single-call ncu profile for both paths at M ∈ {64, 256, 1024, 2048} on the winning tile `128x64x32_w4_v3`.
2. Three metrics drive the call: `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `launch__registers_per_thread`, `smsp__warp_issue_stalled_short_scoreboard_per_inst_executed.ratio`.
3. Decision rule (recorded in REPORT-10):
   - >5% TFLOPS difference at M=2048 AND no >5% regression at M=64 → drop the loser, simplify the launcher.
   - Within 5% → keep both behind an `--int4-unpack {lut,bitshift}` flag; document the tradeoff.

### INT4 BN=256 spill remediation (W3)

Spill cause is likely the simultaneous live set of `FRAG_M=8 × FRAG_N=4 = 32` c_frags across the K-loop. Try, in order:

1. **Reorder WMMA loop** so only `FRAG_N` c_frags are live at a time — write partial sums to SMEM between FRAG_N stripes. Costs SMEM traffic, frees ~12-16 regs.
2. **Apply `__launch_bounds__(WARPS*32, 2)`** (currently INT4 v3 lacks them); force nvcc into a 2-CTA register cap.
3. **Half-epilogue split** — split BN dimension into halves; epilogue runs twice with half the c_frag bank each time.
4. If none reach zero spills, document the blocker in REPORT-10 with the exact PTX counter that didn't move.

Success = bit-correct BN=256 INT4 tile that beats current BN=128 winner by ≥5% at M=2048. Documented blocker is an acceptable Phase-3 outcome.

### Harness changes

- `main.cu` — add 2-3 MoE shape entries for W4 (Phase 5 only). Current sweep is hard-coded N=K=7168.
- `launch_int8.cu` — register v9 dispatch (mirror v4 macro block at line ~178). Add `s.version == 9` to the `smem_bytes` switch (line ~83) and the dispatch switch (line ~374).
- Per-row tolerance gate already exists; no harness change needed.

## Implementation

Phases run sequentially on `tcg-dev`. No Jobs; iterate against the live pod.

### Phase 0 — branch + warm-up (≤ half day)

- Stay on `sprint-016-tensor-unlock`; confirm clean working tree against `main`.
- `kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite`; verify DCGM-exporter pod terminates (`kubectl get pods -n gpu-operator`).
- Rsync repo to `/srv/dev/dsv4-cuda/deepseek-sprint017`; `chown -R` inside the pod to mitigate the `com.apple.provenance` xattr issue.
- Build `make -C /src/tools/tc-grid -j$(nproc)`; reproduce Report 9 baseline:
  `./tc-grid --m-list 64,256,1024,2048 --nk 7168 --dist uniform_small > docs/run_M_baseline.csv`. Confirm within ±1% of Report 9 numbers before any kernel change.

### Phase 1 — W1: v9 mixed-precision INT8 (2-3 days, P0)

1. `cp tools/tc-grid/kernels/v4_kernels.cuh tools/tc-grid/kernels/v9_kernels.cuh`; rename namespace `int8_v4 → int8_v9`, kernel `mm_int8_lut_v4 → mm_int8_lut_v9`.
2. Change `FragC` to `wmma::fragment<accumulator, 16, 16, 16, half>`. Add `float c_acc[FRAG_M][FRAG_N][8]`. Zero both. Add `template <..., int CHUNK_K = 4>`.
3. Wrap existing mma_sync loop with modulo-`CHUNK_K` promote block (see Architecture). Add tail-promote after the K-loop.
4. Replace epilogue's `c_frag → SMEM` with `c_acc → SMEM` (FP32, identical to v3/v4 layout).
5. Wire into `launch_int8.cu`: `#include "v9_kernels.cuh"`, add `s.version == 9` branches to `smem_bytes` switch (line ~83) and the dispatch switch (line ~374); copy the v4 dispatch macro at line ~178 as v9, parameterized by `CHUNK_K`.
6. Add tile entries to `kTiles[]` in `main.cu`:
   ```
   { 128, 128, 32, 4, 8, 2, 9, "128x128x32_w4_v9_c4"  },  // CHUNK_K=4
   { 128, 128, 32, 4, 8, 2, 9, "128x128x32_w4_v9_c8"  },  // CHUNK_K=8
   { 128, 128, 32, 4, 8, 2, 9, "128x128x32_w4_v9_c16" },  // CHUNK_K=16
   ```
7. Smoke test: `./tc-grid --m-list 256 --nk 7168 --dist uniform_small`. Confirm bit-correctness gate. If `rel > 1e-3`, drop `CHUNK_K` and retry.
8. Full sweep INT8 only: `./tc-grid --m-list 64,256,1024,2048 --nk 7168 > docs/run_M_v9_int8.csv`.
9. ncu top-1 v9 vs top-1 v4 at M=2048:
   ```
   ncu --set full --csv \
     --metrics smsp__warp_issue_stalled_short_scoreboard_per_inst_executed.ratio,\
   smsp__warp_issue_stalled_long_scoreboard_per_inst_executed.ratio,\
   sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
   launch__registers_per_thread,smsp__cycles_active.avg \
     ./tc-grid --m-list 2048 --nk 7168 > docs/ncu_M2048_v9_int8.csv
   ```

Exit criteria for Phase 1: rel ≤ 1e-3 at every M, AND (≥3% TFLOPS gain over v4 at M=2048 with no regression at M=64) OR a clear "no win, here's why" with ncu evidence.

### Phase 2 — W2: INT4 LUT vs bitshift (1 day, P0)

1. `scripts/ncu_int4_paths.sh` — wraps the ncu metric call above, runs both paths at four M values.
2. Promote `/tmp/parse_ncu.py` → `scripts/parse_ncu.py`; check in.
3. Full sweep both paths: `./tc-grid --m-list 64,256,1024,2048 --nk 7168 --int4-unpack lut,bitshift > docs/run_M_int4_paths.csv`.
4. ncu top-1 LUT and top-1 bitshift at M=2048 → `docs/ncu_M2048_int4_{lut,bitshift}.csv`.
5. Decision lands in REPORT-10 §"INT4 unpack path".

### Phase 3 — W3: INT4 BN=256 spill remediation (1-2 days, P0)

1. Re-enable the BN=256 INT4 tile in `main.cu` (currently dropped at `src/main.cu:89`).
2. Baseline ncu on the unmodified BN=256 v3 tile → `docs/ncu_M2048_int4_bn256_baseline.csv`. Capture spill bytes + reg count.
3. Apply remediation steps in the order listed in Architecture/W3. After each, re-profile.
4. If spills hit zero (or below the v4 INT8 floor): add tile to grid; full sweep; commit. If blocked: 5-10 line "blocked" note in REPORT-10 citing the unmoved PTX counter.

### Phase 4 — P1 work (only if time remains)

- **W4**: extend `main.cu` shape list with `{N=18432, K=7168}` (FFN1), `{N=7168, K=18432}` (FFN2), `{N=128, K=7168}` (gate). Verify these shapes against the integration team before running. Run v4 / v9 / v3_b1 winners only — not the full grid.
- **W5**: graft v9's chunked-promote into `v5_kernels.cuh` (persistent-CTA). Hypothesis: persistent-CTA may amortize promote cost across more output tiles. Stretch only.

### Phase 5 — REPORT-10 + dispatch table (half day)

`tools/tc-grid/docs/REPORT-10.md`, structure mirroring REPORT-9:

- TL;DR (one paragraph; new winner per (format, M)).
- ms/TFLOPS table at M ∈ {64, 256, 1024, 2048}, all formats, v9 column.
- ncu stall breakdown v9-best vs v4-best (columns: TC%, WARP%, ShtSb%, LngSb%, BarrSt%, Regs, BankConf).
- INT4 unpack path decision + numbers.
- INT4 BN=256 outcome (working tile or blocker note).
- Multi-shape generalization (if Phase 4 ran).
- Corrections to prior reports if applicable.
- Recommended actions for SPRINT-017.

Update `tools/tc-grid/docs/PRIORITIZED-PLAN.md` (or replace with a per-(format, M) dispatch table) to reflect production winners. Promote v4 as default INT8 kernel in any DSv4-Flash integration path still defaulting to v3 (Report 9 recommendation 1).

Re-enable DCGM-exporter: `kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter-`.

## Files Summary

New:

- `tools/tc-grid/kernels/v9_kernels.cuh` — chunked mixed-precision INT8 (~250-300 lines).
- `scripts/ncu_int4_paths.sh` — W2 A/B harness.
- `scripts/parse_ncu.py` — promoted from `/tmp/parse_ncu.py`.
- `tools/tc-grid/docs/REPORT-10.md`.
- `tools/tc-grid/docs/run_M_v9_int8.csv`, `run_M_int4_paths.csv`, `ncu_M2048_v9_int8.csv`, `ncu_M2048_int4_{lut,bitshift}.csv`, `ncu_M2048_int4_bn256_baseline.csv`.

Modified:

- `tools/tc-grid/src/launch_int8.cu` — `#include "v9_kernels.cuh"`; smem_bytes case for v9 (line ~83); dispatch macro mirroring v4 at line ~178; dispatch switch addition at line ~374.
- `tools/tc-grid/src/main.cu` — `kTiles[]` extended with v9 entries + INT4 BN=256 re-enable (line ~89); optional shape-list extension for Phase 4.
- `tools/tc-grid/kernels/int4_kernels.cuh` (or derived `int4_v3b_kernels.cuh`) — BN=256 spill fix from W3.
- `tools/tc-grid/src/launch_int4.cu` — optional `--int4-unpack` flag wiring if decision is "keep both".

Sprint planning:

- `docs/sprints/drafts/SPRINT-016-CLAUDE-DRAFT.md` (this file).
- `docs/sprints/SPRINT-016.md` — finalized after interview (not produced in this draft).

## Definition of Done

1. `kernels/v9_kernels.cuh` exists, compiles clean, runs end-to-end on `tcg-dev`.
2. v9 INT8 either clears `rel ≤ 1e-3` at K=7168 across `uniform_small` / `uniform_wide` / `adversarial` and is benchmarked at M ∈ {64, 256, 1024, 2048}, OR is documented as failing tolerance with a per-`CHUNK_K` rel report.
3. INT4 LUT-vs-bitshift comparison committed: full sweep CSV, ncu CSVs for top-1 each, drop/keep/replace decision in REPORT-10.
4. INT4 BN=256 investigation committed: baseline ncu CSV, at least one remediation attempted, either a working tile entry OR a documented blocker.
5. `run_M{64,256,1024,2048}.csv` v2 regenerated with v9 + INT4 path-B rows.
6. REPORT-10 written; all eight mandatory sections populated.
7. No regression vs REPORT-9 winners at any (format, M) cell — every existing row still passes the tolerance gate and is at least as fast.
8. Memory updated if Phase 1 produces a fact that should propagate to future kernel work (e.g. "CHUNK_K=4 is the sweet spot regardless of M").
9. DCGM-exporter re-enabled on gpu-01 at sprint end.

Hard gate: anything failing the tolerance contract does NOT replace a production winner, regardless of TFLOPS.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| v9 promote doubles c_acc regs; Regs/thread > 200 → occupancy drop eats the win | M | H | Profile after Phase 1 step 5; if regs > 200, restrict to `CHUNK_K=4`, tighten `__launch_bounds__`, or fall back to W7-light (stash c_acc to SMEM) |
| v9 mantissa loss exceeds 1e-3 at all `CHUNK_K` | M | M | Document negative result; v9 may be a wash; flag W6 for SPRINT-017 |
| INT4 BN=256 spill is structural (compiler can't avoid it) | M | L | Phase 3 designed to fail fast; "blocked" with PTX counter is an acceptable outcome |
| ncu on gpu-01 conflicts with DCGM-exporter or other tenants | L | M | Phase 0 pauses DCGM via kubectl label; check `kubectl get pods -n gpu-operator` for stragglers before each ncu run |
| Tile additions push `kTiles[]` past harness compile-time switch limits (long compile times) | M | L | Compile in pod with `make -j$(nproc)`; gate non-winning configs out of Phase 1 sweep if compile > 5 min |
| MoE shape generalization (W4) flips the winner from v4 to something else at non-7168 shapes | L | M | Restrict P0 scope to canonical shape; W4 is informational |
| rsync ownership / `com.apple.provenance` xattr breaks pod build | L | L | Phase 0 chown step; documented mitigation in intent doc |

## Security

No external network surface added. No new secrets, credentials, third-party deps, or data exfiltration paths. All artifacts stay inside the repo and the homelab cluster's `tcg-dev` pod. Kernel changes are pure compute, read-only with respect to user input. CSV/ncu outputs contain only kernel timing/counter data — no PII, no production weights.

The single sensitive cluster operation is pausing DCGM-exporter on gpu-01 for ncu profiling: `kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter=false --overwrite`. Phase 5 cleanup re-enables: `kubectl label node gpu-01 nvidia.com/gpu.deploy.dcgm-exporter-`. No `--no-verify` or hook-skipping; nvcc flags unchanged.

## Dependencies

- **Hardware**: V100 SXM2 32GB on gpu-01 (sm_70), exclusive during ncu runs.
- **Pod**: `tcg-dev` with `/srv/dev/dsv4-cuda/deepseek-sprint017` mounted at `/src` (already up per intent doc).
- **Toolchain**: existing CUDA 12.x on `tcg-dev`, `nvcc` for sm_70, `ncu` (Nsight Compute) ≥ 2023.x, `make` target in `tools/tc-grid/`. No version bumps.
- **In-repo**: `v3_kernels.cuh` / `v4_kernels.cuh` as v9 starting points; existing tolerance gate in `src/launch_*.cu`; existing ncu CSV parser at `/tmp/parse_ncu.py` (promote to `scripts/`).
- **Out-of-tree**: cuBLAS for FP16 ceiling reference (already wired via `src/reference.cu`).
- **Reports**: REPORT-9 is the sprint seed. Prior reports in `tools/tc-grid/docs/REPORT-{1..8}.md` provide historical context.
- **Tolerance**: sprint-015 P2 §7 per-direction FP16 tolerance — `rel ≤ 1e-3` is binding for v9 promotion (see Open Question 1).
- **No upstream blockers.** No other sprints branch off this one mid-flight.

## Open Questions

To resolve before finalizing `docs/sprints/SPRINT-016.md`:

1. **Tolerance binding.** Is `rel ≤ 1e-3` (sprint-015 P2 §7) the hard gate for v9 promotion, or is there a looser DSv4-Flash-only threshold (e.g. `rel ≤ 5e-3`) that would put v8 or aggressive v9 (`CHUNK_K=16`) back in scope?
2. **Escalation policy.** If v9 + W2 + W3 close <50% of the cuBLAS gap (i.e. end near 45 TFLOPS or below), open SPRINT-017 dedicated to W6 (XOR swizzle + manual fragment loads, ~400 LOC × ~4 kernels), or accept the ceiling and shift to integration work?
3. **INT4 path presentation.** If LUT and bitshift land within 5%, keep both behind a dispatch flag (more launcher complexity, per-M dispatch possible) or pick one and document the tradeoff? Affects the consuming DSv4 integration path.
4. **MoE shape coverage (W4).** Which 2-3 DSv4 shapes are canonical? Suggested: FFN1 N=18432 K=7168, FFN2 N=7168 K=18432, gate N=128 K=7168. Confirm with integration team before running Phase 4.
5. **REPORT-10 cadence.** Incremental running doc per phase, or end-of-sprint write-up? Prior reports were end-of-sprint; running gives earlier review signal but more churn.
6. **v8 disposition.** `kernels/v8_kernels.cuh` is kept in-tree as a negative reference. Keep, or delete once v9 lands and REPORT-10 supersedes it?
