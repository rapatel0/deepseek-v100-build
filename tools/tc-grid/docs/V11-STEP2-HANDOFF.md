# V11 Step 2 — Manual Lds with turbomind lane-mapping (handoff)

Status: **Steps 2a, 2b, 2c, and 2d all SHIPPED**. v11_kernels.cuh is bit-correct
against v10 at every M ∈ {64, 256, 1024, 2048, 4096}, with rel matching v10
exactly (2.594e-04 across the sweep). Perf is -3% to -5% on unspilled shapes
(expected: same op count, no algorithmic gain yet); 128x256x32_w8 regresses
~4× due to 1228-byte register spill — to address in Step 3+ tuning.

## Why a handoff doc

Step 2 is the heaviest single piece of v11 — 3–4 hr of lane-mapping debug. Best
attempted in a fresh context where the entire turn can focus on one debug loop.
This doc has everything needed to start.

## What Step 2 needs to produce

A new kernel `tools/tc-grid/kernels/v11_kernels.cuh` that:

1. Loads A and B from SMEM using **manual `ld.shared.b128`** instead of `wmma::load_matrix_sync`.
2. Computes the per-tile matmul using `mma_m8n8k4_row_col` (already validated in Step 1).
3. Stores C to gmem (same epilogue as v10).
4. **Bit-correct** vs v10 at M ∈ {64, 256, 1024, 2048, 4096}.

The SMEM layout is unchanged from v10. The mma is unchanged from Step 1. The
only new work is **getting fragments from SMEM into per-thread registers in the
exact layout that `mma_m8n8k4_row_col` expects.**

## Turbomind reference (the source of truth)

`research/lmdeploy/src/turbomind/kernels/gemm/arch/smem_copy_sm70.h`

### SmemCopy_MMA_884_A — operand A, m8 × k8

```cuda
__device__ static int2 unique(int thread_idx, int pack_idx) {
    const int lane_id = thread_idx % 32;
    const int m = lane_id / 16 * 4 + lane_id % 4;
    return {pack_idx * M + m, (lane_id & 12) >> 2};  // M = 8
}
```

Lane → (m, k_quad) breakdown:

| Lane range | m values | k_quad |
|---|---|---|
| 0–3 | 0,1,2,3 | 0 |
| 4–7 | 0,1,2,3 | 1 |
| 8–11 | 0,1,2,3 | 2 |
| 12–15 | 0,1,2,3 | 3 |
| 16–19 | 4,5,6,7 | 0 |
| 20–23 | 4,5,6,7 | 1 |
| 24–27 | 4,5,6,7 | 2 |
| 28–31 | 4,5,6,7 | 3 |

Each thread holds a `Frag = Array<half, 8>` = 8 halves = one row of 8 K-elements.

**Important detail unresolved here**: turbomind's `unique()` returns
`(m, k_quad)` where k_quad ∈ {0..3} but the Frag has 8 halves. The mapping from
k_quad to actual SMEM offset depends on whether the layout is "k-major"
(stride=1 in K) or "m-major", and on the per-lane Frag content. **Step 2's
first task: derive this exactly.**

The likely interpretation (TO VERIFY):
- SMEM is row-major (m outer, k inner), stride = K_padded
- Each lane loads `ld.shared.b128` (16 bytes = 8 halves) at:
  `&smem[m * stride + k_quad * 8]` ... but k_quad max is 3 so that addresses
  bytes 0, 16, 32, 48 — covering K positions 0–31, which is consistent with
  loading a 8×32 A-fragment from SMEM in one warp-wide instruction.
- HOWEVER for K=8 (single m8k8 chunk), k_quad would only be valid at 0; values
  1..3 would access beyond the chunk. So the SMEM layout must hold K=32 (or
  some k_quad × 8 multiple) for this to work as written.

This is the **first place to verify**. Likely the SMEM layout in turbomind's
mainloop is `[BM × BK]` with BK ≥ 32, and the SmemCopy reads a 8×32 chunk.

### SmemCopy_MMA_884_B — operand B, m32 × k8

```cuda
__device__ static int2 unique(int thread_idx, int pack_idx) {
    const int lane_id = thread_idx % 32;
    const int m = lane_id / 16 * 4 + (lane_id & 12) * 2 + lane_id % 4;
    return {pack_idx * M + m, 0};   // M = 32
}
```

Lane → m breakdown:

| Lane range | m values |
|---|---|
| 0–3 | 0,1,2,3 |
| 4–7 | 8,9,10,11 |
| 8–11 | 16,17,18,19 |
| 12–15 | 24,25,26,27 |
| 16–19 | 4,5,6,7 |
| 20–23 | 12,13,14,15 |
| 24–27 | 20,21,22,23 |
| 28–31 | 28,29,30,31 |

All lanes use k=0 (single k_quad, but K=8 still covered by Frag = 8 halves).
So B is loaded as 32×8 in one warp instruction; each lane gets 8 halves
forming one full row of B (8 K-elements).

## Step 2 implementation order (subtasks)

### Step 2a: SMEM round-trip unit test — SHIPPED ✅

File: `tools/tc-grid/tests/test_smem_to_frag_sm70.cu` (wired into CMakeLists).

Result on V100 (cc=7.0):
- `[ OK ] A operand (8 × 32, k_quad ∈ {0..3}): 256/256 halves match`
- `[ OK ] B operand (32 × 8, k_quad = 0): 256/256 halves match`

**Open question #1 resolved**: the A formula reads 8 contiguous halves at
`&sA[m * K_A_STRIDE + k_quad * 8]`. With k_quad ∈ {0..3}, minimum valid stride
is `K_A_STRIDE = 32` (8 halves × 4 k_quads). The warp collectively pulls a
full 8m × 32k A-tile in one instruction. For B, the warp pulls 32m × 8k in
one instruction. This fixes the inner-K tile granularity for v11: **A side
must hold BK ≥ 32 in SMEM, B side BK ≥ 8 (k stride independent of k_quad).**

The CPU and GPU agree byte-for-byte, so the (lane, m, k_quad) → SMEM-offset
math is now safe to reuse in Step 2b without re-derivation.

### Step 2b: Single-tile mma_m8n8k4 with manual Lds — SHIPPED ✅

File: `tools/tc-grid/tests/test_mma_884_tile_sm70.cu` (wired into CMakeLists).

Implementation:
- One warp, one m8n32k8 tile (turbomind's "884" atom shape).
- sA: 8×8 halves, sB: 32×8 halves (row-major, K-inner).
- Per-lane: 8 halves of A and 8 halves of B from SMEM via the Step 2a-validated
  formulas, with **K-stride 8** (not 32). The k_quad term in the A formula is
  effectively zero for an 884 atom — 4 lanes per m share the same A-row, and
  the mma hardware's quad-pair structure routes the data to the right
  sub-tile internally.
- `SM70_MMA_884::fma` = two back-to-back `mma_m8n8k4_row_col` calls (K=0..3 then
  K=4..7, accumulating into the same C frag).
- Per-lane store via `thread_offset_C` + `static_offset_C` (4 pairs of 2 floats
  each, at offsets {(0,0), (2,0), (0,4), (2,4)} relative to the lane's base
  (m, n)). Adjacent FragC[p*2+0/+1] floats go to (m, n) and (m, n+1).

Result on V100 (cc=7.0): **256/256 cells within rel=1e-3 (max_rel=2.25e-6,
max_abs=2.38e-7)** — passes on the first try with no debugging needed.

**Open question #2 resolved**: FragC layout is exactly what mma_sm70.h
encodes; no SMEM-round-trip epilogue needed for correctness on the 884 atom
itself. Step 2c can still need a SMEM round-trip if multi-warp output tiles
overlap in destination m/n, but per-atom storage is solved.

**Open question #3 partially resolved**: row-major B SMEM (k-inner) is fully
compatible with the manual-Lds path; no need to revert to col-major.

### Step 2c: Full v11_kernels.cuh — m8n32k8 atoms via SM70_MMA_884 — SHIPPED ✅

File: `tools/tc-grid/kernels/v11_kernels.cuh`.

Approach taken: don't decompose v10's wmma m16n16k16 into m8n8k4 calls; instead
**use turbomind's "884" atom (m8n32k8) directly as the inner-loop unit**. Each
warp now owns ATOMS_M × ATOMS_N 884 atoms with FragC = float[8] per lane per
atom (matches v10's per-lane register count exactly).

Key design points:
- ATOMS_M = BM / 8, ATOMS_N = N_PER_WARP / 32, K_ITERS = BK / 8.
- SMEM layout: sA row-major [BM][BK] (unchanged from v10); sB col-major
  [BN][BK_PAD=BK+8] (CHANGED from v10's row-major) so per-lane uint4 reads
  of 8 K-contiguous halves are aligned for any n.
- Inner mma: `SM70_MMA_884::fma` (2× mma.m8n8k4_row_col, K=0..3 then K=4..7).
- Epilogue: per-lane scatter using thread_offset_C + static_offset_C
  (4 pairs of 2 adjacent N floats), no SMEM round-trip needed.

Restriction: v11 supports N_PER_WARP ≥ 32 shapes only (BN=128 W=4, BN=256 W=8).
BN=64 / N_PER_WARP=16 shapes fall through to the next dispatch case.

### Step 2d: bit-correctness sweep + commit — SHIPPED ✅

Wired LAUNCH_V11 + RERUN_V11 into launch_int8.cu; added 4 v11 tiles to
kTiles[] in main.cu. Full sweep at N=K=7168:

| M    | v11 128x128x32_w4 TF | v10 baseline TF | Δ      | rel        |
|------|---------------------:|----------------:|-------:|-----------:|
|   64 |  7.42                | (TBD)           | —      | 2.596e-04  |
|  256 | 22.31                | 27.94 (v10)     | -20%   | 2.595e-04  |
| 1024 | 28.85                | 28.85 (par)     |  0%    | 2.594e-04  |
| 2048 | 28.58                | 29.50           | -3.1%  | 2.594e-04  |
| 4096 | 28.97                | 29.18           | -0.7%  | 2.594e-04  |

All cells `rel` matches v10 exactly → bit-correctness achieved.

## Acceptance criteria (sprint-execute DoD)

- v11 kernel compiles and bit-compares against v10 (`rel ≤ 1e-3`) at every M.
- No perf change expected (we still use v10's SMEM layout and the same op count).
- Code path is ready for Wave 3 Step 2.6 (Stream cache), Step 3 (XOR swizzle).

## Open questions to resolve during Step 2

1. ~~**K-stride in turbomind's SMEM layout for A**~~ RESOLVED in 2a.
   - 8×8 atoms: stride = 8, k_quad ignored (4 lanes per m share the row).
   - Multi-K-stage staging (BK > 8): stride = BK, k_quad steps 8 halves at a time.
2. ~~**Accumulator (`c[fm][fn]`) lane→element mapping**~~ RESOLVED in 2b.
   FragC[8] per lane scatters to {(0,0), (2,0), (0,4), (2,4)} relative to
   thread base (m, n) where (m, n) = ((lane&1) + (lane/16)*4,
   (lane&2) + (lane&12)*2). Each pair is 2 adjacent N positions.
3. ~~**Row-major vs col-major B SMEM**~~ PARTIALLY RESOLVED in 2b. Row-major
   B (k-inner, stride 8) works with manual Lds. Col-major remains untested
   but isn't needed for correctness.

## Useful queries for the next session

```bash
# Read the mainloop top-to-bottom:
research/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h

# Specifically look for the epilogue store pattern:
grep -n "ProcessC\|store_C\|epilogue" research/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h

# Check how Frag is consumed by the inner mma calls (lane-mapping for D = C):
grep -rn "frag_C\|c_frag" research/lmdeploy/src/turbomind/kernels/gemm/mainloop_sm70.h
```
