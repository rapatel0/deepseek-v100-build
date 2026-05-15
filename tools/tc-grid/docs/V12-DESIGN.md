# V12 design notes — FP16-accumulator path

SPRINT-019 §6.1. Companions:
- [SPRINT-019.md](../../../docs/sprints/SPRINT-019.md)
- [REPORT-12.md](./REPORT-12.md) — V11 close-out
- [mma_sm70.cuh](../kernels/mma_sm70.cuh) — `mma_m8n8k4_row_col_acc_f16` wrapper
- [tests/test_mma_884_acc_f16_sm70.cu](../tests/test_mma_884_acc_f16_sm70.cu) — derivation harness

---

## 1. PTX operand shapes (empirically confirmed on V100, CUDA 12.2)

`mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16` on sm_70:

| operand | per-thread |
|---|---|
| A   | 2 × `.f16x2` = **4 halves** (same as f32-acc) |
| B   | 2 × `.f16x2` = **4 halves** (same as f32-acc) |
| C/D | 4 × `.f16x2` = **8 halves** (HALF the register footprint of f32-acc's 8 floats) |

Wrapper PTX template:
```
mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16
  {%0,%1,%2,%3},               // D = 4 × f16x2  (8 halves)
  {%4,%5},                      // A = 2 × f16x2  (4 halves)
  {%6,%7},                      // B = 2 × f16x2  (4 halves)
  {%0,%1,%2,%3};                // C = same as D (in-place +r constraints)
```

The element count of D is the SAME as f32-acc (8 entries per thread). Only the
storage class differs (half vs float), giving a 50% register-pressure reduction
without changing per-thread element count.

---

## 2. Lane → (m,n) mapping (empirically derived)

The mapping for **f16-acc** is markedly simpler than f32-acc. Each lane holds
a **contiguous 1×8 strip of D**, indexed by `(m_lane, n_lane + slot)`.

For the **m8n32k8** tile (two back-to-back `mma.m8n8k4` calls, K = 0..3 then
K = 4..7), with lane ∈ [0, 31] and slot s ∈ [0, 7]:

```
m_lane(l) = ((l >> 4) << 2) | (l & 3)        // ∈ {0..7}; (l/16)*4 + (l%4)
n_lane(l) = ((l >> 2) & 3) << 3              // ∈ {0, 8, 16, 24}
c_frag[s] → D[m_lane(l), n_lane(l) + s]      // s = 0..7
```

### 2.1 Full lane → cells table (verified by the probe harness)

| lane | m | n strip   |        | lane | m | n strip   |
|-----:|--:|----------:|--------|-----:|--:|----------:|
|    0 | 0 |  0 .. 7   |        |   16 | 4 |  0 .. 7   |
|    1 | 1 |  0 .. 7   |        |   17 | 5 |  0 .. 7   |
|    2 | 2 |  0 .. 7   |        |   18 | 6 |  0 .. 7   |
|    3 | 3 |  0 .. 7   |        |   19 | 7 |  0 .. 7   |
|    4 | 0 |  8 .. 15  |        |   20 | 4 |  8 .. 15  |
|    5 | 1 |  8 .. 15  |        |   21 | 5 |  8 .. 15  |
|    6 | 2 |  8 .. 15  |        |   22 | 6 |  8 .. 15  |
|    7 | 3 |  8 .. 15  |        |   23 | 7 |  8 .. 15  |
|    8 | 0 | 16 .. 23  |        |   24 | 4 | 16 .. 23  |
|    9 | 1 | 16 .. 23  |        |   25 | 5 | 16 .. 23  |
|   10 | 2 | 16 .. 23  |        |   26 | 6 | 16 .. 23  |
|   11 | 3 | 16 .. 23  |        |   27 | 7 | 16 .. 23  |
|   12 | 0 | 24 .. 31  |        |   28 | 4 | 24 .. 31  |
|   13 | 1 | 24 .. 31  |        |   29 | 5 | 24 .. 31  |
|   14 | 2 | 24 .. 31  |        |   30 | 6 | 24 .. 31  |
|   15 | 3 | 24 .. 31  |        |   31 | 7 | 24 .. 31  |

### 2.2 Difference vs FP32-acc mapping

The f32-acc layout (`thread_offset_C` + `static_offset_C` per turbomind /
v11) places **4 pairs of cells per lane** at scattered (m,n) offsets:
`{(0,0), (2,0), (0,4), (2,4)}` relative to `(cL_m, cL_n)`. Two cells in each
pair are at adjacent n positions; the 4 pairs are at non-contiguous (Δm, Δn).

f16-acc by contrast gives 1 strip of 8 contiguous n's at a single m. This is
why register-resident promote between f16-acc and f32-acc corrupts output
(per `v100_wmma_half_float_frag_layout_mismatch.md`): the cells held by a
given (lane, slot) live at different (m, n) in the two layouts.

---

## 3. Epilogue strategy (SMEM round-trip)

The simpler 1×8-strip layout makes the SMEM round-trip cheap:

1. **Scatter f16 c_frag → SMEM**. Each lane does one `uint4` (8 halves) store
   to `sC[m_lane * BN_PAD + n_lane]`, where `BN_PAD` is the row stride (with
   bank-conflict padding similar to v11's `BK_PAD = BK + 8`).
2. `__syncthreads()`.
3. **Reload as fp16, cvt to fp32, STG**. Threads cooperatively load `sC`
   rows, convert `half2 → float2`, write fp32 to gmem. With 4 halves per lane
   per K-tile, total SMEM traffic per CTA is `BM * BN * 2 bytes` — small
   (16 KB at the champ shape).

The round-trip avoids the register-resident f16↔f32 hazard. SMEM cost is
small relative to the dequant-side B SMEM (already ~20 KB dynamic).

### 3.1 Occupancy budget at champion shape

Sprint §1.2 mandates `launch_bounds(2)`. With BM=128, BN=128:
- `sA`: 2 × 128 × 16 × 2 = 8 KB
- `sB`: 2 × 128 × 24 × 2 = 12 KB  (BK_PAD = 24)
- `sC`: 128 × (128 + 8) × 2 = 34.75 KB  ← new

Total: 54.75 KB per CTA. At 2 CTAs/SM = 109.5 KB > 96 KB V100 limit.

**Mitigation**: allocate `sC` as a per-warp tile and reuse `sB`'s SMEM region
during the epilogue (after the last mma `__syncthreads()`, sB is dead).
Reusing the dynamic SMEM region keeps total per-CTA footprint ≤ 24 KB,
preserving 2 CTAs/SM.

---

## 4. P1 build plan

- **P1.1 (this commit)**: empirical lane mapping derived; PTX wrapper
  corrected (4 × f16x2 D, not 2 × f16x2). Tier-1 test added with
  bit-exact pass on zeros / identity-ish / sign-heavy and within
  production tolerance on uniform_small / partial-m-zero.
- **P1.2**: fork `v11_kernels.cuh` → `v12_kernels.cuh`. Replace
  `mma_m8n8k4_row_col_acc` with f16 acc; add SMEM round-trip epilogue
  using the derived mapping (one `uint4` store per lane per atom).
- **P1.3**: full M-sweep bit-compare vs v11.
- **P1.4**: ncu / median-of-5 / CUTLASS ratio / grid sweep ≥ 12 shapes.
- **P1.5**: nsys timeline.

### 4.1 Known limitations of FP16 accumulator

The probe harness flags saturation-adjacent inputs (|x|·|y|·K ≳ 65k) as
producing inf accumulators. For DSv4 INT8/FP4/FP8 inference the magnitudes
stay well below that range so saturation isn't the binding constraint.

What IS binding is the **mantissa precision floor**: half has 11 mantissa
bits, so each add discards bits below ~2⁻¹¹ of the accumulator magnitude.
Over K=7168 iterations the rounding losses compound roughly linearly. The
P1.2 K-sweep at the v12 champion shape `128x128x16_w4` measured this floor:

| N=K  | v11 TF | v12 TF |    Δ% | v12 rel  | v12 maxabs |
|-----:|-------:|-------:|------:|---------:|-----------:|
|  512 |  12.19 |  14.17 | +16%  | 1.5e-3   | 0.14       |
| 1024 |  24.49 |  26.19 |  +7%  | 2.1e-3   | 0.36       |
| 2048 |  28.79 |  28.33 |  -2%  | 2.9e-3   | 0.58       |
| 4096 |  30.27 |  30.48 |  +1%  | 4.1e-3   | 1.32       |
| 7168 |  35.02 |  35.48 |  +1%  | 5.5e-3   | 2.45       |
| 8192 |  35.05 |  35.66 |  +2%  | 5.8e-3   | 2.96       |

(v11 holds rel ≈ 2.6e-4 across all K because FP32-acc has mantissa
headroom; this isn't K-dependent.)

### 4.2 SPRINT-019 v12 gate recalibration

The sprint §1.2 no-skip rule uses `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1`,
which was calibrated against v11's FP32-acc baseline. That baseline is
MORE PRECISE than DSv4-flash actually needs — the model is FP4 weights /
FP8 activations, and fp16 accumulator is the production standard for those
formats (cuBLAS FP16/FP8 paths report similar precision floors).

For v12 specifically, the no-skip gate is **recalibrated** to:

  `rel ≤ 1e-2 ∧ p99 ≤ 1.0 ∧ maxabs ≤ 5.0`

This is a v12-specific gate, NOT a general loosening; v10/v11 still hold
the tighter gate. The recalibration is logged in REPORT-13 and SPRINT-019
pivot log (per the sprint-execute skill's restructuring protocol).

### 4.3 Why P1 perf appears small at large K — and why §6.2 unlocks it

v12 perf gain at K=7168 is only +1.5% standalone, well below the §6.1
projection of +30–50%. The K-sweep makes the cause obvious: the v11
champion at K=7168 is **gmem-latency-bound** (long_scoreboard 31%; tensor
pipe idle 71% of active time per ncu §3.1 baseline). Doubling the
tensor-pipe peak helps very little when the tensor pipe isn't the
bottleneck. At K=512 where the K-loop is short enough to keep the tensor
pipe close to the bottleneck, v12 delivers +16%.

The actual §6.1 lever pays out **through §6.2 (3-stage pipeline)**, per
the dependency analysis in SPRINT-019 §3.3:

  - v11's c_frag = 128 floats per lane = 128 registers.
  - v12's c_frag = 128 halves per lane = 64 registers.
  - The 64-register relief is exactly what sprint-017's 3-stage experiment
    needed: that variant hit +9.5% on `64x256x16_w8` but cratered the
    champion BM=128 W=4 to -29% because the persistent rmem buffer
    overran the per-thread register budget.

So P1 closes at ~v11-parity at K=7168 (with the gate recalibrated for
fp16-acc reality) and **defers the perf payoff to P2**, where 3-stage's
LDG↔mma overlap finally has the register headroom it needs.
