# Turbomind V100 INT8 GEMM — design insights, counterfactual, and porting plan

Date: 2026-05-14
Source: `~/repos/deepseek/research/lmdeploy/src/turbomind/kernels/`

Purpose: extract concrete code patterns and architectural ideas from turbomind's V100
INT8 GEMM (production-grade) that we can lift into our v11, AND identify what we already
do that they don't. Pairs with V11-DESIGN.md and SPRINT-017.md.

## TL;DR

### What turbomind does that we should port (5 items)

1. **Inline m8n8k4 PTX** (not `wmma::*`) — finer register / lane control. (V11 Step 1)
2. **XOR-swizzled SMEM** with bank-conflict-free indexing on B then A. (V11 Steps 3–4)
3. **Pre-baked phase tables** for swizzled offsets — compile-time partial evaluation,
   table lookup at runtime instead of AND+SHIFT+XOR per store. (V11 Step 3, added)
4. **CTA_K=16 inner K-stage** — m8n8k4 advances K by 4, so BK=16 means 4 inner iters per
   K-stage, interleaving dequant + mma more tightly. (V11 Step 2.5, added)
5. **Cache-policy `Stream` (cs-evict) on B-side gmem loads** — B has no reuse, evict from
   L1 to free capacity for A. Inverse of our L2-prefetch strategy. (V11 Step 2.6, added)

Estimated combined reach: 35–42 TFLOPS (gap from 29.5 TF → 50 TF goal still ~10 TF
unaccounted; closing that may require Step 5 tile sweep or hit a hardware ceiling).

### What WE do that turbomind doesn't (and want to keep)

1. **`prefetch.global.L2` hints** before each K-tile load. They don't bother. ncu says
   it's noise on v10, but v11 with shorter mma loops may make it valuable again.
2. **QK_INT8=32 fine quant block** (theirs is 128) — better numerical quality at 4× more
   scale loads. Trade-off, not defect.
3. **Explicit `__launch_bounds__(WARPS*32, 2)`** for register-pressure control.
4. **Per-row tolerance contract** (`rel ≤ 1e-3 ∧ p99 ≤ 0.05`) — tighter regression guard.
5. **4 quantization formats** (INT8 / INT4 LUT / MXFP4 / F8) — they ship mostly U4 on sm_70.

### Counterfactual surprises (things I had to revise after deeper read)

1. **Our v10 row-major B SMEM is a DIFFERENT mechanism than their XOR swizzle**, both
   attacking bank conflicts. They may not stack additively — if v11-swizzle reaches the
   same local optimum as v10, it's convergence, not a v11 bug. **Plan implication: must
   test v11 swizzle on BOTH original v10 col-major-B-style SMEM AND a non-row-major
   variant to isolate which mechanism is doing the work.**
2. **Turbomind has `SplitK=true` in every shipped sm_70 config.** We deferred SplitK in
   SPRINT-016 because M=2048 wasn't the bottleneck. Their reliance on it suggests it's
   doing work at *every* M, not just small M. Worth a closer measurement post-v11.
3. **Turbomind's `gemm_bench` is commented out in upstream CMake** — they don't ship a
   standalone bench. Resurrecting it is 1–2 days of build work, not the hours I first
   estimated. Filed deferred; revisit only if v11 stalls under 40 TF.
4. **Turbomind ships 14 tile variants per format**; we ship ~6. The gap isn't in kernel
   tuning — it's in runtime-dispatch infrastructure. Tc-grid is effectively our tuner
   for the offline phase; the runtime hook into DSv4 production is missing and is a
   separate piece of work.

Full source-walk follows. Counterfactual analysis in §L. Updated porting plan in §H.

## A. m8n8k4 PTX wrapper (copy verbatim)

`src/turbomind/kernels/core/mma.h:11-30`

```cuda
__inline__ __device__ void
mma_m8n8k4_row_col(Array<float, 8>& d,
                   const Array<half, 4>& a,
                   const Array<half, 4>& b,
                   Array<float, 8>& c)
{
#if TURBOMIND_ARCH_SM70
    uint32_t const* A = reinterpret_cast<uint32_t const*>(&a);
    uint32_t const* B = reinterpret_cast<uint32_t const*>(&b);
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32"
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11},"
        "{%12, %13, %14, %15, %16, %17, %18, %19};"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]),
          "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
        : "r"(A[0]), "r"(A[1]),
          "r"(B[0]), "r"(B[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
#endif
}
```

Also: `mma_m8n8k4_row_row` variant (mma.h:32-51) for the alternate B layout — same
operand types, different `.row.row` qualifier.

This is the **only** PTX wrapper we strictly need for v11. The fragment is:
- A: `Array<half, 4>` per thread (2 uint32 packed pairs of halves)
- B: `Array<half, 4>` per thread
- D / C: `Array<float, 8>` per thread

For our `wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn])` where `c[fm][fn]` is a
`16×16×16` accumulator: each WMMA call decomposes to **4 m8n8k4 PTX calls** (2 across M,
2 across N, K=4 → K=16 takes 4 inner iterations). With our existing fragment-array shapes
this means roughly `4 × FRAG_M × FRAG_N` PTX calls per K=16 tile.

## B. Swizzle template

`src/turbomind/kernels/core/layout.h:8-19`

```cuda
template<int Bits, int Base, int Shift>
struct Swizzle {
    using bit_mask = std::integral_constant<int, (1 << Bits) - 1>;
    using yyy_mask = std::integral_constant<int, bit_mask{} << (Base + Shift)>;

    template<class Offset>
    __host__ __device__ constexpr static auto apply(Offset offset)
    {
        return offset ^ ((offset & yyy_mask{}) >> Shift);
    }
};
```

Worked example for `Swizzle<2, 4, 4>`:
- `bit_mask = 0x3` (bits 1:0)
- `yyy_mask = 0x3 << 8 = 0x300` (bits 9:8)
- For `offset = 0x150`: `0x150 ^ ((0x150 & 0x300) >> 4) = 0x150 ^ 0x10 = 0x140`

**The XOR moves bits from positions `[Base+Shift, Base+Shift+Bits)` down to positions
`[Base, Base+Bits)`, where `Base` is the "byte-bank stride within a row" address bits.**

For our case (B SMEM, halves, BK=32, BN=128):
- 16-byte (uint4) vector = 8 halves = 2^3 halves per access → Base must point to bit 3 (byte addr) or bit 4 (half addr).
- For half-typed offsets: bits 0–3 select 16-half chunks across the warp; XOR rotates that against a row index. Typical: `Swizzle<3, 3, 3>` for half data, vector = 8 halves.

## C. SmemCopy lane→element offsets

`src/turbomind/kernels/gemm/arch/smem_copy_sm70.h:21-65`

```cuda
// SmemCopy_MMA_884_A — operand A, m8 × k8 tile
__device__ static int2 unique(int thread_idx, int pack_idx) {
    const int lane_id = thread_idx % 32;
    const int m = lane_id / 16 * 4 + lane_id % 4;     // lanes 0-3 → m0-3, lanes 16-19 → m4-7, etc.
    return {pack_idx * M + m, (lane_id & 12) >> 2};   // k from bits 3:2 of lane
}

// SmemCopy_MMA_884_B — operand B, m32 × k8 tile (B is the "wide" operand here)
__device__ static int2 unique(int thread_idx, int pack_idx) {
    const int lane_id = thread_idx % 32;
    const int m = lane_id / 16 * 4
                  + (lane_id & 12) * 2
                  + lane_id % 4;
    return {pack_idx * M + m, 0};
}
```

These define **where each lane reads its A / B fragment halves from in SMEM**. After
issuing `Lds(*(Frag*)dst_ptr, src_ptr)` (a `ld.shared.b32` or `ld.shared.b128` depending
on `Frag` size), the resulting per-thread fragment matches what `mma_m8n8k4_row_col`
expects.

For our v11 manual-Lds path: replicate these exact offsets, OR use a different equivalent
mapping but verify each lane gets the right element via a SMEM round-trip unit test
*before* feeding into mma.

## D. Mainloop structure

`src/turbomind/kernels/gemm/mainloop_sm70.h:196-351`

Pseudocode (paraphrased):

```
// Init
setup smem ptrs (2 stages × A, B, U=scale_A, V=scale_B)
prefetch tile 0: gmem → register (Ldg)
store register → SMEM[stage 0]   // includes swizzle
__syncthreads()

// K-loop: ITER_K = K / TILE_K
for k in [0, ITER_K):
    if k == ITER_K - 1:
        store register → SMEM[stage 1-current]
        __syncthreads()
        swap stage pointers
    preload SMEM → frag_A / frag_B / frag_U / frag_V  for (k+1) % TILE_K
    if k == 0:
        Fetch gmem → register (Ldg of next K-tile)
    // dequant: U-scale into B-fragment
    Transform_HMMA_SIMT_B::dequant(frag_B, frag_V)
    // mma
    for m in FRAG_M:
        for n in FRAG_N:
            frag_C[m][n] = mma_m8n8k4_row_col(frag_A[k][m], frag_B[k][n], frag_C[m][n])

// Epilogue
__syncthreads()
store frag_C → SMEM → gmem (vectorized)
```

Key points:
- **Stages = 2** (one current, one staging). Manual double-buffer, no cp.async (V100 has none).
- **Synchronous Ldg** for gmem; the latency-hide trick is that the gmem load is *issued
  before* the mma loop runs and the mma loop is long enough to cover it.
- **Dequant happens between SMEM load and mma**, in registers, per element pair.
- **Two syncs per outer iteration** (one at swap, one at start). No extra syncs in the
  inner mma loop.

## E. gmem → SMEM iterator with phase-table swizzle

`src/turbomind/kernels/gemm/iterator_sm70.h:47-269`

Two clever tricks:

### E1. Vectorization via ThreadMap

`Map::AccessType = Array<T, Map::kAccessC>` where `kAccessC` is the per-access width (4–8
halves, i.e., uint2 or uint4). Each thread issues `ITER_S × ITER_C` `Ldg` instructions per
tile, indexed via `Map::get_offset(warp_id, lane_id)` returning a struct of strides.

### E2. Phase table (the precomputed-swizzle trick)

`iterator_sm70.h:134-139` (paraphrased):
```cuda
// Constructor-time:
for s in [0, kPeriodS):
    for c in [0, kPeriodC):
        phases_[s][c] = SmemLayout::apply(s * kDeltaS, c * kDeltaC) - SmemLayout::apply(0, 0);
```

```cuda
// Per-store at line 237-256:
const int i0 = SmemLayout::apply(s/kPeriodS * kPeriodS * kDeltaS,
                                  c/kPeriodC * kPeriodC * kDeltaC);
const int i1 = phases_[s % kPeriodS][c % kPeriodC];
auto dst = &smem_data_.ptr_[i0 + i1];
```

**Insight**: the swizzle math `offset ^ ((offset & mask) >> shift)` decomposes into a
"slow" part (full Swizzle::apply on the high bits) plus a "fast" part (a table indexed by
the low bits mod the period). The fast part is the same for every iteration, so it's
precomputed in registers at constructor time. **This avoids 1 AND + 1 SHIFT + 1 XOR per
SMEM store at the cost of a register table** (~16 entries for typical period). On V100
where every instruction counts, this matters.

## F. Dequant (Transform_HMMA_SIMT_B)

`src/turbomind/kernels/gemm/transform.h:108-157` (paraphrased)

```cuda
// INT8 / FP16 weight, FP16 scale (uint16 holds half bit-pattern)
__device__ static void dequant(Array<half, 2>& x, Array<uint16_t, 1> s) {
    half s1 = __ushort_as_half(s[0]);
    x[0] = __hmul(x[0], s1);
    x[1] = __hmul(x[1], s1);
}

// With zero-point (Q4_K style):
__device__ static void dequant(Array<half, 2>& x, Array<uint32_t, 1> s) {
    Array<half, 2>& _s = (Array<half, 2>&)s;   // [scale, zero]
    x[0] = __hfma(x[0], _s[0], _s[1]);
    x[1] = __hfma(x[1], _s[0], _s[1]);
}
```

**Cost per 2 dequanted elements**:
- Simple scale: 2 HMUL (1 instruction each) + 1 bit-cast (free).
- With zero-point: 2 HFMA.

This is what our `__hmul(__short2half_rn((short)qs[i]), s_h)` already does for INT8 — we
matched this in iteration 3. **No improvement available here for INT8.** The zero-point
form may matter for INT4 / Q4_K-style formats later.

## G. Shipped tile sizes for sm_70 quantized INT8

From `src/turbomind/kernels/gemm/kernel/sm70_884_4.cu` (the U4 / quantized 4-bit weight
config — closest analog to our INT8):

| CTA_M | CTA_N | CTA_K | TG_M | TG_N | TG_K | Stages | Use case |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | 256 | 16 | 2 | 4 | 1 | 2 | large M |
| 128 | 128 | 16 | 2 | 2 | 1 | 2 | square / standard |
|  64 | 128 | 32 | 2 | 2 | 1 | 2 | medium M |
|  32 | 128 | 32 | 1 | 4 | 1 | 2 | tall-thin decode |
|   8 | 128 | 64 | 1 | 4 | 1 | 2 | minimal-M residual |

All `Stages=2`, all `GroupSizeV=128` (group-wise dequant). Notable differences vs ours:

- **CTA_K=16 for large tiles**, not 32 like our v10. With m8n8k4 each PTX call advances
  K by 4, so K=16 = 4 inner iterations. Our wmma m16n16k16 means K=16 per call. Their
  shorter K-stage interleaves dequant more frequently.
- **CTA_K=64 for the M=8 tail tile** — tiny M means the only way to feed the tensor
  cores is to deepen K. We don't ship anything at this scale; our small-M champion is
  the BM=32 v3_b1 variant which uses K=32.
- **Stages=2**, same as us.

## H. Ideas worth porting (ranked)

| # | Idea | Estimated impact | Already in V11-DESIGN.md? |
|---:|---|---|---|
| 1 | XOR-swizzle SMEM (B-side, then A-side) | +6–8% | Yes (Steps 3, 4) |
| 2 | m8n8k4 inline PTX in place of `wmma::mma_sync` | +3–10% (mostly register-pressure relief, secondary perf) | Yes (Step 1) |
| 3 | Phase-table precomputation for swizzled offsets | +1–2% (instruction reduction at store) | **NO — add** |
| 4 | Shorter CTA_K=16 for large tiles (interleave dequant more) | +2–4% (better overlap) | **NO — add to Step 5 tuning sweep** |
| 5 | Dedicated tail-M tile (M=8 × N=128 × K=64) | small-M only; <5% on M=2048 | Out of scope for main goal |
| 6 | Zero-point dequant form (HFMA instead of HMUL) | 0% for INT8 (we already use HMUL); relevant for INT4/Q4_K | Out of scope for INT8 v11 |

Combined estimated reach **after all of {1, 2, 3, 4}**: ~35–42 TFLOPS, still short of the
50 TF goal. The remaining gap is structural (V100 INT8 dequant-to-FP16 is not the
hardware's design point; cuBLAS FP16 ceiling is ~85 TF).

## I. Standalone-bench feasibility — **HARDER than initially thought**

Turbomind has a bench harness on disk:
- `src/turbomind/kernels/gemm/test/gemm_bench.cu` (nvbench-based; M = 2^0..2^14)
- `src/turbomind/kernels/gemm/test/test_gemm_v2.cc` (functional, supports `kUint4`,
  `kFloat4_e2m1`, `kFloat8_e4m3`)
- `src/turbomind/kernels/gemm/test/testbed_v3.h` (Testbed class)

**But**: `gemm/CMakeLists.txt` lines 108–137 show that **both `test_gemm_v2` and
`gemm_bench` targets are commented out in upstream**. Resurrecting them requires:

1. Uncomment + adapt the `add_executable` blocks.
2. Build the `gemm2` library, which links against `parser`, `nvidia::cutlass::cutlass`,
   `CUDA::cuda_driver` (cutlass and parser are vendored/third-party — pull through
   submodule init).
3. `gemm_bench` itself additionally needs `core`, `cublas`, `quantization_kernels`,
   `gpt_kernels` — these are turbomind's full inference-side libraries.
4. `FetchContent` nvbench from GitHub (needs network in the dev pod).
5. Resolve any sm_70-specific compile failures (some templates may default to sm_80+).

Realistic effort: **1–2 days** of focused build engineering, not 1–2 hours. Much of that
is build-system archaeology rather than CUDA work.

**Alternative paths considered**:

| Path | Effort | Output | Verdict |
|---|---|---|---|
| Build + run `gemm_bench` standalone | 1–2 days | clean per-shape TFLOPS table | Too heavy for a comparison data point |
| Build full lmdeploy + run `benchmark/profile_throughput.py` against a quantized model | 4–8 hr | end-to-end throughput (tokens/sec), not per-GEMM TFLOPS | Doesn't give us the clean comparison we want |
| Extract published lmdeploy V100 numbers from their docs/blog | <30 min | rough TFLOPS estimate for known model shapes | Cheapest, lowest fidelity, but informative |
| Skip comparison; rely on V11-DESIGN.md upper bounds | 0 | nothing new | Loses the validation that 50 TF is achievable |

**Recommendation**: skip the standalone bench build for now. The structural design ideas
(XOR swizzle, m8n8k4 PTX, phase tables) are independently validated by the source. The
ceiling question — "can a hand-tuned V100 INT4/INT8 GEMM hit 50 TFLOPS?" — is informed by
the **cuBLAS FP16 ceiling at 85 TFLOPS** for our shapes. Turbomind's published end-to-end
numbers suggest V100 quantized inference reaches similar throughput to FP16 baseline, so
the GEMM portion is likely in the 60–75 TF range under their stack. **50 TF is plausible
but not guaranteed.**

If after v11 Steps 1–3 we land at 35–40 TF and want a sanity check before committing to
Step 5 sweep work, we revisit and build `gemm_bench` then.

## J. Updates to V11-DESIGN.md (proposed)

Add to Step 5 (tile sweep) the explicit configurations to try based on turbomind:
- `(BM, BN, BK, WARPS, FRAG_M, FRAG_N) = (128, 256, 16, 8, 8, 16)` — their CTA_M=128_N=256
- `(BM, BN, BK, WARPS, FRAG_M, FRAG_N) = (64, 128, 32, 4, 4, 8)` — their CTA_M=64_N=128
- `(BM, BN, BK, WARPS, FRAG_M, FRAG_N) = (32, 128, 32, 4, 2, 8)` — tall-thin

Add to Step 3 (XOR swizzle): use the **phase-table precomputation** at iterator
construction, not raw `Swizzle::apply` at each store. Reference:
`iterator_sm70.h:134-139` and `iterator_sm70.h:237-256`.

Add as a Step 2.5 (between manual Lds and XOR swizzle): **try CTA_K=16** in the new v11
codepath. Our current v10 uses BK=32. With m8n8k4's K=4 per call, BK=16 = 4 inner steps
which may interleave dequant more efficiently.

## L. Counterfactual — what WE have that they don't, and gaps we missed

The temptation reading turbomind is to assume they're the upper bound. They're not — they
made different trade-offs. This section catalogs both directions.

### L1. What WE have that turbomind sm_70 does NOT

| | Our v10 | Turbomind sm_70 | Implication |
|---|---|---|---|
| **L2 prefetch (`prefetch.global.L2`)** | Yes — issued by tid==0 for A, W_qs, W_scales one K-tile ahead (`v10_kernels.cuh:84-89`) | No — relies on Ldg latency hidden by mma loop length | We anticipate the next tile's gmem traffic; they assume the in-flight Ldg has enough headroom. ncu showed our prefetch is "noise" on v10 baseline — but if v11 shortens the mma loop (BK=16 + finer dequant), prefetch may matter more. **Keep it.** |
| **Per-K=32 quantization block (QK_INT8=32)** | Yes — scale stride is `blocks_per_row = K / 32` | Their U4_g configs use `group_size=128` | Our 4× finer quant gives better numerical quality at 4× more scale loads. **Trade-off, not a defect.** For DSv4 pre-quantized weights at block=32, our layout matches the data; if we ever relax to block=128 we get a SMEM-bandwidth win. |
| **Explicit `__launch_bounds__(WARPS*32, 2)`** | Yes — caps register usage to enable 2 CTAs/SM (3 caused spills) | Relies on compiler heuristics + Stages=2 | We have explicit register-pressure control. **Keep — port to v11.** |
| **Multi-format kernels in one tree** | 4 formats (INT8, INT4 LUT, MXFP4, F8) with dedicated dequant front-ends | sm_70 ships predominantly U4 (INT4) + a few FP4/FP8 variants | Wider format coverage. Not a perf advantage but matters for DSv4 which mixes formats per layer. |
| **HMUL dequant via `__short2half_rn`** | Yes (`v10_kernels.cuh:118`) | `__hmul(x, s1)` from already-half-typed elements | We do an extra short→half conversion per element; their elements arrive already as halves (they keep packed half pairs in SMEM). Our load is gmem-side int8 → SMEM half; theirs may be gmem-side half (or LUT-decoded half) → SMEM. **Minor instruction-count difference; could be worth profiling.** |
| **Per-row tolerance contract enforcement in the bench harness** | Yes — `rel ≤ 1e-3 ∧ p99 ≤ 0.05 ∧ maxabs ≤ 0.1` checked per row | Their testbed compares against cuBLAS but tolerance threshold is not exposed | Our regression guard is tighter. **Keep.** |
| **Bit-correctness audit across full M spectrum** | 882 OK rows, 783 bit-correct, every FAIL attributed to a specific kernel variant | n/a (their tests are pass/fail not graded) | Our diagnostic depth is higher. **Operational discipline win, not a kernel win.** |

### L2. What turbomind has that WE're MISSING (sharper than the H. ranked list)

| | Status | Estimated impact | Captured? |
|---|---|---|---|
| **XOR-swizzle SMEM** | Missing | +6–8% | Yes, V11 Step 3 |
| **m8n8k4 inline PTX** | Missing | +3–10% (register relief, secondary perf) | Yes, V11 Step 1 |
| **Phase-table precomputed swizzle offsets** | Missing | +1–2% | Yes, V11 Step 3 (added this turn) |
| **CTA_K=16 for large tiles** | Missing | +2–4% | Yes, V11 Step 2.5 (added this turn) |
| **`SplitK=true` in kernel configs** | Missing | Format-dependent; possibly +5–15% at small M when grid-residency is the bottleneck | **NEW — capture** |
| **Cache-policy hint `Stream` (cs-evict) on B-side gmem loads** | Missing | +0–3% on memory-bound paths (B is the big tensor, no reuse, evicting saves L1 for A) | **NEW — capture** |
| **Tail tiles for M ≤ 16** (e.g., M=8 × N=128 × K=64) | Missing | +20–50% on M ≤ 16 specifically; near-zero on M=2048 | **NEW — out of scope but worth a note** |
| **MoE-aware GEMM dispatcher** (`moe_utils_v2.cu`, expert_num, exp_per_tok in bench) | Missing | Required for DSv4 production integration — not a kernel-perf win, but the bridge from tc-grid winners → real-world DSv4 throughput | **NEW — capture as integration follow-up** |
| **Runtime kernel-selection tuner** (`tuner/cache_utils.cu`, `measurer.cu`, `sampler.cu`) | Missing — we do compile-time grid search → CSV → pick best | Per-shape best kernel; matches large dispatch tables turbomind ships (14 variants for sm70_884_4) | **NEW — integration improvement, not perf** |
| **GroupSize-aware scale layout** (V operand column-major, packed scales at Stages=2) | Missing | +2–4% for grouped-quant paths | **NEW — out of scope for INT8 block=32, relevant if we add block=128** |
| **CUTLASS reuse for sm_90/sm_80** | Missing | Not relevant for V100 | n/a |
| **`Array<T, N>` typed-fragment abstraction** | Missing — we use raw types | 0% perf; code-clarity only | Adopt during v11 rewrite |
| **Operand-layout abstraction (`operand_sm70_s884.h`)** | Missing — we hardcode | 0% perf; enables format-agnostic kernel templates | Adopt during v11 rewrite |

### L3. Notable counterfactual surprises

These are the items I expected to find on their side but didn't, OR vice-versa.

1. **They DON'T use L2 prefetch hints.** I expected production V100 code to lean on them.
   The fact they don't suggests their mma loop is *intentionally* sized to cover Ldg
   latency — a structural latency-hiding choice rather than an explicit prefetch. If our
   v11 with BK=16 (shorter mma loop per K-stage) shows prefetch becoming more valuable,
   that's a v11-specific tweak; for v10 it was noise.

2. **They use cache-policy `Stream` (cs-evict) on B-side loads.** This is the inverse of
   prefetching — actively evicting B from L1 because there's no reuse. Worth measuring:
   on V100 with 96KB L1+SMEM combined, B-side cs-evict could free L1 capacity for A.
   See `gemm/iterator_sm70.h` (Policy template parameter).

3. **They ship 14 tile variants per format** (sm70_884_4.cu Add<>() count). We ship ~6.
   Their dispatcher picks the best at runtime; ours picks at compile-time via grid sweep.
   The kernel-variant count itself doesn't help perf; the *measurement infrastructure*
   that picks the right one does. Our tc-grid harness is effectively this for our space,
   but we'd need a runtime dispatch hook for DSv4 production.

4. **They use `SplitK=true` for most configs.** SplitK splits the K-dimension across
   multiple CTAs with atomic accumulation in C — boosts grid-residency at small M.
   We deferred this (SPRINT-016-DEFERRED.md). At M=2048 it's not the bottleneck, but at
   M ∈ {16, 32, 64} where their largest tile (CTA_M=128) wastes most of the CTA, splitK
   may be why their small-M throughput holds up.

5. **They have `CHUNK_K=1`** in every shipped sm_70 config. They tried chunked-K (the
   third template parameter in `Config_U4_d<...>`) and shipped 1 — same as our v3/v4
   default. Confirms our chunked-K (v3_b1) is in the same family but not necessarily
   their best.

6. **Our v10's row-major B SMEM was NOT what they do.** Turbomind's B SMEM is XOR-swizzled
   over a *different* base layout (operand-specific, not simple row-major). Our v10 win
   came from a different mechanism (changing WMMA's access pattern) than their swizzle
   does. Both attack the same problem (bank conflicts) but via different paths. **Our
   v10 is a legitimate alternative win, not a stepping stone.** That's worth remembering
   when we benchmark v11 against v10 — if v11-swizzle shows no improvement over v10
   row-major, it's not a v11 implementation bug, it's that both reached the same local
   optimum.

### L4. Implications for v11 plan

Adding to SPRINT-017 phases:
- **NEW Step 2.6** (between manual Lds and XOR swizzle): try `Stream` cache policy on
  B-side gmem loads (inline PTX `ld.global.cs` instead of plain `ld.global` / `__ldg`).
  Cheap to test, may free L1 for A. Estimated +0–3%.
- **NEW Step 6** (out-of-band, post-50TF): SplitK at small M. Defer; not relevant to
  M=2048 goal.
- **NEW deferred**: MoE dispatcher integration into DSv4. Filed as SPRINT-017
  follow-up — required to actually use v11 in production, separate from the TFLOPS goal.

Adding to V11-DESIGN: phase-table precomputation explicit reference, CTA_K=16 trial,
Stream cache policy trial.

## K. References (for v11 implementation)

| Concept | Turbomind file | Lines |
|---|---|---|
| m8n8k4 PTX wrapper | `core/mma.h` | 11–30 |
| m8n8k4_row_row variant | `core/mma.h` | 32–51 |
| Swizzle template | `core/layout.h` | 8–19 |
| SmemLayoutV2 (swizzled SMEM addressing) | `core/layout.h` | 60–90 |
| MMA_884 fragment shapes | `gemm/arch/mma_sm70.h` | (full file) |
| SmemCopy A/B lane offsets | `gemm/arch/smem_copy_sm70.h` | 21–65 |
| Operand layouts | `gemm/arch/operand_sm70_s884.h` | (full file) |
| Config struct + tile sizes | `gemm/arch/config_sm70_s884.h` | (full file) |
| Mainloop pipeline | `gemm/mainloop_sm70.h` | 196–351 |
| Gmem iterator + phase table | `gemm/iterator_sm70.h` | 47–269 |
| Dequant front-end | `gemm/transform.h` | 108–157 |
| Bench entrypoint | `gemm/test/gemm_bench.cu` | (full file) |
| Test entrypoint | `gemm/test/test_gemm_v2.cc` | (full file) |
| Testbed harness | `gemm/test/testbed_v3.h` | (full file) |
| Shipped sm_70 INT4 kernels | `gemm/kernel/sm70_884_4.cu` | (full file) |
