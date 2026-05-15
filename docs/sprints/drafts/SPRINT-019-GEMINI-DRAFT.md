# SPRINT-019 — V11 Performance Push (The 50 TF Goal)

## Overview

SPRINT-019 is the definitive performance engineering push for the v11 INT8 GEMM kernel family on NVIDIA V100 (SM70/Volta). Following the successful production shipment of v11 at 35.08 TFLOPS (+18.9% vs. v10), we have identified a concrete path to the **50 TFLOPS threshold**. 

Our current champion, `mm_int8_lut_v11<128, 128, 16, 4, 8, 2>`, is bottlenecked by a combination of high HBM latency (21.45% long scoreboard stalls) and a 62.5 TFLOPS theoretical limit imposed by FP32 accumulation. To cross the 50 TF barrier, we must transition to **FP16 accumulation**, which doubles the tensor-pipe ceiling to 125 TFLOPS, while simultaneously optimizing the pipeline to hide memory latency.

This sprint is governed by a mandate of **Methodical Discipline**. We will not skip steps. We will not assume correctness. Every lever will be empirically validated using a stack of low-level profiling tools (ncu, nsys), bit-exact correctness checks, and architectural comparison against CUTLASS.

---

## Use Cases

### 1. DeepSeek-V4 Dense Layer Inference
Standard transformer FFN and Attention layers with large M (1024, 2048, 4096) and N=K=7168. These operations are throughput-sensitive and benefit most from the 50 TF push.

### 2. DeepSeek-V4 MoE Expert Routing
Mixture-of-Experts (MoE) layers where expert requests are processed in small, variable batches (M=64, 256). The v11 architecture currently regresses here relative to v10s SplitK; we will close this gap to ensure MoE latency targets are met.

### 3. High-Occupancy Serving
Maximizing wave parallelism and SM occupancy to ensure the GPU remains utilized during multi-tenant workloads. By minimizing the SMEM and register footprint per CTA, we enable more concurrent warps to hide memory and instruction latency.

---

## Architecture Deep-Dive: The v11 SM70 Stack

The v11 kernel architecture is a manual realization of the SM70 Tensor Core pipeline. Unlike v10, which used the `wmma::` abstraction, v11 directly invokes PTX for fine-grained control.

### 1. PTX-Level Tensor Core Control
We utilize `mma.sync.aligned.m8n8k4.row.col.{f32,f16}.f16.f16.{f32,f16}`. 
- **A-matrix**: 8 halves per lane, loaded from SMEM via `ld.shared.b128`.
- **B-matrix**: 8 halves per lane, loaded from SMEM via `ld.shared.b128`.
- **C-matrix**: 8 floats (FP32 acc) or 4 halves (FP16 acc) per lane.

### 2. SMEM Layout and Lane Mapping
The v11 layout uses **Column-Major B** with `BK_PAD = BK + 8` to ensure uint4 alignment for vectorized loads. 
- **SmemCopy_MMA_884_A**: Lane mapping `aL_m = (lane/16)*4 + (lane%4)`.
- **SmemCopy_MMA_884_B**: Lane mapping `bL_n = (lane/16)*4 + (lane&12)*2 + (lane%4)`.
This mapping allows a single `ld.shared.v4.u32` to load 8 contiguous K-elements (for A) or 8 contiguous N-elements (for B) into the thread's fragment registers.

### 3. FP16 Accumulator and the Epilogue Problem
The SM70 `m8n8k4` instruction with FP16 accumulator is twice as fast as the FP32 variant. However, it introduces a **fragment layout mismatch**. The internal register layout of an FP16 accumulator fragment does not map linearly to a 2D matrix tile in a way that is easily dequantized.
- **Solution**: We implement an **SMEM Round-Trip**. After the main K-loop, the FP16 `c_frag` is written to a specialized SMEM tile. It is then read back into registers as FP32. This "layout normalization" allows us to use the existing vectorized dequantization and bias-addition logic.

### 4. Pipeline Evolution
- **2-Stage (Current)**: LDG -> SMEM -> MMA. Relies on double-buffering in SMEM.
- **3-Stage (Proposed)**: LDG -> RMEM -> STS -> SMEM -> MMA. Introduces a register-level buffer (RMEM) to hide the `long_scoreboard` stalls (HBM latency). This requires ~16-32 additional registers per thread.

---

## Glossary of Volta (SM70) Terms

- **MMA (Matrix Multiply-Accumulate)**: The fundamental Tensor Core instruction.
- **Fragment**: A thread-local register array representing a piece of the A, B, or C matrix.
- **LDG (Load Global)**: A load instruction from Global Memory (HBM).
- **STS (Store Shared)**: A store instruction to Shared Memory (SMEM).
- **LDS (Load Shared)**: A load instruction from Shared Memory (SMEM).
- **Long Scoreboard Stall**: A warp stall waiting for data from HBM (LDG).
- **Short Scoreboard Stall**: A warp stall waiting for a register write from a local pipeline (MMA, MIO).
- **MIO (Memory/IO) Throttle**: A stall caused by saturating the SMEM or Texture unit pipelines.
- **Bank Conflict**: A penalty when multiple threads access the same 32-bit SMEM bank.
- **Wave**: A group of CTAs that can run concurrently on the GPU's SMs.

---

## Hardware Constraints for SM70 (Volta)

These are the "hard walls" our grid-search must respect:

| Resource | Limit | Impact of Violation |
|---|---|---|
| **Registers per Thread** | 255 (max), 128 (for 4 CTAs/SM) | Register spilling to stack (local memory) |
| **SMEM per SM** | 96 KB (configurable) | Reduced occupancy (fewer CTAs/SM) |
| **Max Warps per SM** | 64 | Reduced latency hiding capability |
| **Max Threads per CTA** | 1024 | Limits CTA tiling options |
| **Max CTA per SM** | 32 | Limits wave parallelism for small tiles |

---

## Mathematical Foundations of Optimization

### 1. BK vs. Register Pressure Tiling
The choice of `BK` (K-dimension tile size) dictates the SM resources:
- **BK=32**: Fewer K-loop iterations, higher register pressure.
- **BK=16**: More K-loop iterations, lower register pressure. This allows the compiler to keep more of `c_frag` in registers and increases CTA occupancy by reducing SMEM usage from 33KB to 12KB.

### 2. FP16 Accumulator Theoretical Peak
The V100 Tensor Core peak is calculated as:
`Peak = 2 (ops/FMA) * 8 (m) * 8 (n) * 4 (k) * Clocks * SMs / CyclesPerMMA`.
For FP16 accumulation, `CyclesPerMMA` is half that of FP32, doubling the theoretical limit from 62.5 TF to 125 TF.

### 3. Register Spilling Constraint
A 128x128 tile with 4 warps allocates 128 floats per lane for `c_frag`. At 4 bytes/float, this is 512 bytes per thread. The register limit for 4 CTAs/SM is 128 registers (512 bytes). This leaves **zero** headroom for A/B fragments or instruction state. This is why BM=192 or 256 tiles **must** use SMEM spilling to maintain occupancy.

---

## Methodology: The "Methodical" Mandate

We follow a strict **Verification Stack** for every change.

### Tier 1: Isolated Correctness
Before integration into the production dispatcher:
- Build a CPU-reference test in `tests/test_<lever>.cu`.
- Compare GPU output against CPU model (rel ≤ 1e-3).
- Run `compute-sanitizer --tool memcheck`.

### Tier 2: Production Integration
- Wire into `launch_int8.cu` and `main.cu`.
- Regression-test against v10 baseline and previous v11 champion.
- Check for `ptxas` register spill warnings.

### Tier 3: Performance Characterization (Ncu)
- Headline TFLOPS across all M values.
- `ncu` stall breakdown (Long Scoreboard, Short Scoreboard, MIO Throttle).
- Compute utilization (`hmma_cycles_active.pct_of_peak`).

### Tier 4: Comparison against CUTLASS
- Run `kernels::int8_cutlass::Gemm70` (version=40) on the same shape.
- Report the `v11 / CUTLASS` ratio.

---

## Implementation (Phased)

### Phase 6.1: FP16 Accumulator + SMEM Round-Trip
**Strategic Intent**: Attack the 62.5 TF ceiling of FP32 accumulation.
**ROI**: +30–50% (High Risk / High Reward).

**Sub-Phase 6.1.1: Fragment Mapping Derivation**
- **Objective**: Determine the internal layout of FP16 `c_frag` for `m8n8k4`.
- **Step 1**: Review `v100_wmma_half_float_frag_layout_mismatch.md`.
- **Step 2**: Create `tests/test_f16_acc_layout.cu`.
- **Step 3**: Use the `MMA_FP16_ACC` PTX wrapper to initialize a known identity fragment.
- **Step 4**: Derive the index: `lane_to_row = (lane % 4) * 2 + (element / 2)`, `lane_to_col = (lane / 4)`.
- **Decision Gate**: Correctness of the layout map verified by bit-exact match on an 8x8 tile.

**Sub-Phase 6.1.2: SMEM Epilogue Development**
- **Objective**: sidestep the layout mismatch via a round-trip through SMEM.
- **Step 1**: Define `__shared__ half sC_scratch[128][128]` in the kernel template.
- **Step 2**: Implement `store_f16_fragment_to_smem` using the derived map.
- **Step 3**: Implement `load_f16_smem_to_f32_registers` with `__half2float`.
- **Step 4**: Integrate into `mm_int8_lut_v11_f16acc`.
- **Decision Gate**: Pass `compute-sanitizer --tool memcheck` on the integrated kernel.

**Sub-Phase 6.1.3: Benchmarking**
- **Step 1**: Run `tc-grid` grid-search.
- **Step 2**: Profile winner at M=2048.
- **Step 3**: Capture `hmma_cycles_active.pct_of_peak`.
- **Decision Gate**: TFLOPS ≥ 40.

---

### Phase 6.2: Per-Shape 3-Stage Pipeline
**Strategic Intent**: Hide HBM latency (21.45% Long Scoreboard stall).
**ROI**: +5–8%.

**Sub-Phase 6.2.1: Register Budget Analysis**
- **Objective**: Identify tiles that can afford a 3rd stage.
- **Step 1**: Use `ncu --metrics launch__registers_per_thread`.
- **Step 2**: 4 CTAs/SM requires ≤ 128 registers.
- **Step 3**: 3rd stage adds ~16-32 registers (RMEM buffers for A/B).
- **Decision Gate**: Only shapes with baseline < 100 registers are candidates for 3-stage.

**Sub-Phase 6.2.2: Pipeline Implementation**
- **Objective**: Implement `LDG -> RMEM -> STS -> SMEM -> MMA`.
- **Step 1**: Use `#pragma unroll` to encourage the compiler to interleave instructions.
- **Step 2**: Manually insert `__syncthreads()` and double-buffering indices.
- **Step 3**: Verify that `compute-sanitizer --tool racecheck` passes.
- **Decision Gate**: `long_scoreboard` stall must drop by > 5% relative to Phase 6.1.

**Sub-Phase 6.2.3: Nsys Verification**
- **Objective**: Confirm compute/memory overlap.
- **Step 1**: Capture `.nsys-rep` for M=2048.
- **Step 2**: Inspect SM Trace for overlapping LDG and MMA blocks.
- **Decision Gate**: Evidence of overlap documented in `docs/nsys/`.

---

### Phase 6.3: SplitK Port to v11
**Strategic Intent**: Fix the M=64 regression (parity with v10s = 20.31 TF).
**ROI**: +50–80% at small-M.

**Sub-Phase 6.3.1: Kernel Adaptation**
- **Objective**: Implement `grid.z` slicing.
- **Step 1**: Map `blockIdx.z` to the K-dimension sub-range.
- **Step 2**: Each CTA computes a partial `c_frag`.
- **Step 3**: Port `v11` MMA and LDS logic into the SplitK template.
- **Decision Gate**: Isolated test passes for K-slices.

**Sub-Phase 6.3.2: Atomic Epilogue**
- **Objective**: Safe merging of partial sums.
- **Step 1**: Use `atomicAdd(float*, float)` in the epilogue.
- **Step 2**: Ensure dequantization happens *before* the atomic add (since scales vary per CTA).
- **Decision Gate**: Bit-correctness at M=64 against v10 reference.

**Sub-Phase 6.3.3: Optimization**
- **Step 1**: Sweep SplitK factors {2, 4, 8, 16}.
- **Step 2**: Analyze L2 atomic contention in `ncu`.
- **Decision Gate**: M=64 TFLOPS ≥ 20.

---

### Phase 6.4: Larger CTA Tile + c_frag SMEM Spill
**Strategic Intent**: Amortize K-loop overhead for M=4096.
**ROI**: +5–10%.

**Sub-Phase 6.4.1: Spill Architecture**
- **Objective**: Use SMEM as an overflow for registers.
- **Step 1**: Implement a state machine to rotate `c_frag` between registers and SMEM.
- **Step 2**: Implement `sts.b128` (spill) and `lds.b128` (reload) in the K-loop.
- **Decision Gate**: No regression in `maxabs` (verifies spill logic).

**Sub-Phase 6.4.2: Grid Search**
- **Objective**: Find the new champion for large-M.
- **Step 1**: Register BM=192 and BM=256 tile variants.
- **Step 2**: Run `tc-grid --m 4096`.
- **Decision Gate**: Headline TF at M=4096 must improve relative to BM=128.

---

### Phase 6.5: PRMT-Vectorized A-side Load
**Strategic Intent**: Reduce non-tensor instruction pressure.
**ROI**: +1–2%.

**Sub-Phase 6.5.1: Implementation**
- **Step 1**: Replace dequant logic with `prmt.b32` bias trick.
- **Step 2**: Use `__float22half2_rn` for vectorized load+convert.
- **Decision Gate**: Correctness pass.

---

### Phase 6.6: Multi-shape MoE Validation
**Strategic Intent**: Robustness check for DSv4 deployment.

**Sub-Phase 6.6.1: Sweep**
- **Step 1**: Execute `tc-grid` on DSv4 shapes (N=7168 K=7168, N=2048 K=18944, N=18944 K=2048).
- **Step 2**: Identify per-shape champions.
- **Decision Gate**: Results recorded in `REPORT-13.md`.

---

## Appendix A: Technical Deep Dives

### A.1 SmemCopy_MMA_884 Lane Mapping Derivation
The v11 kernel relies on a specific mapping of threads within a warp to matrix elements in SMEM to allow vectorized `ld.shared.b128` loads. 

**For Matrix A (M-major):**
A single `ld.shared.v4.u32` (128 bits) loads 8 `half` elements. In the K-direction, these 8 elements form a "quad".
The mapping is:
- `quad_idx = lane / 16` (0 or 1)
- `m_in_quad = lane % 16`
- `row = (quad_idx * 4) + (m_in_quad % 4)`
- `col_quad = (m_in_quad / 4)`
This results in each thread owning 8 contiguous K-elements at a specific row offset. This mapping is critical for the `mma.m8n8k4` instruction, which expects the A-fragment to be distributed such that each thread has the correct rows for its lane.

**For Matrix B (N-major):**
Similar to A, but with N-major layout in SMEM.
- `quad_idx = lane / 16`
- `n_in_quad = lane % 16`
- `col = (quad_idx * 4) + (n_in_quad % 4)`
- `row_quad = (n_in_quad / 4)`
This mapping ensures that 8 contiguous N-elements are loaded into each thread's B-fragment, aligning with the Tensor Core's expectations for row-major A and column-major B inputs.

---

### A.2 INT8 Dequantization: The PRMT Bias Trick
Converting INT8 to FP16 usually requires a sign-extension and a floating-point multiply. On SM70, we can optimize this using the `prmt.b32` instruction.

**The Math:**
1.  Take an INT8 value `x`.
2.  XOR with `0x80` to convert to an unsigned offset: `y = x ^ 0x80`.
3.  Pack four such bytes into a 32-bit register.
4.  Use `prmt.b32` to interleave these bytes with a constant bias `0x64` (which corresponds to the exponent bits of an FP16 value).
5.  Subtract the bias `1152` in FP16 space.
This sequence replaces a loop of branches and shifts with a single SIMD-style conversion, significantly reducing the `short_scoreboard` stalls caused by ALU-to-Tensor dependencies.

---

### A.3 SplitK Atomic Contention Model
SplitK improves performance by increasing parallelism, but it introduces contention at the L2 cache for the `atomicAdd` operation.

**Performance Model:**
`T_total = T_compute / Factor + T_atomic * Factor`
- `T_compute`: Time for a single CTA to process the entire K-dimension.
- `Factor`: SplitK factor (number of CTAs per output tile).
- `T_atomic`: Overhead of a single atomic operation at the L2.
On SM70, `T_atomic` is significant because the L2 atomic units are shared across multiple SMs. Our goal in Phase 6.3 is to find the `Factor` that minimizes `T_total`. If `Factor` is too high, `T_atomic * Factor` dominates; if too low, `T_compute / Factor` is too large to hide latency.

---

### A.4 SMEM Bank Conflict Analysis: BK=16 vs BK=32
SMEM is organized into 32 banks, each 4 bytes wide. Successive 4-byte words map to successive banks.

**BK=32 (v10 style):**
A row of 32 `half` elements is 64 bytes. In row-major SMEM, thread `i` and thread `i+16` will access the same bank (64 bytes / 4 bytes per bank = 16 banks offset). This causes a 2-way bank conflict for every `ld.shared`.

**BK=16 (v11 style):**
A row of 16 `half` elements is 32 bytes. In row-major SMEM, thread `i` and thread `i+16` map to the exact same bank (32 bytes = 8 banks). This causes a 4-way bank conflict.
**However**, by using **Column-Major B** and `BK_PAD=24`, we change the stride to 48 bytes. Since 48 is not a multiple of 32, the bank mapping rotates, reducing the effective conflict rate to 2-way or even 1-way depending on the access pattern. This is why BK=16 with padding beats BK=32 without padding.

---

### A.5 3-Stage Asynchronous Pipeline Pseudo-code
The 3rd stage allows us to issue HBM loads for the *next* K-tile while the *current* K-tile is being processed by the Tensor Core.

```cpp
// Pseudo-code for the 3rd stage loop
#pragma unroll 1
for (int k = 0; k < K_tiles; ++k) {
    // 1. Issue LDG for tile k+1
    if (k + 1 < K_tiles) {
        load_global_to_rmem(k + 1);
    }
    
    // 2. MMA for tile k
    mma_sync(fragA, fragB, fragC);
    
    // 3. STS for tile k+1 (from rmem)
    if (k + 1 < K_tiles) {
        store_rmem_to_shared(k + 1);
    }
    
    // 4. LDS for tile k+1 (into fragA, fragB)
    __syncthreads();
    load_shared_to_frag(k + 1);
}
```
The compiler's ability to interleave the `mma_sync` with the `load_global_to_rmem` is the key to dropping the 21% `long_scoreboard` stall.

---

## Appendix B: Detailed Benchmarking Commands

### B.1 Full Correctness Sweep
```bash
./build/tc-grid \
  --m-list 64,256,1024,2048,4096 \
  --nk 7168 \
  --dist uniform_small \
  --compare v10 \
  --tolerance 1e-3
```

### B.2 Grid Sweep for Champion Identification
```bash
./build/tc-grid \
  --m 2048 \
  --nk 7168 \
  --grid-sweep \
  --output docs/grid-sweep-SPRINT-019.csv
```

### B.3 Ncu Stall Analysis Workflow
```bash
# Capture full report
ncu --export docs/ncu/phase_6_1_champ.ncu-rep \
    --force-overwrite \
    --target-processes all \
    --kernel-name-filter "mm_int8_lut_v11" \
    ./build/tc-grid --m 2048 --nk 7168 --nk 1

# Export specific metrics to CSV for analysis
ncu --import docs/ncu/phase_6_1_champ.ncu-rep \
    --csv \
    --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct \
    > docs/ncu/long_scoreboard.csv
```

---

## Appendix C: Success Criteria & Decision Matrix

### C.1 Phase-Specific Decision Gates

| Phase | Success Metric | Gate Value | Action on Failure |
|---|---|---|---|
| 6.1 | HMMA Throughput | ≥ 40 TF | Revert; check lane mapping. |
| 6.2 | Long Scoreboard Stall | < 15% | Revert; check Syncthreads. |
| 6.3 | M=64 Throughput | ≥ 20 TF | Revert; check SplitK factor. |
| 6.4 | BM=256 TFLOPS | > BM=128 | Revert; check spill latency. |

### C.2 Champion Selection Matrix
When multiple tile shapes are measured, the "Champion" is selected based on:
1.  **Correctness**: Must pass `rel ≤ 1e-3`.
2.  **Peak TFLOPS**: Highest value at M=2048.
3.  **Warp Occupancy**: Highest `active_warps_per_sm`.
4.  **Bank Conflict Count**: Lowest `shared_bank_conflicts` from Ncu.

---

## Appendix D: Final Sprint Checklist

- [ ] All phases (6.1-6.6) implemented and verified.
- [ ] Bit-correctness sweep passes for all DSv4 shapes.
- [ ] Ncu reports archived for all champions.
- [ ] Nsys timelines confirm compute/memory overlap for 3rd stage.
- [ ] CUTLASS ratio recorded for the final champion.
- [ ] `REPORT-13.md` finalized and committed.
- [ ] No untracked `ptxas` register spill warnings.
- [ ] M=64 parity with v10s achieved.
- [ ] M=2048 goal of 50 TF hit (or ceiling documented).
