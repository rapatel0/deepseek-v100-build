# v11 Design — m8n8k4 inline PTX + XOR-swizzled SMEM (path to 50 TFLOPS)

Status: NOT YET IMPLEMENTED. v10 is the current production champion at 29.49 TFLOPS (M=2048).

## Why v11 is needed

REPORT-11 documents that all wmma::* API optimizations are exhausted at ~29.5 TFLOPS. The remaining ~70% headroom to cuBLAS-FP16's 85 TFLOPS ceiling requires changes that wmma::load_matrix_sync can't accommodate:

1. **XOR-swizzled SMEM** to actually eliminate the 334M bank conflicts (not just shift them as padding does).
2. **Finer fragment control** to interleave dequant work with mma issue.

Both require dropping `wmma::load_matrix_sync` and using manual `ld.shared.b128` to populate fragments. That's the v11 contract.

## Reference implementation (turbomind)

Local clone: `/tmp/turbomind/lmdeploy/src/turbomind/kernels/`

Key files to study, in order:
1. `core/mma.h` — `mma_m8n8k4_row_col` inline PTX wrapper. Already complete; copy verbatim.
2. `core/layout.h` — `Swizzle<Bits, Base, Shift>` template; XOR pattern is `offset ^ ((offset & ((1<<Bits)-1) << (Base+Shift)) >> Shift)`. Apply at SMEM index computation.
3. `gemm/arch/mma_sm70.h` — `SM70_MMA_884` struct defining fragment shapes (M=8, N=32, K=8 per warp call; one m8n8k4 PTX = 8x8 output, two stacked for K=8).
4. `gemm/arch/smem_copy_sm70.h` — `SmemCopy_MMA_884_A`/`_B` define the lane→element mapping for fragment loads via `Lds` (`ld.shared.b32`). These are the lane offsets we must replicate.
5. `gemm/mainloop_sm70.h` — full pipelined mainloop (~350 lines). The structural reference.

## Concrete steps for next session

### Step 1: Bare-bones m8n8k4 mma replacement (NO swizzle yet)

Goal: verify the m8n8k4 PTX produces the same numerical result as wmma::mma_sync. NO performance change expected — this is just a structural prep.

Approach:
- Copy v10_kernels.cuh → v11a_kernels.cuh.
- Keep `wmma::load_matrix_sync` for A and B (same SMEM layout).
- Replace `wmma::mma_sync(c[fm][fn], a[fm], b[fn], c[fm][fn])` with: extract halves from a[fm].x[], b[fn].x[], call 4× `mma_m8n8k4_row_col`, write back to c[fm][fn].x[].
- Build, test bit-correctness against v10. Expect identical output.
- This validates the m8n8k4 PTX wrapper works in our codebase.

### Step 2: Manual Lds fragment load (still NO swizzle)

Goal: reproduce wmma::load_matrix_sync's behavior via manual ld.shared.b128 + thread-local fragment assembly.

Approach:
- For A: each thread loads 8 halves via 2× `ld.shared.b32`. Use turbomind's `SmemCopy_MMA_884_A::unique` to compute lane offsets.
- For B: same pattern, with `SmemCopy_MMA_884_B::unique`.
- Verify bit-correctness vs v10. Numerical match means lane mapping is right.

### Step 3: Add XOR swizzle to SMEM B store + load

Goal: eliminate bank conflicts on B loads.

Approach:
- Define `Swizzle<3, 3, 3>` (or tune via grid search Bits ∈ {1,2,3,4}, Base=3, Shift=3 is canonical for 16-byte vectors).
- Apply swizzle to B SMEM index in BOTH store (in load_tile) and load (in mma loop).
- Verify correctness. Expect bank conflicts on B loads to drop from 334M to ~0.
- Measure perf — target 35-45 TFLOPS at M=2048.

### Step 4: (Optional) Apply same to A

Goal: also reduce A-side bank conflicts (smaller, but stacks).

### Step 5: Tune CTA tile (BM, BN, BK), warps, and Stages

Goal: with swizzle eliminating the bank-conflict floor, larger tiles or wider pipelines may now help.

## Verification gates

- Every step must pass `rel ≤ 1e-3` against the reference matmul (already in tc-grid harness).
- Use `compute-sanitizer --tool memcheck` on first launch to catch OOB.
- Use `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` to confirm swizzle eliminates conflicts.

## Estimated effort

- Step 1: 2 hr (mechanical translation, build/test fail-loop)
- Step 2: 3 hr (lane mapping debugging is finicky)
- Step 3: 2 hr (swizzle math + verify)
- Step 4: 1 hr
- Step 5: 2 hr (grid sweep)

Total: ~10 hr of focused engineering.

## Risk factors

- m8n8k4 lane→element mapping is "opaque" per PTX docs. Turbomind's specific lane offsets work in their codebase; subtle differences in fragment orientation could silently corrupt output. **Mitigation**: bit-compare after every step.
- XOR swizzle must be applied symmetrically (store side and load side). Easy to typo, hard to debug. **Mitigation**: write a unit test that fills SMEM, applies swizzle, reads back, confirms data integrity before integrating with the matmul.
- Compiler may not optimize hand-rolled ld.shared as well as wmma. **Mitigation**: check generated SASS via `cuobjdump --dump-sass`.

## Fallback if v11 doesn't deliver

If after ~10 hr of effort v11 is still < 35 TFLOPS, the remaining options are:
1. **CUTLASS direct integration** — pull in CUTLASS V100 INT8 GEMM as a library. ~4 hr integration. Ceiling: probably 60-75 TFLOPS (theoretical max for V100 INT8 dequant-to-FP16).
2. **Pre-dequant W to FP16, call cuBLAS** — works for benchmarks, but the FP16 weights blow up the memory budget for full DSv4 inference (28GB of FP16 weights for typical model size). Acceptable only for prefill or single-layer benchmarks.
3. **Accept ~30 TFLOPS as the ceiling** — write up the engineering tradeoff explicitly and ship v10 + HMUL.
