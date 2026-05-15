# SPRINT-024 — Grouped MoE Performance Landing (V100 sm70)

**Status:** DRAFT
**Predecessor:** SPRINT-023 (integration infrastructure; ~16.6 t/s on V100)
**Successor:** SPRINT-025 (Multi-slot decode / Speculative decoding)

---

## 1. Overview

SPRINT-024 focuses on landing the performance gains enabled by the SPRINT-023 infrastructure. The primary bottleneck in the current MoE implementation is kernel-launch overhead at M=1 (decode regime), not memory bandwidth or compute. By transitioning from per-expert serial launches to a single grouped MoE dispatch per layer, we aim to reduce the launch frequency by ~6× (the top-k count) and hit the 24+ t/s performance gate.

Secondary objectives include maintaining FP16 activations across the FFN boundary to reduce DRAM traffic and verifying end-to-end quality on a non-MIN model (real weights).

---

## 2. Use Cases

| Phase | Useful output even if sprint stops here |
|---|---|
| P1 | **Grouped MoE Dispatch**: Amortized launches. Immediate TPS lift for all TURBOMIND-enabled models. |
| P2 | **Quality Verification**: Confidence that the turbomind path produces coherent output on real weights. |
| P3 | **FP16 Activation Path**: Reduced DRAM traffic in the FFN loop; further 5-10% TPS lift. |
| P4 | **Final Performance Report**: Detailed profiling (ncu) and verified benchmark numbers. |

---

## 3. Architecture

### 3.1 Grouped MoE Dispatch

The C ABI already exports `ggml_turbomind_mul_mat_grouped`. SPRINT-024 wires this into `ggml_cuda_mul_mat_id`.

**The mechanism:**
1. **Device-side Sorting**: Use the existing host-side token sorting logic but transition the metadata (token_indices, expert_offsets) to device buffers.
2. **StridedPtr Array**: Build a device array of `StridedPtr` structs, one per expert. Each struct contains:
   - `weight`: Pointer to the packed expert weight.
   - `scales`: Pointer to the per-expert scale buffer.
   - `stride`: The **packed leading dimension**. For sm70 `HMMA_884` `OPERAND_B`, `packed_ld = K * 32` (derived from `Packing_v2`).
3. **Single Launch**: Call `ggml_turbomind_mul_mat_grouped` once per layer. This replaces ~6 serial `ggml_turbomind_mul_mat` calls.

### 3.2 FP16 Activation Boundary

Currently, `ggml_cuda_mul_mat_turbomind` performs an FP32→FP16 cast on input $A$ and an FP16→FP32 cast on output $D$ for every expert call. 
- **Change**: Move the cast upstream of the FFN loop. Input activations to the MoE block are cast to FP16 once; the turbomind kernels ingest FP16 and produce FP16; the final sum/output is cast back to FP32 (or kept FP16 if downstream ops support it).

---

## 4. Implementation

### P0 — Preparation & Baseline (1 day)

1. **Re-verify Baseline**: Run `llama-bench -p 128 -n 32` on `DSv4-Flash-MIN-16e` using SPRINT-023 code to confirm the 16.6 t/s baseline.
2. **Device Metadata Buffers**: Pre-allocate device buffers in `ggml-cuda-turbomind.cu` for `token_indices` (size `N`), `expert_offsets` (size `n_experts + 1`), and the `StridedPtr` array (size `n_experts`).

### P1 — Grouped MoE Dispatch (4-5 days)

1. **P1.1 — Implement `ggml_cuda_mul_mat_grouped_turbomind`**:
   - Resides in `ggml-cuda-turbomind.cu`.
   - Accepts the device pointers for metadata and the `StridedPtr` array.
   - Wraps the `ggml_turbomind_mul_mat_grouped` C ABI call.
2. **P1.2 — Wire into `ggml_cuda_mul_mat_id`**:
   - Predicate: If `src0->buffer->buft == CUDA_TURBOMIND`, skip the per-expert slicing loop.
   - Fill the `StridedPtr` array. **Critical**: Set `stride = K * 32` for sm70.
   - Launch the grouped helper.
3. **P1.3 — Correctness**: Run `test_correctness.cpp` (extended in SPRINT-023) to ensure grouped output matches per-expert output within FP16 ULP.
4. **P1.4 — Performance Gate**: Confirm `tg ≥ 24 t/s` on MIN-16e.

**P1 Gate**: Grouped path is functional; correctness tests pass; decode TPS ≥ 24 t/s.

### P2 — Quality Verification (2 days)

1. **P2.1 — Model Setup**: Download `DSv4-Flash-AVG-16e` (real weights).
2. **P2.2 — Greedy Decode Test**: 
   - Compare 32-token completions vs `exps=CPU` (host reference).
   - **Gate**: ≥75% token ID match on a set of 5 standard prompts.
3. **P2.3 — Perplexity (Optional)**: If time permits, run a short `perplexity` pass on WikiText-2 to ensure no catastrophic divergence.

**P2 Gate**: Real-model completions are coherent and match CPU baseline.

### P3 — FP16 Activation Boundary (2-3 days)

1. **P3.1 — Upstream Cast**: Modify the FFN dispatch in `ggml-cuda.cu` to cast the input activation tensor to FP16 before the MoE routing.
2. **P3.2 — Direct FP16 Dispatch**: Update `ggml_cuda_mul_mat_grouped_turbomind` to accept FP16 $A$ and return FP16 $D$ without internal casting.
3. **P3.3 — Performance Delta**: Measure the speedup from reduced DRAM traffic. Target: +1-2 t/s.

**P3 Gate**: Casts removed from the hot loop; 5-10% performance gain measured.

### P4 — Final Measurement & Close-out (2 days)

1. **P4.1 — Full Sweep**: `llama-bench` on MIN-8e, 16e, 32e.
2. **P4.2 — NCU Profiling**: Capture `HMMA active %` and `cudaLaunchKernel` latency. Verify launch overhead is no longer the dominant cost.
3. **P4.3 — Documentation**: Write `SPRINT-024-SUMMARY.md` with final numbers.

---

## 5. Files Summary

### Modified Files

| Path | Change |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cu` | Add grouped dispatch helper; manage device metadata buffers. |
| `ggml/src/ggml-cuda/ggml-cuda-turbomind.cuh` | Export grouped helper. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_mul_mat_id` predicate + metadata prep; move FP16 boundary. |
| `ggml/vendor/turbomind/api.cc` | Ensure `ggml_turbomind_mul_mat_grouped` uses `packed_ld` correctly. |

---

## 6. Definition of Done

### Ship-blockers

1. ✅ Grouped MoE dispatch path exists and is used by default for `CUDA_TURBOMIND` tensors.
2. ✅ Decode TPS on `DSv4-Flash-MIN-16e` ≥ **24 t/s** on V100.
3. ✅ Output matches per-expert path within FP16 ULP.
4. ✅ `DSv4-Flash-AVG-16e` (real model) produces coherent output matching CPU reference (≥75% tokens).
5. ✅ FP16 activations maintained across the FFN boundary.
6. ✅ No regression in `test_correctness.cpp`.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | `mul_mat_grouped` has hidden overhead | Low | Medium | P0 microbench already showed launch-bound nature; amortization is mathematically sound. |
| 2 | StridedPtr alignment issues on device | Medium | Medium | Use `ggml_cuda_pool` for the metadata buffers to ensure alignment. |
| 3 | Quality match < 75% on real weights | Low | High | Investigate epsilon drift in the sum reduction; check scale precision. |

---

## 8. Security

- No changes to the security posture. 
- The C ABI boundary remains the same. 
- Metadata buffers are internal and sized based on model parameters.

---

## 9. Dependencies

- `libggml-turbomind.so` (from SPRINT-023).
- `DSv4-Flash-AVG-16e` GGUF for quality testing.
- V100-SXM2-32GB hardware.

---

## 10. Open Questions

1. **Multi-slot impact**: How much does `mul_mat_grouped` gain when `M > 1` (parallel sequences)?
2. **F8_E4M3_B128 dense**: Should we enable TURBOMIND for non-expert dense layers in this sprint? (Intent says only if compute fraction is high).
3. **Absolute vs Relative gates**: Should we finalize the move to relative error gates for all MoE tests?
