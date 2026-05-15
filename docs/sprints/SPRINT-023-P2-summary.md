# SPRINT-023 P2 — Pack + dispatch correctness, PASS

**Date:** 2026-05-15

## Result

`test_ggml_turbomind_correctness ./libggml-turbomind.so` →

```
[F8_E4M3_B128] PASS — rel=1.67e-04, max_abs=4.78e-01 (gate 2.0 = 2-ULP at max_ref=1563)
[MXFP4]       PASS — rel=1.71e-04, max_abs=1.12e-02 (gate 6.25e-02 = 2-ULP at max_ref=42)
```

Both quant paths produce outputs within FP16 precision of a host-dequant
reference matmul. Random fixture, M=8 N=256 K=256, FP16 activations.

## Two fixes that unlocked the gate

1. **Packed B `ld` is K×32, not K.**
   `convert_v3.cu` runs `Packing_v2<HMMA_884 | OPERAND_B | _1, kRowMajor>::apply({m,k}) = {m/32, k*32}` on the descriptor it returns, so the packed output's leading dimension is 32× the unpacked source ld. We were passing `Bdesc.ld = K` at mul-mat time, which only happens to align reads modulo the warp-N tile — explaining why the first 32 columns of the kernel output matched ref and the rest were garbage.
   - `api.cc:ggml_turbomind_mul_mat` now re-derives the packed ld via the same `Packing_v2` formula (`b_packed_ld = packed_cols` for kRowMajor, `packed_rows` for kColMajor).

2. **MXFP4 scales need `AdjustUe8m0ScaleForHalf` before convert.**
   Raw E8M0 bytes encode `2^(e-127)`; the sm70 transform pipeline reads them directly as FP16 exponent fields, which expects bias 15. Without the adjust, e=127 → FP16 exp 127 (overflow → inf), explaining the all-NaN MXFP4 output.
   - `api.cc:pack_weight_expert` now calls `turbomind::AdjustUe8m0ScaleForHalf(raw_scales, n_scales, stream)` for `GGML_TM_DTYPE_MXFP4` between deinterleave and conv_s.
   - FP8 doesn't need it — our deinterleave already converts E8M0 → FP16 inline (`fp16_bits = (e - 112) << 10`).

## Things ruled out / abandoned

- Adding non-grouped `Config_E4M3<-1>` / `Config_MXF4<-1>` kernel registrations to `sm70_884_{4,8}.cu`. Tried, made FP8 worse, reverted — the grouped variant (`group_axis=0`, `MODE_A=kIndexed`) works fine for non-grouped problems via `accept(kIndexed, kFlat) == true` once the ld is correct.
- Routing single-expert calls through `ggml_turbomind_mul_mat_grouped` to "match the kernel's authored mode." Built the test path (StridedPtr per-expert layout, identity token_indices, [0,M] offsets), but the symptom was identical — the cols 32+ break was the ld bug, not the dispatch path. Test reverted to plain `mul_mat`.

## Gate philosophy

The original SPRINT-015 P2 gate (`max_abs ≤ 2e-2`, `p99 ≤ 1e-2`) is unsuitable for outputs at this magnitude — FP16's ULP at `|ref|=1563` is already 1.0. The test now:

- Hard limit on **relative error**: `rel = sum|diff| / sum|ref| ≤ 1e-3`
- Soft floor on **max_abs**: `max(2e-2, 2 × FP16_ULP_at_max_ref)`

This stays tight at low magnitudes (matches the original gate for MXFP4 with max_ref=42) and scales correctly at large magnitudes (FP8 with max_ref=1563).

## Files touched

- `ggml/vendor/turbomind/api.cc` — packed-ld fix in `mul_mat` and `mul_mat_grouped`; E8M0 adjust call in `pack_weight_expert`; both mul-mat paths now derive Bdesc/Vdesc from converter rather than hard-coding orders.
- `ggml/vendor/turbomind/ggml-turbomind-deinterleave.{cu,h}` — kernels emit uint16 [K, N] directly (skips `extend_to_u16`), scales in [K/g, N] (FP16 for FP8, raw E8M0 byte for MXFP4 — adjustment happens in `api.cc`).
- `ggml/vendor/turbomind/test_correctness.cpp` — gate updated; FP8 fixture skips 0x7F/0xFF (NaN sentinel); diagnostic prints removed.
- `ggml/vendor/turbomind/CMakeLists.txt` — adds `test_ggml_turbomind_correctness` target.

## Next: P3-P5

P2 close-out unblocks:
- P3 — `CUDA_TURBOMIND` buffer type + upload hook in ggml-cuda
- P4 — Dispatcher integration in `mmq.cu`/`mmvq.cu`
- P5 — REPORT-17 with `llama-bench` on the new path
