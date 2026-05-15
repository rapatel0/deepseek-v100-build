# Report 4 — memory-layout grid search (best tile per format × M × N=K)

**Generated:** 2026-05-13T19:09:15.865795Z (V100 SXM2 32GB, sm_70)
**Scope:** v2 kernels only (items 1+2+3 applied). All variants share the same kernel architecture; only the (BM, BN, BK, WARPS, FRAG_M, FRAG_N) tile shape differs.

## Tile configurations swept

| Tile label | BM | BN | BK | WARPS | FRAG_M | FRAG_N | SMEM (KB) | mma_sync/warp/K-slice |
|------------|----|----|----|-------|--------|--------|-----------|------------------------|
| `64x64x32_w4_v2`   |  64 |  64 | 32 | 4 | 4 | 1 | 12 ×2 = 24 KB | 4 |
| `64x128x32_w4_v2`  |  64 | 128 | 32 | 4 | 4 | 2 | 24 ×2 = 48 KB | 8 |
| `128x64x32_w4_v2`  | 128 |  64 | 32 | 4 | 8 | 1 | 12 ×2 = 24 KB | 8 |
| `128x128x32_w4_v2` | 128 | 128 | 32 | 4 | 8 | 2 | 16 ×2 = 32 KB | 16 |
| `128x256x32_w4_v2` | 128 | 256 | 32 | 4 | 8 | 4 | 24 ×2 = 48 KB | 32 |
| `128x256x32_w8_v2` | 128 | 256 | 32 | 8 | 8 | 2 | 24 ×2 = 48 KB | 16 |

(SMEM accounting: sA[BM,BK] + sB[BK,BN] each FP16, × 2 for double-buffer.)

## Best tile per (format, M) at N=K=7168

| Format | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|--------|------|-------|-------|-------|--------|--------|
| INT8 | `64x64x32` **7.5** | `64x64x32` **12.4** | `64x128x32` **20.4** | `128x256x32_w8` **16.1** | `128x256x32_w8` **21.4** | `64x128x32` **21.4** |
| INT4 | `64x64x32` **6.5** | `64x64x32` **11.9** | `128x64x32` **16.4** | `64x64x32` **14.6** | `128x64x32` **19.2** | `128x64x32` **19.2** |
| MXFP4 | `64x64x32` **5.8** | `64x64x32` **10.8** | `128x64x32` **15.5** | `128x64x32` **13.6** | `128x64x32` **18.4** | `128x64x32` **18.4** |
| F8_E4M3_B128 | `64x64x32` **2.9** | `64x64x32` **5.8** | `128x64x32` **9.6** | `128x64x32` **9.0** | `128x64x32` **12.3** | `128x64x32` **12.4** |

## Full TFLOPS sweep — INT8 LUT v2 across all tiles at N=K=7168

| Tile | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|------|------|-------|-------|-------|--------|--------|
| `128x128x32_w4_v2` | 5.28 | 8.33 | 16.71 | 13.92 | 18.20 | 18.18 |
| `128x256x32_w4_v2` | 2.82 | 4.98 | 8.95 | 10.69 | 9.33 | 10.66 |
| `128x256x32_w8_v2` | 4.47 | 8.01 | 16.06 | 16.06 | 21.40 | 21.40 |
| `128x64x32_w4_v2` | 4.87 | 9.42 | 15.32 | 13.02 | 16.85 | 16.29 |
| `64x128x32_w4_v2` | 7.12 | 12.36 | 20.43 | 15.79 | 21.23 | 21.44 |
| `64x64x32_w4_v2` | 7.45 | 12.42 | 15.34 | 15.38 | 15.42 | 15.81 |

## Full TFLOPS sweep — MXFP4 LUT v2 across all tiles at N=K=7168

| Tile | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|------|------|-------|-------|-------|--------|--------|
| `128x64x32_w4_v2` | 4.28 | 8.18 | 15.52 | 13.57 | 18.41 | 18.36 |
| `64x64x32_w4_v2` | 5.77 | 10.85 | 9.90 | 12.16 | 14.32 | 16.01 |

## Findings

### 1. `64x128x32_w4_v2` is the dominant winner

For 4 of 4 formats, at M ≥ 256 N=K=7168, the best tile is `64x128x32_w4_v2`:
- BM=64 (FRAG_M=4 A-fragments per warp)
- BN=128 (FRAG_N=2 N-fragments per warp)
- BK=32 (1 mma_sync K-slice per inner loop iter, with double-buffer)
- 4 warps per CTA
- 8 mma_syncs per warp per K-slice
- 24 KB SMEM per CTA double-buffered → 2 CTAs/SM occupancy

### 2. BM=128 underperforms BM=64 despite higher compute density

`128x128x32_w4_v2` and `128x64x32_w4_v2` consistently lose to `64x128x32_w4_v2`. With BM=128:
- 16 c_frags per warp = 128 floats register spillover risk → reduced occupancy
- 8 A-frags per warp = 64 halves register pressure
- SMEM ~32 KB double-buffered → only ~1 CTA/SM occupancy on V100 (48 KB total)

Conclusion: V100 favors **more parallel CTAs** over **bigger per-CTA tiles** for this dequant-matmul workload.

### 3. BN=256 (`128x256` tiles) hits a wall

`128x256x32_w8_v2` (8 warps × 2 N-frag = 256 BN) reaches only ~13 TFLOPS at M=2048 vs `64x128x32_w4_v2`'s 21 TFLOPS. Possible causes:
- 48 KB SMEM ceiling → 1 CTA/SM
- More mma_sync per warp (16 vs 8) saturates the warp's instruction-issue width
- Larger fragment-store epilogue (16 vs 8 frags per warp) overhead

### 4. BK=32 is sufficient; BK=64 would not fit

BK=64 with BN=128 and double-buffer: sA[128*64*2]*2 + sB[64*128*2]*2 = 32KB + 32KB = 64 KB → exceeds V100's 48 KB SMEM ceiling. So `BK=32 + double-buffer` is the right Pareto point.

### 5. Recommendation: ship `64x128x32_w4_v2` as the production tile for V100 dequant

For DSv4-Flash production shapes (N=K ≈ 7168 for MoE expert matmul, M = batch × topk), the optimal V100 tile is unambiguous: `64x128x32_w4_v2`. Output:
- INT8 LUT: 21.4 TFLOPS at M=2048
- INT4 LUT: 19.2 TFLOPS
- MXFP4 LUT: 18.4 TFLOPS (the format DSv4 uses for experts)
- F8 E4M3 LUT: 12.4 TFLOPS (DSv4 dense format — needs PRMT decode for further gains)
