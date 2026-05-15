# SPRINT-016 Deferred

## XOR-swizzle + manual fragment load — PROMOTED TO ACTIVE (SPRINT-017)

- **Status**: No longer deferred. v9 mixed-precision was broken-by-design and v10 (row-major
  B SMEM, the simpler "v10" we ended up using) only closed ~6% of the gap. Remaining ~70%
  gap to cuBLAS-FP16 ceiling (85 TF) requires this work. Active in SPRINT-017.
- **What**: Bypass `wmma::load_matrix_sync` for B; populate fragments via inline PTX
  `ld.shared.b128` + m8n8k4 HMMA from XOR-swizzled SMEM. Roadmap in
  `tools/tc-grid/docs/V11-DESIGN.md`. Estimated ~10 hr engineering.
- **Files**: will create `tools/tc-grid/kernels/v11a_kernels.cuh` (Step 1), then evolve.

## int8_v9-alt: store partial c to SMEM mid-K-loop

- **What**: Instead of mixed-precision, free c_frag regs by writing partial c to SMEM every K-chunk and re-loading later. Trades SMEM bandwidth for register pressure.
- **Why deferred**: Higher complexity than v9 mixed-precision; only worth it if v9 fails. SMEM bandwidth on V100 is ~14 TB/s but our SMEM is already 80%+ utilized by A/B tiles in the BM=128 BN=128 config.
- **Target sprint**: SPRINT-018+.
- **Prerequisites**: v9 fails or saturates.
- **Files**: would create `kernels/v9alt_kernels.cuh`.

## Persistent CTA grid-stride output loop revisit (was Tier A3 / v5)

- **What**: v5 underperformed v3 in Tier A measurements. With fresh register-pressure lens, may revisit — does persistent CTA help when v9 reduces regs below the 1-CTA/SM ceiling?
- **Why deferred**: Conditional on v9 succeeding and the c_frag count dropping enough to enable >1 CTA/SM. Only meaningful if occupancy >12% becomes feasible.
- **Target sprint**: Conditional SPRINT-017 follow-up.
- **Prerequisites**: v9 ships AND ncu shows occupancy headroom.
- **Files**: would touch `tools/tc-grid/kernels/v5_kernels.cuh` (already exists).

## Split-K (was Tier B2)

- **What**: K-direction split with atomic accumulation. Helps when M is small and K is large (more CTAs per problem).
- **Why deferred**: At DSv4 production shape (M=2048, K=7168) we already have plenty of CTAs (grid.x × grid.y ≈ 1700+ for BM=BN=128). Split-K only helps when M is the bottleneck.
- **Target sprint**: When small-M (M<32) becomes a target.
- **Prerequisites**: Real-world DSv4 small-M profile.
- **Files**: would touch `launch_int8.cu` and add new kernel.

## Summary table

| Item | Target Sprint | Blocker |
|---|---|---|
| XOR-swizzle + manual fragment load | **SPRINT-017 (active)** | none — unconditional after v9 failure |
| v9-alt SMEM partial-c store | SPRINT-018+ | v11 fails or saturates |
| Persistent CTA revisit | SPRINT-018+ | v11 ships + ncu shows occupancy headroom |
| Split-K | Future | Production M<32 use case |
