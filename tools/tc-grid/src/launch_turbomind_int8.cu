// SPRINT-020 P2.1 — tc-grid → turbomind GEMM bridge (skeleton).
//
// Adapts tc_grid::LaunchSpec to turbomind::gemm::Gemm::Run so the
// turbomind kernels can be measured apples-to-apples through the tc-grid
// harness (same data generator, same reference GEMM, same tolerance gate).
//
// THIS IS A SKELETON. Full implementation requires P1's gemm_bench build
// to succeed (so the turbomind gemm2 library is actually linkable). The
// current state:
//
//   - File compiles standalone but launch_turbomind_int8() returns
//     LaunchResult{R.ok=false, R.note="turbomind not yet wired"}.
//   - When P1 lands, replace the stub body with the actual mapping from
//     LaunchSpec to turbomind::gemm::Operation / MatrixLayout / Gemm::Run.
//   - Build guard: TCGRID_ENABLE_TURBOMIND_GEMM CMake option (added per
//     SPRINT-020 §3.2).
//
// Data-layout validation (sprint §1.3 P2 no-skip gate) MUST happen
// before any benchmark trust: verify turbomind kernels can consume
// tc-grid's INT8 weight tensors without an expensive per-launch re-pack.
// See SPRINT-020-P1-gemm_bench-audit.md §"External libraries needed" for
// the turbomind weight-layout / scale-granularity questions.

#include "tc_grid.h"
#include "dispatch.h"

#include <cstdio>

#ifdef TCGRID_ENABLE_TURBOMIND_GEMM
// TODO P1.5: confirm these headers exist post-build of turbomind.
// #include "research/lmdeploy/src/turbomind/kernels/gemm/gemm.h"
// #include "research/lmdeploy/src/turbomind/kernels/gemm/types.h"
#endif

namespace tc_grid {

LaunchResult launch_turbomind_int8(const LaunchSpec & s, const void * d_W,
                                    const float * d_act, float * d_dst,
                                    const float * d_ref) {
    LaunchResult R; R.ok = false;
    (void) s; (void) d_W; (void) d_act; (void) d_dst; (void) d_ref;

#ifdef TCGRID_ENABLE_TURBOMIND_GEMM
    // P2.1 TODO: adapter from LaunchSpec to turbomind::gemm::Operation
    //   - Operation.dispatch_policy = DispatchPolicy::kAppraise (use the
    //     measured dispatch_cache if available)
    //   - Operation.epilogue = Epilogue::kNone (no fused activation)
    //   - Operation.quant_a/quant_b — depends on tc-grid's quantization
    //     compat (P2.2 validation gate)
    //
    // P2.2 data-layout validation hook lives here too: refuse to run if
    // the LaunchSpec's weight layout requires expensive re-pack.
    R.note = "turbomind launcher not yet wired — P2.1 skeleton (placeholder)";
    return R;
#else
    R.note = "TCGRID_ENABLE_TURBOMIND_GEMM=OFF";
    return R;
#endif
}

}  // namespace tc_grid
