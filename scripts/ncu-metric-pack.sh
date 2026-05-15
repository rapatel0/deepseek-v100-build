#!/usr/bin/env bash
# ncu-metric-pack.sh — canonical SPRINT-020 §2.4 ncu metric pack invocation.
#
# Usage:
#   ncu-metric-pack.sh <kernel-regex> <m> <output-csv> [extra tc-grid args...]
#
# Examples:
#   ncu-metric-pack.sh mm_int8_lut_v12_ms3 2048 out.csv --nk 7168 --dist uniform_small
#   ncu-metric-pack.sh mm_int8_lut_v12s    64   out.csv --nk 7168 --dist uniform_small
#
# Required env: BENCH_TCGRID_BIN  (default ./build/tc-grid)
#               BENCH_KUBECTL     (set if running through kubectl exec)
#                                 example: "kubectl exec -n llm tcg-dev --"
#
# Uses the CORRECT `-k regex:NAME` form per SPRINT-019-FOLLOWUPS item 4.
# Sprint §2.5's `--kernel-id ::name:1` template was BROKEN (only matches
# launch invocation 1 globally).

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <kernel-regex> <m> <output-csv> [extra tc-grid args...]" >&2
    exit 2
fi

KERNEL_REGEX="$1"; shift
M_VAL="$1"; shift
OUT_CSV="$1"; shift

TCGRID="${BENCH_TCGRID_BIN:-./build/tc-grid}"
PRE="${BENCH_KUBECTL:-}"

# Canonical SPRINT-020 §2.4 metric set.
METRICS=(
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
    smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct
    smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    l1tex__t_sector_hit_rate.pct
    launch__registers_per_thread
    launch__shared_mem_per_block_dynamic
    launch__waves_per_multiprocessor
    sm__pipe_tensor_op_hmma_cycles_active.sum
)

METRICS_CSV=$(IFS=,; echo "${METRICS[*]}")

if [[ -n "$PRE" ]]; then
    # shellcheck disable=SC2086
    $PRE ncu -k "regex:${KERNEL_REGEX}" --metrics "$METRICS_CSV" --csv \
        "$TCGRID" --m-list "$M_VAL" "$@" > "$OUT_CSV"
else
    ncu -k "regex:${KERNEL_REGEX}" --metrics "$METRICS_CSV" --csv \
        "$TCGRID" --m-list "$M_VAL" "$@" > "$OUT_CSV"
fi

echo "[ncu-metric-pack] $(wc -l < "$OUT_CSV") lines → $OUT_CSV" >&2
