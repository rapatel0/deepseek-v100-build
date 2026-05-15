#!/usr/bin/env bash
# bench-median.sh — run tc-grid N times and emit median TF per (tile,M) row.
#
# Usage:
#   bench-median.sh [--runs N] [--out CSV] -- <tc-grid args...>
#
# Examples:
#   bench-median.sh --runs 5 --out baseline.csv -- --m-list 2048 --nk 7168 --dist uniform_small
#   bench-median.sh -- --m-list 1,8,32,64,256,1024,2048,4096 --nk 7168
#
# Output CSV columns:
#   format,path,dist,M,N,K,tile,tflops_median,tflops_min,tflops_max,
#   tflops_mean,ms_median,rel,maxabs,p99,n_runs
#
# Only rows with status=OK across ALL runs are emitted. Rows that vary in
# SKIP-state between runs are dropped with a stderr note. rel/maxabs/p99 are
# taken from the median-of-5 *run* (not row-wise medians) for traceability.
#
# Required env: BENCH_TCGRID_BIN  (path to tc-grid binary; defaults to ./build/tc-grid)
#               BENCH_KUBECTL     (set if running through kubectl exec)
#                                 example: "kubectl exec -n llm tcg-dev --"
set -euo pipefail

RUNS=5
OUT=""
declare -a ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)  RUNS="$2"; shift 2;;
    --out)   OUT="$2";  shift 2;;
    --)      shift; ARGS=("$@"); break;;
    *)       echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ ${#ARGS[@]} -gt 0 ]] || { echo "no tc-grid args provided after --" >&2; exit 2; }

TCGRID="${BENCH_TCGRID_BIN:-./build/tc-grid}"
PRE="${BENCH_KUBECTL:-}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for i in $(seq 1 "$RUNS"); do
  echo "[bench-median] run $i/$RUNS ..." >&2
  if [[ -n "$PRE" ]]; then
    # shellcheck disable=SC2086
    $PRE "$TCGRID" "${ARGS[@]}" > "$TMPDIR/run-$i.csv"
  else
    "$TCGRID" "${ARGS[@]}" > "$TMPDIR/run-$i.csv"
  fi
done

python3 - "$TMPDIR" "$RUNS" "${OUT:-/dev/stdout}" <<'PY'
import csv, re, statistics, sys
from collections import defaultdict
tmp, runs, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# tc-grid emits free-form CSV:
#   format,path,dist,M,N,K,tile,status,detail
# `dist` is a label that itself contains commas (e.g. "U(-1,1)"). We anchor
# parsing on the first OK|SKIP token and walk fixed positions from each side.
STATUS_RE = re.compile(r",(OK|SKIP),")
def parse_row(line):
    m = STATUS_RE.search(line)
    if not m:
        return None
    head = line[:m.start()].split(",")
    if len(head) < 7:
        return None
    status = m.group(1)
    detail = line[m.end():].rstrip("\n")
    # Last 5 head fields are M,N,K,tile -- with dist absorbing the rest.
    M, N, K, tile = head[-4], head[-3], head[-2], head[-1]
    fmt = head[0]
    path = head[1]
    dist = ",".join(head[2:-4])
    kvs = {}
    if status == "OK":
        for tok in detail.split(","):
            if "=" in tok:
                k, v = tok.split("=", 1)
                kvs[k.strip()] = v.strip()
    try:
        return (fmt, path, dist, int(M), int(N), int(K), tile), status, kvs
    except ValueError:
        return None

# (key) -> [ (status, kvs), ... ] across runs
rows = defaultdict(list)
for i in range(1, runs+1):
    with open(f"{tmp}/run-{i}.csv") as f:
        for line in f:
            r = parse_row(line)
            if r is None:
                continue
            k, st, kvs = r
            rows[k].append((st, kvs))

# Emit only rows that are OK in every run.
fields = ["format","path","dist","M","N","K","tile",
         "tflops_median","tflops_min","tflops_max","tflops_mean",
         "ms_median","rel","maxabs","p99","n_runs"]
fh = sys.stdout if out == "/dev/stdout" else open(out, "w")
w = csv.writer(fh)
w.writerow(fields)
dropped = 0
for k, lst in rows.items():
    if len(lst) != runs or any(s != "OK" for s, _ in lst):
        dropped += 1
        continue
    tflops = [float(kvs["tflops"]) for _, kvs in lst]
    mss    = [float(kvs["ms"])     for _, kvs in lst]
    tflops_sorted = sorted(tflops)
    median = statistics.median(tflops_sorted)
    # rel/maxabs/p99 from the run whose tflops is median (or nearest)
    target_idx = sorted(range(len(tflops)), key=lambda i: abs(tflops[i]-median))[0]
    kvs_med = lst[target_idx][1]
    w.writerow([*k,
                f"{median:.3f}",
                f"{min(tflops):.3f}",
                f"{max(tflops):.3f}",
                f"{statistics.mean(tflops):.3f}",
                f"{statistics.median(mss):.4f}",
                kvs_med.get("rel",""),
                kvs_med.get("maxabs",""),
                kvs_med.get("p99",""),
                runs])
if fh is not sys.stdout: fh.close()
print(f"[bench-median] emitted {len(rows)-dropped} rows, dropped {dropped} (non-uniform OK)", file=sys.stderr)
PY
