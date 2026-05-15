# SPRINT-019 follow-ups

Items discovered during execution. Each has a severity (Critical /
Important / Nice-to-have), the file(s) affected, and a suggested target
sprint.

---

## v12s compute-sanitizer race/initcheck not run

**What**: SPRINT-019 §1.2 #2 mandates `compute-sanitizer --tool racecheck`
and `--tool initcheck` on atomic kernels. v12s (P3) ships with the
atomicAdd-based SplitK epilogue but only memcheck was run on its atom
test, not on the full production-shape launch. The single-run
correctness at M=64 showed rel within v12's recalibrated gate but the
sanitizer evidence is missing.

**Why discovered**: Scope/context constraint during execution; commit
f4263b71b notes the deferral.

**Severity**: Important (atomic kernels are race-vulnerable by
construction; the gate is in the sprint plan).

**Suggested sprint**: SPRINT-020 P0 (clear before any v12s-derived
work or any new atomic kernel).

**Files**: `tools/tc-grid/kernels/v12_kernels.cuh` (mm_int8_lut_v12s),
`tools/tc-grid/src/launch_int8.cu` (LAUNCH_V12S).

---

## tc-grid CLI lacks asymmetric N≠K support

**What**: tc-grid's `--nk` flag forces N=K. To validate v12_ms3 against
DSv4-flash MoE expert layers (7168×18944 FFN-up, 18944×7168 FFN-down,
2048×7168 attn-out) the harness needs `--n-list` and `--k-list` separate
flags. P6 was therefore tested on 3 square shapes (4096²/7168²/8192²)
instead of the 6-shape catalog in sprint §6.6.

**Why discovered**: P6 setup; sprint planned MoE catalog assumed
arbitrary N,K.

**Severity**: Important (sprint §6.6 explicitly required multi-shape
validation across DSv4 expert dims; the deferred test is the
binding-readiness gate).

**Suggested sprint**: SPRINT-020 P0.

**Files**: `tools/tc-grid/src/main.cu` (parse_int_list + nk_list + outer
loops), `tools/tc-grid/src/data_gen.cu` (verify A/B/C allocation handles
asymmetric).

---

## nsys PNG timeline cannot be produced headless

**What**: SPRINT-019 §1.2 #10 mandates a Nsight Systems timeline PNG
screenshot for P1/P2/P4. The gpu-01 pod is headless; nsys can render
.nsys-rep + sqlite + CSV summary but the PNG step requires the GUI.
P0.6 and P2.5 commit the kernel-summary CSV as a text proxy and the
.nsys-rep stays in the pod for later GUI inspection.

**Why discovered**: P0.6 attempted PNG export; no GUI in the dev pod.

**Severity**: Nice-to-have (the CSV summary covers the same data; the
PNG is for human visualization only).

**Suggested sprint**: SPRINT-020 if a GUI workstation is wired in;
otherwise document the kern_sum CSV pattern as the standing convention.

**Files**: `tools/tc-grid/docs/nsys/*.nsys-rep` (pod-only),
`tools/tc-grid/docs/nsys/*-kern_sum.csv` (committed proxy).

---

## ncu kernel-id template in sprint §2.5 is invalid

**What**: SPRINT-019 §2.5's canonical ncu command uses
`--kernel-id ::mm_int8_lut_v<X>:1`. The `:1` invocation suffix means
"invocation index 1 globally," so only ONE kernel launch is captured
across the entire tc-grid run — the first v11 launch happened to be
the BK=32 variant, not the champion. The correct syntax for capturing
all v11 instantiations is `-k 'regex:mm_int8_lut_v<X>'`, then filter
post-hoc by template args.

**Why discovered**: P0.5 first attempt produced only one kernel's
data; debugging traced it to the kernel-id suffix.

**Severity**: Important (the standardized profiling protocol is broken
as documented; every phase that follows the protocol would mis-capture
unless this is corrected).

**Suggested sprint**: SPRINT-020 P0 (update sprint template).

**Files**: `docs/sprints/SPRINT-019.md` §2.5 — for future sprint plans,
use the corrected `-k regex:...` template.

---

## SMEM bank-conflict mitigation on sB beyond +8 padding

**What**: v12_ms3's mio_throttle bottleneck (25.73%, P2 ncu) is driven
mainly by SMEM bank conflicts on sB. The current BK_PAD = BK + 8 gives
a 4-way conflict at the BK=16 champ. Sprint-017 tried Swizzle<3,3,3>
on sB (REPORT-12 §4.2) and got +14% on BK=32 but -1% at BK=16 champ.
With v12_ms3's new perf profile (SMEM is now the wall), the swizzle
revisit may be net positive.

**Why discovered**: P2 ncu showed mio_throttle as the new wall;
sprint-017's swizzle data is the obvious next experiment.

**Severity**: Nice-to-have (cleanest path to push past 39 TF M=2048).

**Suggested sprint**: SPRINT-020 P1.

**Files**: `tools/tc-grid/kernels/v12_kernels.cuh` (sB load/store
indices in load_tile + mainloop).

---

## BM=192/256 c_frag SMEM-spill rotation not implemented

**What**: SPRINT-019 §6.4 originally specified a c_frag SMEM-spill
rotation pattern to enable BM=192/256 within launch_bounds(2). P4
discovered that BM=192/256 already FIT (v12's reg relief from §6.1
was enough), but they UNDERPERFORM (-11% to -17% at M=2048) because
the kernel is mio-bound. Implementing the spill rotation would further
worsen mio_throttle — likely net loss. The rotation pattern stays
unimplemented; the negative result is documented.

**Why discovered**: P4 direct test of BM=192/256 v12_ms3.

**Severity**: Nice-to-have (only relevant if mio_throttle is mitigated
first).

**Suggested sprint**: SPRINT-021+ (conditional on mio_throttle work).

**Files**: would have been `tools/tc-grid/kernels/v12_bm192_kernels.cuh`
per sprint §5; not created.

---

## Per-(M, shape) dispatch logic not implemented

**What**: P6 confirmed the champion is shape-sensitive (NK=4096 at -14%
vs NK=7168). The sprint §6.6 decision gate said "underperforms by >10%
on some shape → ship per-shape dispatch logic this sprint." The
square-shape variance is intrinsic to K-amortization (not tile-fit
mismatch), so per-(M, shape) dispatch wouldn't change the champion
choice — but the formal dispatcher rule is not yet encoded in
launch_int8.cu. tc-grid currently relies on the user picking (M, tile)
explicitly via the test driver; production deployment in DSv4
inference would need the formal rule.

**Why discovered**: P6 multi-shape sweep.

**Severity**: Important (blocks DSv4 inference integration; tc-grid
is a benchmark, not a runtime, but the runtime layer needs the rule).

**Suggested sprint**: SPRINT-020 (DSv4 inference integration).

**Files**: would create new `tools/tc-grid/include/dispatch.h` with
per-M champion table.

---

## sprint-017 step-2.3 finding re-discovered in P5

**What**: SPRINT-019 §6.5's "4× __floats2half_rn → __float22half2_rn"
lever was ALREADY landed in sprint-017 step 2.3 (commit c78106eb6).
P5 closed as a documented no-op. Future sprint plans should grep for
existing primitive use before listing a lever as "to be implemented."

**Why discovered**: P5 SASS verification.

**Severity**: Nice-to-have (process improvement).

**Suggested sprint**: Future planning sprints.

**Files**: SPRINT-020+ planning workflow.

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| v12s sanitizer race/initcheck | Important | SPRINT-020 P0 | `v12_kernels.cuh`, `launch_int8.cu` |
| Asymmetric N≠K CLI | Important | SPRINT-020 P0 | `main.cu`, `data_gen.cu` |
| nsys PNG export | Nice-to-have | SPRINT-020 (if GUI wired) | `docs/nsys/` |
| Sprint §2.5 ncu template invalid | Important | SPRINT-020 P0 | future sprint plans |
| sB SMEM swizzle revisit | Nice-to-have | SPRINT-020 P1 | `v12_kernels.cuh` |
| BM=192/256 spill rotation | Nice-to-have | SPRINT-021+ | (new file) |
| Per-(M, shape) dispatcher | Important | SPRINT-020 | new `dispatch.h` |
| §6.5 already-done re-discovery | Nice-to-have | future planning | sprint plans |
