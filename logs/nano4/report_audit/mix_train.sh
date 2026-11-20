#!/usr/bin/env bash
# Section 4L.4: same-rank mixed-player training arm. Waits for the unit test and the five feature caches, then submits
# C-mix50 / T-mix50 (folds 1-4, seeds 0-2) and C-mix100 (fold 2, seeds 0-2) under a concurrency cap, waits, and runs the compares.
# args: <mixtest job> <rmix1> <rmix2> <rmix3> <rmix4> <rmix2f>
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=14
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
TEST=$1; shift
wait_jobs "$TEST,$(IFS=,; echo "$*")"
echo "mixtest $TEST: $(state $TEST)"
for j in "$@"; do echo "cache $j: $(state $j)"; done
[ "$(state $TEST)" = "COMPLETED" ] || { echo "unit test failed, not training"; exit 1; }
for f in results/trunk_seq/r_val1_mix50.npz results/trunk_seq/r_val2_mix50.npz results/trunk_seq/r_val3_mix50.npz results/trunk_seq/r_val4_mix50.npz results/trunk_seq/r_val2_mix100.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
grep -h "mixed-player questions" logs/nano4/rmix*-*.out
submit() {  # respects the cap on this user's jobs
  while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 60; done
  sbatch --parsable $EX -t 02:30:00 -J "$1" scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train "${@:2}"
}
ids=""
for v in 1 2 3 4; do
  for arm in C T; do
    for seed in 0 1 2; do
      j=$(submit mx$arm$v$seed --arm $arm --val-fold $v --seed $seed --inner-frac 0.1 --mix 0.5 --out results/trunk_seq/es)
      echo "mx$arm$v$seed $j"; ids="$ids,$j"
    done
  done
done
for seed in 0 1 2; do
  j=$(submit mxC2f$seed --arm C --val-fold 2 --seed $seed --inner-frac 0.1 --mix 1.0 --out results/trunk_seq/es)
  echo "mxC2f$seed $j"; ids="$ids,$j"
done
echo "jobs ${ids#,}" | tee /home/u2507115/.cache/report_audit/mix_train_jobs.txt
wait_jobs "${ids#,}"
for j in ${ids//,/ }; do echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/mx*-$j.out | sed 's/.*stopped/stopped/')"; done
C1=$(sbatch --parsable $EX -c 12 -t 01:00:00 -J mixcmp scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 1 2 3 4 --arms C T C-mix50 T-mix50 --seeds 0 1 2 --pairs C-mix50:C T-mix50:T --tag mix_ --out results/trunk_seq/es)
C2=$(sbatch --parsable $EX -c 12 -t 01:00:00 -J mixcmp2 scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 2 --arms C C-mix50 C-mix100 --seeds 0 1 2 --pairs C-mix100:C C-mix100:C-mix50 --tag mix100_ --out results/trunk_seq/es)
wait_jobs "$C1,$C2"
echo "== compare $C1 $(state $C1)"; grep -v "^==\|^+" logs/nano4/mixcmp-$C1.out
echo "== compare fold 2 mix100 $C2 $(state $C2)"; grep -v "^==\|^+" logs/nano4/mixcmp2-$C2.out
echo "compare done"
