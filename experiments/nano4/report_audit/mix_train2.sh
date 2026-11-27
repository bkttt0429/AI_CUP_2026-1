#!/usr/bin/env bash
# Second wave of section 4L.4 (the first hit the QOS submit limit): folds 3-4 C-mix50 / T-mix50 and fold 2 C-mix100, submitted
# only while this user has fewer than CAP jobs in the queue; then the two compares.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=10
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 60; done
    j=$(sbatch --parsable $EX -t 02:30:00 -J "$1" scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train "${@:2}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
ids=""
for v in 3 4; do
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
echo "jobs ${ids#,}" | tee /home/u2507115/.cache/report_audit/mix_train2_jobs.txt
wait_jobs "${ids#,}"
for j in ${ids//,/ }; do echo "$j $(state $j) $(grep -h 'stopped at step' logs/nano4/mx*-$j.out | sed 's/.*stopped/stopped/')"; done
C1=$(sbatch --parsable $EX -c 12 -t 01:00:00 -J mixcmp scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 1 2 3 4 --arms C T C-mix50 T-mix50 --seeds 0 1 2 --pairs C-mix50:C T-mix50:T --tag mix_ --out results/trunk_seq/es)
C2=$(sbatch --parsable $EX -c 12 -t 01:00:00 -J mixcmp2 scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py compare --val-folds 2 --arms C C-mix50 C-mix100 --seeds 0 1 2 --pairs C-mix100:C C-mix100:C-mix50 --tag mix100_ --out results/trunk_seq/es)
wait_jobs "$C1,$C2"
echo "== compare $C1 $(state $C1)"; grep -v "^==\|^+" logs/nano4/mixcmp-$C1.out
echo "== compare fold 2 mix100 $C2 $(state $C2)"; grep -v "^==\|^+" logs/nano4/mixcmp2-$C2.out
echo "compare done"
