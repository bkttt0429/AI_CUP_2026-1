#!/usr/bin/env bash
# Section 4M M7 production members (2026-11-04, user: 開始測測看): r_all_s5-10 cache, then 12 train-all models with --train-sizes 5 10
# -> results/models/ct/all_{C,T}-s5-10_s{0..5}.pt (+ .json). Cap-limited submissions.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=18
TR="python -I scripts/trunk_seq_check.py"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 120; }
  done
  echo "$j"
}
echo "== $(date '+%F %T') M7 production start"
R=$(submit rall510 "-c 12 -t 02:00:00" $TR rfeat --val-fold all --train-sizes 5 10)
echo "cache $R"; wait_jobs "$R"; echo "cache $(state $R) $(grep -h 'cached\|questions' logs/nano4/rall510-$R.out | tail -1)"
[ "$(state $R)" = "COMPLETED" ] || { tail -20 logs/nano4/rall510-$R.out; exit 1; }
ids=""
for arm in C T; do for s in 0 1 2 3 4 5; do
  ids="$ids,$(submit all510$arm$s "-t 02:30:00" $TR train --arm $arm --val-fold all --seed $s --inner-frac 0.1 --train-sizes 5 10 --out results/trunk_seq)"
done; done
echo "train jobs ${ids#,}"
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j) $(grep -h 'stopped at step\|saved' logs/nano4/all510*-$j.out 2>/dev/null | sed 's/.*stopped/stopped/' | tr '\n' ' ')"; done
ls -la results/models/ct/all_*-s5-10_s*.pt
echo "== $(date '+%F %T') M7 production done"
