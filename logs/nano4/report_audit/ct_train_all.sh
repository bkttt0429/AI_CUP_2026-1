#!/usr/bin/env bash
# After the r_all feature cache exists: the submission's 12 trunk + TCN models (C and T, seeds 0-5, every fold's players).
# arg: <rall job id>
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
wait_jobs() { for id in ${1//,/ }; do until [ -z "$(squeue -h -j $id 2>/dev/null)" ]; do sleep 60; done; done; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs "$1"
echo "rall $1: $(state $1)"
[ "$(state $1)" = "COMPLETED" ] && [ -f results/trunk_seq/r_all.npz ] || { echo "r_all cache missing"; exit 1; }
ids=""
for arm in C T; do
  for seed in 0 1 2 3 4 5; do
    j=$(sbatch --parsable $EX -t 04:00:00 -J all$arm$seed scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm $arm --val-fold all --seed $seed --inner-frac 0.1 --out results/trunk_seq/es)
    echo "all$arm$seed $j"
    ids="$ids,$j"
  done
done
echo "jobs ${ids#,}" | tee /home/u2507115/.cache/report_audit/ct_train_all_jobs.txt
wait_jobs "${ids#,}"
for j in ${ids//,/ }; do [ -n "$j" ] && echo "$j $(state $j) $(grep -h 'stopped at step\|saved' logs/nano4/all*-$j.out | tr '\n' ' ')"; done
ls -la results/models/ct/
