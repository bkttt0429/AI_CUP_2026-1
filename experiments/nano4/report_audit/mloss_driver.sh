#!/bin/bash
# Section 4F: the smoke job first; then per fold the tree + MLP job and the C / T jobs (seed 0, kernel and es-ft); then the evaluation.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
wait_jobs() { until [ -z "$(squeue -h -j ${1#,} 2>/dev/null)" ]; do sleep 60; done; }
j=$(submit $EX -c 12 -t 00:50:00 -J ml-smoke scripts/nano4/run.sbatch bash $HOME/.cache/report_audit/mloss_smoke.sh)
echo "smoke job $j"
wait_jobs $j
grep -q "SMOKE OK" logs/nano4/ml-smoke-$j.out || { echo "smoke failed (job $j)"; tail -40 logs/nano4/ml-smoke-$j.out; exit 1; }
echo "smoke ok ($j)"
ids=""
for v in 1 2 3 4; do
  ids="$ids,$(submit $EX -c 12 -t 04:00:00 -J mlr$v scripts/nano4/run.sbatch python -I logs/nano4/metric_loss_r.py fit --val-fold $v)"
  for arm in C T; do
    ids="$ids,$(submit $EX -c 12 -t 03:30:00 -J ml${arm}k$v scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed 0 --inner-frac 0.1 --loss kernel --out results/trunk_seq/es)"
    ids="$ids,$(submit $EX -c 12 -t 03:30:00 -J ml${arm}e$v scripts/nano4/run.sbatch python -I scripts/trunk_seq_check.py train --arm $arm --val-fold $v --seed 0 --inner-frac 0.1 --loss es-ft --out results/trunk_seq/es)"
  done
done
echo "fit jobs ${ids#,}"
wait_jobs $ids
echo "fits: $(sacct -j ${ids#,} -X --format=JobName%8,State -n | tr -s ' ' | tr '\n' ';')"
echo "result files: $(ls results/metric_loss/r_val*.npz results/trunk_seq/es/val*/[CT]-k_s0.npz results/trunk_seq/es/val*/[CT]-es_s0.npz 2>/dev/null | wc -l) of 20"
j=$(submit $EX -c 12 -t 01:00:00 -J ml-eval scripts/nano4/run.sbatch python -I logs/nano4/metric_loss_r.py eval)
wait_jobs $j
grep -v -E '^(\+|== 2026)' logs/nano4/ml-eval-$j.out
