#!/bin/bash
# Section 4D: the XGBoost search and MLP residual per fold, then the C + T residual and the paired evaluation.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
ids=""
for v in 1 2 3 4; do
  ids="$ids,$(submit $EX -c 12 -t 03:00:00 -J gbt$v scripts/nano4/run.sbatch python -I logs/nano4/gbdt_tune.py fit --val-fold $v)"
done
until [ -z "$(squeue -h -j ${ids#,} 2>/dev/null)" ]; do sleep 60; done
echo "fits: $(sacct -j ${ids#,} -X --format=JobName%5,State -n | tr -s ' ' | tr '\n' ';')"
for v in 1 2 3 4; do grep -h -E '^fold|Error|error' logs/nano4/gbt$v-*.out | tail -4; done
ls results/gbdt_hd/tune_val*.npz | wc -l | grep -qx 4 || { echo "missing fold results, stopping"; exit 1; }
j=$(submit $EX -c 12 -t 02:00:00 -J gbteval scripts/nano4/run.sbatch python -I logs/nano4/gbdt_tune.py eval)
until [ -z "$(squeue -h -j $j 2>/dev/null)" ]; do sleep 30; done
grep -v -E '^(\+|== 2026)' logs/nano4/gbteval-$j.out
