#!/bin/bash
# Section 4B: four XGBoost (GPU, .venv) and four LightGBM (CPU, ~/.cache/lgbm_venv) fits, then the paired evaluation.
cd $HOME/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,039,062]'
submit() { until j=$(sbatch --parsable "$@" 2>/dev/null); do sleep 120; done; echo "$j"; }
ids=""
for v in 1 2 3 4; do
  ids="$ids,$(submit $EX -c 12 -t 03:00:00 -J gbx$v scripts/nano4/run.sbatch python -I logs/nano4/gbdt_hd_fit.py --lib xgb --val-fold $v)"
  ids="$ids,$(submit $EX -c 12 -t 06:00:00 -J gbl$v scripts/nano4/run.sbatch $HOME/.cache/lgbm_venv/bin/python -I logs/nano4/gbdt_hd_fit.py --lib lgb --val-fold $v)"
done
until [ -z "$(squeue -h -j ${ids#,} 2>/dev/null)" ]; do sleep 60; done
echo "fits: $(sacct -j ${ids#,} -X --format=JobName%5,State -n | tr -s ' ' | tr '\n' ';')"
for v in 1 2 3 4; do for lib in xgb lgb; do grep -h 'chose' logs/nano4/gb${lib:0:1}$v-*.out | tail -1; done; done
j=$(submit $EX -c 12 -t 02:00:00 -J gbeval scripts/nano4/run.sbatch python -I logs/nano4/gbdt_hd_eval.py)
until [ -z "$(squeue -h -j $j 2>/dev/null)" ]; do sleep 30; done
grep -v -E '^(\+|== 2026)' logs/nano4/gbeval-$j.out
