#!/bin/bash
# Section 4F smoke (one GPU job): unit tests of the loss module, the XGBoost objectives and the MLP stages on a sample, and the
# C/T trainer's two loss options on a tiny run. Prints SMOKE OK at the end; the driver checks for it.
set -euo pipefail
cd $HOME/BK_AICUP_2026-1
python -I -m pytest -q tests/test_metric_loss.py tests/test_metric.py 2>&1 | tail -3
python -I logs/nano4/metric_loss_r.py smoke
OUT=$HOME/.cache/report_audit/smoke_4f
rm -rf $OUT; mkdir -p $OUT
python -I scripts/trunk_seq_check.py train --arm T --val-fold 1 --seed 0 --inner-frac 0.1 --loss kernel --limit 640 --epochs 1 --inner-questions 400 --out $OUT
python -I scripts/trunk_seq_check.py train --arm T --val-fold 1 --seed 0 --inner-frac 0.1 --loss es-ft --limit 640 --epochs 1 --ft-epochs 1 --inner-questions 400 --out $OUT
ls $OUT/val1
grep -o '"ft_best_step": [^,]*' $OUT/val1/T-es_s0.json
echo SMOKE OK
