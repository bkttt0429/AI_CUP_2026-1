#!/usr/bin/env bash
# M33 production steps (pre-registered): 1. fold-1 100-question mock consistency (val1 C/T-pc s0, existing oh shards, heads3_B):
# max |dp| <= 0.001 and 0 decisions differ; 2. only then submission_rank_ct9.csv from the 24 pc members (ct7's other inputs).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
IN="data/mock_tests/fold1_100/rank_prediction_test_public.csv data/mock_tests/fold1_100/rank_prediction_test_private.csv"
EXPECT="--expect-rows 100 --expect-public 50 --expect-private 50"
OUT=results/mock_fold1_100_pc; mkdir -p $OUT
M="results/models/ct_cv_m33/val1_C-pc-f0-x40-oh_s0.pt results/models/ct_cv_m33/val1_T-pc-f0-x40-oh_s0.pt"
python -I scripts/trunk_predict.py predict --inputs $IN $EXPECT --shards results/mock_fold1_100_oh/submission_rank.csv --nshards 4 --vec results/mock_fold1_100/vec --models $M --l1b-names heads3_B --fallback-submission results/mock_fold1_100/submission_rank.csv --out $OUT/submission_rank_ct.csv || { echo "mock predict FAILED"; exit 1; }
python -I scripts/trunk_predict.py offline --mock-dir data/mock_tests/fold1_100 --fold 1 --models $M --out $OUT/offline_ct_proba.csv || { echo "mock offline FAILED"; exit 1; }
python -I scripts/trunk_predict.py compare --a $OUT/submission_rank_ct_proba.csv --b $OUT/offline_ct_proba.csv --answers data/mock_tests/fold1_100/answers.csv --report $OUT/ct_check.json
python -I -c "
import json, sys; r = json.load(open('$OUT/ct_check.json')); print({k: r[k] for k in r if k in ('max_abs_dp', 'mean_abs_dp', 'decisions_differ', 'pass', 'tolerance')})
sys.exit(0 if r['pass'] and r['decisions_differ'] == 0 else 1)" || { echo "MOCK CHECK FAIL: ct9 not produced"; exit 1; }
echo "MOCK CHECK PASS"
MODELS=""; for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do MODELS="$MODELS results/models/ct_cv_m33/val${v}_$arm-pc-f0-x40-oh_s$s.pt"; done; done; done
echo "checkpoints present: $(ls $MODELS 2>/dev/null | wc -l)/24"
python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $MODELS --fallback-submission submission_rank_ct3.csv --out submission_rank_ct9.csv || { echo "ct9 predict FAILED"; exit 1; }
md5sum submission_rank_ct9.csv submission_rank_ct7.csv
python -I -c "
import pandas as pd
a, b = pd.read_csv('submission_rank_ct9.csv'), pd.read_csv('submission_rank_ct7.csv')
assert list(a.columns) == list(b.columns) and len(a) == len(b) == 800 and (a.iloc[:, 0] == b.iloc[:, 0]).all()
d = a.iloc[:, 1] != b.iloc[:, 1]
print(f'ct9 vs ct7: {int(d.sum())}/800 decisions differ (public rows {int(d.iloc[:400].sum())}, private rows {int(d.iloc[400:].sum())})')
print('ct9 predicted counts', a.iloc[:, 1].value_counts().sort_index().to_dict())"
