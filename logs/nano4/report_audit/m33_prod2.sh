#!/usr/bin/env bash
# M33 production, amended mock criterion (decisions identical, scores equal, max |dp| <= ct7's precedent 0.0040): diagnose the largest
# |dp| questions of the existing mock run, then submission_rank_ct9.csv from the 24 pc members, then ct9 vs ct7.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
OUT=results/mock_fold1_100_pc
python -I -c "
import json, sys, numpy as np, pandas as pd
r = json.load(open('$OUT/ct_check.json'))
a = pd.read_csv('$OUT/submission_rank_ct_proba.csv').set_index('question_id'); b = pd.read_csv('$OUT/offline_ct_proba.csv').set_index('question_id').loc[a.index]
cols = [c for c in a.columns if c.startswith('p_')]
d = (a[cols] - b[cols]).abs().max(axis=1).sort_values(ascending=False)
for q in d.index[:5]: print(f'{q}: max |dp| {d[q]:.4f}, max probability {a.loc[q, cols].max():.3f}')
print(f'questions above 1e-3: {int((d > 1e-3).sum())}/100')
ok = r['decisions_differ'] == 0 and abs(r['score_a'] - r['score_b']) < 1e-12 and r['max_abs_dp'] <= 0.0040
print('AMENDED MOCK CHECK', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)" || exit 1
MODELS=""; for v in 1 2 3 4; do for arm in C T; do for s in 0 1 2; do MODELS="$MODELS results/models/ct_cv_m33/val${v}_$arm-pc-f0-x40-oh_s$s.pt"; done; done; done
echo "checkpoints present: $(ls $MODELS 2>/dev/null | wc -l)/24"
python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $MODELS --fallback-submission submission_rank_ct3.csv --out submission_rank_ct9.csv || { echo "ct9 predict FAILED"; exit 1; }
md5sum submission_rank_ct9.csv submission_rank_ct7.csv
python -I -c "
import pandas as pd
a, b = pd.read_csv('submission_rank_ct9.csv'), pd.read_csv('submission_rank_ct7.csv')
assert list(a.columns) == list(b.columns) and len(a) == len(b) == 800 and (a.iloc[:, 0] == b.iloc[:, 0]).all()
pub = a.iloc[:, 0].str.startswith('pub')
d = a.iloc[:, 1] != b.iloc[:, 1]
print(f'ct9 vs ct7: {int(d.sum())}/800 decisions differ (public {int(d[pub].sum())}, private {int(d[~pub].sum())})')
print('ct9 predicted counts', a.iloc[:, 1].value_counts().sort_index().to_dict())"
