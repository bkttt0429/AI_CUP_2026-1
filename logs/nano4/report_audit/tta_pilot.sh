#!/usr/bin/env bash
# 8-symmetry TTA pilot on the fold-1 100-question mock (2026-11-05): for k = 1..7 transform the mock CSVs, run shards (4 GPUs) ->
# trunk extract -> predict with the 12 production f0-x40 checkpoints, then compare decisions / |dp| against symmetry 0
# (results/mock_fold1_100/submission_rank_ct3_proba.csv) and evaluate the 8-way probability average. Stop rule: decision change
# rate across symmetries < 2% -> no test-side TTA.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
AUD=/home/u2507115/.cache/report_audit
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() { local j=""; while [ -z "$j" ]; do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }; done; echo "$j"; }
M="results/models/ct/all_C-f0-x40_s0.pt results/models/ct/all_C-f0-x40_s1.pt results/models/ct/all_C-f0-x40_s2.pt results/models/ct/all_C-f0-x40_s3.pt results/models/ct/all_C-f0-x40_s4.pt results/models/ct/all_C-f0-x40_s5.pt results/models/ct/all_T-f0-x40_s0.pt results/models/ct/all_T-f0-x40_s1.pt results/models/ct/all_T-f0-x40_s2.pt results/models/ct/all_T-f0-x40_s3.pt results/models/ct/all_T-f0-x40_s4.pt results/models/ct/all_T-f0-x40_s5.pt"
SRC=data/mock_tests/fold1_100
EXPECT="--expect-rows 100 --expect-public 50 --expect-private 50"
echo "== $(date '+%F %T') TTA pilot start"
ids=""
for k in 1 2 3 4 5 6 7; do
  D=data/mock_tests/fold1_100_sym$k; OUT=results/mock_fold1_100_sym$k
  IN="$D/rank_prediction_test_public.csv $D/rank_prediction_test_private.csv"
  j=$(submit tta$k "--gres=gpu:4 -c 48 -t 02:00:00" bash -c "python -I logs/nano4/sgf_symmetry.py --inputs $SRC/rank_prediction_test_public.csv $SRC/rank_prediction_test_private.csv --k $k --out-dir $D --copy $SRC/answers.csv && bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs $IN $EXPECT --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --workers 10 && bash scripts/nano4/shards.sh 4 python -I scripts/trunk_predict.py extract --inputs $IN --vec $OUT/vec --workers 10 && python -I scripts/trunk_predict.py predict --inputs $IN $EXPECT --shards $OUT/submission_rank.csv --nshards 4 --vec $OUT/vec --models $M --fallback-submission results/mock_fold1_100/submission_rank.csv --out $OUT/submission_rank_ct3.csv")
  echo "sym $k job $j"; ids="$ids,$j"
done
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; done
A=$(submit ttaeval "-c 12 -t 00:30:00" python -I -c "
import pandas as pd, numpy as np, json
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from aicup.metric import LEVELS, expected_score_decision, score
cols = [f'p_{l}' for l in LEVELS]
base = pd.read_csv('results/mock_fold1_100/submission_rank_ct3_proba.csv')
ans = pd.read_csv('data/mock_tests/fold1_100/answers.csv', dtype={'question_id': str, 'rank': str}).set_index('question_id').loc[base['question_id']]
y = ans['rank'].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
P0 = base[cols].to_numpy(); d0 = np.array([expected_score_decision(p) for p in P0])
rep = {'symmetries': {}}
Ps = [P0]
for k in range(1, 8):
    f = Path(f'results/mock_fold1_100_sym{k}/submission_rank_ct3_proba.csv')
    if not f.exists():
        rep['symmetries'][k] = 'missing'; continue
    P = pd.read_csv(f)[cols].to_numpy(); Ps.append(P)
    d = np.array([expected_score_decision(p) for p in P])
    rep['symmetries'][k] = {'decision_change': float((d != d0).mean()), 'mean_abs_dp': float(np.abs(P - P0).mean()), 'max_abs_dp': float(np.abs(P - P0).max()), 'score': float(np.mean([score(int(a), int(b)) for a, b in zip(d, y)]))}
Pavg = np.mean(Ps, axis=0); davg = np.array([expected_score_decision(p) for p in Pavg])
rep['n_symmetries'] = len(Ps)
rep['score_sym0'] = float(np.mean([score(int(a), int(b)) for a, b in zip(d0, y)]))
rep['score_avg'] = float(np.mean([score(int(a), int(b)) for a, b in zip(davg, y)]))
rep['avg_decision_change_vs_sym0'] = float((davg != d0).mean())
print(json.dumps(rep, indent=1))
Path('results/r_variants/tta_pilot.json').write_text(json.dumps(rep, indent=1))
")
wait_jobs "$A"; echo "== TTA eval $A $(state $A)"; grep -v "^+" logs/nano4/ttaeval-$A.out
echo "== $(date '+%F %T') TTA pilot done"
