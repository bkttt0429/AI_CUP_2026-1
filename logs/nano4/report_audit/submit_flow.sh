#!/usr/bin/env bash
# 11/04 submission flow (NANO4 4.13 / 4.20 / 4.21). args: <public csv> <private csv> (any location; copied to data/tests/ under the official names)
# Produces in the repo root: submission_rank.csv (main: rank_model_fused_opp_diff_r.json), submission_rank_ct.csv (C+T 12 models),
# results/sub_hd/submission_rank.csv (HD document, only when every game carries HD[]), results/sub_old/submission_rank.csv (old fused_opp_diff),
# results/data/test_check.json (distribution + train overlap). Nothing is uploaded.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
PUB=data/tests/rank_prediction_test_public.csv; PRIV=data/tests/rank_prediction_test_private.csv
HEADS="--l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
sub() { local j=""; while [ -z "$j" ]; do j=$(sbatch --parsable $EX "$@" 2>/dev/null) || { j=""; echo "sbatch retry" >&2; sleep 60; }; done; echo "$j"; }
mkdir -p data/tests
[ "$(readlink -f "$1")" = "$(readlink -f $PUB)" ] || cp "$1" $PUB
[ "$(readlink -f "$2")" = "$(readlink -f $PRIV)" ] || cp "$2" $PRIV
md5sum $PUB $PRIV
echo "== $(date '+%F %T') start"
CHK=$(sub -c 12 -t 02:00:00 -J tcheck scripts/nano4/run.sbatch python -I logs/nano4/test_check.py --test $PUB $PRIV --out results/data/test_check.json --train-overlap)
SH=$(sub -p 8gpus --gres=gpu:8 -c 96 -t 06:00:00 -J sub8 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --inputs $PUB $PRIV --out submission_rank.csv $HEADS --workers 10)
echo "test check $CHK, shard scoring $SH"
wait_jobs "$CHK"; echo "tcheck $CHK: $(state $CHK)"; grep -v "^+ \|^== 2026" logs/nano4/tcheck-$CHK.out
wait_jobs "$SH"; echo "sub8 $SH: $(state $SH)"; tail -3 logs/nano4/sub8-$SH.out
[ "$(state $SH)" = "COMPLETED" ] || { echo "shard scoring failed: see logs/nano4/shard*-$SH.log"; exit 1; }
MG=$(sub -c 12 -t 01:00:00 -J submg scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8 --inputs $PUB $PRIV --out submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff_r.json)
for alt in hd old; do
  mkdir -p results/sub_$alt
  for i in 0 1 2 3 4 5 6 7; do ln -sf ../../submission_rank_shard${i}of8.npz results/sub_$alt/submission_rank_shard${i}of8.npz; done
done
MH=$(sub -c 12 -t 01:00:00 -J submghd scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8 --inputs $PUB $PRIV --out results/sub_hd/submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff_hd_r.json)
MO=$(sub -c 12 -t 01:00:00 -J submgold scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8 --inputs $PUB $PRIV --out results/sub_old/submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff.json)
VEC=$(sub -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J ctvec scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/trunk_predict.py extract --inputs $PUB $PRIV --vec results/ct_submission/vec --workers 10)
echo "merges $MG $MH $MO, trunk vectors $VEC"
wait_jobs "$MG,$MH,$MO,$VEC"
for j in $MG $MH $MO $VEC; do echo "$j $(state $j)"; done
grep -h "wrote\|WARNING" logs/nano4/submg-$MG.out logs/nano4/submghd-$MH.out logs/nano4/submgold-$MO.out
CT=$(sub -t 01:00:00 -J ctpred scripts/nano4/run.sbatch python -I scripts/trunk_predict.py predict --inputs $PUB $PRIV --shards submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models results/models/ct/all_C_s0.pt results/models/ct/all_C_s1.pt results/models/ct/all_C_s2.pt results/models/ct/all_C_s3.pt results/models/ct/all_C_s4.pt results/models/ct/all_C_s5.pt results/models/ct/all_T_s0.pt results/models/ct/all_T_s1.pt results/models/ct/all_T_s2.pt results/models/ct/all_T_s3.pt results/models/ct/all_T_s4.pt results/models/ct/all_T_s5.pt --fallback-submission submission_rank.csv --out submission_rank_ct.csv)
wait_jobs "$CT"; echo "ctpred $CT: $(state $CT)"; grep -v "^+ \|^== 2026" logs/nano4/ctpred-$CT.out | tail -6
echo "== files"; for f in submission_rank.csv submission_rank_ct.csv results/sub_hd/submission_rank.csv results/sub_old/submission_rank.csv; do [ -f $f ] && echo "$(md5sum $f | cut -c1-12) $(wc -l < $f) lines  $f"; done
python - <<'PY'
import pandas as pd, itertools
fs = {"main": "submission_rank.csv", "ct": "submission_rank_ct.csv", "hd": "results/sub_hd/submission_rank.csv", "old": "results/sub_old/submission_rank.csv"}
d = {k: pd.read_csv(v, dtype=str).set_index("question_id")["pred_rank"] for k, v in fs.items() if __import__("os").path.exists(v)}
for a, b in itertools.combinations(d, 2):
    print(f"{a} vs {b}: {(d[a] != d[b].reindex(d[a].index)).sum()} of {len(d[a])} decisions differ")
for k, s in d.items():
    print(k, s.value_counts().sort_index().to_dict())
PY
echo "== $(date '+%F %T') flow done"
