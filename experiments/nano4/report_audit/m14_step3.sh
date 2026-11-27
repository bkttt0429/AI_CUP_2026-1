#!/usr/bin/env bash
# M14 step 3 (only after the step-2 gate passed): 12 production models of the adopted variant, the fold-1 mock consistency check of the
# test-side opponent-HSL path, and submission_rank_ct6.csv. args: <variant: oh|ohr> <sub8oh job id (test-set shards with the 18-profile opponent pass)>
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
CAP=19
AUD=/home/u2507115/.cache/report_audit
TR="python -I scripts/trunk_seq_check.py"
VAR=$1; SUB=$2
FLAG="--opp-hsl"; [ "$VAR" = "ohr" ] && FLAG="--opp-hsl-r"
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
ok_all() { for id in ${1//,/ }; do [ -z "$id" ] && continue; [ "$(state $id)" = "COMPLETED" ] || return 1; done; return 0; }
submit() {
  local j=""
  while [ -z "$j" ]; do
    while [ "$(squeue -u "$USER" -h | wc -l)" -ge "$CAP" ]; do sleep 45; done
    j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }
  done
  echo "$j"
}
echo "== $(date '+%F %T') M14 step 3: variant $VAR ($FLAG), test-set shards job $SUB"
# 1. production models (fold 0 trains too), packed 8 per node, concurrently with the mock check
rm -f $AUD/m14p_cmds_*.txt
for arm in C T; do for s in 0 1 2 3 4 5; do
  echo "$TR train --arm $arm --val-fold all --seed $s --inner-frac 0.1 --with-fold0 --extra-games 40 $FLAG --out results/trunk_seq" >> $AUD/m14p_cmds_all.txt
done; done
split -l 6 -d $AUD/m14p_cmds_all.txt $AUD/m14p_cmds_
ids=""; for f in $AUD/m14p_cmds_0*; do ids="$ids,$(submit ohprod$(basename $f | sed 's/.*_//') "--gres=gpu:6 -c 72 -t 05:00:00" bash $AUD/pack8.sh $f)"; done
echo "production packs ${ids#,}" | tee $AUD/m14_prod_jobs.txt
# 2. mock consistency check (fold-1 100-question mock): shards with the 18-profile opponent pass, the fold-1 seed-0 CV weights of the variant
IN="data/mock_tests/fold1_100/rank_prediction_test_public.csv data/mock_tests/fold1_100/rank_prediction_test_private.csv"
EXPECT="--expect-rows 100 --expect-public 50 --expect-private 50"
OUT=results/mock_fold1_100_oh
mkdir -p $OUT
M="results/models/ct_check/val1_C-x40-${VAR}_s0.pt results/models/ct_check/val1_T-x40-${VAR}_s0.pt"
J1=$(submit mockshoh "-p 8gpus --gres=gpu:4 -c 48 -t 04:00:00" bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs $IN $EXPECT --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --workers 10)
echo "mock shards $J1"; wait_jobs "$J1"; echo "mockshoh $J1 $(state $J1)"
if [ "$(state $J1)" = "COMPLETED" ]; then
  J4=$(submit mockctoh "-t 01:00:00" bash -c "python -I scripts/trunk_predict.py predict --inputs $IN $EXPECT --shards $OUT/submission_rank.csv --nshards 4 --vec results/mock_fold1_100/vec --models $M --l1b-names heads3_B --fallback-submission results/mock_fold1_100/submission_rank.csv --out $OUT/submission_rank_ct.csv && python -I scripts/trunk_predict.py offline --mock-dir data/mock_tests/fold1_100 --fold 1 --models $M --out $OUT/offline_ct_proba.csv && python -I scripts/trunk_predict.py compare --a $OUT/submission_rank_ct_proba.csv --b $OUT/offline_ct_proba.csv --answers data/mock_tests/fold1_100/answers.csv --report $OUT/ct_check.json")
  echo "mock predict/offline/compare $J4"; wait_jobs "$J4"; echo "mockctoh $J4 $(state $J4)"; grep -v "^+" logs/nano4/mockctoh-$J4.out | tail -25 | cut -c1-400
else
  echo "mock shard scoring failed"; tail -5 logs/nano4/mockshoh-$J1.out
fi
# 3. the production models, then ct6 from the test-set shards
wait_jobs "$ids"
for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; grep -h "slot .* done\|FAILED" logs/nano4/ohprod*-$j.out | cut -c1-140; done
ok_all "$ids" || { echo "a production pack failed: no ct6"; exit 1; }
wait_jobs "$SUB"; echo "sub8oh $SUB $(state $SUB)"; [ "$(state $SUB)" = "COMPLETED" ] || { echo "test-set shard scoring failed: no ct6"; exit 1; }
MODELS=""; for arm in C T; do for s in 0 1 2 3 4 5; do MODELS="$MODELS results/models/ct/all_$arm-f0-x40-${VAR}_s$s.pt"; done; done
ls $MODELS | wc -l
P=$(submit ctpred6 "-t 01:00:00" python -I scripts/trunk_predict.py predict --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --shards results/ct6/submission_rank.csv --nshards 8 --vec results/ct_submission/vec --models $MODELS --fallback-submission submission_rank_ct3.csv --out submission_rank_ct6.csv)
wait_jobs "$P"; echo "== ctpred6 $P $(state $P)"; grep -v "^+" logs/nano4/ctpred6-$P.out | tail -4 | cut -c1-700; md5sum submission_rank_ct6.csv
echo "== $(date '+%F %T') M14 step 3 done"
