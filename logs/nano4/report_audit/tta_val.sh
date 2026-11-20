#!/usr/bin/env bash
# TTA validation (2026-11-05): fold 1's 1,592 n=20 pilot questions as a mock test, 8 symmetries x (shards 8 GPUs -> trunk extract ->
# predict with the 6 fold-1 validation checkpoints, heads3_B tokens), then the paired comparison 8-average vs symmetry 0.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
AUD=/home/u2507115/.cache/report_audit
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do [ -z "$id" ] && continue; until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() { local j=""; while [ -z "$j" ]; do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) || { j=""; sleep 90; }; done; echo "$j"; }
echo "== $(date '+%F %T') TTA validation start"
M0=$(grep mockn20 $AUD/tta_val_prep.txt | awk '{print $2}'); R=$(grep "val1 members" $AUD/tta_val_prep.txt | awk '{print $3}')
wait_jobs "$M0,$R"; echo "mock $M0 $(state $M0); members $R $(state $R)"
[ "$(state $M0)" = "COMPLETED" ] && [ "$(state $R)" = "COMPLETED" ] || { echo "prerequisites failed: abort"; exit 1; }
ls results/models/ct_check/val1_*_s*.pt | head -12
M=$(ls results/models/ct_check/val1_C_s0.pt results/models/ct_check/val1_C_s1.pt results/models/ct_check/val1_C_s2.pt results/models/ct_check/val1_T_s0.pt results/models/ct_check/val1_T_s1.pt results/models/ct_check/val1_T_s2.pt | tr '\n' ' ')
SRC=data/mock_tests/fold1_n20
NQ=$(($(wc -l < $SRC/answers.csv) - 1)); NP=$(($(wc -l < $SRC/rank_prediction_test_public.csv) - 1)); NV=$((NQ - NP))
EXPECT="--expect-rows $NQ --expect-public $NP --expect-private $NV"
echo "questions $NQ (public $NP, private $NV)"
run_sym() {  # run_sym <k>: transform (k > 0), shards, extract, predict; the per-symmetry vectors (~3 GB) are deleted after the prediction
  local k=$1 OUT=results/tta_val/sym$1 D PRE IN
  if [ "$k" = 0 ]; then D=$SRC; PRE=":"; else D=data/mock_tests/fold1_n20_sym$k; PRE="python -I logs/nano4/sgf_symmetry.py --inputs $SRC/rank_prediction_test_public.csv $SRC/rank_prediction_test_private.csv --k $k --out-dir $D --copy $SRC/answers.csv"; fi
  IN="$D/rank_prediction_test_public.csv $D/rank_prediction_test_private.csv"
  submit ttav$k "--gres=gpu:8 -c 96 -t 06:00:00" bash -c "$PRE && bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --inputs $IN $EXPECT --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --workers 10 && bash scripts/nano4/shards.sh 8 python -I scripts/trunk_predict.py extract --inputs $IN --vec $OUT/vec --workers 10 && python -I scripts/trunk_predict.py predict --inputs $IN $EXPECT --shards $OUT/submission_rank.csv --nshards 8 --vec $OUT/vec --models $M --l1b-names heads3_B --allow-fallback --out $OUT/submission_rank_ct.csv; rc=\$?; rm -rf $OUT/vec; exit \$rc"
}
for wave in "0 1 2 3" "4 5 6 7"; do  # two waves of four: peak disk ~13 GB of vectors
  ids=""
  for k in $wave; do j=$(run_sym $k); echo "sym $k job $j"; ids="$ids,$j"; done
  wait_jobs "$ids"
  for j in ${ids//,/ }; do [ -z "$j" ] && continue; echo "$j $(state $j)"; done
  df -h /home/u2507115 | tail -1
done
E=$(submit ttavaleval "-c 12 -t 00:30:00" python -I logs/nano4/tta_val_eval.py --root results/tta_val --out results/r_variants/tta_val.json)
wait_jobs "$E"; echo "== TTA validation eval $E $(state $E)"; grep -v "^+" logs/nano4/ttavaleval-$E.out
echo "== $(date '+%F %T') TTA validation done"
