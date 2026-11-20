#!/usr/bin/env bash
# fold-1 100-question mock through the main shard step, then the C+T test-side path, then the offline reference and the comparison.
# args: <mock1 job id> <ckC1 job id> <ckT1 job id>
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
IN="data/mock_tests/fold1_100/rank_prediction_test_public.csv data/mock_tests/fold1_100/rank_prediction_test_private.csv"
EXPECT="--expect-rows 100 --expect-public 50 --expect-private 50"
OUT=results/mock_fold1_100
wait_jobs() { for id in ${1//,/ }; do until [ -z "$(squeue -h -j $id 2>/dev/null)" ]; do sleep 60; done; done; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }

wait_jobs "$1"
echo "mock1 $1: $(state $1)"
[ "$(state $1)" = "COMPLETED" ] || { echo "mock build failed"; exit 1; }
J1=$(sbatch --parsable $EX -p 8gpus --gres=gpu:4 -c 48 -t 04:00:00 -J mocksh scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 4 python -I scripts/predict_submission.py --inputs $IN $EXPECT --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --workers 10)
echo "shard scoring $J1"
wait_jobs "$J1"
echo "mocksh $J1: $(state $J1)"
[ "$(state $J1)" = "COMPLETED" ] || { echo "shard scoring failed"; exit 1; }
J2=$(sbatch --parsable $EX -c 12 -t 00:30:00 -J mockmg scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 4 --inputs $IN $EXPECT --out $OUT/submission_rank.csv --l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt --fused-model results/models/rank_model_fused_opp_diff_r.json)
echo "merge $J2"
wait_jobs "$J2"
echo "mockmg $J2: $(state $J2)"
J3=$(sbatch --parsable $EX -t 01:00:00 -J mockvec scripts/nano4/run.sbatch python -I scripts/trunk_predict.py extract --inputs $IN --vec $OUT/vec --workers 10)
echo "extract $J3"
wait_jobs "$J3"
echo "mockvec $J3: $(state $J3)"
[ "$(state $J3)" = "COMPLETED" ] || { echo "extract failed"; exit 1; }
wait_jobs "$2,$3"
echo "ckC1 $2: $(state $2), ckT1 $3: $(state $3)"
M="results/models/ct_check/val1_C_s0.pt results/models/ct_check/val1_T_s0.pt"
J4=$(sbatch --parsable $EX -t 01:00:00 -J mockct scripts/nano4/run.sbatch bash -c "python -I scripts/trunk_predict.py predict --inputs $IN $EXPECT --shards $OUT/submission_rank.csv --nshards 4 --vec $OUT/vec --models $M --l1b-names heads3_B --fallback-submission $OUT/submission_rank.csv --out $OUT/submission_rank_ct.csv && python -I scripts/trunk_predict.py offline --mock-dir data/mock_tests/fold1_100 --fold 1 --models $M --out $OUT/offline_ct_proba.csv && python -I scripts/trunk_predict.py compare --a $OUT/submission_rank_ct_proba.csv --b $OUT/offline_ct_proba.csv --answers data/mock_tests/fold1_100/answers.csv --out $OUT/ct_check.json")
echo "predict/offline/compare $J4"
wait_jobs "$J4"
echo "mockct $J4: $(state $J4)"
tail -30 logs/nano4/mockct-$J4.out
