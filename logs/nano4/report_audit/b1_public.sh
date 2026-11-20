#!/usr/bin/env bash
# B1 support run (section 4M M62, 2026-11-08): competition 1's old Public candidates (400 x 100 black games -> 2,000 chunks of 20) and its
# 400 questions through ct9's test-side path. Outputs under results/b1_public/: submission_rank_ct9_proba.csv is the 10-dim probability
# file the identification session evaluates B1 with. Nothing is uploaded; nothing feeds competition 2.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
OUT=results/b1_public
IN="$OUT/inputs/cands.csv $OUT/inputs/quest.csv"
HEADS="--l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt"
EXP="--expect-rows 2400 --expect-public 2400 --expect-private 0"
MODELS=$(ls results/models/ct_cv_m33/val{1,2,3,4}_{C,T}-pc-f0-x40-oh_s{0,1,2}.pt | tr '\n' ' ')
sub() { sbatch --parsable $EX "$@"; }
mkdir -p "$OUT/inputs" "$OUT/vec"
B=$(sub -c 4 -t 00:30:00 -J b1build scripts/nano4/run.sbatch python -I logs/nano4/b1_build_inputs.py)
S=$(sub --dependency=afterok:$B --gres=gpu:8 -c 96 -t 08:00:00 -J b1sub8 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py --inputs $IN --out $OUT/submission_rank.csv $HEADS --workers 10 $EXP)
V=$(sub --dependency=afterok:$B --gres=gpu:8 -c 96 -t 04:00:00 -J b1vec scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I scripts/trunk_predict.py extract --inputs $IN --vec $OUT/vec --workers 10 $EXP)
M=$(sub --dependency=afterok:$S -c 12 -t 01:00:00 -J b1merge scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 8 --inputs $IN --out $OUT/submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff_hd_r.json --allow-missing-hd $EXP)
P=$(sub --dependency=afterok:$M:$V -c 12 -t 01:00:00 -J b1pred scripts/nano4/run.sbatch python -I scripts/trunk_predict.py predict --inputs $IN --shards $OUT/submission_rank.csv --nshards 8 --vec $OUT/vec --models $MODELS --fallback-submission $OUT/submission_rank.csv --allow-missing-hd --out $OUT/submission_rank_ct9.csv $EXP)
echo "build $B, scoring $S, vectors $V, merge $M, predict $P"
echo "$B $S $V $M $P" > "$OUT/jobs.txt"
