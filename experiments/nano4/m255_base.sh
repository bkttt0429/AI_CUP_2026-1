#!/usr/bin/env bash
# M255 (2026-11-20): the 20 B questions + 6 control test questions through ct9's test-side path (copy of ~/.cache/report_audit/b1_public.sh,
# one GPU per stage, HD tags present so no --allow-missing-hd). Outputs under results/m255/base/. Nothing is uploaded.
main() {
  set -uo pipefail
  cd /home/u2507115/BK_AICUP_2026-1 || return 1
  EX='--exclude=25a-hgpn[001,003,004,006-010,039,062,069,082,083,145-147,155,159,161,166,167,171,172,174,177]'
  OUT=results/m255/base
  IN="results/m255/inputs/bq.csv results/m255/inputs/ctl.csv"
  HEADS="--l1b-heads results/l1b/heads3_A.pt results/l1b/heads3_B.pt"
  EXP="--expect-rows 26 --expect-public 26 --expect-private 0"
  MODELS=$(ls results/models/ct_cv_m33/val{1,2,3,4}_{C,T}-pc-f0-x40-oh_s{0,1,2}.pt | tr '\n' ' ')
  [ "$(echo $MODELS | wc -w)" = 24 ] || { echo "24 members expected"; return 1; }
  sub() { sbatch --parsable -p 8gpus $EX "$@"; }
  mkdir -p "$OUT/vec"
  B=$(sub --gres=gpu:1 -c 4 -t 00:20:00 -J m255in scripts/nano4/run.sbatch python -I logs/nano4/m255_inputs.py) || return 1
  S=$(sub --dependency=afterok:$B --gres=gpu:1 -c 12 -t 03:00:00 -J m255sub scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 1 python -I scripts/predict_submission.py --inputs $IN --out $OUT/submission_rank.csv $HEADS --workers 10 $EXP) || return 1
  V=$(sub --dependency=afterok:$B --gres=gpu:1 -c 12 -t 02:00:00 -J m255vec scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 1 python -I scripts/trunk_predict.py extract --inputs $IN --vec $OUT/vec --workers 10 $EXP) || return 1
  M=$(sub --dependency=afterok:$S --gres=gpu:1 -c 12 -t 01:00:00 -J m255merge scripts/nano4/run.sbatch python -I scripts/predict_submission.py --merge --nshards 1 --inputs $IN --out $OUT/submission_rank.csv $HEADS --fused-model results/models/rank_model_fused_opp_diff_hd_r.json $EXP) || return 1
  P=$(sub --dependency=afterok:$M:$V --gres=gpu:1 -c 12 -t 01:00:00 -J m255pred scripts/nano4/run.sbatch python -I scripts/trunk_predict.py predict --inputs $IN --shards $OUT/submission_rank.csv --nshards 1 --vec $OUT/vec --models $MODELS --fallback-submission $OUT/submission_rank.csv --out $OUT/submission_rank_ct9.csv $EXP) || return 1
  echo "inputs $B, scoring $S, vectors $V, merge $M, predict $P"
  echo "$B $S $V $M $P" > results/m255/base_jobs.txt
}
main "$@"; exit $?
