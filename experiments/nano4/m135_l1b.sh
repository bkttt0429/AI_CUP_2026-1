#!/usr/bin/env bash
# M135, one 8-GPU job: the external set's L1b scores (heads3_A), the target's moves then the opponent's, eight shards each.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
E=/work/u2507115/ext_rank
main() {
  export AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv
  F="--folds 9 --folds-csv $E/player_folds.csv --workers 10"
  echo "== $(date '+%T') L1b target";   bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt $F --out $E/l1b/scores3_hd
  for i in 0 1 2 3 4 5 6 7; do tail -n 1 logs/nano4/shard$i-$SLURM_JOB_ID.log | cut -c1-160; done
  echo "== $(date '+%T') L1b opponent"; bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt $F --out $E/l1b/scores3_opp_hd --opponent
  for i in 0 1 2 3 4 5 6 7; do tail -n 1 logs/nano4/shard$i-$SLURM_JOB_ID.log | cut -c1-160; done
  echo "== $(date '+%T') done"
}
main "$@"; exit $?
