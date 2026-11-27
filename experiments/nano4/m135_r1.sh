#!/usr/bin/env bash
# M135 recovery 1 (one 8-GPU job) after the two unreplayable records were cut: phase A in parallel - the L1b target shards 5 and 7
# resume, the Human SL target games of the dead shard 7/16 as six sub-shards (ids 7, 23, ... of 96); phase B - the L1b opponent pass.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
E=/work/u2507115/ext_rank; A=/home/u2507115/.cache/report_audit
main() {
  echo "== $(date '+%T') phase A"; bash $A/pack8.sh $A/m135_r1_a.txt; a=$?; echo "phase A exit $a"
  for i in 0 1 2 3 4 5 6 7; do tail -n 1 logs/nano4/pack-$SLURM_JOB_ID-$i.log | cut -c1-160; done
  export AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv
  echo "== $(date '+%T') phase B: L1b opponent"; bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt --folds 9 --folds-csv $E/player_folds.csv --workers 10 --out $E/l1b/scores3_opp_hd --opponent; b=$?; echo "phase B exit $b"
  for i in 0 1 2 3 4 5 6 7; do tail -n 1 logs/nano4/shard$i-$SLURM_JOB_ID.log | cut -c1-160; done
  echo "== $(date '+%T') done"; [ $a -eq 0 ] && [ $b -eq 0 ]
}
main "$@"; exit $?
