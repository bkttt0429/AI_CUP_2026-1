#!/usr/bin/env bash
# M135 smoke (one 8-GPU job): write the external game tables, then run the five scoring passes on two units per rank into a
# scratch root, build its score array and check it (alignment, the int8 trunk vectors against fp16, sanity of the scores by rank).
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
E=/work/u2507115/ext_rank; S=/work/u2507115/ext_rank_smoke
main() {
  [ -f $E/player_folds.csv ] || python -I logs/nano4/m135_ext_prep.py
  cat $E/units.json | head -c 1500; echo
  export AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv
  K="--game-keys $E/smoke_keys.csv"; F="--folds 9 --folds-csv $E/player_folds.csv --workers 10"
  mkdir -p $S; cp -f $E/player_folds.csv $E/hd_table.csv $S/
  echo "== $(date '+%T') trunk vectors (int8)";  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py $F --out $S/trunk_vec --quant8 results/m90/glstats.npz $K
  echo "== $(date '+%T') trunk vectors (fp16)";  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py $F --out $S/trunk_vec_fp16 $K
  echo "== $(date '+%T') L1b target";            bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt $F --out $S/l1b/scores3_hd $K
  echo "== $(date '+%T') L1b opponent";          bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads3_A.pt $F --out $S/l1b/scores3_opp_hd --opponent $K
  echo "== $(date '+%T') Human SL target";       bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 9 --workers 10 --out $S/scores_hd $K
  echo "== $(date '+%T') Human SL opponent";     bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 9 --workers 10 --out $S/scores_opp_hd --opponent $K
  echo "== $(date '+%T') score array";           python -I scripts/trunk_seq_check.py build --ext $S
  echo "== $(date '+%T') checks";                python -I logs/nano4/m135_smoke_check.py $S
  echo "== $(date '+%T') smoke scoring done"
}
main "$@"; exit $?
