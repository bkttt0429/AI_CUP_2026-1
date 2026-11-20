#!/usr/bin/env bash
# M135: submit the external set's trunk-vector extraction (int8, 16 shards on two 8-GPU jobs) if /work has room; prints the job ids.
#   bash logs/nano4/m135_trunk.sh [min free GB, default 10]
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,069,155,177]'
A=/home/u2507115/.cache/report_audit; E=/work/u2507115/ext_rank
ENVV="AICUP_TRAIN_DIR=$E/training AICUP_FOLDS_CSV=$E/player_folds.csv"
main() {
  need=${1:-10}; free=$(df --output=avail -BG /work/u2507115 | tail -n 1 | tr -dc '0-9'); echo "/work free: ${free} GB (need ${need})"; du -sh $E
  [ "$free" -ge "$need" ] || { echo "not enough free space on /work: nothing submitted"; return 1; }
  rm -f $A/m135_trunk_*.txt
  for k in $(seq 0 15); do
    echo "$ENVV python -I scripts/trunk_extract.py --folds 9 --folds-csv $E/player_folds.csv --out $E/trunk_vec --workers 10 --quant8 results/m90/glstats.npz --shard $k --nshards 16" >> $A/m135_trunk_$((k / 8)).txt
  done
  v0=$(sbatch --parsable $EX -p 8gpus --gres=gpu:8 -c 96 -t 04:00:00 -J m135v0 scripts/nano4/run.sbatch bash $A/pack8.sh $A/m135_trunk_0.txt)
  v1=$(sbatch --parsable $EX -p 16gpus --gres=gpu:8 -c 96 -t 04:00:00 -J m135v1 scripts/nano4/run.sbatch bash $A/pack8.sh $A/m135_trunk_1.txt)
  echo "trunk jobs $v0 $v1"
}
main "$@"; exit $?
