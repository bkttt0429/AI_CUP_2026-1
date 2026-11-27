#!/usr/bin/env bash
# Isolated, full-data throughput spike; never writes production checkpoints.
set -euo pipefail
workers=${1:?expected 1 or 4}
[[ "$workers" == 1 || "$workers" == 4 ]] || exit 2
r2=/home/u2507115/BK_AICUP_2026-2
x=/work/u2507115/ext_fox
out=/work/u2507115/pid_v2/o26_parallel_probe_20261116/n${workers}
[[ ! -e "$out" ]] || { echo "refusing to reuse probe output: $out"; exit 1; }
mkdir -p "$out"
sha256sum "$r2/scripts/pid_seq_train.py" "$r2/scripts/nano4/pid_seq_train_fork.sh"
nvidia-smi --query-gpu=index,name,uuid,power.limit --format=csv
data="--ext-ply $x/pilot/ply $x/scale1/ply_a $x/scale1/ply_b $r2/data/ext_ply/scale1_c $r2/data/ext_ply/scale2_top $r2/data/ext_ply/scale2_5d --ext-select $x/pilot/select.csv $x/scale1/select.csv $x/scale2_top/select.csv $x/scale2_5d/select.csv"
common="--steps 1500 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --drop-fold 4 --out $out"
specs=("-1 o26probe_n${workers}_Ss0 $common --seed 180")
if [[ "$workers" == 4 ]]; then
  specs+=("-1 o26probe_n4_Ss1 $common --seed 181"
          "-1 o26probe_n4_SMs0 $common --drop-open 0.3 --seed 184"
          "-1 o26probe_n4_SMs1 $common --drop-open 0.3 --seed 185")
fi
bash "$r2/scripts/nano4/pid_seq_train_fork.sh" "$data" "${specs[@]}"
