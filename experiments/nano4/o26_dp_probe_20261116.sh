#!/usr/bin/env bash
set -euo pipefail
r2=/home/u2507115/BK_AICUP_2026-2
x=/work/u2507115/ext_fox
out=/work/u2507115/pid_v2/o26_parallel_probe_20261116/dp2
python logs/nano4/test_o26_parallel_forward.py
[[ ! -e "$out" ]] || { echo "output already exists: $out"; exit 1; }
mkdir -p "$out"
IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#cards[@]} == 2 ]] || { echo 'expected two allocated GPUs'; exit 1; }
spec=$(mktemp)
trap 'rm -f "$spec"' EXIT
printf '%s\n' "${cards[0]},${cards[1]} logs/nano4/pidseq_-1_o26probe_dp2_Ss0-${SLURM_JOB_ID}.log --net -1 --tag o26probe_dp2_Ss0 --steps 1500 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --drop-fold 4 --out $out --seed 180" > "$spec"
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I logs/nano4/o26_dp_train_20261116.py --fork "$spec" \
  --ext-ply "$x/pilot/ply" "$x/scale1/ply_a" "$x/scale1/ply_b" "$r2/data/ext_ply/scale1_c" "$r2/data/ext_ply/scale2_top" "$r2/data/ext_ply/scale2_5d" \
  --ext-select "$x/pilot/select.csv" "$x/scale1/select.csv" "$x/scale2_top/select.csv" "$x/scale2_5d/select.csv"
