#!/usr/bin/env bash
# M90 repair: re-run one shard of the ply extraction, its parts split over the GPUs of this job, then one
# plain pass for anything a copy left behind (a copy on a broken GPU), then a completeness check.
#   sbatch --gres=gpu:8 -c 96 ... scripts/nano4/run.sbatch bash logs/nano4/m90_fix_shard.sh <job> <shard> <parts expected in all three directories>
set -uo pipefail
JOB="$1"; SHARD="$2"; EXPECT="$3"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}; pids=()
for ((r = 0; r < N; r++)); do
  log="logs/nano4/m90fix_r$r-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I logs/nano4/m90_ply_extract.py --job "$JOB" --njobs 3 --shard "$SHARD" --nshards 8 --workers 10 --sub "$r" --nsub "$N" > "$log" 2>&1 &
  pids+=($!); echo "copy $r on GPU ${GPUS[$r]}: pid ${pids[$r]}, log $log"
done
for r in "${!pids[@]}"; do if wait "${pids[$r]}"; then echo "copy $r done"; else echo "copy $r FAILED (GPU ${GPUS[$r]})"; fi; done
for g in "${GPUS[@]}"; do  # sweep: existing parts are skipped, so this is quick when every copy finished
  if CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I logs/nano4/m90_ply_extract.py --job "$JOB" --njobs 3 --shard "$SHARD" --nshards 8 --workers 40 > "logs/nano4/m90fix_sweep-${SLURM_JOB_ID:-local}.log" 2>&1; then echo "sweep on GPU $g done"; break; else echo "sweep on GPU $g FAILED"; fi
done
have=$(ls results/m90_ply0/ply_*.npz results/m90_ply1/ply_*.npz results/m90_ply2/ply_*.npz | grep -vc '\.tmp\.npz$')
tmp=$(ls results/m90_ply0/ results/m90_ply1/ results/m90_ply2/ | grep -c '\.tmp\.npz$')
echo "parts: $have of $EXPECT, unfinished temporary files: $tmp"
[ "$have" -eq "$EXPECT" ] && [ "$tmp" -eq 0 ]
