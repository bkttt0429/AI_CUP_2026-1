#!/usr/bin/env bash
set -eu
ulimit -Su "$(ulimit -Hu)"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
count=${#devices[@]}
pids=()
for ((j=0;j<count;j++)); do
  (
    export CUDA_VISIBLE_DEVICES="${devices[$j]}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    for ((k=j;k<8;k+=count)); do
      /home/u2507115/BK_AICUP_2026-1/.venv/bin/python -u search.py "$k" > "judge_${SLURM_JOB_ID}_$k.log" 2>&1
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
