#!/usr/bin/env bash
# Run N copies of a sharded command, one per allocated GPU, and wait for all of them:
#   bash scripts/nano4/shards.sh 8 python -I scripts/score_folds.py --folds 0 --workers 10
# Copy i sees only the i-th GPU of the allocation and gets "--shard i --nshards N" appended.
# Its output goes to logs/nano4/shard<i>-<jobid>.log. Exits non-zero if any copy fails.
set -uo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <n_shards> <command> [args...]" >&2
  exit 2
fi
N="$1"
shift
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((N - 1)))}"
if [ "${#GPUS[@]}" -lt "$N" ]; then
  echo "need $N GPUs but the job has ${#GPUS[@]} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})" >&2
  exit 1
fi
mkdir -p logs/nano4
pids=()
for ((i = 0; i < N; i++)); do
  log="logs/nano4/shard$i-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$@" --shard "$i" --nshards "$N" > "$log" 2>&1 &
  pids+=($!)
  echo "shard $i on GPU ${GPUS[$i]}: pid ${pids[$i]}, log $log"
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "shard $i done"
  else
    echo "shard $i FAILED, see logs/nano4/shard$i-${SLURM_JOB_ID:-local}.log"
    failed=1
  fi
done
exit "$failed"
