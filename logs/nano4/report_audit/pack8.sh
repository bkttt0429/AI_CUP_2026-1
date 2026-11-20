#!/usr/bin/env bash
# Run the commands listed in $1 (one per line, at most 8) in parallel inside one 8-GPU allocation, one GPU each; exit non-zero if any fails.
cd /home/u2507115/BK_AICUP_2026-1
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
i=0; pids=(); names=()
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  log="logs/nano4/pack-${SLURM_JOB_ID:-local}-$i.log"
  echo "slot $i on GPU ${GPUS[$i]}: $cmd -> $log"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash -c "$cmd" > "$log" 2>&1 &
  pids+=($!); names+=("$log"); i=$((i + 1))
done < "$1"
fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then echo "slot $k done: $(grep -h 'stopped at step' "${names[$k]}" | tail -1 | cut -c1-120)"; else echo "slot $k FAILED (${names[$k]})"; tail -5 "${names[$k]}"; fail=1; fi
done
exit $fail
