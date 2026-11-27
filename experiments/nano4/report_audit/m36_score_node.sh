#!/usr/bin/env bash
# M36 step 2 on one 8-GPU node: shards NODE*8 .. NODE*8+7 of NSHARDS, one GPU and 11 replay workers each.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
NODE=$1; NSH=$2
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
pids=()
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=1 python -I logs/nano4/e2e_pilot.py score --games results/r_variants/attn_games.csv \
    --heads results/r_variants/attn_heads/attn_A.pt results/r_variants/attn_heads/attn_B.pt --out results/attn_scores \
    --shard $((NODE * 8 + i)) --nshards $NSH --workers 11 --part-size 500 > logs/nano4/m36score-${SLURM_JOB_ID:-local}-$i.log 2>&1 &
  pids+=($!)
done
fail=0; for k in "${!pids[@]}"; do wait "${pids[$k]}" || { echo "shard slot $k FAILED"; tail -3 logs/nano4/m36score-${SLURM_JOB_ID:-local}-$k.log; fail=1; }; done
exit $fail
