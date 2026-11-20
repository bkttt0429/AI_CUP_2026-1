#!/usr/bin/env bash
# Regenerate results/trunk_vec_x80 (deleted by my mistake on 2026-11-15 15:15; it holds the per-ply trunk vectors of 149,949
# training games that m88_seqid.load_games reads): the original command of job 704400 with the same 8 shards (same file
# layout: games_s<shard>_p<part>.csv / moves_s<shard>_p<part>.npz, keys sorted inside a shard, 100 games per part), on the GPUs given.
#   bash logs/nano4/x80_regen.sh "2 3 4 5 6"
main() {
  local gpus=($1) i pids=() failed=0
  export PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
  for ((i = 0; i < 8; i++)); do
    CUDA_VISIBLE_DEVICES="${gpus[$((i % ${#gpus[@]}))]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -I scripts/trunk_extract.py --folds 0 1 2 3 4 --allow-fold0 --max-games 80 --game-keys results/data/x80_keys.csv --out results/trunk_vec_x80 --workers 10 --shard "$i" --nshards 8 > "logs/nano4/x80regen-shard$i.log" 2>&1 &
    pids+=($!)
    echo "shard $i on GPU ${gpus[$((i % ${#gpus[@]}))]}: pid ${pids[$i]}"
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "shard $i done"; else echo "shard $i FAILED, see logs/nano4/x80regen-shard$i.log"; failed=1; fi
  done
  return $failed
}
main "$@"; exit $?
