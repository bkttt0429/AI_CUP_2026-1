#!/usr/bin/env bash
# M216 (input up to 384 plies) on ONE node of a multi-node job, 2026-11-17:  bash logs/nano4/m216_node.sh <seed>
# smoke (tail attach checks, step time, GPU memory -> number of checkpointed layers) -> list of the truncated games -> tail
# extraction on the node's 8 GPUs -> the five networks of this seed on 5 GPUs. Tail parts and running checkpoints live in the
# node's memory disk (/dev/shm/m216_s<seed>); they are copied to /work only after logs/nano4/multi_spool/sync_ok exists and
# /work has >= 8 GB free (multi_lib.sh).
main() {
  local seed=${1:?seed}; source logs/nano4/multi_lib.sh
  local L=/dev/shm/m216_s$seed KEYS=results/m216/tail_keys_s$seed.npy SK=results/m216_smoke/tail_keys_s$seed.npy FLAG=logs/nano4/multi_spool/sync_ok
  export SLURM_JOB_ID="${SLURM_JOB_ID:-local}n$seed" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  local LOGS=logs/nano4/m216_net-1_s$seed-$SLURM_JOB_ID.log G1; G1=$(first_gpus 1)
  mkdir -p $L/tail $L/ckpt $L/smoke_tail $L/smoke_ckpt results/m216 results/m216_smoke || return 1
  echo "== $(date '+%F %T') M216 seed $seed on $(hostname); GPUs $CUDA_VISIBLE_DEVICES; scratch $L; /dev/shm free $(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9) GB; /work free $(work_free_gb) GB"
  python -I logs/nano4/m216_tail_extract.py --list --first-files 20 --list-file $SK --workers 10 || return 1
  CUDA_VISIBLE_DEVICES=$G1 python -I logs/nano4/m216_tail_extract.py --list-file $SK --out $L/smoke_tail --workers 10 || return 1
  echo "smoke tail parts: $(ls $L/smoke_tail | wc -l), $(du -sh $L/smoke_tail | cut -f1)"
  CUDA_VISIBLE_DEVICES=$G1 python -I logs/nano4/m216_check.py || return 1
  local k CKL="" peak
  for k in 0 3 6; do
    rm -f $L/smoke_ckpt/*
    if CUDA_VISIBLE_DEVICES=$G1 python -I logs/nano4/m216_prod.py --jobs all:$seed --ckpt-layers $k --smoke --tail $L/smoke_tail --tail-list $SK --ck-dir $L/smoke_ckpt; then
      peak=$(grep -o 'peak GPU memory [0-9]* GB' $LOGS | tail -1 | tr -dc 0-9)
      echo "== smoke with $k checkpointed layers passed: peak GPU memory ${peak:-?} GB; $(grep -o '[0-9.]* s/step' $LOGS | tail -1)"
      if [ -n "$peak" ] && [ "$peak" -le 128 ]; then CKL=$k; break; fi
    else
      echo "== smoke with $k checkpointed layers FAILED"; tail -n 12 $LOGS | cut -c1-300
    fi
  done
  grep -E "tail attached|held-out own games|best row-weighted|outputs written" $LOGS | tail -n 8 | cut -c1-420
  [ -n "$CKL" ] || { echo "== no smoke configuration fits the GPU: stopping"; return 1; }
  rm -rf $L/smoke_tail $L/smoke_ckpt
  python -I logs/nano4/m216_tail_extract.py --list --list-file $KEYS --workers 24 || return 1
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/m216_tail_extract.py --workers 10 --out $L/tail --list-file $KEYS || return 1
  echo "== $(date '+%F %T') tail parts: $(ls $L/tail | wc -l), $(du -sh $L/tail | cut -f1); training with $CKL checkpointed layers"
  CUDA_VISIBLE_DEVICES=$(first_gpus 5) python -I logs/nano4/m216_prod.py --jobs 1:$seed 2:$seed 3:$seed 4:$seed all:$seed --ckpt-layers $CKL --steps 40000 --tail $L/tail --tail-list $KEYS --ck-dir $L/ckpt &
  local pid=$!; save_loop $L/ckpt /work/u2507115/m216_ckpt $FLAG $pid & wait $pid; local rc=$?
  echo "== $(date '+%F %T') training ended with exit code $rc"
  save_final $L/ckpt /work/u2507115/m216_ckpt $FLAG 20; [ "$seed" = 30 ] && save_final $L/tail /work/u2507115/m216_tail $FLAG 20
  return $rc
}
main "$@"; exit $?
