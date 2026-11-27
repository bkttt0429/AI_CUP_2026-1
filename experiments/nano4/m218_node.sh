#!/usr/bin/env bash
# One M218 arm (five networks) on ONE node of a multi-node job, 2026-11-17:  bash logs/nano4/m218_node.sh <arm> <ext share> <outlier weight> [seed]
# Checkpoints stay in the node's memory disk; after the training the script waits for logs/nano4/multi_spool/keep_<arm>
# (copy the five slim checkpoints to /work/u2507115/m218_ckpt/<arm> once /work has >= 8 GB free) or keep_<arm>.drop (leave).
main() {
  local arm=${1:?arm} share=${2:?share} oe=${3:?oe} seed=${4:-30}; source logs/nano4/multi_lib.sh; local L=/dev/shm/m218/$arm
  export SLURM_JOB_ID="${SLURM_JOB_ID:-local}$arm" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p $L || return 1
  echo "== $(date '+%F %T') M218 arm $arm (external share $share, outlier weight $oe, seed $seed) on $(hostname); GPUs $CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES=$(first_gpus 5) python -I logs/nano4/m218_prod.py --jobs 1:$seed 2:$seed 3:$seed 4:$seed all:$seed --ext-share $share --oe $oe --out $arm --ck-dir $L --no-periodic; local rc=$?
  echo "== $(date '+%F %T') arm $arm training ended with exit code $rc"
  [ $rc = 0 ] && save_final $L /work/u2507115/m218_ckpt/$arm logs/nano4/multi_spool/keep_$arm 20
  return $rc
}
main "$@"; exit $?
