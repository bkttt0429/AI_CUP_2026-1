#!/usr/bin/env bash
# One launcher of M219 (identity networks with the window input) on a node of a multi-node job, 2026-11-17:
#   bash logs/nano4/m219_node.sh <arm> <second window 0|1> <GPU fields of the node's list, e.g. 1-5> <NET:SEED> [<NET:SEED> ...]
# Checkpoints stay in the node's memory disk (/dev/shm/m219/<arm>); nothing is written to /work. Several launchers may share
# a node (different GPU fields; the task redirects each one's output to its own file).
main() {
  local arm=${1:?arm} prev=${2:?second window} fields=${3:?GPU fields}; shift 3; [ $# -gt 0 ] || { echo "no NET:SEED jobs given"; return 2; }
  local L=/dev/shm/m219/$arm gpus rc; gpus=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" | cut -d, -f"$fields")
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p "$L" || return 1
  echo "== $(date '+%F %T') M219 arm $arm (second window $prev), jobs $*, on $(hostname), GPUs $gpus"
  CUDA_VISIBLE_DEVICES=$gpus python -I logs/nano4/m219_prod.py --out "$arm" --win-prev "$prev" --jobs "$@" --ck-dir "$L" --no-periodic; rc=$?
  echo "== $(date '+%F %T') arm $arm jobs $*: training ended with exit code $rc; in $L: $(ls "$L" | tr '\n' ' ')"
  return $rc
}
main "$@"; exit $?
