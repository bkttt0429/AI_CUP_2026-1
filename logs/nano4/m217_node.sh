#!/usr/bin/env bash
# One node of a multi-node job, 2026-11-17: the M217 diagnosis (GPU 0), the M218 smoke run (GPU 1) and the M218 evaluation of
# the production control checkpoints with the stranger queries (GPUs 2-6). Nothing is written to /work.
main() {
  source logs/nano4/multi_lib.sh; local J=${SLURM_JOB_ID:-local}; IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p /dev/shm/m218/smoke /dev/shm/m218/control
  echo "== $(date '+%F %T') on $(hostname); GPUs ${G[*]}"
  SLURM_JOB_ID=${J}d CUDA_VISIBLE_DEVICES=${G[0]} python -I logs/nano4/m217_generic.py > logs/nano4/m217_generic-$J.out 2>&1 & local p1=$!
  SLURM_JOB_ID=${J}k CUDA_VISIBLE_DEVICES=${G[1]} python -I logs/nano4/m218_prod.py --jobs all:30 --ext-share 0.25 --oe 0.5 --out smoke --smoke --ck-dir /dev/shm/m218/smoke > logs/nano4/m218_smoke-$J.out 2>&1 & local p2=$!
  SLURM_JOB_ID=${J}c CUDA_VISIBLE_DEVICES=${G[2]},${G[3]},${G[4]},${G[5]},${G[6]} python -I logs/nano4/m218_prod.py --jobs 1:30 2:30 3:30 4:30 all:30 --out control --eval-ckpt /work/u2507115/m161q_ckpt --ck-dir /dev/shm/m218/control > logs/nano4/m218_control-$J.out 2>&1 & local p3=$!
  local rc=0; wait $p2; echo "== $(date '+%F %T') M218 smoke: exit $?"; tail -n 6 logs/nano4/m218_smoke_net-1_s30-${J}k.log 2>/dev/null | cut -c1-600
  wait $p1 || rc=1; echo "== $(date '+%F %T') M217: exit $rc"; grep -E "^(seen|unseen|fewgames)" logs/nano4/m217_generic-$J.out | cut -c1-400
  wait $p3 || rc=1; echo "== $(date '+%F %T') M218 control evaluation done"; tail -n 6 logs/nano4/m218_control-$J.out | cut -c1-600
  return $rc
}
main "$@"; exit $?
