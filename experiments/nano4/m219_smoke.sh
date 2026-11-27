#!/usr/bin/env bash
# M219 smoke (2026-11-17): the window builders against a plain replay, then 80 training steps of each arm on two GPUs.
main() {
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; local g; g=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | cut -d, -f1-2)
  echo "== $(date '+%F %T') on $(hostname)"; python -I logs/nano4/pid_win.py || return 1; python -I logs/nano4/pid_win2.py || return 1
  local arm prev
  for arm in w9:0 w9p:1; do
    prev=${arm#*:}; arm=${arm%:*}; rm -rf /dev/shm/m219_ckpt_smoke/$arm
    CUDA_VISIBLE_DEVICES=$g python -I logs/nano4/m219_prod.py --smoke --out $arm --win-prev $prev --jobs 1:30 all:30 || { echo "== smoke of $arm FAILED"; return 1; }
    echo "== $(date '+%F %T') smoke of $arm passed"; grep -h "window input\|window encoder\|step 80/80\|step 80:" logs/nano4/m219_${arm}_net*_s30-${SLURM_JOB_ID}.log | cut -c1-420
  done
  rm -rf /dev/shm/m219_ckpt_smoke
}
main "$@"; exit $?
