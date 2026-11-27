#!/usr/bin/env bash
# One seed of M221 (window input + external strangers; five networks) on ONE node of a multi-node job, 2026-11-18:
#   bash logs/nano4/m221_node.sh <arm> <seed>
# Checkpoints are written to the node's memory disk and copied to results/m221/<arm>_ckpt (on /home, about 0.55 GB per seed)
# right after the training when /home has at least 4 GB free - the job ends at a fixed time, nothing may stay only on the node.
main() {
  local arm=${1:?arm} seed=${2:?seed} L D rc free; L=/dev/shm/m221/$arm; D=results/m221/${arm}_ckpt
  export SLURM_JOB_ID="${SLURM_JOB_ID:-local}$arm" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2; mkdir -p "$L" || return 1
  echo "== $(date '+%F %T') M221 arm $arm seed $seed on $(hostname); GPUs $CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}" | cut -d, -f1-5) python -I logs/nano4/m221_prod.py --jobs 1:$seed 2:$seed 3:$seed 4:$seed all:$seed --ext-share 0.25 --oe 0 --out "$arm" --ck-dir "$L" --no-periodic; rc=$?
  echo "== $(date '+%F %T') arm $arm seed $seed: training ended with exit code $rc"
  [ $rc = 0 ] || return $rc
  free=$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc 0-9)
  if [ "$free" -ge 4000 ]; then mkdir -p "$D" && cp "$L"/ckpt_*_s"$seed".pt "$D"/ && echo "== $(date '+%F %T') copied the five checkpoints of seed $seed to $D ($(du -sh "$D" | cut -f1)); /home free $free MB"; else echo "== /home has only $free MB free: checkpoints NOT copied (they are in $L on $(hostname))"; fi
}
main "$@"; exit $?
