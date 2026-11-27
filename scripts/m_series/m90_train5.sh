#!/usr/bin/env bash
# M90: cross-fit networks of one seed of the all-games sequence identity model, one per GPU of the job.
#   sbatch --gres=gpu:5 -c 60 ... scripts/nano4/run.sbatch bash scripts/m_series/m90_train5.sh <seed> [net ...]   (default nets: 1 2 3 4 -1)
# Each GPU is tried first (2026-11-12: one GPU of 25a-hgpn177 was not visible to CUDA and the failed job cancelled the
# whole chain). Networks go round-robin over the GPUs that work, so with one broken GPU two networks share a card
# (slower, but the job finishes). Training resumes from its checkpoint when the job is run again.
set -uo pipefail
SEED="$1"; shift
if [ $# -gt 0 ]; then NETS=("$@"); else NETS=(1 2 3 4 -1); fi
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "${#NETS[@]}" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for ${#NETS[@]} networks" >&2; exit 1; fi
i=0; pids=()
for v in "${NETS[@]}"; do
  g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/m90s${SEED}_net${v}-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I scripts/m_series/m88_seqid.py --stage train --net "$v" --full --seed "$SEED" > "$log" 2>&1 &
  pids+=($!); echo "seed $SEED net $v on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))
done
failed=0
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done
exit "$failed"
