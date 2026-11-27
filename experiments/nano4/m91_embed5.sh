#!/usr/bin/env bash
# M91: both-side identity vectors of all games under the M90 networks, one cross-fit network index per GPU (its three seeds in turn).
#   sbatch --gres=gpu:5 -c 60 ... scripts/nano4/run.sbatch bash logs/nano4/m91_embed5.sh [net ...]   (default nets: 1 2 3 4 -1)
# Each GPU is tried first; networks go round-robin over the GPUs that work. Finished files are skipped when the job is run again.
set -uo pipefail
if [ $# -gt 0 ]; then NETS=("$@"); else NETS=(1 2 3 4 -1); fi
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "${#NETS[@]}" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for ${#NETS[@]} networks" >&2; exit 1; fi
i=0; pids=()
for v in "${NETS[@]}"; do
  g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/m91emb_net${v}-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I logs/nano4/m91_embed.py --net "$v" > "$log" 2>&1 &
  pids+=($!); echo "net $v on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))
done
failed=0
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done
ls -la results/m91/ | tail -n 25
exit "$failed"
