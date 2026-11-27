#!/usr/bin/env bash
# M101: the M90 sequence identity model with a shorter schedule, with or without external players (Fox Go dataset accounts; per-ply
# parts of the player-identification session, read-only), one cross-fit network per GPU of the job. Launcher logic of m90_train5.sh.
#   sbatch --gres=gpu:5 -c 60 ... scripts/nano4/run.sbatch bash scripts/m_series/m101_train5.sh <seed> <ext-share|none> <steps> <out folder:VAR> [net ...]
set -uo pipefail
SEED="$1"; SHARE="$2"; STEPS="$3"; OUT="$4"; shift 4
EXT_PLY="${EXT_PLY:-/work/u2507115/ext_fox/pilot/ply}"; EXT_SEL="${EXT_SEL:-/work/u2507115/ext_fox/pilot/select.csv}"
if [ "$SHARE" = "none" ]; then EXT=(); else EXT=(--ext-ply "$EXT_PLY" --ext-select "$EXT_SEL" --ext-share "$SHARE"); fi
if [ $# -gt 0 ]; then NETS=("$@"); else NETS=(1 2 3 4 -1); fi
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "${#NETS[@]}" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for ${#NETS[@]} networks" >&2; exit 1; fi
i=0; pids=(); name="${OUT%%:*}"
for v in "${NETS[@]}"; do
  g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/${name}s${SEED}_net${v}-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I scripts/m_series/m88_seqid.py --stage train --net "$v" --full --seed "$SEED" --steps "$STEPS" --out "$OUT" "${EXT[@]}" > "$log" 2>&1 &
  pids+=($!); echo "seed $SEED net $v on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))
done
failed=0
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done
exit "$failed"
