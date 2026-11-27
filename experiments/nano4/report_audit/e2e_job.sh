#!/usr/bin/env bash
# M34 / M35 driver inside one allocation: prepare (replay into /dev/shm) -> one train process per GPU -> stats. MODE smoke | full.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
MODE=${1:-smoke}; DATA=/dev/shm/e2e_${SLURM_JOB_ID:-local}; OUT=results/r_variants/e2e$([ "$MODE" = smoke ] && echo _smoke)
df -h /dev/shm | tail -1
rm -rf "$DATA" "$OUT"
if [ "$MODE" = smoke ]; then python -I logs/nano4/e2e_pilot.py prepare --data "$DATA" --workers 20 --smoke || exit 1
else python -I logs/nano4/e2e_pilot.py prepare --data "$DATA" --workers 88 || exit 1; fi
du -sh "$DATA"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [ "$MODE" = smoke ]; then RUNS=("mean 0" "ftattn 0"); EXTRA="--epochs 1 --log-every 2 --batch 8"; else RUNS=("mean 0" "mean 1" "attn 0" "attn 1" "ft 0" "ft 1" "ftattn 0" "ftattn 1"); EXTRA=""; fi
pids=(); i=0
for r in "${RUNS[@]}"; do set -- $r
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=8 python -I logs/nano4/e2e_pilot.py train --data "$DATA" --arm $1 --seed $2 --out "$OUT" $EXTRA > logs/nano4/e2e-${SLURM_JOB_ID:-local}-$1-$2.log 2>&1 &
  pids+=($!); i=$((i + 1))
done
fail=0; for k in "${!pids[@]}"; do wait "${pids[$k]}" || { echo "run ${RUNS[$k]} FAILED"; fail=1; }; done
for f in logs/nano4/e2e-${SLURM_JOB_ID:-local}-*.log; do echo "== $f"; grep -vE "Warning|warnings.warn" "$f" | tail -4 | cut -c1-220; done
python -I logs/nano4/e2e_pilot.py stats --data "$DATA" --out-dir "$OUT" --out "$OUT.json"
rm -rf "$DATA"
exit $fail
