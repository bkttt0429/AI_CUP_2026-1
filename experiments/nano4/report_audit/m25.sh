#!/usr/bin/env bash
# M25 (strong net + search). Stage 1: 40-game throughput check on 1 GPU; choose visits by the pre-registered rule; stage 2: the 8-GPU pilot + eval.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EXC='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
submit() { for i in $(seq 1 200); do j=$(sbatch --parsable "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
waitj() { until state "$1" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; state "$1"; }
J1=$(submit $EXC -J m25bench -c 12 -t 00:40:00 scripts/nano4/run.sbatch env APPIMAGE_EXTRACT_AND_RUN=1 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits 100 --benchmark-games 40 --out results/ai_search)
echo "bench $J1"; S1=$(waitj $J1); echo "bench $S1"; grep -E "saved|katago" logs/nano4/m25bench-$J1.out | tail -5
VPS=$(grep -oE "[0-9]+ visits/s" logs/nano4/m25bench-$J1.out | tail -1 | grep -oE "^[0-9]+")
[ -n "$VPS" ] || { echo "no throughput reading"; exit 1; }
HOURS=$(python3 -c "print(2.7e6 * 100 / (8 * $VPS) / 3600)")
V=$(python3 -c "print(100 if $HOURS <= 3 else 50)"); echo "1-GPU visits/s $VPS -> 8 GPUs at 100 visits: $HOURS h -> visits $V"
J2=$(submit $EXC -J m25 --gres=gpu:8 -c 96 -t 08:00:00 scripts/nano4/run.sbatch env APPIMAGE_EXTRACT_AND_RUN=1 bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits $V --out results/ai_search)
echo "pilot $J2"; S2=$(waitj $J2); echo "pilot $S2"
J3=$(submit $EXC -J m25eval -c 8 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/ai_search_eval.py --root results/ai_search/fold1 --nosearch results/ai_quality/fold1 --out results/r_variants/ai_search_pilot.json)
echo "eval $J3"; S3=$(waitj $J3); echo "eval $S3"; grep -vE "Warning|warnings.warn|^\+" logs/nano4/m25eval-$J3.out | tail -25
