#!/usr/bin/env bash
# M29: the M25 search (same settings) on M23's 600 players with per-move saving, then the behaviour features.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
waitj() { until state "$1" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; state "$1"; }
submit() { for i in $(seq 1 300); do j=$(sbatch --parsable $EX "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
J=$(submit -J m29 --gres=gpu:8 -c 96 -t 04:00:00 scripts/nano4/run.sbatch bash -c 'ulimit -u $(ulimit -Hu); cd tools/katago_v1182/bin && rm -rf squashfs-root && ./katago --appimage-extract > /dev/null && cd ../../.. && bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_search_pilot.py --fold 1 --per-class 60 --visits 100 --config tools/katago_v1182/analysis_m25_t128.cfg --save-moves results/ai_search_moves --out results/ai_search_m29')
echo "search $J"; S=$(waitj $J); echo "search $S"; grep -h "saved" logs/nano4/shard*-$J.log | head -8
[ "$S" = COMPLETED ] || exit 1
E=$(submit -J m29eval -c 12 -t 01:00:00 scripts/nano4/run.sbatch python -I logs/nano4/psych_eval.py --moves results/ai_search_moves/fold1 --out results/r_variants/psych_pilot.json)
echo "eval $E"; S2=$(waitj $E); echo "eval $S2"; grep -vE "Warning|warnings.warn|^\+|^==" logs/nano4/m29eval-$E.out | tail -22
