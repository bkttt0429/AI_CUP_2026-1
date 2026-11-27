#!/usr/bin/env bash
# M26 (rules / komi pilot): 8-GPU scoring of the M18/M23 sample under cn75 / cnk / jp, then the evaluation.
set -euo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EXC='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
submit() { for i in $(seq 1 300); do j=$(sbatch --parsable "$@" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
waitj() { until state "$1" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; state "$1"; }
J1=$(submit $EXC -J m26 --gres=gpu:8 -c 96 -t 04:00:00 scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I logs/nano4/rules_pilot_score.py --fold 1 --per-class 60 --workers 10 --out results/rules_pilot)
echo "score $J1"; S1=$(waitj $J1); echo "score $S1"; tail -3 logs/nano4/shard0-$J1.log || true
J2=$(submit $EXC -J m26eval -c 8 -t 00:30:00 scripts/nano4/run.sbatch python -I logs/nano4/rules_pilot_eval.py --root results/rules_pilot/fold1 --fold 1 --out results/r_variants/rules_pilot.json)
echo "eval $J2"; S2=$(waitj $J2); echo "eval $S2"; grep -vE "Warning|warnings.warn|^\+" logs/nano4/m26eval-$J2.out | tail -15
