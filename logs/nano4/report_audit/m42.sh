#!/usr/bin/env bash
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
J=""; for i in $(seq 1 900); do J=$(sbatch --parsable $EX -c 12 -t 01:00:00 -J m42 scripts/nano4/run.sbatch python -I logs/nano4/winrate_pilot.py --out results/r_variants/m42_winrate.json 2>/dev/null) && [[ "$J" =~ ^[0-9]+$ ]] && break; sleep 60; done
echo "== $(date '+%F %T') M42 job $J"
until state $J | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done
echo "== $(state $J)"; grep -vE 'Warning|warnings.warn|^\+|^== ' logs/nano4/m42-$J.out | tail -16 | cut -c1-230
