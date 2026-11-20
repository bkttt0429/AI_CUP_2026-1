#!/usr/bin/env bash
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
submit() { local j=""; for i in $(seq 1 900); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
J1=$(submit m43a "-c 12 -t 01:00:00" python -I logs/nano4/m43_gate.py --seeds 0 1 2 --out results/trunk_seq/es/m43_s012.json)
J2=$(submit m43b "-c 12 -t 01:00:00" python -I logs/nano4/m43_gate.py --seeds 3 4 5 --out results/trunk_seq/es/m43_s345.json)
echo "== $(date '+%F %T') M43 jobs $J1 $J2"
for J in $J1 $J2; do until state $J | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 30; done; echo "== $J $(state $J)"; grep -vE 'Warning|warnings.warn|^\+|^== ' logs/nano4/m43*-$J.out | tail -10 | cut -c1-230; done
echo "== $(date '+%F %T') M43 done"
