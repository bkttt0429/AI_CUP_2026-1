#!/usr/bin/env bash
# M41: MINIROCKET pilot -- smoke, then the full run (each submitted whenever the QOS lets it in).
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
EX='--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]'
state() { sacct -n -X -j "$1" -o State | head -1 | tr -d ' '; }
wait_jobs() { for id in ${1//,/ }; do until state "$id" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL"; do sleep 60; done; done; }
submit() { local j=""; for i in $(seq 1 900); do j=$(sbatch --parsable $EX $2 -J "$1" scripts/nano4/run.sbatch "${@:3}" 2>/dev/null) && [[ "$j" =~ ^[0-9]+$ ]] && { echo "$j"; return 0; }; sleep 60; done; return 1; }
echo "== $(date '+%F %T') M41 smoke"
S=$(submit m41smk "-c 12 -t 01:00:00" python -I logs/nano4/minirocket_pilot.py --smoke --out /home/u2507115/.cache/report_audit/m41_smoke.json); echo "smoke job $S"
wait_jobs "$S"; echo "smoke $(state $S)"; grep -vE 'Warning|warnings.warn|^\+' logs/nano4/m41smk-$S.out | tail -8 | cut -c1-220
[ "$(state $S)" = "COMPLETED" ] || { echo "smoke failed: stop"; exit 1; }
echo "== $(date '+%F %T') M41 full run"
F=$(submit m41 "-c 12 -t 03:00:00" python -I logs/nano4/minirocket_pilot.py --out results/r_variants/m41_minirocket.json); echo "job $F"
wait_jobs "$F"; echo "full $(state $F)"; grep -vE 'Warning|warnings.warn|^\+' logs/nano4/m41-$F.out | tail -10 | cut -c1-240
echo "== $(date '+%F %T') M41 done"
