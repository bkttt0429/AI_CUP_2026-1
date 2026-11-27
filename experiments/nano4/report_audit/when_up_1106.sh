#!/usr/bin/env bash
# Wait until the scheduler accepts a submission, then submit the identification debug job (once).
cd /home/u2507115/BK_AICUP_2026-1
F=/home/u2507115/.cache/report_audit/identdbg.job
sleep 90
while true; do
  if [ -s "$F" ]; then echo "already submitted: $(cat $F)"; exit 0; fi
  j=$(sbatch --parsable --exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155] -J identdbg -c 32 --mem=120G -t 01:00:00 scripts/nano4/run.sbatch python -I logs/nano4/ident_pool.py --out results/r_variants/ident_pool_dbg.json --limit-players 1200 --workers 30 2>/dev/null)
  if [[ "$j" =~ ^[0-9]+$ ]]; then echo "$j" > "$F"; echo "identdbg $j submitted at $(date '+%F %T')"; exit 0; fi
  echo "$(date '+%T') scheduler still down"; sleep 60
done
