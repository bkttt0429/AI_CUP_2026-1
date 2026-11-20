#!/usr/bin/env bash
cd /home/u2507115/BK_AICUP_2026-1
F=/home/u2507115/.cache/report_audit/m24.job
while true; do
  if grep -qE '^[0-9]+$' "$F" 2>/dev/null; then echo "already submitted: $(cat $F)"; exit 0; fi
  j=$(sbatch --parsable --exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155] -J m24 -c 32 --mem=160G -t 02:00:00 scripts/nano4/run.sbatch bash /home/u2507115/.cache/report_audit/m24_nano4.sh 2>/dev/null)
  if [[ "$j" =~ ^[0-9]+$ ]]; then echo "$j" > "$F"; echo "m24 $j submitted at $(date '+%F %T')"; exit 0; fi
  echo "$(date '+%T') scheduler still down"; sleep 120
done
