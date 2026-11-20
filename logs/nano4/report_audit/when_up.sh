#!/usr/bin/env bash
# Wait until the scheduler accepts a submission, then run the queue.
cd /home/u2507115/BK_AICUP_2026-1
while true; do
  j=$(sbatch --parsable -J probe -t 00:01:00 scripts/nano4/run.sbatch true 2>/dev/null)
  if [[ "$j" =~ ^[0-9]+$ ]]; then scancel "$j" 2>/dev/null; echo "scheduler accepting at $(date '+%F %T') (probe $j cancelled)"; break; fi
  sleep 120
done
bash /home/u2507115/.cache/report_audit/queue_1105.sh
