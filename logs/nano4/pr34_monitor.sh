#!/usr/bin/env bash
# PR #34 job watcher (login node, sacct/squeue only): one line per job that ends or can never start.
# Exits when every job in pr34_jobs.txt has ended or can never start. usage: pr34_monitor.sh
cd ~/BK_AICUP_2026-1 || exit 1
L=logs/nano4/pr34_jobs.txt
declare -A seen
# already ended before this watcher started: reported by an earlier one
for j in $(cut -d= -f2 "$L"); do
  st=$(sacct -n -X -j "$j" -o State%12 2>/dev/null | head -1 | awk '{print $1}')
  case "$st" in COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*) seen[$j]=1 ;; esac
done
while true; do
  left=0
  for j in $(cut -d= -f2 "$L"); do
    [ -n "${seen[$j]}" ] && continue
    s=$(sacct -n -X -j "$j" -o JobName%12,State%12,ExitCode,Elapsed,NodeList%14 2>/dev/null | head -1)
    st=$(echo "$s" | awk '{print $2}')
    case "$st" in
      COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*) echo "END $j $s"; seen[$j]=1 ;;
      *)
        if [ "$(squeue -h -j "$j" -o %r 2>/dev/null)" = "DependencyNeverSatisfied" ]; then
          echo "NEVER $j $s"; seen[$j]=1
        else
          left=$((left + 1))
        fi ;;
    esac
  done
  [ "$left" = 0 ] && break
  sleep 60
done
echo ALL_DONE
