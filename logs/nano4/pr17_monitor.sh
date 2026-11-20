#!/usr/bin/env bash
# PR #17 job watcher (login node, sacct/squeue only): one line per job that ends, per job never satisfiable,
# per x5/xfit job in the queue that pr17_jobs.txt does not list, and when the feeder process exits.
# usage: pr17_monitor.sh FEEDER_PID
cd ~/BK_AICUP_2026-1 || exit 1
L=logs/nano4/pr17_jobs.txt
FEEDER=$1
declare -A seen
for j in $(grep -oE '[0-9]{6}' "$L" | sort -u); do
  st=$(sacct -n -X -j "$j" -o State | awk 'NR==1{print $1}')
  case "$st" in COMPLETED|FAILED|CANCELLED*) seen[$j]=1 ;; esac
done
fd=0
while true; do
  for j in $(grep -oE '[0-9]{6}' "$L" | sort -u); do
    [ -n "${seen[$j]}" ] && continue
    s=$(sacct -n -X -j "$j" -o JobName%10,State%12,ExitCode,Elapsed,NodeList%14 2>/dev/null | head -1)
    st=$(echo "$s" | awk '{print $2}')
    case "$st" in
      COMPLETED*|FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_ME*|NODE_FAIL*|PREEMPTED*) echo "END $j $s"; seen[$j]=1 ;;
      PENDING) [ "$(squeue -h -j "$j" -o %r 2>/dev/null)" = "DependencyNeverSatisfied" ] && { echo "NEVER $j $s"; seen[$j]=1; } ;;
    esac
  done
  for x in $(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /^(x5|xfit)/{print $1}'); do
    grep -q "$x" "$L" || { [ -z "${seen[u$x]}" ] && { echo "UNTRACKED job $x"; seen[u$x]=1; }; }
  done
  if [ $fd = 0 ] && ! kill -0 "$FEEDER" 2>/dev/null; then echo "FEEDER exited: $(tail -1 logs/nano4/pr17_feeder5.log)"; fd=1; fi
  x=$(grep -oE 'xfit=[0-9]+' "$L" | cut -d= -f2)
  [ -n "$x" ] && [ -n "${seen[$x]}" ] && break
  sleep 60
done
echo ALL_DONE
