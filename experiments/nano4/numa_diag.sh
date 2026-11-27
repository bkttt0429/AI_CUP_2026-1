#!/usr/bin/env bash
# Per trainer process on this node: CPU of the main thread, user / kernel ticks of the main thread over 4 s, allowed CPUs.
#   bash logs/nano4/numa_diag.sh <pgrep -f pattern> [show | pin]      pin: a main thread with more kernel than user time has
#   all its threads moved to the other NUMA side (node0 = CPUs 0-55, node1 = 56-111; cut to the allowed list by the kernel)
pat=${1:?pattern}; mode=${2:-show}; pids=$(pgrep -u "$USER" -f "$pat"); declare -A u0 s0
echo "== $(date '+%F %T') $(hostname): $(echo $pids | wc -w) process(es) match '$pat'; $(numactl -H 2>/dev/null | grep -E 'free' | tr '\n' ' ')"
for p in $pids; do read -r u s < <(awk '{print $14, $15}' /proc/$p/task/$p/stat 2>/dev/null); u0[$p]=${u:-0}; s0[$p]=${s:-0}; done
sleep 4
for p in $pids; do
  read -r u s cpu < <(awk '{print $14, $15, $39}' /proc/$p/task/$p/stat 2>/dev/null) || continue
  du=$((u - u0[$p])); ds=$((s - s0[$p])); al=$(awk '/Cpus_allowed_list/{print $2}' /proc/$p/status); gpu=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
  echo "pid $p: main thread on cpu $cpu, user +$du / kernel +$ds ticks in 4 s; allowed $al; threads $(ls /proc/$p/task | wc -l); GPU $gpu; $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-60)"
  if [ "$mode" = pin ] && [ "$ds" -gt "$du" ] && [ "$ds" -gt 100 ]; then
    if [ "$cpu" -lt 56 ]; then tgt=56-111; else tgt=0-55; fi
    taskset -acp $tgt $p > /dev/null 2>&1 && echo "   -> all threads of $p pinned to $tgt" || echo "   -> pinning $p to $tgt FAILED"
  fi
done
