#!/usr/bin/env bash
# Operational handoff: observe the existing jobs and quarantine measurement-only CSVs.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1 || exit 1
printf '%s\n' "$$" > logs/nano4/o26_multinode_monitor_20261116.pid
r2=/home/u2507115/BK_AICUP_2026-2
end_at=$((SECONDS + 57600))
while (( SECONDS < end_at )); do
  date -Is
  for mount in /work/u2507115 /home/u2507115; do
    free_mb=$(df -BM --output=avail "$mount" 2>/dev/null | awk 'NR==2 {gsub(/M/, ""); print $1}')
    limit=3000
    [[ "$mount" == /home/* ]] && limit=1200
    if [[ "$free_mb" =~ ^[0-9]+$ ]]; then
      echo "SPACE $mount ${free_mb} MiB"
      (( free_mb < limit )) && echo "ALERT LOW SPACE $mount ${free_mb} MiB < ${limit} MiB"
    else
      echo "ALERT quota read failed: $mount"
    fi
  done
  states=$(sacct -X -n -P -j 785721,785655,785723,785724,785732 -o JobID,State,ExitCode,Elapsed)
  printf '%s\n' "$states"
  chain_state=$(printf '%s\n' "$states" | awk -F'|' '$1==785724 {print $2}')
  cmp_state=$(printf '%s\n' "$states" | awk -F'|' '$1==785732 {print $2}')
  terminal='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)'
  if [[ "$chain_state" =~ $terminal && "$cmp_state" =~ $terminal ]]; then
    shopt -s nullglob
    candidates=("$r2"/upload/pid_v2o26m_*.csv)
    if (( ${#candidates[@]} )); then
      mkdir -p "$r2/results/final/not_for_upload" || exit 1
      for f in "${candidates[@]}"; do
        target="$r2/results/final/not_for_upload/${f##*/}"
        if [[ -e "$target" ]]; then
          echo "ALERT quarantine destination exists; retained source: $f"
        else
          mv -- "$f" "$target" || exit 1
          echo "QUARANTINED $target"
        fi
      done
    fi
    for f in v2_offsim_v2o26m_x_vs_v2bgm_d2_s0_1_2_3.json v2_offsim_v2o26m_only_vs_v2o26m_ref1_s0_1_2_3.json v2_oldsim_v2o26m_only.json; do
      if [[ -s "$r2/results/final/$f" ]]; then
        echo "RESULT PRESENT $r2/results/final/$f"
      else
        echo "ALERT RESULT MISSING $r2/results/final/$f"
      fi
    done
    rg -n 'Traceback|FAILED|No space|a .*failed|not built|no checkpoint' logs/nano4/p_chain_v2o26m-785724.out logs/nano4/p_o26cmp-785732.out logs/nano4/newnets-785724-*.log || true
    md5sum "$r2/upload/pid_v2bgm_d2_a2k_p3.csv"
    [[ "$chain_state" == COMPLETED && "$cmp_state" == COMPLETED ]] || echo 'ALERT pipeline did not complete successfully'
    echo 'WATCH ENDED: terminal jobs; result contents and worker logs still require review.'
    exit 0
  fi
  sleep 60
done
echo 'ALERT WATCH WINDOW EXPIRED: inspect job states and renew if needed.'
