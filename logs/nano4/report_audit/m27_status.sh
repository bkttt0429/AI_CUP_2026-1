#!/usr/bin/env bash
# One progress snapshot of M27 (login node: file counts and logs only).
cd /home/u2507115/BK_AICUP_2026-1
echo "== $(date '+%H:%M') M27 status"
grep -E "^== |scoring jobs|build jobs|pack jobs|INCOMPLETE|failed|FAILED" ~/.cache/report_audit/m27.log | tail -6
for p in "scores_hd_jp:scores_hd" "scores_opp_hd_jp:scores_opp_hd" "scores_hd_x40_jp:scores_hd_x40" "scores_opp_hd_x40_jp:scores_opp_hd_x40"; do
  j=${p%:*}; c=${p#*:}
  nj=$(cat results/$j/fold*/games_s*_p*.csv 2>/dev/null | grep -vc "^game_key"); nc=$(cat results/$c/fold*/games_s*_p*.csv 2>/dev/null | grep -vc "^game_key")
  echo "$j: $nj / $nc games ($(( 100 * nj / (nc > 0 ? nc : 1) ))%)"
done
squeue -u $USER -h -o "%i %j %T %M" | sort | tr '\n' ';'; echo
