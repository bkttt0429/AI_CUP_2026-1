#!/usr/bin/env bash
# Pause, resume or inspect the full-fold Human SL scoring run. Run on lab01:
#   scripts/scores_ctl.sh pause    each shard stops after the part in progress (about a minute)
#   scripts/scores_ctl.sh resume   removes the pause file and restarts any stopped shard
#   scripts/scores_ctl.sh status   running/stopped per shard, last log line, parts per fold
# Resume never rescores a finished game (see scripts/score_folds.py).
set -e
cd "$(dirname "$0")/.."
OUT=results/scores
FOLDS="1 2 3 4 0"
running() { pgrep -f "score_[f]olds.py .*--shard $1 " >/dev/null; }
case "$1" in
  pause)
    touch "$OUT/PAUSE"
    echo "pause requested: each shard stops after its current part (about a minute)";;
  resume)
    rm -f "$OUT/PAUSE"
    for s in 0 1; do
      if running "$s"; then echo "shard $s already running"; continue; fi
      echo "=== resumed $(date '+%m-%d %H:%M:%S') ===" >> "$OUT/log_shard$s.txt"
      CUDA_VISIBLE_DEVICES=$s setsid nohup .venv/bin/python -I scripts/score_folds.py --folds $FOLDS --shard "$s" --nshards 2 \
        --workers 10 --part-size 100 >> "$OUT/log_shard$s.txt" 2>&1 < /dev/null &
      echo "shard $s started"
    done;;
  status)
    for s in 0 1; do
      state=$(running "$s" && echo running || echo stopped)
      echo "shard $s: $state | $(grep -v Warning "$OUT/log_shard$s.txt" | tail -n 1)"
    done
    if [ -e "$OUT/PAUSE" ]; then echo "PAUSE file present"; fi
    for f in $FOLDS; do
      echo "fold $f: $(ls "$OUT/fold$f"/games_s*_p*.csv 2>/dev/null | grep -vc tmp) parts written, done markers $(ls "$OUT/fold$f"/done_s* 2>/dev/null | wc -l)/2"
    done;;
  *)
    echo "usage: $0 pause|resume|status"; exit 1;;
esac
