#!/usr/bin/env bash
# M55 (B): 13 probe files, each = ct13 + one r=2 push (logs/nano4/m53_write_probe.py), in the pre-registered order.
cd /home/u2507115/BK_AICUP_2026-1
k=16
for push in "2D 1D" "3D 2D" "4D 3D" "5D 4D" "6D 5D" "A B" "B C" "C D" "D C" "C B" "1D 2D" "3D 4D" "4D 5D"; do
  echo "=== ct$k: push $push"
  python -I logs/nano4/m53_write_probe.py --base submission_rank_ct13.csv --push $push --r 2 --out submission_rank_ct$k.csv | grep -v "^flipped"
  k=$((k + 1))
done
python -I - <<'PY'
import pandas as pd
base = pd.read_csv("submission_rank_ct13.csv", dtype=str).set_index("question_id")["pred_rank"]
sets = {}
for k in range(16, 29):
    f = pd.read_csv(f"submission_rank_ct{k}.csv", dtype=str).set_index("question_id")["pred_rank"].reindex(base.index)
    sets[k] = set(base.index[f != base])
over = [(a, b, len(sets[a] & sets[b])) for a in sets for b in sets if a < b and sets[a] & sets[b]]
print("overlapping flip sets between probes:", over or "none")
PY
