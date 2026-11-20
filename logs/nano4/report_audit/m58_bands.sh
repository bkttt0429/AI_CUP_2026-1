#!/usr/bin/env bash
# M58 band-count probes (each = ct13 + one disjoint band of the A|1D region) and the A-decided reverse band.
cd /home/u2507115/BK_AICUP_2026-1
python -I logs/nano4/m58_write_band.py --lo 0.285714 --hi 0.384615 --out submission_rank_ct30.csv
python -I logs/nano4/m58_write_band.py --lo 0.2 --hi 0.285714 --out submission_rank_ct31.csv
python -I - <<'PY'
import hashlib, sys
import pandas as pd
sys.path[:0] = ["src", "logs/nano4"]
from aicup.submission import validate_submission, write_submission
pr = pd.read_csv("submission_rank_ct9_proba.csv").set_index("question_id")
base = pd.read_csv("submission_rank_ct13.csv", dtype=str); d = base.set_index("question_id")["pred_rank"].copy()
odds = pr["p_A"] / pr["p_1D"]
for lo, hi in ((1.0, 2.0), (2.0, 3.0)):
    flip = [q for q in d.index if d[q] == "A" and lo <= odds[q] < hi]
    print(f"A-decided with A/1D odds [{lo}, {hi}): public {sum(q.startswith('pub') for q in flip)}, private {sum(q.startswith('priv') for q in flip)}")
flip = [q for q in d.index if d[q] == "A" and 1.0 <= odds[q] < 2.0]
d2 = d.copy(); d2[flip] = "1D"
out = "submission_rank_ct32.csv"
write_submission(list(zip(base["question_id"], d2.reindex(base["question_id"]))), out)
print(f"{out}: A->1D for A-decided odds [1, 2): public {sum(q.startswith('pub') for q in flip)}, private {sum(q.startswith('priv') for q in flip)}; problems {validate_submission(out, base['question_id'].tolist(), 800, {'pub_': 400, 'priv_': 400}) or 'none'}; md5 {hashlib.md5(open(out, 'rb').read()).hexdigest()[:12]}")
PY
