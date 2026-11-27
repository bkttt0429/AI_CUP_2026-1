"""Section 4M M71 Public readout (2026-11-10): ct40 scored 0.7761668 on Public (ct28 0.7827471, -0.0065803 = -2.632 points).
Which Public questions did ct40 change against ct28, and what does the ledger already know about them?
 - the 8 questions whose truth is determined (A, the 1D->A flips of ct13)
 - the 9 questions ct28 flipped 4D->5D (7 right, 2 wrong, identity unknown)
 - the ledger's MAP truth before ct40 for every changed question that some probe touched
Splits the changes into: made by the anchors on ct9's decision (ct39 vs ct9) / made or undone by the re-computed pushes.
  python -I logs/nano4/m71_public_readout.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

R = {lv: i for i, lv in enumerate(LEVELS)}
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank  # noqa: E731
c9, c13, c28, c39, c40 = (sub(k) for k in ("ct9", "ct13", "ct28", "ct39", "ct40"))
led = json.load(open(ROOT / "results/trunk_seq/es/public_ledger_decoded_before_ct40.json")); mapa, det = led["map"], set(led["determined"])
pub = c9.index[c9.index.str.startswith("pub_")]
sc = lambda d, t: 1.0 if d == t else (float(np.exp(-1)) if abs(R[d] - R[t]) == 1 else 0.0)  # noqa: E731
ch = [q for q in pub if c40[q] != c28[q]]
print(f"ct40 vs ct28 on Public: {len(ch)} questions; observed delta -2.632 points = 3 one-level losses (0.632) + 2 adjacent-to-far losses (0.368), net")
flips13 = [q for q in pub if c13[q] != c9[q]]; flips28 = [q for q in pub if c28[q] != c13[q]]
print(f"the 8 determined questions (truth A): ct28 answers {[c28[q] for q in sorted(det)]}")
print(f"                                      ct40 answers {[c40[q] for q in sorted(det)]}  -> known loss {sum(sc(c40[q], 'A') - sc(c28[q], 'A') for q in det):+.3f} points")
print(f"the 9 questions ct28 flipped 4D->5D (7 truly 5D): ct40 answers {[c40[q] for q in flips28]}  ({sum(c40[q] == '5D' for q in flips28)} kept at 5D)")
rows = []
for q in ch:
    src = "anchor" if c39[q] != c9[q] else "push"
    t = mapa.get(q); rows.append({"q": q, "ct9": c9[q], "ct28": c28[q], "ct39": c39[q], "ct40": c40[q], "by": src, "ledger_map": t, "determined": q in det,
                                  "delta_if_map": (sc(c40[q], t) - sc(c28[q], t)) if t else np.nan, "in13": q in flips13, "in28": q in flips28})
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
k = T[T.ledger_map.notna()]
print(f"changed questions the ledger had a MAP truth for: {len(k)} of {len(T)}; summed delta if the MAP were the truth: {k.delta_if_map.sum():+.3f} points (determined ones alone: {T[T.determined].delta_if_map.sum():+.3f})")
print("by source: " + "; ".join(f"{s}: {len(g)} questions, MAP-known {int(g.ledger_map.notna().sum())}, delta if MAP {g.delta_if_map.sum():+.3f}" for s, g in T.groupby("by")))
und = T[T.in13 | T.in28]
print(f"changes that touch a question ct13 or ct28 had pushed: {len(und)} -> " + str(und[["q", "ct9", "ct28", "ct40", "by", "ledger_map"]].values.tolist()))
