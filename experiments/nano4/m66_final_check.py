"""Section 4M M66 (2026-11-10): final-file check before the last upload. (1) Format: header, 800 unique ids = the test ids, ranks valid.
(2) Mechanical transfer: ct28 vs ct13 and ct13 vs ct9, per half and per move, so the Private side is exactly the pushes read on Public.
(3) Probability that the 4D->5D push gains on Private (11 flips) under the priors used on 11-09 (beta-binomial, integrated over p).
  python -I logs/nano4/m66_final_check.py
"""
import sys
from math import comb, lgamma, exp
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

ids = pd.concat([pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{h}.csv", usecols=["question_id"], dtype=str) for h in ("public", "private")]).question_id
f = {k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str) for k in ("ct28", "ct13", "ct9")}
d = f["ct28"]
print("header", list(d.columns), "| rows", len(d), "| unique ids", d.question_id.nunique(), "| ids == test ids", set(d.question_id) == set(ids),
      "| invalid ranks", int((~d.pred_rank.isin(LEVELS)).sum()), "| missing", int(d.pred_rank.isna().sum()))
def diff(a, b):
    A = f[a].set_index("question_id").pred_rank; B = f[b].set_index("question_id").pred_rank.reindex(A.index)
    ch = A[A != B]
    for half in ("pub_", "priv_"):
        c = ch[ch.index.str.startswith(half)]
        moves = pd.Series([f"{B[i]}->{c[i]}" for i in c.index]).value_counts().to_dict() if len(c) else {}
        print(f"  {a} vs {b} {half[:-1]:4s}: {len(c):2d} changed {moves}")
diff("ct28", "ct13"); diff("ct13", "ct9")
def bb_tail(k, n, a, b):  # P(X >= k), X ~ BetaBinomial(n, a, b)
    lb = lambda x, y: lgamma(x) + lgamma(y) - lgamma(x + y)
    return sum(comb(n, i) * exp(lb(i + a, n - i + b) - lb(a, b)) for i in range(k, n + 1))
n = 11
print("P(Private 4D->5D flips gain, i.e. >= 6 of 11 right) and mean precision:")
for lab, (a, b) in {"pure noise p=0.38 (fixed)": (None, None), "12-push pooled prior + 7/9": (4.2 + 7, 6.8 + 2), "prior incl. the 8/8 band + 7/9": (2.1 + 7, 2.9 + 2),
                    "flat prior + 7/9": (1 + 7, 1 + 2), "flat + both directions 13/16": (1 + 13, 1 + 3)}.items():
    if (a, b) == (None, None) or lab.startswith("pure"):
        p = 0.38; t = sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(6, n + 1)); print(f"  {lab:34s} mean p 0.380  P(gain) {t:.2f}  EV {(2*p-1)*0.63212*n/400:+.4f}"); continue
    m = a / (a + b); print(f"  {lab:34s} mean p {m:.3f}  P(gain) {bb_tail(6, n, a, b):.2f}  EV {(2*m-1)*0.63212*n/400:+.4f}")
