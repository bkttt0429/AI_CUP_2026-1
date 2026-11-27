"""Identification overlay (AI_CUP_2026-1 issue #22; the user's decision whether to use it): replace a submission's pred_rank by the
identified training player's rank where the identification margin >= tau. Also the overlap diagnostic: agreement between the identified
rank and the base model's prediction, by margin band (players really in the pool -> agreement near the model's hit rate; not in the
pool -> near chance).
  python -I combine_id.py --base submission_rank_ct.csv --table rank_overlap_test.csv --tau 16.58 --out submission_rank_id.csv"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/u2507115/BK_AICUP_2026-1")
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--table", required=True)
ap.add_argument("--tau", type=float, default=16.58)
ap.add_argument("--out", default=None)
a = ap.parse_args()
base = pd.read_csv(a.base, dtype=str, keep_default_na=False).set_index("question_id")
t = pd.read_csv(a.table, dtype={"question_id": str}, keep_default_na=False)
print("table columns:", list(t.columns))
rank_col = next(c for c in t.columns if c.endswith("rank") and c != "pred_rank")
margin_col = "margin"
n_col = next((c for c in t.columns if c in ("n", "n_games", "num_games")), None)
t = t.set_index("question_id").loc[base.index]
ident = t[rank_col].astype(str)
m = t[margin_col].astype(float)
RANK = {lv: i for i, lv in enumerate(LEVELS)}
bi, ii = base["pred_rank"].map(RANK), ident.map(RANK)
print(f"questions {len(t)}; margin quantiles {np.round(m.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy(), 2).tolist()}; share >= tau {float((m >= a.tau).mean()):.3f}")
bands = [(0, 1), (1, 2.7), (2.7, 5.9), (5.9, 9.4), (9.4, 16.58), (16.58, 1e9)]
for lo, hi in bands:
    sel = (m >= lo) & (m < hi)
    if sel.sum():
        ex = float((bi[sel] == ii[sel]).mean()); adj = float((np.abs(bi[sel] - ii[sel]) <= 1).mean())
        print(f"  margin [{lo:>5}, {hi if hi < 1e9 else 'inf':>5}): {int(sel.sum()):4d} q  identified rank == model {ex:.3f}, within 1 {adj:.3f}")
chance = float(np.mean([(bi == np.roll(ii.to_numpy(), k)).mean() for k in range(1, 50)]))
print(f"  chance level (shuffled pairing) exact {chance:.3f}")
if n_col:
    print("  share >= tau by n:", {int(k): round(float(v), 3) for k, v in (m >= a.tau).groupby(t[n_col].astype(int)).mean().items()})
if a.out:
    out = base.copy()
    use = m >= a.tau
    out.loc[use, "pred_rank"] = ident[use]
    out.reset_index().to_csv(a.out, index=False)
    print(f"wrote {a.out}: {int(use.sum())} questions replaced; predicted counts {out['pred_rank'].value_counts().sort_index().to_dict()}")
