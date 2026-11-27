#!/usr/bin/env python
"""Section 4M M173 diagnostic (2026-11-15, report only): in even games, does the target's colour say who is the stronger of the
two? Validation, identifier given: d = candidate label - target label by the target's colour, per band group; and the share
of games played as Black by the target's level (all even games).
  python -I logs/nano4/m173_colour.py SX4L~m15
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; tag = sys.argv[1]
z = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); tb = z["target_black"]; ev = z["hd"] == 0; y = z["y"].astype(int); d = z["f_opp_lab"].astype(int) - y
edges = {True: np.quantile(z["f_imp_m"][tb], QS), False: np.quantile(z["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"])); rng = np.random.default_rng(0)
print(f"{tag}: even games {int(ev.sum())}; the target plays Black in {float(tb[ev].mean()):.3f}; by the target's level (D..6D): " + "  ".join(f"{float(tb[ev & (y == c)].mean()):.3f}" for c in range(10)))
print("one-stone games: the target plays Black in " + f"{float(tb[z['hd'] == 1].mean()):.3f}; by level: " + "  ".join(f"{float(tb[(z['hd'] == 1) & (y == c)].mean()):.3f}" for c in range(10)))
for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8)):
    s = ev & (band >= lo) & (band <= hi); a, b_ = d[s & tb], d[s & ~tb]; bs = [rng.choice(a, len(a)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(1000)]
    print(f"  {nm:22s} target Black: n {len(a)}, mean d {a.mean():+.3f}, d<0 {float((a < 0).mean()):.3f}, d=0 {float((a == 0).mean()):.3f}, d>0 {float((a > 0).mean()):.3f} | target White: n {len(b_)}, mean d {b_.mean():+.3f}, d<0 {float((b_ < 0).mean()):.3f}, d=0 {float((b_ == 0).mean()):.3f}, d>0 {float((b_ > 0).mean()):.3f} | difference {a.mean() - b_.mean():+.3f} [{np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}]")
