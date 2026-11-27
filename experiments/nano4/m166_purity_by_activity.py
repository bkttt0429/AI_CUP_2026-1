#!/usr/bin/env python
"""Section 4M M166 diagnostic (2026-11-15, report only): are anchors on busy candidates cleaner readings of the target's level?

For an identifier (m84_<tag>~m15_val.npz): validation opponent-side queries of even games, by the margin band (quantiles of the
impostor margins, as m71_step2.fit) and by the number of training games of the first candidate (< 100 / 100-299 / >= 300):
the share of queries, P(candidate label = target label), P(within one). The same for the impostor side (the null).

  python -I logs/nano4/m166_purity_by_activity.py SPAL~m15 SPALm~m15 SX4L~m15
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; GR = [(0, 100), (100, 300), (300, 10 ** 9)]
pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int)
for tag in sys.argv[1:]:
    z = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); ev = z["hd"] == 0; ng = pf.reindex(z["f_opp_best"].astype(str)).to_numpy()
    edges = {True: np.quantile(z["f_imp_m"][z["target_black"]], QS), False: np.quantile(z["f_imp_m"][~z["target_black"]], QS)}
    band = np.where(z["target_black"], np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"])); d = z["f_opp_lab"].astype(int) - z["y"].astype(int)
    print(f"\n=== {tag}: even games {int(ev.sum())}; first candidate's games: share of queries / same level / within one, per band group")
    for nm, lo, hi in (("bands < q80", 0, 1), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8)):
        m = ev & (band >= lo) & (band <= hi); row = []
        for a, b_ in GR:
            s = m & (ng >= a) & (ng < b_); row.append(f"{a}-{'' if b_ > 10 ** 8 else b_}: {s.sum() / max(m.sum(), 1):.2f} / {float((d[s] == 0).mean()) if s.any() else float('nan'):.3f} / {float((np.abs(d[s]) <= 1).mean()) if s.any() else float('nan'):.3f}")
        print(f"  {nm:12s} ({int(m.sum()):6d} queries, share {m.sum() / ev.sum():.3f}): " + "   ".join(row) + f"   | all: {float((d[m] == 0).mean()):.3f} / {float((np.abs(d[m]) <= 1).mean()):.3f}")
