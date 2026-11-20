#!/usr/bin/env python
"""Section 4M M167 diagnostic (2026-11-15, report only): is the margin-band calibration of an identifier uneven across game lengths?

For an identifier (m84_<tag>~m15_val.npz), validation games by the number of plies (< 150 / 150-199 / 200-255 / >= 256):
  the impostor side: share of the own-side (null) margins above the pooled band edges q80 / q95 / q99 (nominal 0.20 / 0.05 / 0.01)
  - if these differ by length, band edges pooled over lengths put short and long games on different scales;
  the opponent side (even games): share of queries per band group, P(candidate label = target label), P(within one).

  python -I logs/nano4/m167_purity_by_length.py SPALm~m15 SX4L~m15
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; LB = [(0, 150), (150, 200), (200, 256), (256, 10 ** 6)]
tags = sys.argv[1:]; z0 = dict(np.load(RV / f"m84_{tags[0]}_val.npz", allow_pickle=True)); want = set(z0["key"].astype(str)); n_ply = {}; pat = re.compile(r";[BW]\["); csv.field_size_limit(10 ** 9)
for f in sorted((ROOT / "data/training").glob("train_*.csv")):
    with open(f, newline="") as fh:
        r = csv.reader(fh); next(r)
        for pid, gid, rank, color, sgf in r:
            k = f"{rank}:{gid}"
            if k in want:
                n_ply[k] = len(pat.findall(sgf))
L = np.array([n_ply.get(k, -1) for k in z0["key"].astype(str)]); print(f"validation games {len(L)}, plies found for {int((L >= 0).sum())}; quartiles " + " / ".join(f"{x:.0f}" for x in np.quantile(L[L >= 0], [0.1, 0.25, 0.5, 0.75, 0.9])))
np.save(RV / "m167_val_plies.npy", L); lb = np.array([next((i for i, (lo, hi) in enumerate(LB) if lo <= n < hi), -1) for n in L])
for tag in tags:
    z = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); assert np.array_equal(z["key"], z0["key"]); ev = z["hd"] == 0; tb = z["target_black"]
    edges = {True: np.quantile(z["f_imp_m"][tb], QS), False: np.quantile(z["f_imp_m"][~tb], QS)}
    iband = np.where(tb, np.searchsorted(edges[True], z["f_imp_m"]), np.searchsorted(edges[False], z["f_imp_m"]))  # the impostor is the target's own side: the target's colour
    band = np.where(tb, np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"])); d = z["f_opp_lab"].astype(int) - z["y"].astype(int)
    print(f"\n=== {tag}\n  null (own side) above the pooled q80 / q95 / q99 edges, by plies " + " | ".join(f"{lo}-{'' if hi > 10 ** 5 else hi - 1}: " + " / ".join(f"{float((iband[lb == i] >= b_).mean()):.3f}" for b_ in (2, 4, 6)) for i, (lo, hi) in enumerate(LB)) + "   (nominal 0.200 / 0.050 / 0.010)")
    print("  opponent side, even games: share of the length group's queries in the band group / same level / within one")
    for nm, lo_, hi_ in (("bands < q80", 0, 1), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8)):
        row = []
        for i, (lo, hi) in enumerate(LB):
            g = ev & (lb == i); s = g & (band >= lo_) & (band <= hi_); row.append(f"{lo}-{'' if hi > 10 ** 5 else hi - 1}: {s.sum() / max(g.sum(), 1):.3f} / {float((d[s] == 0).mean()) if s.any() else float('nan'):.3f} / {float((np.abs(d[s]) <= 1).mean()) if s.any() else float('nan'):.3f}")
        print(f"  {nm:12s}: " + "   ".join(row))
    print("  games per length group (even): " + " / ".join(str(int((ev & (lb == i)).sum())) for i in range(len(LB))))
