#!/usr/bin/env python
"""Section 4M M169 (2026-11-15, report only): the kernel by outcome with a CLEAN outcome - the pilot games that have a strong
engine's final win rate (results/ai_search_moves/fold1, 600 players of fold 1), identifier given.

  python -I logs/nano4/m169_clean.py SX4L~m15
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; tag = sys.argv[1]
z = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); keys = z["key"].astype(str); pk = {k: i for i, k in enumerate(keys)}; lm = np.load(RV / "m169_last_val.npz")["last_is_target"]; rws, seen = [], set()
for npz in sorted(glob.glob(str(ROOT / "results/ai_search_moves/fold1/shard*.npz"))):
    games = pd.read_csv(npz.replace(".npz", "_games.csv"), dtype={"player_id": str, "game_key": str}); zz = np.load(npz); df = pd.DataFrame({k: zz[k] for k in ("game", "t", "side", "win", "lead")}).sort_values(["game", "t"])
    for g, s in df.groupby("game"):
        key = games["game_key"].iloc[g]
        if key in seen or key not in pk:
            continue
        seen.add(key); tc = 0 if games["color"].iloc[g] == "B" else 1; mine = s["side"].to_numpy() == tc; w = s["win"].to_numpy(float); ld = s["lead"].to_numpy(float); rws.append((pk[key], float(w[-1] if mine[-1] else 1 - w[-1]), float(ld[-1] if mine[-1] else -ld[-1])))
ii = np.array([r[0] for r in rws]); wt = np.array([r[1] for r in rws]); lead = np.array([r[2] for r in rws]); win = wt > 0.5
tb = z["target_black"]; edges = {True: np.quantile(z["f_imp_m"][tb], QS), False: np.quantile(z["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"]))[ii]
d = (z["f_opp_lab"].astype(int) - z["y"].astype(int))[ii]; ev = (z["hd"] == 0)[ii]; rng = np.random.default_rng(0)
print(f"{tag}: pilot games {len(ii)} (even {int(ev.sum())}); the target is ahead at the end in {float(win.mean()):.3f}; agreement with the last-move proxy {float((lm[ii] == win).mean()):.3f}")
for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q50-q95", 1, 3), (">= q95", 4, 8), (">= q99", 6, 8)):
    s = ev & (band >= lo) & (band <= hi); a, b_ = d[s & win], d[s & ~win]; diff = a.mean() - b_.mean(); bs = [rng.choice(a, len(a)).mean() - rng.choice(b_, len(b_)).mean() for _ in range(2000)]
    print(f"  {nm:22s} won: n {len(a)}, mean d {a.mean():+.3f}, d<0 {float((a < 0).mean()):.3f}, d=0 {float((a == 0).mean()):.3f}, d>0 {float((a > 0).mean()):.3f} | lost: n {len(b_)}, mean d {b_.mean():+.3f}, d<0 {float((b_ < 0).mean()):.3f}, d=0 {float((b_ == 0).mean()):.3f}, d>0 {float((b_ > 0).mean()):.3f} | difference {diff:+.3f} [{np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}]")
s = ev & (band >= 4); print("  >= q95: share of games the target is ahead at the end, by d: " + "  ".join(f"d={x:+d}: {float(win[s & (d == x)].mean()):.3f} (n {int((s & (d == x)).sum())})" for x in (-2, -1, 0, 1, 2)))
