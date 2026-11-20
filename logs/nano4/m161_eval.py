#!/usr/bin/env python
"""Section 4M M161 (2026-11-15): one table for the pilot's variants (design doc M161).

For every all-players network given (results/m161/out_-1_<variant>.npz; references: results/m120/out_-1.npz = M120 seed 0 at
12,000 steps, out_-1_s7.npz = 40,000 steps):
  held-out own games: top-1 / top-5, by the player's number of training games, weighted by the rows (the registered metric);
  the TEST games, label-free: the margin (best minus second-best cosine) of the opponent side against the same statistic of the
  target side (a test player is never a training player: every target-side match is an impostor) - the share of opponent sides
  above the target side's q90 / q98 / q99.5 minus the nominal 10% / 2% / 0.5%, and the largest excess over all thresholds
  (a lower bound of the share of test opponents that are recognised). This is the quantity the anchors live on.

  python -I logs/nano4/m161_eval.py [--also results/m161/out_-1_x.npz ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--also", nargs="*", default=[]); a_ = ap.parse_args()
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); C = len(players); Q, H = int(q["Q"]), int(q["H"]); truth = q["held_truth"].astype(int); qtest, side = q["qtest"], q["side"]
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ng = pf.n_games.astype(int).reindex(players).to_numpy()
    share = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum(); band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]])
    files = [("M120 seed 0 (12k, reference)", ROOT / "results/m120/out_-1.npz"), ("M120 seed 7 (40k, reference)", ROOT / "results/m120/out_-1_s7.npz")] + [(f"variant {f.stem.split('_', 2)[2]}", f) for f in sorted((ROOT / "results/m161").glob("out_-1_*.npz"))] + [(f"variant {f.parent.name}", f) for f in sorted((ROOT / "results/m161").glob("*/out_-1_s*.npz"))] + [(f"production {f.stem[4:]}", f) for f in sorted((ROOT / "results/m161p").glob("out_-1_s*.npz"))] + [(Path(f).stem, Path(f)) for f in a_.also]
    rep = {}; print("network                          held-out top-1  top-5 | by games: " + "  ".join(f"{lo}+" for lo, _ in BANDS) + " | row-weighted || test opponents above the target-side q90 / q98 / q99.5 (excess) | largest excess")
    for name, f in files:
        if not f.exists():
            print(f"{name}: {f} missing"); continue
        z = np.load(f); mine = z["qidx"]; pos = {int(g): i for i, g in enumerate(mine)}; P = z["protos"].astype(np.float32); V = z["qvec"].astype(np.float32)
        S = V[[pos[2 * Q + i] for i in range(H)]] @ P.T; r = (S > S[np.arange(H), truth][:, None]).sum(1); t1 = np.array([float((r[band == k] == 0).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1); w1 = float((share[ok] * t1[ok]).sum() / share[ok].sum())
        te = np.flatnonzero(qtest[mine] >= 0); mg = np.zeros(len(te), np.float32); top = np.zeros(len(te), np.float32)
        for i in range(0, len(te), 4096):
            s = V[te[i:i + 4096]] @ P.T; p2 = np.partition(s, -2, axis=1)[:, -2:]; mg[i:i + 4096] = p2[:, 1] - p2[:, 0]; top[i:i + 4096] = p2[:, 1]
        sd = side[mine][te]; opp, tgt = mg[sd == 1], mg[sd == 0]; ex = {}
        for qq in (0.90, 0.98, 0.995):
            ex[qq] = float((opp > np.quantile(tgt, qq)).mean() - (1 - qq))
        grid = np.quantile(tgt, np.linspace(0.5, 0.999, 500)); sup = float(max((opp > t).mean() - (tgt > t).mean() for t in grid))
        grid2 = np.quantile(top[sd == 0], np.linspace(0.5, 0.999, 500)); sup2 = float(max((top[sd == 1] > t).mean() - (top[sd == 0] > t).mean() for t in grid2))
        rep[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "row_weighted": w1, "test_excess": {str(k): v for k, v in ex.items()}, "test_sup_excess_margin": sup, "test_sup_excess_top": sup2, "test_opp": int(len(opp)), "test_tgt": int(len(tgt))}
        print(f"{name:32s} {rep[name]['top1']:.3f}  {rep[name]['top5']:.3f} | " + "  ".join("  -  " if not np.isfinite(x) else f"{x:.3f}" for x in t1) + f" | {w1:.3f} || {ex[0.90]:+.3f} / {ex[0.98]:+.3f} / {ex[0.995]:+.3f} | margin {sup:.3f}, best score {sup2:.3f}")
    (RV / "m161_eval.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m161_eval.json")


if __name__ == "__main__":
    main()
