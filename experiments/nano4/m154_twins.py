#!/usr/bin/env python
"""Section 4M M154 (2026-11-15, report only): is one person split into several player ids, one per level? (design doc M154)

Every level file holds exactly 100,000 games, exactly half of them with the listed player as Black, and no player id appears in
two files. If the ids were made per file, a person who played at 2D and later at 3D is two "players". The identity prototypes
would show it: such twins are far more similar than strangers and - unlike strangers of the same strength - can never carry the
SAME level. Test: for every player the nearest other prototype (all-players networks, seeds 0, 1, 3 averaged); the distribution
of the level difference by similarity. Strangers: the same level is the most frequent difference at every similarity. Twins:
at the highest similarities the same level is depleted and +-1 dominates.

  python -I logs/nano4/m154_twins.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); C = len(players)
pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ng = pf.n_games.astype(int).reindex(players).to_numpy(); fold = pf.fold.astype(int).reindex(players).to_numpy()
LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; rep = {}


def table(S, title, mask_rows=None, mask_cols=None):
    S = S.copy(); np.fill_diagonal(S, -2)
    if mask_cols is not None:
        S[:, ~mask_cols] = -2
    j = S.argmax(1); s = S[np.arange(C), j]; d = plab[j] - plab; rows = np.ones(C, bool) if mask_rows is None else mask_rows
    S2 = S.copy(); S2[np.arange(C), j] = -2; s2 = S2.max(1)
    print(f"\n=== {title}: {int(rows.sum())} players; similarity of the nearest other prototype: quartiles " + " / ".join(f"{x:.3f}" for x in np.quantile(s[rows], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])) + f"; gap to the second nearest: median {np.median((s - s2)[rows]):.3f}, q90 {np.quantile((s - s2)[rows], 0.9):.3f}, q99 {np.quantile((s - s2)[rows], 0.99):.3f}")
    qs = np.quantile(s[rows], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]); out = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = rows & (s >= lo) & (s <= hi)
        r = {"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "same": float((d[m] == 0).mean()), "pm1": float((np.abs(d[m]) == 1).mean()), "far": float((np.abs(d[m]) >= 2).mean()), "up": float((d[m] == 1).mean()), "down": float((d[m] == -1).mean())}
        out.append(r); print(f"   similarity {lo:.3f}-{hi:.3f} ({r['n']:5d} players): nearest has the same level {r['same']:.3f}, one level apart {r['pm1']:.3f} (higher {r['up']:.3f} / lower {r['down']:.3f}), two or more {r['far']:.3f}")
    gap = s - s2; qg = np.quantile(gap[rows], [0, 0.5, 0.9, 0.97, 0.99, 1.0])
    for lo, hi in zip(qg[:-1], qg[1:]):
        m = rows & (gap >= lo) & (gap <= hi); print(f"   gap to the second nearest {lo:.3f}-{hi:.3f} ({int(m.sum()):5d} players): same level {float((d[m] == 0).mean()):.3f}, one apart {float((np.abs(d[m]) == 1).mean()):.3f}, two or more {float((np.abs(d[m]) >= 2).mean()):.3f}; median games of the player {np.median(ng[m]):.0f}, of the neighbour {np.median(ng[j[m]]):.0f}")
    mut = rows & (j[j] == np.arange(C)); print(f"   mutual nearest pairs: {int(mut.sum()) // 2 if mask_rows is None else int(mut.sum())} ; same level {float((d[mut] == 0).mean()):.3f}, one apart {float((np.abs(d[mut]) == 1).mean()):.3f}, two or more {float((np.abs(d[mut]) >= 2).mean()):.3f}")
    print("   by the player's level, nearest prototype's level difference (-1 | 0 | +1): " + "  ".join(f"{LV[c]} {float((d[rows & (plab == c)] == -1).mean()):.2f}|{float((d[rows & (plab == c)] == 0).mean()):.2f}|{float((d[rows & (plab == c)] == 1).mean()):.2f}" for c in range(10)))
    return out, j, s


P = [np.load(ROOT / f"results/m120/out_-1{t}.npz")["protos"].astype(np.float64) for t in ("", "_s1", "_s3")]
S = sum(p @ p.T for p in P) / len(P); big = ng >= 30
rep["all"], j, s = table(S, "all-players networks, every player against every other")
rep["big"], _, _ = table(S, "players with >= 30 games against players with >= 30 games", mask_rows=big, mask_cols=big)
base = np.bincount(plab, minlength=10) / C; exp_same = float((base ** 2).sum()); print(f"\nreference: two random players have the same level with probability {exp_same:.3f}, one level apart {float(2 * (base[:-1] * base[1:]).sum()):.3f}")
# cross-fitted: a fold's players are unseen by that fold's network; nearest SEEN prototype
for v in (1, 2, 3, 4):
    Pv = [np.load(ROOT / f"results/m120/out_{v}{t}.npz")["protos"].astype(np.float64) for t in ("", "_s1", "_s3")]; Sv = sum(p @ p.T for p in Pv) / len(Pv)
    rep[f"net{v}"], _, _ = table(Sv, f"network {v}: its unseen players (fold {v}) against the players it saw", mask_rows=fold == v, mask_cols=fold != v)
(ROOT / "results/r_variants/m154_twins.json").write_text(json.dumps(rep, indent=1)); print("\nwritten results/r_variants/m154_twins.json")
