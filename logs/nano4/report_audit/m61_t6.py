# M61 (research T6): does the sub-population of training players WITHOUT any HD>=2 game in their history look like the test?
# Per validation player: share of HD>=2 and HD1 games over ALL their training games (hd_table). Then ct9's A|1D first-band precision and
# the HD1 share, for players with zero HD>=2 games vs the rest; and the same by fraction thresholds.
import sys
import numpy as np, pandas as pd
sys.path[:0] = ["src", "logs/nano4"]
from m43_gate import FOLDS, decide
from ens_compare import ES, load_fold
A, D1 = 3, 4
hd = pd.read_csv("results/data/hd_table.csv", dtype={"game_key": str, "player_id": str}, keep_default_na=False, na_values={"hd": [""]})
hd["hd"] = pd.to_numeric(hd["hd"], errors="coerce").fillna(0)
pl = hd.groupby("player_id").agg(games=("hd", "size"), hd2=("hd", lambda h: (h >= 2).mean()), hd1=("hd", lambda h: (h == 1).mean()), rank=("rank", "first"))
print(f"training players {len(pl)}; with zero HD>=2 games {int((pl.hd2 == 0).sum())} ({(pl.hd2 == 0).mean():.3f}); their HD1 share (game-weighted) "
      f"{np.average(pl.hd1[pl.hd2 == 0], weights=pl.games[pl.hd2 == 0]):.3f} vs others {np.average(pl.hd1[pl.hd2 > 0], weights=pl.games[pl.hd2 > 0]):.3f}")
print("share of players with zero HD>=2 by rank: " + " ".join(f"{r} {(g.hd2 == 0).mean():.2f}" for r, g in pl.groupby("rank")))
print("median games per player, zero-HD2 vs others: " + f"{pl.games[pl.hd2 == 0].median():.0f} vs {pl.games[pl.hd2 > 0].median():.0f}")
rows = []
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    d = decide(P)
    rows.append(pd.DataFrame({"pid": pid, "y": y, "n": n, "d": d, "odds": P[:, A] / P[:, D1]}))
V = pd.concat(rows); V["zero2"] = V.pid.map(pl.hd2).fillna(1) == 0; V["hd1"] = V.pid.map(pl.hd1)
for name, s in (("zero HD>=2 players", V[V.zero2]), ("other players", V[~V.zero2])):
    band = s[(s.d == D1) & (s.odds >= 0.5)]
    print(f"{name}: {s.pid.nunique()} players, {len(s)} questions; first band {len(band)} q, precision A {(band.y == A).mean():.3f}; "
          f"true-A share among 1D-decided {((s.d == D1) & (s.y == A)).sum() / max((s.d == D1).sum(), 1):.3f}")
for lo in (0.0, 0.2, 0.3):
    s = V[V.zero2 & (V.hd1 > lo)]; band = s[(s.d == D1) & (s.odds >= 0.5)]
    print(f"zero-HD2 players with HD1 share > {lo}: {s.pid.nunique()} players; first band {len(band)} q, precision A {(band.y == A).mean() if len(band) else float('nan'):.3f}")
