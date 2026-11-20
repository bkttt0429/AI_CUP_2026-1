# M61b (audit S13 reopened): player activity (total training games) vs ct9's A|1D first-band precision, HD1 share and residuals.
import sys
import numpy as np, pandas as pd
sys.path[:0] = ["src", "logs/nano4"]
from m43_gate import FOLDS, decide, qscore
from ens_compare import ES, load_fold
A, D1 = 3, 4
hd = pd.read_csv("results/data/hd_table.csv", dtype={"game_key": str, "player_id": str}, keep_default_na=False, na_values={"hd": [""]})
hd["hd"] = pd.to_numeric(hd["hd"], errors="coerce").fillna(0)
pl = hd.groupby("player_id").agg(games=("hd", "size"), hd2=("hd", lambda h: (h >= 2).mean()), hd1=("hd", lambda h: (h == 1).mean()))
rows = []
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    d = decide(P); E = (P * np.arange(10)).sum(1)
    rows.append(pd.DataFrame({"pid": pid, "y": y, "n": n, "d": d, "odds": P[:, A] / P[:, D1], "res": E - y, "s": qscore(d, y)}))
V = pd.concat(rows); V["games"] = V.pid.map(pl.games); V["hd1"] = V.pid.map(pl.hd1); V["zero2"] = V.pid.map(pl.hd2) == 0
V["gbin"] = pd.cut(V.games, [0, 25, 40, 80, 160, 100000])
print("by player's total training games: players, HD1 share, first-band precision A, residual by true class (kyu A / dan 1D / dan 2D+)")
for b, s in V.groupby("gbin", observed=True):
    band = s[(s.d == D1) & (s.odds >= 0.5)]
    print(f"  games {str(b):14s}: {s.pid.nunique():5d} players, HD1 {s.hd1.mean():.3f}, zero-HD2 {s.zero2.mean():.2f}; first band {len(band):4d} q precision A "
          f"{(band.y == A).mean() if len(band) else float('nan'):.3f}; residual A {s[s.y == A].res.mean():+.2f} 1D {s[s.y == D1].res.mean():+.2f} "
          f"2D+ {s[s.y > D1].res.mean():+.2f} kyu<A {s[s.y < A].res.mean():+.2f}; true-class mix " + " ".join(f"{c}:{(s.y == c).mean():.2f}" for c in range(10)))
