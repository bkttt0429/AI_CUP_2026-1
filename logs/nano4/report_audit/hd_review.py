"""Numbers for the review of the 2026-11-03 discussion report: what handicap and colour alone say about the label (per game,
every player weighted equally), and how the label relates to the opponent's estimated level plus the signed handicap gap
(question level, folds 1-4 validation questions). Read-only; prints JSON-ish lines."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402

t = pd.read_csv(ROOT / "results/data/hd_table.csv")
t["y"] = t["rank"].map(RANK_TO_IDX)
t["w"] = 1.0 / t.groupby("player_id")["game_key"].transform("count")  # every player one unit of weight
t["bucket"] = pd.cut(t["hd"], [-0.5, 0.5, 1.5, 3.5, 99], labels=["0", "1", "2-3", "4+"])
print(f"games {len(t)}, players {t.player_id.nunique()}, HD>0 share (game-weighted) {(t.hd > 0).mean():.3f}, player-weighted {np.average(t.hd > 0, weights=t.w):.3f}")
rows = []
for (b, c), g in t.groupby(["bucket", "color"], observed=True):
    dist = np.array([g.loc[g.y == k, "w"].sum() for k in range(10)])
    dist = dist / dist.sum()
    order = np.argsort(-dist)
    cum = np.cumsum(dist[order])
    k99, k95 = int(np.searchsorted(cum, 0.99) + 1), int(np.searchsorted(cum, 0.95) + 1)
    ent = float(-(dist[dist > 0] * np.log(dist[dist > 0])).sum())
    rows.append({"hd": b, "color": c, "games": len(g), "player_weight": float(g.w.sum()), "entropy_nats": ent, "levels_for_95pct": k95, "levels_for_99pct": k99,
                 "dist": [round(float(x), 3) for x in dist], "mode": LEVELS[int(order[0])]})
    print(f"HD {b:>3s} {c}: games {len(g):7d}  H {ent:.2f} nats (uniform 2.30)  95% needs {k95} levels, 99% needs {k99}  mode {LEVELS[int(order[0])]}  dist {' '.join(f'{x:.2f}' for x in dist)}")
# per level: share of games where the target gives / receives stones
for k, name in enumerate(LEVELS):
    g = t[t.y == k]
    give = np.average((g.hd >= 1) & (g.color == "W"), weights=g.w)
    recv = np.average((g.hd >= 1) & (g.color == "B"), weights=g.w)
    print(f"level {name:2s}: gives handicap {give:.3f}  receives {recv:.3f}  (player-weighted shares of games)")

# question level: label vs opponent level estimate and signed handicap gap
X, y, n = [], [], []
for v in (1, 2, 3, 4):
    z = np.load(ROOT / f"results/trunk_seq/r_val{v}.npz")
    X.append(z["X_va"].astype(float))
    y.append(z["y_va"].astype(int))
    q = json.loads((ROOT / f"results/trunk_seq/r_val{v}.json").read_text())["val"]
    n.append(np.asarray(q["n_games"]))
X, y, n = np.vstack(X), np.concatenate(y), np.concatenate(n)
opp_minus_own, opp, gap = X[:, 56], X[:, 57], X[:, 62] - X[:, 63]  # diff features; hd_gap_up - hd_gap_down (mean over the games)
own = opp - opp_minus_own
print(f"questions {len(y)}; corr(y, own pmi) {np.corrcoef(y, own)[0, 1]:.3f}, corr(y, opp pmi) {np.corrcoef(y, opp)[0, 1]:.3f}, corr(y, signed gap) {np.corrcoef(y, gap)[0, 1]:.3f}")
A = np.column_stack([np.ones(len(y)), opp, gap])
beta, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ beta
print(f"OLS y ~ opp_pmi + signed_gap: intercept {beta[0]:.3f}, per pmi unit {beta[1]:.3f}, per stone {beta[2]:.3f} levels; R2 {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
A2 = np.column_stack([np.ones(len(y)), own])
b2, *_ = np.linalg.lstsq(A2, y, rcond=None)
print(f"OLS y ~ own_pmi: R2 {1 - ((y - A2 @ b2) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
A3 = np.column_stack([np.ones(len(y)), own, opp, gap])
b3, *_ = np.linalg.lstsq(A3, y, rcond=None)
print(f"OLS y ~ own + opp + gap: own {b3[1]:.3f}, opp {b3[2]:.3f}, per stone {b3[3]:.3f}; R2 {1 - ((y - A3 @ b3) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
# does the handicap match the estimated strength gap? mean(own - opp pmi) by the question's mean signed gap
for lo, hi, lab in ((-9, -1.5, "receives >=2"), (-1.5, -0.01, "receives (mean gap in (-1.5, 0))"), (-0.01, 0.01, "even"), (0.01, 1.5, "gives (0, 1.5)"), (1.5, 9, "gives >=2")):
    m = (gap > lo) & (gap <= hi)
    if m.sum():
        print(f"signed gap {lab:32s}: questions {m.sum():6d}  mean(own - opp pmi) {(-opp_minus_own[m]).mean():+.3f}  mean label {y[m].mean():.2f}")
