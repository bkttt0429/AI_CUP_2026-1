#!/usr/bin/env python
"""Section 4M M183b (2026-11-16, report only): is what is left a property of the PLAYER, and does anything observable predict it?

From m183_dump.npz (ct112 rule, seeds 0-2). Every validation player has one 20-game question, its two disjoint 10-game halves
and its four disjoint 5-game quarters. e = expected level of the pooled posterior minus the label (levels).
(a) replication: correlation / covariance of e between the two halves (and between quarters) - the covariance is the
    variance of the part that does not depend on which games were drawn; the same for the base alone. Decisions: how often
    the two halves agree, and how often the label is the level they agree on (four quarters: unanimous / 3 of 4).
(b) who carries the persistent part d = mean of e over the two halves: by true level (mean, sd), by the player's number of
    games, by the player's share of handicap games; the direction of the misses when both halves agree and the label is
    elsewhere, by agreed level.
(c) can it be predicted from what a test question also has: out-of-fold regression (players split in five) of the label
    minus the pooled expected level (20-game question) on - the gaps between the channels (base, anchors, look-alike vs
    pooled), the entropy, the number of anchor games, the pooled expected level itself; R^2 out of fold, and the score if the
    posterior were shifted by the prediction.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); E1 = float(np.exp(-1.0)); LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def main():
    z = np.load(ROOT / "results/r_variants/m183_dump.npz", allow_pickle=True); name = "SQ4"; y, n, pid = z["y"].astype(int), z["n"].astype(int), z["pid"].astype(str); post = z["anat_post"]; logp = z["anat_logp"]; R = z["anat_R"]; Lall = z["anat_Lall"]; AN = z["anat_AN"].astype(int); dec = z["anat_plain"].astype(int); base = z["anat_base"].astype(int)
    lev = np.arange(10.0); sm = lambda a: (e := np.exp(a - a.max(1, keepdims=True))) / e.sum(1, keepdims=True)  # noqa: E731
    ep = post @ lev; eb = np.exp(logp) @ lev; ea = sm(Lall) @ lev; er = sm(R) @ lev; ent = -(post * np.log(np.maximum(post, 1e-12))).sum(1)
    df = pd.DataFrame({"pid": pid, "n": n, "y": y, "e": ep - y, "eb": eb - y, "dec": dec, "base": base, "row": np.arange(len(y))})
    # ---------------------------------------------------------------- (a)
    print("(a) replication on disjoint games of the same player")
    for size, k in ((10, 2), (5, 4)):
        g = df[df.n == size].groupby("pid"); ok = g.size() == k; ids = ok.index[ok]; sub = df[(df.n == size) & df.pid.isin(ids)].sort_values(["pid", "row"]); E = sub.e.to_numpy().reshape(-1, k); EB = sub.eb.to_numpy().reshape(-1, k); D = sub.dec.to_numpy().reshape(-1, k); B = sub.base.to_numpy().reshape(-1, k); Y = sub.y.to_numpy().reshape(-1, k)[:, 0]
        inner = (Y >= 1) & (Y <= 8); cov = lambda M: float(np.mean([np.cov(M[inner, i], M[inner, j])[0, 1] for i in range(k) for j in range(i + 1, k)]))  # noqa: E731
        cor = lambda M: float(np.mean([np.corrcoef(M[inner, i], M[inner, j])[0, 1] for i in range(k) for j in range(i + 1, k)]))  # noqa: E731
        print(f"  {k} disjoint sets of {size} games, {len(Y)} players (true levels C-5D for the moments): pooled - total variance {float(E[inner].var()):.3f}, covariance between sets {cov(E):.3f} (sd of the player part {np.sqrt(max(cov(E), 0)):.3f} levels), correlation {cor(E):.3f}; base alone - variance {float(EB[inner].var()):.3f}, covariance {cov(EB):.3f} (sd {np.sqrt(max(cov(EB), 0)):.3f}), correlation {cor(EB):.3f}")
        if k == 2:
            ag = D[:, 0] == D[:, 1]; print(f"    final decisions of the two halves agree {float(ag.mean()):.3f}; when they agree the label is that level {float((D[ag, 0] == Y[ag]).mean()):.3f} (one level off {float((np.abs(D[ag, 0] - Y[ag]) == 1).mean()):.3f}); when they differ the label is one of the two {float(((D[~ag, 0] == Y[~ag]) | (D[~ag, 1] == Y[~ag])).mean()):.3f}; a single half is right {float((D == Y[:, None]).mean()):.3f}")
            halves = pd.DataFrame({"pid": sub.pid.to_numpy().reshape(-1, k)[:, 0], "d": E.mean(1), "db": EB.mean(1), "y": Y, "agree": ag, "lvl": D[:, 0]})
        else:
            mode = np.array([np.bincount(r, minlength=10).argmax() for r in D]); cnt = (D == mode[:, None]).sum(1)
            print("    four quarters: " + "; ".join(f"{c} of 4 give the same level in {float((cnt == c).mean()):.3f} of the players, the label is that level {float((mode[cnt == c] == Y[cnt == c]).mean()):.3f}" for c in (4, 3, 2)))
    # ---------------------------------------------------------------- (b)
    print("\n(b) who carries the player part d (mean of the two halves' pooled expected level minus the label)")
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); hd = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["player_id", "hd"], dtype={"player_id": str}); hs = (hd.assign(h=hd.hd.astype(float) > 0).groupby("player_id").h.mean())
    halves["ng"] = pf.n_games.astype(int).reindex(halves.pid).to_numpy(); halves["hs"] = hs.reindex(halves.pid).to_numpy()
    print("  by true level (mean / sd of d): " + "  ".join(f"{LV[c]} {halves.d[halves.y == c].mean():+.2f}/{halves.d[halves.y == c].std():.2f}" for c in range(10)))
    mid = halves[(halves.y >= 1) & (halves.y <= 8)]
    for lab, col, edges in (("games under the player's name", "ng", [20, 30, 60, 100, 200, 400, 10 ** 6]), ("share of handicap games", "hs", [0, 0.02, 0.1, 0.25, 0.5, 1.01])):
        print(f"  by {lab} (true levels C-5D; players | mean d | sd of d | base: mean / sd): " + "  ".join(f"[{edges[i]}, {edges[i + 1]}): {int(m.sum())} | {mid.d[m].mean():+.3f} | {mid.d[m].std():.3f} | {mid.db[m].mean():+.3f} / {mid.db[m].std():.3f}" for i in range(len(edges) - 1) if (m := (mid[col] >= edges[i]) & (mid[col] < edges[i + 1])).sum() >= 50))
    print(f"  correlation of d with log games {float(np.corrcoef(mid.d, np.log(mid.ng))[0, 1]):+.3f}, of |d| with log games {float(np.corrcoef(mid.d.abs(), np.log(mid.ng))[0, 1]):+.3f}; of d with the handicap share {float(np.corrcoef(mid.d, mid.hs.fillna(0))[0, 1]):+.3f}")
    miss = halves[halves.agree & (halves.lvl != halves.y)]; print(f"  both halves agree and the label is elsewhere: {len(miss)} players ({len(miss) / max(int(halves.agree.sum()), 1):.3f} of the agreeing ones); label above / below the agreed level, by agreed level: " + "  ".join(f"{LV[c]} {int(((miss.lvl == c) & (miss.y > c)).sum())}/{int(((miss.lvl == c) & (miss.y < c)).sum())}" for c in range(10)))
    # ---------------------------------------------------------------- (c)
    print("\n(c) does anything a question carries predict the miss (20-game questions; out of fold over players)")
    m20 = n == 20; X = np.column_stack([eb - ep, ea - ep, er - ep, ent, AN[:, 1], AN[:, 2], ep, ep ** 2, (eb - ep) ** 2, np.abs(ea - ep)])[m20]; t = (y - ep)[m20]; pp = pid[m20]; yy = y[m20]; P20 = post[m20]
    up = np.unique(pp); rng = np.random.default_rng(0); fold = dict(zip(up, rng.integers(0, 5, len(up)))); fo = np.array([fold[p] for p in pp]); pred = np.zeros(len(t)); Xs = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / X.std(0)])
    for k in range(5):
        tr = fo != k; beta = np.linalg.lstsq(Xs[tr].T @ Xs[tr] + 1e-3 * np.eye(Xs.shape[1]), Xs[tr].T @ t[tr], rcond=None)[0]; pred[~tr] = Xs[~tr] @ beta
    r2 = 1 - float(((t - pred) ** 2).mean() / ((t - t.mean()) ** 2).mean()); print(f"  linear, 10 features: out-of-fold R^2 {r2:+.4f} (variance of the miss {float(t.var()):.3f}; sd of the prediction {float(pred.std()):.3f})")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        pg = np.zeros(len(t))
        for k in range(5):
            tr = fo != k; pg[~tr] = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05).fit(X[tr], t[tr]).predict(X[~tr])
        print(f"  boosted trees: out-of-fold R^2 {1 - float(((t - pg) ** 2).mean() / ((t - t.mean()) ** 2).mean()):+.4f}"); pred2 = pg
    except Exception as e:  # noqa: BLE001
        print(f"  (boosted trees not run: {e})"); pred2 = pred
    SMx = np.eye(10) + E1 * (np.eye(10, k=1) + np.eye(10, k=-1)); d0 = (P20 @ SMx).argmax(1); s0 = np.where(d0 == yy, 1.0, np.where(np.abs(d0 - yy) == 1, E1, 0.0))
    for lab, pr in (("linear", pred), ("trees", pred2)):
        best = None
        for lam in (0.5, 1.0, 2.0):
            Ps = P20 * np.exp(lam * pr[:, None] * (lev[None] - (P20 @ lev)[:, None])); Ps /= Ps.sum(1, keepdims=True); d1 = (Ps @ SMx).argmax(1); s1 = np.where(d1 == yy, 1.0, np.where(np.abs(d1 - yy) == 1, E1, 0.0)); best = max(best or (-9, 0, 0), (float((s1 - s0).mean()), lam, float((d1 != d0).mean())))
        print(f"  posterior tilted by the {lab} prediction: best unweighted score change {best[0]:+.4f} (strength {best[1]}, answers changed {best[2]:.3f}) - an optimistic upper reading (strength picked on the same data)")


if __name__ == "__main__":
    main()
