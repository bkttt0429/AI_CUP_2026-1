#!/usr/bin/env python
"""Section 4M M170 step 1 (2026-11-15, report only): each labelled candidate's own offset between its label and the labels of
the opponents in its own games; does correcting the anchor reading by it tighten the kernel? (design doc M170)

Data: results/m150/pilot.npz (all 1,000,000 training rows: owner, first labelled candidate of the opponent side, m15 margin),
results/data/hd_table.csv (handicap per game), an identifier's validation file.
Writes results/r_variants/m170_offsets.npz: u_all [C], n_all [C], u_fold [4, C] (computed without the opponents of fold v),
n_fold [4, C], kappa.

  python -I logs/nano4/m170_offset.py SX4L~m15
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; FOLDS = (1, 2, 3, 4)


def main():
    tag = sys.argv[1]; z = np.load(ROOT / "results/m150/pilot.npz", allow_pickle=True); own, c1, m15, key = z["owner"], z["c1"], z["m15"], z["key"].astype(str)
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); C = len(players)
    pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).set_index("game_key").hd.reindex(key).to_numpy(); ev = hd == 0; dl = (plab[c1] - plab[own]).astype(np.float64)
    print(f"training rows {len(own)}; even {float(ev.mean()):.3f}; handicap missing {int(np.isnan(hd).sum())}")
    rng = np.random.default_rng(170); half = rng.integers(0, 2, len(own)); best = None
    for qq in (0.80, 0.90, 0.95):
        s = ev & (m15 >= np.quantile(m15, qq)); n = np.bincount(own[s], minlength=C); sm = np.bincount(own[s], weights=dl[s], minlength=C); has = n >= 4
        a = [np.bincount(own[s & (half == h)], weights=dl[s & (half == h)], minlength=C) / np.maximum(np.bincount(own[s & (half == h)], minlength=C), 1) for h in (0, 1)]; nh = [np.bincount(own[s & (half == h)], minlength=C) for h in (0, 1)]
        ok = (nh[0] >= 4) & (nh[1] >= 4); r = float(np.corrcoef(a[0][ok], a[1][ok])[0, 1]); w = n[has] / n[has].sum(); mu = sm[has] / n[has]; gm = float((w * mu).sum())
        var_row = float(np.sum((dl[s] - (sm / np.maximum(n, 1))[own[s]]) ** 2) / max(s.sum() - int((n > 0).sum()), 1)); var_between = max(float(np.var(mu) - np.mean(var_row / n[has])), 1e-9); kappa = var_row / var_between
        print(f"rows above the m15 q{int(qq * 100)}: {int(s.sum())} (even); owners with >= 4 such rows {int(has.sum())} (median rows {np.median(n[has]):.0f}); mean of (opponent label - own label) {gm:+.3f}; sd of the owners' means {np.std(mu):.3f}; "
              f"split-half correlation {r:.3f} ({int(ok.sum())} owners); within-owner sd of a row {np.sqrt(var_row):.3f}; true between-owner sd {np.sqrt(var_between):.3f}; kappa {kappa:.1f}")
        if qq == 0.90:
            best = (s, kappa, gm)
    s, kappa, gm = best; opp_fold = pfold[c1]; U = {}
    for v in (0, *FOLDS):  # 0 = all opponents (the test side); v = without the opponents of fold v
        m = s & (opp_fold != v) if v else s; n = np.bincount(own[m], minlength=C); sm = np.bincount(own[m], weights=dl[m] - gm, minlength=C); U[v] = (n / (n + kappa) * sm / np.maximum(n, 1), n)
    np.savez(RV / "m170_offsets.npz", u_all=U[0][0], n_all=U[0][1], u_fold=np.stack([U[v][0] for v in FOLDS]), n_fold=np.stack([U[v][1] for v in FOLDS]), kappa=kappa, centre=gm, players=players)
    print(f"offsets written (kappa {kappa:.1f}, centred at {gm:+.3f}); shrunk offsets: sd {np.std(U[0][0]):.3f}, quantiles 5 / 25 / 75 / 95: " + " / ".join(f"{x:+.2f}" for x in np.quantile(U[0][0], [0.05, 0.25, 0.75, 0.95])) + f"; by the candidate's level (mean offset): " + "  ".join(f"{c}: {U[0][0][plab == c].mean():+.2f}" for c in range(10)))
    # ---------------------------------------------------------------- validation: does the offset predict d?
    val = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); pidx = {p: i for i, p in enumerate(players)}; ci = np.array([pidx[p] for p in val["f_opp_best"].astype(str)]); fo = val["fold"].astype(int); tb = val["target_black"]
    u = np.zeros(len(ci)); nn = np.zeros(len(ci))
    for j, v in enumerate(FOLDS):
        m = fo == v; u[m] = U[v][0][ci[m]]; nn[m] = U[v][1][ci[m]]
    edges = {True: np.quantile(val["f_imp_m"][tb], QS), False: np.quantile(val["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], val["f_opp_m"]), np.searchsorted(edges[True], val["f_opp_m"])); d = (val["f_opp_lab"].astype(int) - val["y"].astype(int)).astype(np.float64); e0 = val["hd"] == 0
    print(f"\n{tag}, validation even games: d = candidate label - target label; u = the candidate's shrunk offset computed without the game's fold")
    for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8), (">= q95", 4, 8)):
        m = e0 & (band >= lo) & (band <= hi); x, y_ = u[m], d[m]; slope = float(np.cov(x, y_)[0, 1] / max(np.var(x), 1e-12)); lam = -slope; adj = y_ + x; adj2 = y_ + lam * x
        print(f"  {nm:22s} n {int(m.sum())}; offset sd {x.std():.3f}; slope of d on u {slope:+.3f} (corr {np.corrcoef(x, y_)[0, 1]:+.3f}); sd(d) {y_.std():.3f} -> sd(d + u) {adj.std():.3f} ({adj.std() / y_.std() - 1:+.1%}) -> best scale {lam:.2f}: {adj2.std():.3f} ({adj2.std() / y_.std() - 1:+.1%}); "
              f"same level {float((y_ == 0).mean()):.3f} -> {float((np.rint(adj) == 0).mean()):.3f}; within one {float((np.abs(y_) <= 1).mean()):.3f} -> {float((np.abs(np.rint(adj)) <= 1).mean()):.3f}; rounded label changed {float((np.rint(x) != 0).mean()):.3f}")
    m = e0 & (band >= 4); print("  >= q95 by the size of the offset (|u| < 0.15 / 0.15-0.35 / >= 0.35): " + "  ".join(f"n {int((m & g_).sum())}, mean d {d[m & g_].mean():+.3f} (mean u {u[m & g_].mean():+.3f})" for g_ in (np.abs(u) < 0.15, (u <= -0.15) & (u > -0.35), u <= -0.35, (u >= 0.15) & (u < 0.35), u >= 0.35)))


if __name__ == "__main__":
    main()
