#!/usr/bin/env python
"""Section 4M M171 (2026-11-15): candidate-specific readings - the level distribution of each labelled candidate's own
opponents (design doc M171).

From results/m150/pilot.npz (all training rows: owner, first labelled candidate of the opponent side, m15) and the handicap
table: N[c, t] = number of the owner c's even games (margin in the top 10%) whose recognised opponent has label t; computed
with all opponents (the test side) and, per validation fold v, without the opponents of fold v.
pbar[lab, t] = the same pooled over the owners of one label; multiplier M[c, t] = (N[c, t] / pbar[lab_c, t] + kappa) /
(n_c + kappa): 1 for a candidate without data, the candidate's own lift of level t otherwise.
Writes results/r_variants/m171_cand.npz (M_all [C, 10], M_fold [4, C, 10], n_all, n_fold, pbar_all).
Report: on the validation side of an identifier, the mean log-likelihood of the true level under the pooled reading and under
the candidate's reading, by margin band.

  python -I logs/nano4/m171_cand.py SX4L~m15
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; FOLDS = (1, 2, 3, 4); KAPPA = 4.0


def main():
    tag = sys.argv[1]; z = np.load(ROOT / "results/m150/pilot.npz", allow_pickle=True); own, c1, m15, key = z["owner"], z["c1"], z["m15"], z["key"].astype(str)
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); plab = q["plab"].astype(int); C = len(players)
    pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).set_index("game_key").hd.reindex(key).to_numpy(); s = (hd == 0) & (m15 >= np.quantile(m15, 0.90)); lo_ = plab[c1]; of = pfold[c1]; out = {}
    for v in (0, *FOLDS):
        m = s & (of != v) if v else s; N = np.zeros((C, 10)); np.add.at(N, (own[m], lo_[m]), 1.0); n = N.sum(1); pb = np.zeros((10, 10))
        for lab in range(10):
            pb[lab] = N[plab == lab].sum(0) + 0.5
        pb /= pb.sum(1, keepdims=True); M = np.clip((N / pb[plab] + KAPPA) / (n[:, None] + KAPPA), 0.05, 20.0); out[v] = (M, n, pb)
    np.savez(RV / "m171_cand.npz", M_all=out[0][0], n_all=out[0][1], pbar_all=out[0][2], M_fold=np.stack([out[v][0] for v in FOLDS]), n_fold=np.stack([out[v][1] for v in FOLDS]), pbar_fold=np.stack([out[v][2] for v in FOLDS]), kappa=KAPPA, players=players)
    n = out[0][1]; print(f"rows used {int(s.sum())}; candidates with >= 1 / 10 / 30 recognised even opponents: {int((n >= 1).sum())} / {int((n >= 10).sum())} / {int((n >= 30).sum())} of {C}; multiplier quantiles 5 / 50 / 95: " + " / ".join(f"{x:.2f}" for x in np.quantile(out[0][0], [0.05, 0.5, 0.95])))
    print("pooled reading pbar[label of the candidate -> label of the opponent], rows = candidate level D..6D (same level / within one): " + "  ".join(f"{out[0][2][l, l]:.2f}/{out[0][2][l, max(l - 1, 0):l + 2].sum():.2f}" for l in range(10)))
    # ---------------------------------------------------------------- validation: log-likelihood of the true level
    val = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); pidx = {p: i for i, p in enumerate(players)}; ci = np.array([pidx[p] for p in val["f_opp_best"].astype(str)]); fo = val["fold"].astype(int); tb = val["target_black"]; y = val["y"].astype(int)
    edges = {True: np.quantile(val["f_imp_m"][tb], QS), False: np.quantile(val["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], val["f_opp_m"]), np.searchsorted(edges[True], val["f_opp_m"])); e0 = val["hd"] == 0
    lp0 = np.zeros(len(y)); lp1 = np.zeros(len(y)); nn = np.zeros(len(y))
    for v in FOLDS:
        m = fo == v; M, n_, pb = out[v]; base = pb[plab[ci[m]]]; cand = base * M[ci[m]]; cand /= cand.sum(1, keepdims=True); r = np.arange(int(m.sum())); lp0[m] = np.log(base[r, y[m]]); lp1[m] = np.log(cand[r, y[m]]); nn[m] = n_[ci[m]]
    print(f"\n{tag}, validation even games: mean log-likelihood of the target's true level given the candidate - pooled reading -> the candidate's own reading (gain), by band and by the candidate's number of recognised opponents")
    for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8), (">= q95", 4, 8)):
        m = e0 & (band >= lo) & (band <= hi); row = []
        for a, b_ in ((0, 10), (10, 30), (30, 10 ** 9)):
            g = m & (nn >= a) & (nn < b_); row.append(f"n_c {a}-{'' if b_ > 10 ** 8 else b_ - 1}: {int(g.sum())} games, {lp0[g].mean():.3f} -> {lp1[g].mean():.3f} ({lp1[g].mean() - lp0[g].mean():+.3f})")
        print(f"  {nm:22s} all {int(m.sum())}: {lp0[m].mean():.3f} -> {lp1[m].mean():.3f} ({lp1[m].mean() - lp0[m].mean():+.3f}) | " + " | ".join(row))


if __name__ == "__main__":
    main()
