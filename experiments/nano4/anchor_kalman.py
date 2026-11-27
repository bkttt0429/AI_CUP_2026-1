"""Gemini's O(1/n) claim (2026-11-05): the target's own strength estimate carries an O(1) persistent offset, while the n opponents' offsets
are independent draws, so the matchmaking anchor (opponent rank estimate corrected by the HD rule) should beat own evidence at large n.
Test on folds 1-4 players with 80 scored games (L1b per-game expected ranks on the 0-9 label scale):
  per n in {5, 10, 20, 40, 80}: sd and mean of (own mean - y) and (anchor mean - y), their correlation across players, the best fused weight
  w (grid) vs Gemini's Kalman w(n), exact accuracy (round, class-balanced) of own-only / anchor-only / fused; the 40/40 split-half
  correlation of the anchor error (persistence => the opponents' offsets are not independent of the target's).
  python -I logs/nano4/anchor_kalman.py --out results/r_variants/anchor_kalman.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table, signed_gap  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
LC = [f"sum_{lv}" for lv in LEVELS]
NS = (5, 10, 20, 40, 80)


def per_game(f, hd):
    own = pd.concat([read_scored_games(ROOT / "results/l1b" / (ts.L1B[f] + sfx) / f"fold{f}") for sfx in ("", "_x40", "_x80")], ignore_index=True)
    opp = pd.concat([read_scored_games(ROOT / "results/l1b" / (ts.OPP[f] + sfx) / f"fold{f}") for sfx in ("", "_x40", "_x80")], ignore_index=True).set_index("game_key")
    g = own[["game_key", "player_id", "rank", "color"]].copy()
    g["e_own"] = posterior_mean_index(own[LC].to_numpy(float))
    g["e_opp"] = posterior_mean_index(opp.loc[g.game_key, LC].to_numpy(float))
    g["hd"] = g.game_key.map(hd).fillna(0).to_numpy()
    g["anchor"] = g.e_opp + signed_gap(g.hd.to_numpy(), g.color.to_numpy())  # y_opp + (own - opp rank gap) = the target's displayed rank
    g["y"] = g["rank"].map(RANK)
    return g


def balanced_exact(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(len(LEVELS)) if (y == c).any()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    G = pd.concat([per_game(f, hd) for f in (1, 2, 3, 4)], ignore_index=True)
    cnt = G.groupby("player_id").size(); players = cnt[cnt >= 80].index
    G = G[G.player_id.isin(players)]
    print(f"{len(players)} players with >= 80 games, {len(G)} games; per-game sd(e_own - y) {(G.e_own - G.y).std():.3f}, sd(anchor - y) {(G.anchor - G.y).std():.3f}, "
          f"mean(anchor - y) {(G.anchor - G.y).mean():+.3f}, HD>=1 share {(G.hd >= 1).mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    rows = {n: [] for n in NS}; half = []
    for pid, s in G.groupby("player_id"):
        s = s.iloc[rng.permutation(len(s))[:80]]
        y = int(s.y.iat[0]); eo, an = s.e_own.to_numpy(), s.anchor.to_numpy()
        for n in NS:
            rows[n].append((y, eo[:n].mean(), an[:n].mean()))
        half.append((y, an[:40].mean(), an[40:80].mean(), eo[:40].mean(), eo[40:80].mean()))
    H = np.array(half); rep = {"players": int(len(players)), "by_n": {}}
    rep["split_half_40_40"] = {"anchor_error_corr": float(np.corrcoef(H[:, 1] - H[:, 0], H[:, 2] - H[:, 0])[0, 1]), "own_error_corr": float(np.corrcoef(H[:, 3] - H[:, 0], H[:, 4] - H[:, 0])[0, 1])}
    print(f"40/40 split-half error correlation: own {rep['split_half_40_40']['own_error_corr']:.3f}, anchor {rep['split_half_40_40']['anchor_error_corr']:.3f}")
    # within-class (class means of each error removed: the shrinkage of the edge classes cannot drive these)
    yv = H[:, 0].astype(int)
    def dm(e):
        out = e.copy()
        for c in np.unique(yv):
            out[yv == c] -= e[yv == c].mean()
        return out
    ea1, ea2, eo1, eo2 = dm(H[:, 1] - H[:, 0]), dm(H[:, 2] - H[:, 0]), dm(H[:, 3] - H[:, 0]), dm(H[:, 4] - H[:, 0])
    rep["split_half_40_40_within_class"] = {"anchor": float(np.corrcoef(ea1, ea2)[0, 1]), "own": float(np.corrcoef(eo1, eo2)[0, 1]), "own_vs_anchor_same_half": float(np.corrcoef(eo1, ea1)[0, 1]),
                                            "own_half1_vs_anchor_half2": float(np.corrcoef(eo1, ea2)[0, 1]),
                                            "by_class_anchor": {LEVELS[c]: float(np.corrcoef(ea1[yv == c], ea2[yv == c])[0, 1]) for c in np.unique(yv) if (yv == c).sum() > 30},
                                            "by_class_own": {LEVELS[c]: float(np.corrcoef(eo1[yv == c], eo2[yv == c])[0, 1]) for c in np.unique(yv) if (yv == c).sum() > 30}}
    w = rep["split_half_40_40_within_class"]
    print(f"within-class split-half: own {w['own']:.3f}, anchor {w['anchor']:.3f}; own vs anchor same half {w['own_vs_anchor_same_half']:.3f}, own half 1 vs anchor half 2 {w['own_half1_vs_anchor_half2']:.3f}")
    print("  by class anchor: " + " ".join(f"{k}:{v:.2f}" for k, v in w["by_class_anchor"].items())); print("  by class own:    " + " ".join(f"{k}:{v:.2f}" for k, v in w["by_class_own"].items()))
    sp2, sgo2, sga2 = 0.45, 1.2, 1.2  # Gemini's constants
    for n in NS:
        A = np.array(rows[n]); y, eo, an = A[:, 0].astype(int), A[:, 1], A[:, 2]
        e1, e2 = eo - y, an - y
        grid = np.linspace(0, 1, 101); rmse = [np.sqrt(np.mean(((1 - w) * eo + w * an - y) ** 2)) for w in grid]; w_best = float(grid[int(np.argmin(rmse))])
        w_gem = (n * sp2 + sgo2) / ((n + 1) * sp2 + sgo2 + sga2)
        fused = (1 - w_best) * eo + w_best * an
        r = {"sd_own_err": float(e1.std()), "mean_own_err": float(e1.mean()), "sd_anchor_err": float(e2.std()), "mean_anchor_err": float(e2.mean()), "corr_own_anchor_err": float(np.corrcoef(e1, e2)[0, 1]),
             "best_w_opp": w_best, "gemini_w_opp": float(w_gem), "rmse_own": float(np.sqrt(np.mean(e1 ** 2))), "rmse_anchor": float(np.sqrt(np.mean(e2 ** 2))), "rmse_fused": float(min(rmse)),
             "exact_own": balanced_exact(np.clip(np.rint(eo), 0, 9).astype(int), y), "exact_anchor": balanced_exact(np.clip(np.rint(an), 0, 9).astype(int), y), "exact_fused": balanced_exact(np.clip(np.rint(fused), 0, 9).astype(int), y)}
        rep["by_n"][str(n)] = r
        print(f"n={n:2d}: own err sd {r['sd_own_err']:.3f} (mean {r['mean_own_err']:+.3f}) | anchor err sd {r['sd_anchor_err']:.3f} (mean {r['mean_anchor_err']:+.3f}) | corr {r['corr_own_anchor_err']:+.3f} | "
              f"best w_opp {w_best:.2f} (Gemini {w_gem:.2f}) rmse own/anchor/fused {r['rmse_own']:.3f}/{r['rmse_anchor']:.3f}/{r['rmse_fused']:.3f} | exact own/anchor/fused {r['exact_own']:.3f}/{r['exact_anchor']:.3f}/{r['exact_fused']:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
