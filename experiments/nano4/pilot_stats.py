"""Signal report shared by the step-0 pilots of architecture v2 section 4L (geo_pilot.py, traj_pilot.py).

Input: one row per game with player_id, rank, fold, own_pmi, opp_pmi and candidate statistics. For every candidate column:
  spearman_game      Spearman with the rank index over games;
  spearman_half      the same over player halves (each player's games split at random into two halves of floor(n/2) games, the
                     column averaged per half: a 10-game question when the player has 20 games);
  adjacent_auc       mean over the 9 adjacent-rank pairs of max(AUC, 1 - AUC) between the half means of the two ranks;
  split_half_r       Pearson correlation between a player's two half means (reliability), and w_between = ½ mean (zA - zB)² over
                     var(½ (zA + zB)) (pure noise: 2, a perfectly stable trait: 0; section 3.3(b)'s definition);
  partial_rho        partial Spearman with the rank given the half means of own_pmi and opp_pmi (the information beyond the L1b channels).
Gate G0 of section 4L.1: split_half_r >= 0.75 and |partial_rho| >= 0.05 (written before any result was seen)."""
import json

import numpy as np
import pandas as pd

from aicup.evidence import auc
from aicup.metric import LEVELS

RANK = {lv: i for i, lv in enumerate(LEVELS)}
G0_RELIABILITY, G0_PARTIAL, SCREEN_AUC = 0.75, 0.05, 0.60


def _ranks(x):
    return pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(_ranks(a[ok]), _ranks(b[ok]))[0, 1]) if ok.sum() > 2 else float("nan")


def partial_spearman(x, y, Z):
    """Spearman of x and y after regressing both (rank-transformed) on the rank-transformed columns of Z."""
    x, y, Z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(Z, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 10:
        return float("nan")
    A = np.column_stack([np.ones(ok.sum())] + [_ranks(Z[ok, j]) for j in range(Z.shape[1])])
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def halves(games, cols, seed=7, min_games=10):
    """Per player two random halves of floor(n / 2) games each (players with >= min_games games); the mean of every column per half."""
    rows = []
    for pid, g in games.groupby("player_id", sort=True):
        if len(g) < min_games:
            continue
        rng = np.random.RandomState((int(pid[:8], 16) + seed) % 2**32)
        order = rng.permutation(len(g))
        k = len(g) // 2
        for half, idx in ((0, order[:k]), (1, order[k:2 * k])):
            part = g.iloc[idx]
            rows.append({"player_id": pid, "rank": g["rank"].iloc[0], "half": half, "n_games": k,
                         **{c: float(np.nanmean(part[c].to_numpy(dtype=float))) if np.isfinite(part[c].to_numpy(dtype=float)).any() else np.nan for c in cols}})
    return pd.DataFrame(rows)


def report(games, cols, own="own_pmi", opp="opp_pmi"):
    """{col: metrics} plus a sorted table; `games` holds player_id, rank, own, opp and the candidate columns."""
    games = games.copy()
    games["y"] = games["rank"].map(RANK).astype(int)
    H = halves(games, list(cols) + [own, opp])
    H["y"] = H["rank"].map(RANK).astype(int)
    A, B = H[H.half == 0].set_index("player_id"), H[H.half == 1].set_index("player_id")
    both = A.index.intersection(B.index)
    out = {}
    for c in cols:
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10:
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            w, between = 0.5 * np.mean((a[ok] - b[ok]) ** 2), np.var(0.5 * (a[ok] + b[ok]))
            wb = float(w / between) if between > 0 else float("nan")
        else:
            r, wb = float("nan"), float("nan")
        hv = H[c].to_numpy(dtype=float)
        aucs = []
        for k in range(9):
            lo, hi = hv[(H.y == k) & np.isfinite(hv)], hv[(H.y == k + 1) & np.isfinite(hv)]
            if len(lo) > 5 and len(hi) > 5:
                u = auc(hi, lo)
                aucs.append(max(u, 1 - u))
        out[c] = {"n_games": int(np.isfinite(games[c].to_numpy(dtype=float)).sum()), "coverage": float(np.isfinite(games[c].to_numpy(dtype=float)).mean()),
                  "mean": float(np.nanmean(games[c])), "spearman_game": spearman(games[c], games["y"]), "spearman_half": spearman(H[c], H["y"]),
                  "adjacent_auc": float(np.mean(aucs)) if aucs else float("nan"), "split_half_r": r, "w_between": wb,
                  "partial_rho": partial_spearman(H[c], H["y"], H[[own, opp]].to_numpy()),
                  "corr_own_pmi": spearman(H[c], H[own]), "corr_opp_pmi": spearman(H[c], H[opp])}
        out[c]["g0"] = bool(np.isfinite(r) and r >= G0_RELIABILITY and abs(out[c]["partial_rho"]) >= G0_PARTIAL)
        out[c]["screen_auc"] = bool(np.isfinite(out[c]["adjacent_auc"]) and out[c]["adjacent_auc"] >= SCREEN_AUC)
    ref = {}
    for c in (own, opp):
        a, b = A.loc[both, c].to_numpy(dtype=float), B.loc[both, c].to_numpy(dtype=float)
        ref[c] = {"split_half_r": float(np.corrcoef(a, b)[0, 1]), "spearman_half": spearman(H[c], H["y"])}
    table = pd.DataFrame(out).T.sort_values("partial_rho", key=lambda s: -s.abs())
    return out, ref, table, H


def print_report(out, ref, table, title):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(f"== {title}: {len(out)} statistics; reference own_pmi split-half {ref['own_pmi']['split_half_r']:.3f} (spearman {ref['own_pmi']['spearman_half']:.3f}), "
          f"opp_pmi {ref['opp_pmi']['split_half_r']:.3f} ({ref['opp_pmi']['spearman_half']:.3f})")
    print(table[["coverage", "spearman_game", "spearman_half", "adjacent_auc", "split_half_r", "w_between", "partial_rho", "corr_own_pmi", "g0", "screen_auc"]]
          .astype({"coverage": float, "spearman_game": float, "spearman_half": float, "adjacent_auc": float, "split_half_r": float, "w_between": float,
                   "partial_rho": float, "corr_own_pmi": float}).round(3).to_string())
    g0 = [c for c, m in out.items() if m["g0"]]
    scr = [c for c, m in out.items() if m["screen_auc"] and not m["g0"]]
    print(f"G0 pass: {g0 or 'none'}; screen (AUC >= {SCREEN_AUC}) only: {scr or 'none'}")


def dump(path, out, ref, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"statistics": out, "reference": ref, "g0": {"split_half_r": G0_RELIABILITY, "partial_rho": G0_PARTIAL, "screen_auc": SCREEN_AUC},
                                **(extra or {})}, indent=1, default=float))
