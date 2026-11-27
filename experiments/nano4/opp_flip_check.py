"""Section 4M M14: definitive check that `score_folds.py --opponent` scored the OTHER player's moves.

For every game present in both tables of one fold, the per-move ply indices `t` of the target table
(results/scores_hd) and the opponent table (results/scores_opp_hd) must be disjoint, and their union
must be the contiguous range 0..T-1 (all plies of the game). Also reports the parity pattern per
target colour and the correlation of the per-game 1d sums.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aicup.scoring import completed_parts, part_paths  # noqa: E402


def load(fold_dir):
    out = {}
    for s, k in completed_parts(fold_dir):
        csv, npz = part_paths(fold_dir, s, k)
        g = pd.read_csv(csv, dtype={"game_key": str, "color": str}, keep_default_na=False)
        z = np.load(npz)
        t, row = z["t"], z["game_row"]
        lp = z["logp"]
        for i, key in enumerate(g["game_key"]):
            m = row == i
            out[key] = (np.sort(t[m]), lp[m], g["color"].iat[i], int(g["n_moves"].iat[i]), float(g["sum_1d"].iat[i]))
    return out


def main(fold, max_games):
    own = load(ROOT / "results/scores_hd" / f"fold{fold}")
    opp = load(ROOT / "results/scores_opp_hd" / f"fold{fold}")
    shared = sorted(set(own) & set(opp))
    print(f"fold {fold}: own {len(own)} games, opp {len(opp)} games, shared {len(shared)}", flush=True)
    rng = np.random.default_rng(0)
    keys = shared if len(shared) <= max_games else list(rng.choice(shared, max_games, replace=False))
    disjoint = contiguous = n_ok = same_logp = 0
    parity = {"B": [], "W": []}
    s_own, s_opp = [], []
    bad = []
    for key in keys:
        t_o, lp_o, col_o, n_o, s1 = own[key]
        t_p, lp_p, col_p, n_p, s2 = opp[key]
        inter = np.intersect1d(t_o, t_p)
        union = np.union1d(t_o, t_p)
        d = len(inter) == 0
        c = len(union) == union.max() + 1 if len(union) else False
        ok = len(t_o) == n_o and len(t_p) == n_p
        disjoint += d; contiguous += c; n_ok += ok
        if not (d and c and ok):
            bad.append((key, len(inter), len(union), int(union.max()) + 1, n_o, len(t_o), n_p, len(t_p)))
        if len(t_o) and len(t_p):
            parity[col_o].append((float(np.mean(t_o % 2)), float(np.mean(t_p % 2))))
        if col_o == col_p and len(lp_o) == len(lp_p) and np.allclose(lp_o, lp_p):
            same_logp += 1
        s_own.append(s1); s_opp.append(s2)
    n = len(keys)
    print(f"checked {n} games: disjoint plies {disjoint}/{n}, union contiguous 0..T-1 {contiguous}/{n}, "
          f"len(t)==n_moves both tables {n_ok}/{n}, identical logp arrays {same_logp}/{n}")
    for col, v in parity.items():
        if v:
            a = np.array(v)
            print(f"target colour {col}: mean(t%2) target {a[:,0].mean():.3f}  opponent {a[:,1].mean():.3f}  ({len(v)} games)")
    print(f"corr(sum_1d own, sum_1d opp) over checked games: {np.corrcoef(s_own, s_opp)[0,1]:.4f}")
    print(f"colour column equal in both tables: {np.mean([own[k][2] == opp[k][2] for k in keys]):.3f} (by design: the target's colour)")
    if bad:
        print(f"{len(bad)} games violate a check; first 10 (key, |inter|, |union|, T, n_own, len_t_own, n_opp, len_t_opp):")
        for b in bad[:10]:
            print("  ", b)
    verdict = "PASS" if disjoint == n and contiguous == n and n_ok == n and same_logp == 0 else "FAIL"
    print(f"flip check fold {fold}: {verdict}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
