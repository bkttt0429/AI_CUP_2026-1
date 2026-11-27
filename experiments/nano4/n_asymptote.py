"""How far does per-game evidence carry as the number of games grows past the test's 20?  Validation players (folds 1-4) with 80
scored games (base 20 + x40 games 21-40 + x80 games 41-80): the n-agnostic decoder of vote_check (mean of per-game L1b posteriors,
expected-score decision) at n = 5, 10, 20, 40, 80 games, class-balanced official score and exact rate; the persistent error at
n = 80 and the 40/40 split-half correlation of the expected rank; a saturation fit score(n) = a - b n^(-1/2).
  python -I logs/nano4/n_asymptote.py --out results/r_variants/n_asymptote.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
NS = (5, 10, 20, 40, 80)
DRAWS = {5: 8, 10: 8, 20: 4, 40: 2, 80: 1}


def softmax(L):
    e = np.exp(L - L.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def balanced(s, y):
    return float(np.mean([s[y == k].mean() for k in range(10) if (y == k).any()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    lcols = sum_cols(LEVELS)
    x40 = pd.read_csv(ROOT / "results/data/x40_keys.csv", dtype={"player_id": str})
    x80 = pd.read_csv(ROOT / "results/data/x80_keys.csv", dtype={"player_id": str})
    rng = np.random.default_rng(0)
    recs = []  # one per (player, n, draw)
    per_player = []
    for f in (1, 2, 3, 4):
        base = read_scored_games(ROOT / "results/l1b" / ts.L1B[f] / f"fold{f}")
        tabs = [base] + [read_scored_games(ROOT / "results/l1b" / (ts.L1B[f] + sfx) / f"fold{f}") for sfx in ("_x40", "_x80")]
        L = pd.concat(tabs).drop_duplicates("game_key").set_index("game_key")
        P = softmax(L[lcols].to_numpy(dtype=float)); row = {k: i for i, k in enumerate(L.index)}
        ids80 = x80.loc[x80.fold == f].groupby("player_id")["game_key"].apply(list)
        ids40 = x40.loc[x40.fold == f].groupby("player_id")["game_key"].apply(list)
        keys_base = base.groupby("player_id")["game_key"].apply(list)
        rank_of = dict(zip(base["player_id"], base["rank"]))
        for pid, k80 in ids80.items():
            if len(k80) < 40 or pid not in ids40 or len(ids40[pid]) < 20 or pid not in keys_base or len(keys_base[pid]) < 20:
                continue
            keys = [k for k in keys_base[pid][:20] + ids40[pid][:20] + k80[:40] if k in row]
            if len(keys) < 80:
                continue
            y = RANK[rank_of[pid]]; idx = np.array([row[k] for k in keys[:80]])
            for n in NS:
                for d in range(DRAWS[n]):
                    sel = idx if n == 80 else rng.choice(idx, n, replace=False)
                    mp = P[sel].mean(axis=0); pred = int(expected_score_decision(mp))
                    recs.append({"fold": f, "player_id": pid, "y": y, "n": n, "pred": pred, "E": float((mp * np.arange(10)).sum()), "s": score(pred, y)})
            h = rng.permutation(idx); E0 = (P[h[:40]].mean(axis=0) * np.arange(10)).sum(); E1 = (P[h[40:]].mean(axis=0) * np.arange(10)).sum()
            per_player.append({"fold": f, "y": y, "E0": E0, "E1": E1, "E80": (P[idx].mean(axis=0) * np.arange(10)).sum()})
    T, Q = pd.DataFrame(recs), pd.DataFrame(per_player)
    by_n = {}
    for n in NS:
        t = T[T.n == n]; by_n[n] = {"balanced": balanced(t["s"].to_numpy(), t["y"].to_numpy()), "exact": float((t.pred == t.y).mean()),
                                   "within1": float((np.abs(t.pred - t.y) <= 1).mean()), "sd_E_minus_y": float((t.E - t.y).std())}
    ns = np.array(NS, float); ys = np.array([by_n[n]["balanced"] for n in NS])
    A = np.stack([np.ones_like(ns), -ns ** -0.5], axis=1); a_, b_ = np.linalg.lstsq(A, ys, rcond=None)[0]
    e80 = Q.E80 - Q.y
    rep = {"players": int(len(Q)), "by_n": by_n, "fit_a_minus_b_over_sqrt_n": {"a_asymptote": float(a_), "b": float(b_), "fitted": [float(a_ - b_ * n ** -0.5) for n in NS]},
           "persistent_at_80": {"sd_E80_minus_y": float(e80.std()), "mean": float(e80.mean()), "share_abs_ge_0.5": float((e80.abs() >= 0.5).mean()),
                                "splithalf_r_E_40_40": float(np.corrcoef(Q.E0 - Q.y, Q.E1 - Q.y)[0, 1]), "slope_e80_on_y": float(np.polyfit(Q.y, e80, 1)[0])},
           "exact_by_rank_at_80": [float((T[(T.n == 80) & (T.y == k)].pred == k).mean()) for k in range(10)]}
    print(json.dumps(rep, indent=1)); Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
