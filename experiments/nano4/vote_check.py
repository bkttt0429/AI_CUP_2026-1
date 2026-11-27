"""Per-game 'local model' + voting vs log-likelihood summation on the validation questions (folds 1-4): the per-game L1b
10-head log p sums as the local evidence. Decoders: (a) majority vote of per-game argmax (ties -> lowest), (b) vote with +-1
tolerance (count a game for its argmax and its neighbours, half weight), (c) argmax of the summed log-likelihood, (d) expected-score
decision on the softmax of the summed log-likelihood, (e) mean of per-game posteriors then expected-score. Official score by n.
  python -I logs/nano4/vote_check.py --out results/r_variants/vote_check.json
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
from aicup.qeval import balanced_mean_over_n  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def softmax(L):
    e = np.exp(L - L.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    folds_df = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str})
    rows = []
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        players = l1b_cv.players_from_scores(hsl, folds_df)
        val = l1b_cv.pilot_questions(players)
        li = l1b.set_index("game_key")
        L = li[sum_cols(LEVELS)].to_numpy(dtype=float); row = {k: i for i, k in enumerate(li.index)}
        for q in val.itertuples():
            G = L[[row[k] for k in q.game_keys]]  # [n, 10] per-game log-likelihood sums
            y = RANK[q.rank]; n = len(G)
            am = G.argmax(axis=1)
            counts = np.bincount(am, minlength=10)
            vote = int(np.argmax(counts))
            tol = counts.astype(float).copy()
            tol[1:] += 0.5 * counts[:-1]; tol[:-1] += 0.5 * counts[1:]
            vote_tol = int(np.argmax(tol))
            S = G.sum(axis=0)
            sum_arg = int(np.argmax(S))
            sum_es = int(expected_score_decision(softmax(S)))
            mean_post = softmax(G).mean(axis=0)
            mean_es = int(expected_score_decision(mean_post))
            rows.append({"n": n, "y": y, "vote": vote, "vote_tol": vote_tol, "sum_argmax": sum_arg, "sum_es": sum_es, "meanpost_es": mean_es,
                         "game_exact": float((am == y).mean()), "game_adj": float((np.abs(am - y) <= 1).mean())})
    T = pd.DataFrame(rows)
    rep = {"questions": int(len(T)), "per_game_exact": float(T["game_exact"].mean()), "per_game_within1": float(T["game_adj"].mean()), "decoders": {}}
    for d in ("vote", "vote_tol", "sum_argmax", "sum_es", "meanpost_es"):
        s = np.array([score(int(p), int(t)) for p, t in zip(T[d], T["y"])])
        rep["decoders"][d] = {"balanced": float(balanced_mean_over_n(s, T["y"].to_numpy(), T["n"].to_numpy())), "exact": float((T[d] == T["y"]).mean()),
                              "by_n": {str(m): float(s[T["n"] == m].mean()) for m in (5, 10, 20)}}
    print(f"{len(T)} validation questions; single-game exact {rep['per_game_exact']:.3f}, within 1 {rep['per_game_within1']:.3f}")
    for d, v in rep["decoders"].items():
        print(f"{d:12s} balanced {v['balanced']:.4f} exact {v['exact']:.3f}  by n " + " ".join(f"{m}: {x:.4f}" for m, x in v["by_n"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
