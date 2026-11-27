"""GPT's decision-level bound B (2026-11-05): on the CV oh ensemble (folds 1-4), take the questions whose two best expected-score actions are a
given adjacent pair {c, c+1}; B = (1/N) sum over those of [max(u(a1, y), u(a2, y)) - u(a1, y)] -- the most a perfect local two-way decision
could add to the official score (denominator: all N questions, class-balanced over n as the metric). Also: share of such questions, how often
the second action was right, and the within-class split-half error correlation of 1D / 2D / 3D (floor_decomp.players.csv, 10/10 halves).
  python -I logs/nano4/oracle_B.py --out results/r_variants/oracle_B.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
SM = np.array([[score(k, j) for j in range(10)] for k in range(10)])  # SM[action, truth]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    ys, ns, Ps = [], [], []
    for v in (1, 2, 3, 4):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        ys.append(meta["rank"].map(RANK).to_numpy()); ns.append(meta.n_games.to_numpy()); Ps.append(P)
    y, n, P = np.concatenate(ys), np.concatenate(ns), np.concatenate(Ps)
    ES = P @ SM.T  # expected score of each action
    order = np.argsort(-ES, axis=1); a1, a2 = order[:, 0], order[:, 1]
    u1 = SM[a1, y]; u2 = SM[a2, y]; gain = np.maximum(u1, u2) - u1
    base = balanced_mean_over_n(u1, y, n)
    rep = {"questions": int(len(y)), "base_score": base, "pairs": {}}
    print(f"{len(y)} questions; base (ES decision) {base:.4f}; a second-action oracle over ALL questions would add {balanced_mean_over_n(u1 + gain, y, n) - base:+.4f}")
    for c in range(9):
        sel = ((a1 == c) & (a2 == c + 1)) | ((a1 == c + 1) & (a2 == c))
        u_or = u1.copy(); u_or[sel] = np.maximum(u1, u2)[sel]
        B = balanced_mean_over_n(u_or, y, n) - base
        rep["pairs"][f"{LEVELS[c]}-{LEVELS[c+1]}"] = {"share_of_questions": float(sel.mean()), "B": B, "second_action_right_share": float((u2[sel] > u1[sel]).mean()) if sel.any() else None,
                                                     "truth_in_pair_share": float(np.isin(y[sel], [c, c + 1]).mean()) if sel.any() else None}
        r = rep["pairs"][f"{LEVELS[c]}-{LEVELS[c+1]}"]
        print(f"top-2 = {LEVELS[c]:>2s}/{LEVELS[c+1]:<2s}: {r['share_of_questions']*100:5.1f}% of questions, truth in the pair {r['truth_in_pair_share']:.3f}, 2nd action better {r['second_action_right_share']:.3f}, B = {B:+.4f}")
    # by n for the 1D-2D pair
    sel = ((a1 == 4) & (a2 == 5)) | ((a1 == 5) & (a2 == 4)); u_or = u1.copy(); u_or[sel] = np.maximum(u1, u2)[sel]
    rep["B_1D2D_by_n"] = {str(m): balanced_mean_over_n(u_or, y, n, sizes=(m,)) - balanced_mean_over_n(u1, y, n, sizes=(m,)) for m in (5, 10, 20)}
    print("B(1D-2D) by n:", {k: round(v, 4) for k, v in rep["B_1D2D_by_n"].items()})
    # within-class split-half error correlation (floor_decomp: e0 / e1 are the two 10-game halves' signed errors)
    fd = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str})
    rep["split_half_within_class"] = {}
    for lv in LEVELS:
        s = fd[fd["y"] == RANK[lv]] if fd["y"].dtype != object else fd[fd["y"] == lv]
        if len(s) > 20:
            rep["split_half_within_class"][lv] = {"players": int(len(s)), "corr_e0_e1": float(np.corrcoef(s.e0, s.e1)[0, 1])}
    print("within-class split-half error corr (10/10): " + " ".join(f"{k}:{v['corr_e0_e1']:.2f}" for k, v in rep["split_half_within_class"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
