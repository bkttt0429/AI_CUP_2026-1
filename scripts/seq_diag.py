#!/usr/bin/env python
"""Does the order of the moves carry rank information beyond their sum? Diagnostics on the per-move L1b log p (issue #19).

For every game of the CV folds, the per-move posterior-mean rank index pmi_t (softmax of the 10 L1b log p of the move,
expected index) and the move's surprisal s_t = -max_k log p_t,k give two scalar sequences. Per game: mean, std, lag-1 and
lag-2 autocorrelation (of the deviations from the game mean), second-half minus first-half mean, OLS slope over the
normalised move index, mean surprisal and its lag-1 autocorrelation, log n_moves. Reported per rank (means), and tested:
a class-balanced multinomial logistic regression per game on folds 1-4 (train three, test one) with the mean alone
(mean_pmi, log_moves) against mean + shape; the gain in per-game balanced accuracy and NLL is the size of the "shape"
signal the sums cannot see. A sequence model (scripts/seq_crossfit.py) is worth training only if this gain is clearly
positive.

  python -I scripts/seq_diag.py --l1b results/l1b/scores3 results/l1b/scores3_B --out results/seq/seq_diag.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.rankmodel import SoftmaxModel, class_weights  # noqa: E402
from aicup.seqdata import read_moves  # noqa: E402
import l1b_cv  # noqa: E402

FEATURES = ["mean_pmi", "log_moves", "std_pmi", "ac1_pmi", "ac2_pmi", "half_diff_pmi", "slope_pmi", "mean_s", "ac1_s"]
MEAN_ONLY = ["mean_pmi", "log_moves"]


def _ac(x, lag):
    x = x - x.mean()
    if len(x) <= lag + 1 or np.var(x) < 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / np.var(x))


def game_shape(logp):
    """The FEATURES of one game from its per-move L1b log p [n, 10]."""
    lp = np.asarray(logp, dtype=np.float64)
    n = len(lp)
    if n == 0:
        return np.zeros(len(FEATURES))
    z = lp - lp.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pmi = p @ np.arange(lp.shape[1])
    s = -lp.max(axis=1)
    t = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    half = n // 2
    half_diff = float(pmi[half:].mean() - pmi[:half].mean()) if half else 0.0
    slope = float(np.polyfit(t, pmi, 1)[0]) if n > 2 else 0.0
    return np.array([pmi.mean(), np.log(n), pmi.std(), _ac(pmi, 1), _ac(pmi, 2), half_diff, slope, s.mean(), _ac(s, 1)])


def fold_table(fold_dir):
    games, moves = read_moves(fold_dir)
    rows = np.stack([game_shape(moves[k][1]) for k in games["game_key"]])
    out = pd.DataFrame(rows, columns=FEATURES)
    out.insert(0, "y", games["rank"].map(RANK_TO_IDX).to_numpy())
    out.insert(0, "player_id", games["player_id"].to_numpy(dtype=str))
    return out


def _balanced(pred_proba, y):
    y = np.asarray(y)
    acc, nll = [], []
    for c in range(len(LEVELS)):
        m = y == c
        if m.any():
            acc.append(float((pred_proba[m].argmax(axis=1) == c).mean()))
            nll.append(float(-np.log(np.clip(pred_proba[m, c], 1e-12, 1)).mean()))
    return float(np.mean(acc)), float(np.mean(nll))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories (each fold in exactly one)")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    try:
        where = l1b_cv.locate_folds(a.l1b, folds)
        tables = {f: fold_table(Path(where[f]) / f"fold{f}") for f in folds}
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    allg = pd.concat(tables.values(), ignore_index=True)
    by_rank = allg.groupby("y")[FEATURES].mean()
    by_rank.index = [LEVELS[i] for i in by_rank.index]
    print("per-rank means of the per-game shape features:")
    print(by_rank.round(3).to_string())
    corr = allg[FEATURES].corr().loc["mean_pmi"].round(3).to_dict()
    results = {"mean_only": [], "mean_plus_shape": []}
    for v in folds:
        tr = pd.concat([tables[f] for f in folds if f != v], ignore_index=True)
        te = tables[v]
        for name, cols in (("mean_only", MEAN_ONLY), ("mean_plus_shape", FEATURES)):
            m = SoftmaxModel(l2=1e-3).fit(tr[cols].to_numpy(), tr["y"].to_numpy(), w=class_weights(tr["y"].to_numpy()))
            acc, nll = _balanced(m.predict_proba(te[cols].to_numpy()), te["y"].to_numpy())
            results[name].append({"fold": v, "balanced_acc": acc, "balanced_nll": nll, "n_games": int(len(te))})
        print(f"fold {v}: mean-only acc {results['mean_only'][-1]['balanced_acc']:.4f} NLL {results['mean_only'][-1]['balanced_nll']:.4f} | "
              f"mean+shape acc {results['mean_plus_shape'][-1]['balanced_acc']:.4f} NLL {results['mean_plus_shape'][-1]['balanced_nll']:.4f}", flush=True)
    gain_acc = [b["balanced_acc"] - a_["balanced_acc"] for a_, b in zip(results["mean_only"], results["mean_plus_shape"])]
    gain_nll = [a_["balanced_nll"] - b["balanced_nll"] for a_, b in zip(results["mean_only"], results["mean_plus_shape"])]
    print(f"shape gain per fold: acc {' '.join(f'{g:+.4f}' for g in gain_acc)}  NLL {' '.join(f'{g:+.4f}' for g in gain_nll)}  mean acc {np.mean(gain_acc):+.4f} NLL {np.mean(gain_nll):+.4f}")
    out = {"folds": folds, "games": int(len(allg)), "features": FEATURES, "per_rank_means": by_rank.to_dict(orient="index"),
           "corr_with_mean_pmi": corr, "per_game_logistic": results, "shape_gain": {"balanced_acc": gain_acc, "balanced_nll": gain_nll,
                                                                                    "mean_acc": float(np.mean(gain_acc)), "mean_nll": float(np.mean(gain_nll))},
           "l1b_dirs": a.l1b, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
