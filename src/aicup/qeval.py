"""Question-level evaluation and paired model comparison on pooled evaluation questions.

A question is scored by the official metric of the expected-score decision taken from its
class probabilities (see aicup.metric). All headline numbers are class-balanced: the mean of
per-class means, per question size, then averaged over the sizes. Classes are the competition's
10 levels (index 0 = D ... 9 = 6D, see aicup.metric.LEVELS).
"""
import numpy as np

from aicup.evidence import bootstrap_ci_mean_over_sizes
from aicup.metric import LEVELS, expected_score_decision, score

SIZES = (5, 10, 20)
N_CLASSES = len(LEVELS)


def _balanced(values, y, mask=None):
    """Mean over classes present (inside `mask`) of the per-class mean of `values`."""
    mask = np.ones(len(y), dtype=bool) if mask is None else mask
    return float(np.mean([values[mask & (y == k)].mean() for k in range(N_CLASSES) if (mask & (y == k)).any()]))


def _balanced_by_n(values, y, n_games, sizes):
    return {int(m): _balanced(values, y, n_games == m) for m in sizes}


def balanced_mean_over_n(scores, y, n_games, sizes=SIZES):
    """Class-balanced mean of `scores` per question size, averaged over `sizes`."""
    scores, y, n_games = np.asarray(scores, dtype=float), np.asarray(y, dtype=int), np.asarray(n_games)
    return float(np.mean(list(_balanced_by_n(scores, y, n_games, sizes).values())))


def evaluate(proba, y, qs, sizes=SIZES):
    """Official scores and diagnostics of class probabilities `proba` [Q, 10] for true classes `y` [Q].

    qs: DataFrame with an n_games column (question size). Returns (metrics, scores [Q]).
    `nll_balanced` is the class-balanced mean of -log p(true class) (infinite when a true class
    gets probability 0, as for one-hot models) and `brier` the class-balanced mean of the
    multi-class Brier score sum_k (p_k - 1[k = y])^2.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = np.array([expected_score_decision(p) for p in proba])
    s = np.array([score(p, t) for p, t in zip(pred, y)])
    n = qs["n_games"].to_numpy()

    out = {"balanced_by_n": _balanced_by_n(s, y, n, sizes)}
    out["balanced_mean_over_n"] = float(np.mean(list(out["balanced_by_n"].values())))
    out["question_mean"] = float(s.mean())
    out["per_class"] = {LEVELS[k]: float(s[y == k].mean()) for k in range(N_CLASSES)}
    out["exact"] = float((pred == y).mean())
    out["within1"] = float((np.abs(pred - y) <= 1).mean())
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(conf, (y, pred), 1)
    out["confusion_rows_truth_cols_pred"] = conf.tolist()
    with np.errstate(divide="ignore"):
        nll = -np.log(proba[np.arange(len(y)), y])
    out["nll_balanced"] = _balanced(nll, y)
    out["brier"] = _balanced(((proba - np.eye(N_CLASSES)[y]) ** 2).sum(axis=1), y)
    return out, s


def paired_diff(s_a, s_b, y, player_id, n_games, sizes=SIZES, n=5000, seed=42):
    """Paired difference A - B of the class-balanced mean over question sizes, with a player bootstrap CI.

    s_a, s_b: per-question scores of the two models on the same questions. The statistic is linear
    in the scores, so the bootstrap interval of s_a - s_b is the paired interval of the difference
    (players are resampled within each class, once per replicate and shared by every size).
    Only questions whose size is in `sizes` enter, the difference and the interval alike.
    Returns {"diff", "ci95": [lo, hi], "by_n": {size: difference at that size}}.
    """
    s_a, s_b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)
    y, player_id, n_games = np.asarray(y, dtype=int), np.asarray(player_id), np.asarray(n_games)
    keep = np.isin(n_games, list(sizes))
    s_a, s_b, y, player_id, n_games = s_a[keep], s_b[keep], y[keep], player_id[keep], n_games[keep]
    by_a, by_b = _balanced_by_n(s_a, y, n_games, sizes), _balanced_by_n(s_b, y, n_games, sizes)
    lo, hi = bootstrap_ci_mean_over_sizes(s_a - s_b, player_id, y, n_games, n=n, seed=seed)
    return {"diff": balanced_mean_over_n(s_a, y, n_games, sizes) - balanced_mean_over_n(s_b, y, n_games, sizes),
            "ci95": [lo, hi],
            "by_n": {m: by_a[m] - by_b[m] for m in by_a}}
