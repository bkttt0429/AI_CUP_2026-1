"""Turn per-profile log-likelihoods into class decisions and separability statistics.

Profiles are the 18 Human SL ranks (index 0 = 12k ... 17 = 6d, see aicup.humansl.PROFILES);
classes are the competition's 10 levels (index 0 = D ... 9 = 6D, see aicup.metric.LEVELS).
"""
import numpy as np

from aicup.metric import score

# Competition class -> profile indices, using the nominal kyu ranges (D = 10-12k ... A = 1-3k).
HARD_BINS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], [13], [14], [15], [16], [17]]


def class_scores_hard_bins(L):
    """[..., 18] log-likelihoods -> [..., 10]: each class scored by its best profile."""
    L = np.asarray(L, dtype=float)
    return np.stack([L[..., b].max(axis=-1) for b in HARD_BINS], axis=-1)


def posterior_mean_index(L):
    """Mean profile index under softmax(L) along the last axis (uniform prior over profiles)."""
    L = np.asarray(L, dtype=float)
    p = np.exp(L - L.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    return (p * np.arange(L.shape[-1])).sum(axis=-1)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(pos == neg) over all pairs (Mann-Whitney)."""
    pos, neg = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def fit_thresholds(s, y, n_classes=10, weights=None):
    """Monotone cut points on statistic s maximising the (weighted) mean official score (exact DP).

    Equal s values always land in the same class. Returns cuts [n_classes - 1] with
    class = number of cuts strictly below s (see apply_thresholds); unused classes get
    collapsed cuts (+inf above the highest used class, -inf below the lowest).
    """
    s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    values, inverse = np.unique(s, return_inverse=True)
    gain = np.zeros((len(values), n_classes))
    for c in range(n_classes):
        gain[:, c] = np.bincount(inverse, weights=[score(c, t) * wi for t, wi in zip(y, w)], minlength=len(values))
    best = np.empty_like(gain)
    back = np.zeros(gain.shape, dtype=int)
    best[0] = gain[0]
    for b in range(1, len(values)):
        run_arg = 0
        for c in range(n_classes):
            if best[b - 1, c] > best[b - 1, run_arg]:
                run_arg = c
            back[b, c] = run_arg
            best[b, c] = best[b - 1, run_arg] + gain[b, c]
    cls = np.empty(len(values), dtype=int)
    cls[-1] = int(np.argmax(best[-1]))
    for b in range(len(values) - 1, 0, -1):
        cls[b - 1] = back[b, cls[b]]
    cuts = np.empty(n_classes - 1)
    for j in range(1, n_classes):
        below, above = values[cls < j], values[cls >= j]
        if len(above) == 0:
            cuts[j - 1] = np.inf
        elif len(below) == 0:
            cuts[j - 1] = -np.inf
        else:
            cuts[j - 1] = (below.max() + above.min()) / 2
    return cuts


def apply_thresholds(s, cuts):
    return (np.asarray(s, dtype=float)[:, None] > np.asarray(cuts)[None, :]).sum(axis=1)


def bootstrap_ci(scores, players, ranks, n=1000, seed=42, alpha=0.05):
    """95% CI of the mean question score, resampling players with replacement within each rank."""
    scores, players, ranks = np.asarray(scores, dtype=float), np.asarray(players), np.asarray(ranks)
    rng = np.random.RandomState(seed)
    by_rank = {}
    for r in np.unique(ranks):
        ps = np.unique(players[ranks == r])
        by_rank[r] = [scores[(players == p) & (ranks == r)] for p in ps]
    means = np.empty(n)
    for i in range(n):
        picked = [groups[j] for groups in by_rank.values() for j in rng.randint(len(groups), size=len(groups))]
        means[i] = np.concatenate(picked).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def question_matrix(games, questions, profiles):
    """Per question: summed log p [Q, P], per-game-normalised sums [Q, P], target-move counts [Q].

    games: DataFrame with game_key, n_moves and sum_<profile> columns.
    questions: DataFrame with a game_keys column holding lists of game keys.
    """
    g = games.set_index("game_key")
    sums = g[[f"sum_{p}" for p in profiles]].to_numpy(dtype=float)
    n = g["n_moves"].to_numpy(dtype=float)
    row = {k: i for i, k in enumerate(g.index)}
    L_sum = np.zeros((len(questions), len(profiles)))
    L_norm = np.zeros_like(L_sum)
    n_moves = np.zeros(len(questions), dtype=int)
    for qi, keys in enumerate(questions["game_keys"]):
        idx = [row[k] for k in keys]
        L_sum[qi] = sums[idx].sum(axis=0)
        L_norm[qi] = (sums[idx] / n[idx, None]).sum(axis=0)
        n_moves[qi] = int(n[idx].sum())
    return L_sum, L_norm, n_moves
