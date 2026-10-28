"""Question features from a strength head's per-game statistics (issue #14, option b).

scripts/se_score.py stores, per game, the moves judged and statistics of the strength scores s of the target's (or,
with --opponent, the opponent's) moves: mean, standard deviation, the 10 / 50 / 90 % quantiles and the mean by
stage (moves with t < 40, 40 <= t < 100, t >= 100; a stage without moves takes the game's mean). se_features
turns a question's games into SE_FEATURE_NAMES: the move-weighted mean of the per-game means, the move-weighted
standard deviation of the per-game means (between games), the move-weighted mean of the per-game standard deviations
(within games), and the move-weighted means of the quantiles and stage means. build_se_matrix has
qfeatures.build_matrix's shape for a scored-games table with SE_STATS columns.
"""
import numpy as np

from aicup.metric import RANK_TO_IDX

SE_STATS = ("se_mean", "se_std", "se_q10", "se_q50", "se_q90", "se_s1", "se_s2", "se_s3")
SE_FEATURE_NAMES = ["se_mean", "se_between_std", "se_within_std", "se_q10", "se_q50", "se_q90", "se_s1", "se_s2", "se_s3"]
STAGES = ((0, 40), (40, 100), (100, 10 ** 9))


def game_stats(s, t):
    """SE_STATS of one game from its per-move scores s [n] and move indices t [n] (zeros for a game without moves)."""
    s, t = np.asarray(s, dtype=np.float64), np.asarray(t)
    if len(s) == 0:
        return dict.fromkeys(SE_STATS, 0.0)
    mean = float(s.mean())
    q10, q50, q90 = (float(v) for v in np.quantile(s, [0.1, 0.5, 0.9]))
    stages = []
    for lo, hi in STAGES:
        m = (t >= lo) & (t < hi)
        stages.append(float(s[m].mean()) if m.any() else mean)
    return {"se_mean": mean, "se_std": float(s.std()), "se_q10": q10, "se_q50": q50, "se_q90": q90,
            "se_s1": stages[0], "se_s2": stages[1], "se_s3": stages[2]}


def se_features(stats, n_moves):
    """[9] SE_FEATURE_NAMES from a question's games: stats [m, 8] in SE_STATS order and n_moves [m]."""
    stats = np.asarray(stats, dtype=np.float64).reshape(-1, len(SE_STATS))
    n = np.asarray(n_moves, dtype=np.float64)
    w = n / max(n.sum(), 1.0)
    mean_w = float((stats[:, 0] * w).sum())
    between = float(np.sqrt(max((w * (stats[:, 0] - mean_w) ** 2).sum(), 0.0)))
    within = float((stats[:, 1] * w).sum())
    rest = [float((stats[:, k] * w).sum()) for k in range(2, len(SE_STATS))]
    return np.array([mean_w, between, within, *rest])


def build_se_matrix(games, questions):
    """X [Q, 9], y [Q] (-1 when the rank is unknown), games per question -- for a scored-games table with SE_STATS."""
    g = games.set_index("game_key")
    stats = g[list(SE_STATS)].to_numpy(dtype=float)
    n = g["n_moves"].to_numpy(dtype=float)
    row = {k: i for i, k in enumerate(g.index)}
    X = np.stack([se_features(stats[idx], n[idx]) for idx in ([row[k] for k in keys] for keys in questions["game_keys"])])
    if "rank" in questions:
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(int).to_numpy()
    else:
        y = np.full(len(questions), -1)
    return X, y, questions["game_keys"].map(len).to_numpy()
