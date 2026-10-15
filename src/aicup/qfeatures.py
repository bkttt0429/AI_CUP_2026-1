"""Question-level features (L2/L3 inputs) from per-game Human SL log-likelihood sums.

A question is a set of games by one player; each game contributes, per profile, the sum of
log p over the target player's moves and the number of those moves.
"""
import numpy as np
import pandas as pd

from aicup.evidence import posterior_mean_index
from aicup.humansl import PROFILES
from aicup.metric import RANK_TO_IDX

SUM_COLS = [f"sum_{p}" for p in PROFILES]
FEATURE_NAMES = [f"rel_{p}" for p in PROFILES] + ["best_logp", "pmi_sum", "peak_interp", "log_moves", "log_games",
                                                  "game_argmax_std"]


def _peak_interp(curve):
    """Argmax refined by a parabola through it and its two neighbours (profile-index units)."""
    i = int(np.argmax(curve))
    if 0 < i < len(curve) - 1:
        a, b, c = curve[i - 1], curve[i], curve[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def question_features(sums, n_moves):
    """sums [games, 18] and n_moves [games] -> feature vector aligned with FEATURE_NAMES.

    rel_*: per-move mean log p under each profile minus its maximum (curve shape);
    best_logp: that maximum; pmi_sum: posterior-mean profile index of the summed log p;
    peak_interp: parabola-refined peak of the per-move curve; log_moves, log_games:
    evidence amount; game_argmax_std: spread of the best profile across games.
    """
    sums = np.asarray(sums, dtype=float)
    n = np.asarray(n_moves, dtype=float)
    total = sums.sum(axis=0)
    moves = max(n.sum(), 1.0)
    mean = total / moves
    best = mean.max()
    per_game = sums / np.maximum(n, 1.0)[:, None]
    return np.concatenate([mean - best, [best, posterior_mean_index(total), _peak_interp(mean), np.log(moves),
                                         np.log(len(sums)), per_game.argmax(axis=1).std()]])


def build_matrix(games, questions):
    """Feature matrix X [Q, F], labels y [Q] (-1 when the rank is unknown) and games per question."""
    g = games.set_index("game_key")
    sums = g[SUM_COLS].to_numpy(dtype=float)
    n = g["n_moves"].to_numpy(dtype=float)
    row = {k: i for i, k in enumerate(g.index)}
    X = np.stack([question_features(sums[idx], n[idx]) for idx in ([row[k] for k in keys] for keys in questions["game_keys"])])
    if "rank" in questions:
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(int).to_numpy()
    else:
        y = np.full(len(questions), -1)
    return X, y, questions["game_keys"].map(len).to_numpy()


def sample_questions(players, per_player, rng, sizes=range(5, 21)):
    """Random training questions: per player, `per_player` subsets of its scored games."""
    rows = []
    for p in players.itertuples():
        keys = list(p.game_keys)
        allowed = [s for s in sizes if s <= len(keys)]
        if not allowed:
            continue
        for j in range(per_player):
            size = int(rng.choice(allowed))
            pick = rng.choice(len(keys), size=size, replace=False)
            rows.append({"question_id": f"s_{p.player_id[:8]}_{j}", "rank": p.rank, "player_id": p.player_id,
                         "n_games": size, "game_keys": [keys[i] for i in pick]})
    return pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])
