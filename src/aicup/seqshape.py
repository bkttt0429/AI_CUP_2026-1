"""Per-game "shape" features of the per-move L1b log p (issue #19 step 0; issue #21 fused_opp_shape).

One game's per-move L1b log p [n, 10] gives two scalar sequences: pmi_t, the expected rank index under the softmax of move
t's ten log p, and s_t = -max_k log p_t,k, the surprisal of the move's best rank. FEATURES per game: mean_pmi, log_moves,
std_pmi, ac1_pmi, ac2_pmi (lag-1 / lag-2 autocorrelation of pmi about the game mean), half_diff_pmi (second half minus
first half), slope_pmi (OLS slope over the normalised move index), mean_s, ac1_s. SHAPE is FEATURES without log_moves
(the question features already carry n_moves). The sums behind today's features see none of these except through the
mean, so a question-level mean of SHAPE over the question's games is an independent channel for within-game order.
"""
import numpy as np
import pandas as pd

from aicup.seqdata import read_moves

FEATURES = ["mean_pmi", "log_moves", "std_pmi", "ac1_pmi", "ac2_pmi", "half_diff_pmi", "slope_pmi", "mean_s", "ac1_s"]
SHAPE = [f for f in FEATURES if f != "log_moves"]


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


def fold_shape_table(fold_dir):
    """game_key, player_id, rank, n_moves and the FEATURES of every game of one scored fold (an l1b_score.py fold directory)."""
    games, moves = read_moves(fold_dir)
    rows = np.stack([game_shape(moves[k][1]) for k in games["game_key"]]) if len(games) else np.zeros((0, len(FEATURES)))
    out = pd.DataFrame(rows, columns=FEATURES)
    for c in ("n_moves", "rank", "player_id", "game_key"):
        out.insert(0, c, games[c].to_numpy())
    return out


def question_shape(table, questions, cols=SHAPE, with_std=False):
    """X [Q, k] (or [Q, 2k] with the standard deviation over the games appended): the mean of `cols` over each question's games."""
    g = table.set_index("game_key")
    vals = g[list(cols)].to_numpy(dtype=float)
    row = {k: i for i, k in enumerate(g.index)}
    k = len(cols)
    out = np.zeros((len(questions), 2 * k if with_std else k))
    for q, keys in enumerate(questions["game_keys"]):
        v = vals[[row[key] for key in keys]]
        out[q, :k] = v.mean(axis=0)
        if with_std:
            out[q, k:] = v.std(axis=0)
    return out
