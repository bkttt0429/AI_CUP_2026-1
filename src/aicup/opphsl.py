"""Section 4M M14: the opponent's moves scored under the 18 Human SL profiles as question features.

The training side scores them with scripts/score_folds.py --opponent (results/scores_opp_hd, results/scores_opp_hd_x40); the
test side with predict_submission.py's opponent pass (the shards' hsl_opp_sums / hsl_opp_move_logp). Both hold, per game, the
opponent's move count and its 18 log p sums (one per profile, own = opp = profile metadata, as the target's own HSL scores).

question_features gives the 39 columns of one question: the mean and std over its games of the per-game per-move mean log p
curve centred by its max (18 + 18), the mean and std over games of the opponent's HSL posterior-mean index (2), and the
target's own HSL posterior-mean index mean minus the opponent's (1). opp_hsl_features is the table form (scored-game tables
as the training side reads them). The R-level ablation (experiments/nano4/r_variants.py opphsl) was +0.0200 [+0.0156, +0.0245].
"""
import numpy as np

from aicup.evidence import posterior_mean_index
from aicup.humansl import PROFILES

SUM_COLS = [f"sum_{p}" for p in PROFILES]
OPP_HSL_FEATURE_NAMES = ([f"opp_hsl_mean_{p}" for p in PROFILES] + [f"opp_hsl_std_{p}" for p in PROFILES]
                         + ["opp_hsl_pmi_mean", "opp_hsl_pmi_std", "own_minus_opp_hsl_pmi"])
N_OPP_HSL = len(OPP_HSL_FEATURE_NAMES)


def per_game_curve(opp_sums, opp_n):
    """[G, 18]: the per-move mean log p of the opponent's moves under each profile, centred by the per-game maximum."""
    oc = np.asarray(opp_sums, dtype=float) / np.maximum(np.asarray(opp_n, dtype=float), 1.0)[:, None]
    return oc - oc.max(axis=1, keepdims=True)


def question_features(own_sums, opp_sums, opp_n):
    """[N_OPP_HSL] of one question: own_sums [G, 18] (the target's HSL sums), opp_sums [G, 18] and opp_n [G] (the opponent's)."""
    own_sums, opp_sums = np.asarray(own_sums, dtype=float), np.asarray(opp_sums, dtype=float)
    if own_sums.shape != opp_sums.shape or own_sums.ndim != 2 or own_sums.shape[1] != len(PROFILES) or len(opp_n) != len(opp_sums):
        raise ValueError(f"own sums {own_sums.shape}, opponent sums {opp_sums.shape}, {len(opp_n)} move counts: expected [G, {len(PROFILES)}] each")
    C = per_game_curve(opp_sums, opp_n)
    pm, hm = posterior_mean_index(opp_sums), posterior_mean_index(own_sums)
    return np.concatenate([C.mean(axis=0), C.std(axis=0), [pm.mean(), pm.std(), hm.mean() - pm.mean()]])


def opp_hsl_features(hsl, opp_hsl, questions):
    """[Q, N_OPP_HSL] from the scored-game tables (game_key, n_moves, sum_<profile>) of the target (hsl) and of the opponent
    (opp_hsl, whose n_moves is the opponent's); questions: a frame with game_keys lists. ValueError for a game missing from a table."""
    o, h = opp_hsl.set_index("game_key"), hsl.set_index("game_key")
    orow, hrow = {k: i for i, k in enumerate(o.index)}, {k: i for i, k in enumerate(h.index)}
    O, On, H = o[SUM_COLS].to_numpy(dtype=float), o["n_moves"].to_numpy(dtype=float), h[SUM_COLS].to_numpy(dtype=float)
    out = np.zeros((len(questions), N_OPP_HSL))
    for qi, keys in enumerate(questions["game_keys"]):
        keys = list(keys)
        missing = [k for k in keys if k not in orow or k not in hrow]
        if missing:
            raise ValueError(f"question {qi}: {len(missing)} games are not in both HSL tables, e.g. {missing[:3]}")
        oi, hi = [orow[k] for k in keys], [hrow[k] for k in keys]
        out[qi] = question_features(H[hi], O[oi], On[oi])
    return out
