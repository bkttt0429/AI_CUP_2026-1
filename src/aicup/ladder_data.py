"""Question packing and the static rungs (0)(1)(2) of the L3 aggregator ladder.

pack_questions turns a question DataFrame into a padded set of per-game tokens over the vectors
of aicup.gamevec (`vec`: game_key, staged, whole, sums, n_moves). static_features turns packed
questions into the fixed-length question features of the static rungs, in float64; every kind is
a symmetric function of the question's games, so the order of game_keys does not matter.

Notation (per question): m games, sums_g [18] and n_g target moves of game g, L = sum_g sums_g,
n = sum_g n_g, c_g = sums_g / n_g (per-move curve), r_g = c_g - mean_r c_g (its shape over profiles).
log m and log n are the evidence amount; counts=False is the game-length ablation (METHOD_NOTES
§2.6): it drops the explicit count inputs (token n_eff, present and log1p(n_g); qglob beyond log m;
log n; q24's log_moves), as listed in pack_questions and static_features.
"""
import numpy as np

from aicup.gamevec import SEGMENTS, WHOLE, vec_names
from aicup.metric import RANK_TO_IDX
from aicup.qfeatures import FEATURE_NAMES, question_features
from aicup.rankmodel import MLPModel, SoftmaxModel

QGLOB_NAMES = ["log_games", "log_target_moves", "log1p_seg1", "log1p_seg2", "log1p_seg3", "log1p_seg4"]
REPS = {"staged": SEGMENTS, "whole": WHOLE}
CLIP_QUANTILE = 0.95


def _stat_cols(names, stat):
    """Columns of one per-stage field (e.g. "n_eff"), in stage order: names are {prefix}_{stat}."""
    return [i for i, name in enumerate(names) if name.split("_", 1)[1] == stat]


_STAGED = vec_names(SEGMENTS)
SEG_N_EFF, SEG_PRESENT = _stat_cols(_STAGED, "n_eff"), _stat_cols(_STAGED, "present")


def count_cols(rep):
    """Token columns zeroed by counts=False: each stage's n_eff and present, and log1p(n_moves_g) (the last)."""
    names = vec_names(REPS[rep])
    return _stat_cols(names, "n_eff") + _stat_cols(names, "present") + [len(names)]


def pack_questions(questions, vec, rep="staged", counts=True, max_games=20):
    """Questions -> padded game sets over the vectors in `vec` (see the module docstring).

    Returns tokens float32 [Q, max_games, D + 1] (the game's `rep` vector, D = 300 staged or 75
    whole, then log1p(n_moves_g); padding rows are 0), mask bool [Q, max_games] (True = a game),
    qglob float32 [Q, 6] aligned with QGLOB_NAMES (log m, log n and log1p of each stage's total
    target moves, always from the staged n_eff), y int64 [Q] (-1 without a rank), idx (per question
    the vec rows of its games, in game_keys order), player_id, n_games and question_id arrays.
    counts=False zeroes the count_cols(rep) token columns and qglob[:, 1:].
    """
    if rep not in REPS:
        raise ValueError(f"rep must be one of {list(REPS)}, got {rep!r}")
    feats = vec[rep]
    D = len(vec_names(REPS[rep]))
    if feats.shape[1] != D:
        raise ValueError(f"vec[{rep!r}] has {feats.shape[1]} columns, expected {D}")
    keys = vec["game_key"].tolist()
    row = {k: i for i, k in enumerate(keys)}
    if len(row) != len(keys):
        raise ValueError(f"vec has {len(keys) - len(row)} duplicate game keys")
    idx = []
    for qid, gk in zip(questions["question_id"], questions["game_keys"]):
        if not 1 <= len(gk) <= max_games:
            raise ValueError(f"question {qid}: {len(gk)} games, need 1..{max_games}")
        missing = [k for k in gk if k not in row]
        if missing:
            raise KeyError(f"question {qid}: game {missing[0]} is not in the vectors")
        idx.append(np.array([row[k] for k in gk], dtype=np.int64))
    n_games = np.array([len(ix) for ix in idx], dtype=np.int64)
    if "n_games" in questions and not np.array_equal(questions["n_games"].to_numpy(dtype=np.int64), n_games):
        raise ValueError("n_games column differs from the number of game_keys")

    n_moves, seg = vec["n_moves"], vec["staged"][:, SEG_N_EFF].astype(np.float64)
    tokens = np.zeros((len(idx), max_games, D + 1), dtype=np.float32)
    mask = np.zeros((len(idx), max_games), dtype=bool)
    qglob = np.zeros((len(idx), len(QGLOB_NAMES)))
    for q, ix in enumerate(idx):
        m = len(ix)
        tokens[q, :m, :D] = feats[ix]
        tokens[q, :m, D] = np.log1p(n_moves[ix])
        mask[q, :m] = True
        qglob[q] = [np.log(m), np.log(max(n_moves[ix].sum(), 1)), *np.log1p(seg[ix].sum(axis=0))]
    if not counts:
        tokens[:, :, count_cols(rep)] = 0.0
        qglob[:, 1:] = 0.0
    if "rank" in questions:
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(np.int64).to_numpy()
    else:
        y = np.full(len(idx), -1, dtype=np.int64)
    return {"tokens": tokens, "mask": mask, "qglob": qglob.astype(np.float32), "y": y, "idx": idx,
            "player_id": questions["player_id"].to_numpy(dtype=str), "n_games": n_games,
            "question_id": questions["question_id"].to_numpy(dtype=str)}


def _profile_shape(vec, ix):
    """r_g = c_g - mean_r c_g [m, 18] with c_g = sums_g / n_g (float64)."""
    c = vec["sums"][ix].astype(np.float64) / np.maximum(vec["n_moves"][ix].astype(np.float64), 1.0)[:, None]
    return c - c.mean(axis=1, keepdims=True)


def _evidence(vec, ix, counts):
    """[log m, log n], or [log m] when counts=False."""
    m, n = len(ix), max(float(vec["n_moves"][ix].sum()), 1.0)
    return [np.log(m), np.log(n)] if counts else [np.log(m)]


def _robust(vec, ix, kind, clip_k):
    """18 robust per-profile means of r_g: clipped around the median ("clip") or 20% trimmed ("trim")."""
    r = _profile_shape(vec, ix)
    if kind == "clip":
        med = np.median(r, axis=0)
        return med + np.clip(r - med, -clip_k, clip_k).mean(axis=0)
    cut = len(r) // 5  # floor(0.2 m) games off each side
    return np.sort(r, axis=0)[cut:len(r) - cut].mean(axis=0)


def _sum(vec, ix):
    """The calibrated-sum form of §2.3, linearised: (L - mean_r L) / sqrt(n) and / n."""
    L = vec["sums"][ix].astype(np.float64).sum(axis=0)
    n = max(float(vec["n_moves"][ix].sum()), 1.0)
    d = L - L.mean()
    return np.concatenate([d / np.sqrt(n), d / n])


def _q24(packed, vec, counts):
    X = np.stack([question_features(vec["sums"][ix], vec["n_moves"][ix]) for ix in packed["idx"]])
    return X if counts else np.delete(X, FEATURE_NAMES.index("log_moves"), axis=1)


def _pooled(packed, vec, counts):
    """Mean token over the question's games, qglob and q24; counts=False strips the count columns."""
    tok, mask = packed["tokens"], packed["mask"]
    reps = {len(vec_names(s)) + 1: r for r, s in REPS.items()}
    if tok.shape[2] not in reps:
        raise ValueError(f"tokens have {tok.shape[2]} columns, not a packed rep")
    # per question, so the [Q, 20, D + 1] tokens are never copied to float64 as a whole
    mean = np.stack([tok[q, mask[q]].astype(np.float64).mean(axis=0) for q in range(len(tok))])
    qglob = packed["qglob"].astype(np.float64)
    if counts:
        if np.any(mean[:, -1] == 0):
            raise ValueError("pooled with counts=True needs questions packed with counts=True")
    else:
        mean[:, count_cols(reps[tok.shape[2]])] = 0.0
        qglob[:, 1:] = 0.0
    return np.hstack([mean, qglob, _q24(packed, vec, counts)])


def _length(vec, ix):
    """Game-length statistics only (§2.6): log m, log n, mean and std (ddof=0) of n_g, per-stage
    fraction of games with the stage present and mean n_eff."""
    n_g = vec["n_moves"][ix].astype(np.float64)
    st = vec["staged"][ix].astype(np.float64)
    return np.concatenate([_evidence(vec, ix, True), [n_g.mean(), n_g.std()], st[:, SEG_PRESENT].mean(axis=0),
                           st[:, SEG_N_EFF].mean(axis=0)])


def static_features(packed, vec, kind, counts=True, clip_k=None):
    """Question features X float64 [Q, F] of one static kind (see the module docstring for notation).

    sum: [(L - mean_r L)/sqrt(n) (18), (L - mean_r L)/n (18), log m, log n]
    clip: [med + mean_g clip(r_g - med, -k, k) (18), log m, log n], med = per-profile median_g r_g,
          k = clip_k (from fit_clip_k on training questions only)
    trim: [per-profile 20% trimmed mean of r_g, floor(0.2 m) games off each side (18), log m, log n]
    q24: qfeatures.question_features of the question's games, identical to qfeatures.build_matrix
    pooled: [mean token over the valid games (D + 1), qglob (6), q24 (24)]
    length: [log m, log n, mean and std of n_g, per-stage present fraction (4), per-stage mean n_eff (4)]
    counts=False: sum/clip/trim drop log n, q24 drops log_moves, pooled zeroes the token count columns
    and qglob[:, 1:] and uses the reduced q24; length is refused.
    """
    idx = packed["idx"]
    if kind == "q24":
        return _q24(packed, vec, counts)
    if kind == "pooled":
        return _pooled(packed, vec, counts)
    if kind == "length":
        if not counts:
            raise ValueError("the length-only kind is made of counts; it has no counts=False form")
        return np.stack([_length(vec, ix) for ix in idx])
    if kind == "sum":
        return np.stack([np.concatenate([_sum(vec, ix), _evidence(vec, ix, counts)]) for ix in idx])
    if kind in ("clip", "trim"):
        if kind == "clip" and clip_k is None:
            raise ValueError("kind 'clip' needs clip_k (fit_clip_k on the training questions)")
        return np.stack([np.concatenate([_robust(vec, ix, kind, clip_k), _evidence(vec, ix, counts)]) for ix in idx])
    raise ValueError(f"unknown static feature kind {kind!r}")


def fit_clip_k(packed, vec):
    """Clip width k: the 95th percentile of |r_g - med| over every game and profile of the (training) questions."""
    dev = []
    for ix in packed["idx"]:
        r = _profile_shape(vec, ix)
        dev.append(np.abs(r - np.median(r, axis=0)).ravel())
    return float(np.quantile(np.concatenate(dev), CLIP_QUANTILE))


# rung -> (static feature kind, vector rep, zero-argument model factory)
STATIC_RUNGS = {
    "r0_sum": ("sum", "staged", lambda: SoftmaxModel(l2=1e-3)),
    "r1_clip": ("clip", "staged", lambda: SoftmaxModel(l2=1e-3)),
    "r1_trim": ("trim", "staged", lambda: SoftmaxModel(l2=1e-3)),
    "r2_mlp24": ("q24", "staged", lambda: MLPModel(hidden=32)),  # the current production model
    "r2_staged": ("pooled", "staged", lambda: MLPModel(hidden=64)),
    "r2_whole": ("pooled", "whole", lambda: MLPModel(hidden=64)),
    "r2_length_only": ("length", "staged", lambda: MLPModel(hidden=32)),
}
