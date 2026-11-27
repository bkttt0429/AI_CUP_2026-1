"""TSFEL features of the per-move score series of a game (issue #36: the user's "let TSFEL generate the features").

Each game gives two scalar series over its moves (aicup.seqshape): pmi_t, the posterior-mean rank index of the move's ten
L1b log p, and s_t, the surprisal of the move's best rank. TSFEL (Time Series Feature Extraction Library, 0.2.0) turns a
series into its statistical, temporal, spectral and fractal descriptors; game_tsfel runs it on both series and prefixes the
names with pmi_ / s_. The four wavelet features are left out (they break under pandas 3: duplicate column labels); games
shorter than MIN_MOVES, and any feature TSFEL cannot compute for a series, give NaN. Some TSFEL features have a length-
dependent number of outputs, so a fold's table takes the union of the names and NaN where a game has none; the CV side
standardises with the training questions' statistics and imputes NaN by the training mean (standardise_impute), then
reduces the block with a PCA fitted on the training questions only (pca_fit / pca_apply). fold_tsfel_table caches nothing
itself; experiments/tsfel_check.py writes the per-fold tables to disk because extraction costs about 11 ms per series.
"""
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd

from aicup.seqdata import read_moves
from aicup.seqshape import move_pmi

MIN_MOVES = 12  # TSFEL's LPCC / MFCC need at least this many samples; shorter games give NaN features
WAVELET_BROKEN = ("Wavelet absolute mean", "Wavelet energy", "Wavelet standard deviation", "Wavelet variance")
SERIES = ("pmi", "s")
_CFG = None


def tsfel_config():
    """TSFEL's full configuration minus WAVELET_BROKEN (cached)."""
    global _CFG
    if _CFG is None:
        import tsfel
        cfg = tsfel.get_features_by_domain()
        for name in WAVELET_BROKEN:
            cfg.get("spectral", {}).pop(name, None)
        _CFG = cfg
    return _CFG


def game_series(logp):
    """(pmi_t, s_t) of one game from its per-move L1b log p [n, 10]."""
    lp = np.asarray(logp, dtype=np.float64)
    if len(lp) == 0:
        return np.zeros(0), np.zeros(0)
    return move_pmi(lp), -lp.max(axis=1)


def series_features(x, prefix):
    """{prefix_<TSFEL name>: value} of one series; {} when it is too short or TSFEL fails on it."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < MIN_MOVES:
        return {}
    import tsfel
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = tsfel.time_series_features_extractor(tsfel_config(), pd.DataFrame({prefix: x}), fs=1, verbose=0)
    except Exception:  # noqa: BLE001 -- a feature that cannot be computed for this series is NaN, not a crash
        return {}
    row = df.iloc[0]
    return {str(k): float(v) for k, v in row.items() if np.isfinite(v)}


def game_tsfel(logp):
    """The TSFEL features of both series of one game (names prefixed pmi_ and s_)."""
    pmi, s = game_series(logp)
    out = series_features(pmi, "pmi")
    out.update(series_features(s, "s"))
    return out


def _one(args):
    key, logp = args
    return key, game_tsfel(logp)


def fold_tsfel_table(fold_dir, workers=1):
    """game_key, player_id, rank, n_moves and the union of the games' TSFEL features (NaN where a game has none) of one
    scored fold (an l1b_score.py fold directory)."""
    games, moves = read_moves(fold_dir)
    keys = games["game_key"].tolist()
    tasks = [(k, moves[k][1]) for k in keys]
    if workers and workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            rows = dict(pool.imap_unordered(_one, tasks, chunksize=64))
    else:
        rows = dict(map(_one, tasks))
    feats = pd.DataFrame([rows[k] for k in keys])
    feats = feats.reindex(sorted(feats.columns), axis=1)
    out = pd.concat([games[["game_key", "player_id", "rank", "n_moves"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out


def save_tsfel_table(path, table):
    """One fold's table as an npz (keys, player_id, rank, n_moves, cols, X); npz files under results/seq are not committed."""
    cols = feature_columns(table)
    np.savez_compressed(path, keys=table["game_key"].to_numpy(dtype=str), player_id=table["player_id"].to_numpy(dtype=str),
                        rank=table["rank"].to_numpy(dtype=str), n_moves=table["n_moves"].to_numpy(dtype=np.int64),
                        cols=np.asarray(cols, dtype=str), X=table[cols].to_numpy(dtype=np.float32))


def load_tsfel_table(path):
    with np.load(path, allow_pickle=False) as z:
        cols = [str(c) for c in z["cols"]]
        out = pd.DataFrame({"game_key": z["keys"].astype(str), "player_id": z["player_id"].astype(str), "rank": z["rank"].astype(str), "n_moves": z["n_moves"]})
        feats = pd.DataFrame(z["X"].astype(np.float64), columns=cols)
    return pd.concat([out, feats], axis=1)


def feature_columns(table):
    """The TSFEL columns of a fold_tsfel_table (everything after the four game columns)."""
    return [c for c in table.columns if c not in ("game_key", "player_id", "rank", "n_moves")]


def question_tsfel(table, questions, cols):
    """X [Q, k]: the nanmean of `cols` over each question's games (NaN when no game of the question has the feature)."""
    g = table.set_index("game_key")
    vals = g.reindex(columns=list(cols)).to_numpy(dtype=float)
    row = {k: i for i, k in enumerate(g.index)}
    out = np.full((len(questions), len(cols)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for q, keys in enumerate(questions["game_keys"]):
            out[q] = np.nanmean(vals[[row[key] for key in keys]], axis=0)
    return out


def standardise_impute(X_tr, X_va):
    """Standardise both with the training nan-mean / nan-std, impute NaN by 0 (the training mean) and drop the columns that
    are constant or all-NaN in training; returns (Z_tr, Z_va, kept column indices)."""
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0)
    keep = np.where(np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12))[0]
    mu, sd = mu[keep], sd[keep]

    def z(X):
        Z = (X[:, keep] - mu) / sd
        return np.where(np.isfinite(Z), Z, 0.0)

    return z(X_tr), z(X_va), keep


def pca_fit(Z, k):
    """Principal directions [d, k'] of the standardised training block (k' = min(k, rank)) by SVD of the centred data."""
    Zc = Z - Z.mean(axis=0)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    r = int((s > 1e-9 * max(s[0], 1e-300)).sum()) if len(s) else 0
    k = max(0, min(int(k), r))
    return {"mean": Z.mean(axis=0), "components": vt[:k].T, "explained": (s[:k] ** 2 / max((s ** 2).sum(), 1e-300)).tolist()}


def pca_apply(pca, Z):
    return (Z - pca["mean"]) @ pca["components"]
