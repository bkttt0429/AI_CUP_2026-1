"""Per-game staged evidence vectors: the input of the L3 aggregator ladder.

A game's per-move log-likelihoods ll [moves, 18] (one column per Human SL profile) are
reduced to a fixed-length vector per game stage. Within a stage, every move is made relative
to its profile mean, d = ll - mean_r ll, so the vector keeps the *shape* of the evidence over
profiles and drops the move's overall level l = mean_r ll; the level survives only as the
stage's mean surprisal. There is no unit-length normalisation.

Stages are defined on the game's move number t + 1 (t is the 0-based index in the game, the
target colour's moves only). Each stage contributes SEG_DIM = 75 fields:
  d_mean 18 | d_std 18 (ddof=0) | d_q10 18 | d_q90 18 (linear quantiles) | n_eff | mean_surprisal | present
n_eff is the number of target moves in the stage, mean_surprisal = -mean(l), present = 1 when
the stage has any move. A stage without moves is all zeros with present = 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.humansl import PROFILES
from aicup.qfeatures import SUM_COLS
from aicup.scoring import completed_parts, part_paths

SEGMENTS = ((1, 40), (41, 100), (101, 180), (181, None))  # inclusive move-number ranges; None = no upper bound
WHOLE = ((1, None),)
SEG_DIM = 4 * len(PROFILES) + 3
CSV_TOL = 1e-2  # max |sum of per-move ll - CSV sum| per game and profile (the CSV sums are float32 sums)


def _segment_block(ll):
    """The SEG_DIM fields of one stage from its per-move ll [n >= 1, 18] (float64)."""
    level = ll.mean(axis=1)
    d = ll - level[:, None]
    q10, q90 = np.quantile(d, [0.1, 0.9], axis=0, method="linear")
    return np.concatenate([d.mean(axis=0), d.std(axis=0, ddof=0), q10, q90, [len(ll), -level.mean(), 1.0]])


def game_vector(logp, t, segments=SEGMENTS):
    """logp [n, 18] and 0-based move index t [n] -> float32 vector of SEG_DIM * len(segments) fields."""
    ll = np.asarray(logp, dtype=np.float64)
    number = np.asarray(t, dtype=np.int64) + 1
    if ll.ndim != 2 or ll.shape[1] != len(PROFILES) or number.shape != (len(ll),):
        raise ValueError(f"logp {ll.shape} and t {number.shape} do not describe n moves x {len(PROFILES)} profiles")
    out = np.zeros((len(segments), SEG_DIM))
    for i, (lo, hi) in enumerate(segments):
        inside = (number >= lo) if hi is None else (number >= lo) & (number <= hi)
        if inside.any():
            out[i] = _segment_block(ll[inside])
    return out.reshape(-1).astype(np.float32)


def vec_names(segments=SEGMENTS):
    """Field names aligned with game_vector: s1_mean_12k ... s4_present (prefix w_ for WHOLE)."""
    names = []
    for i in range(len(segments)):
        p = "w" if tuple(segments) == WHOLE else f"s{i + 1}"
        names += [f"{p}_{stat}_{prof}" for stat in ("mean", "std", "q10", "q90") for prof in PROFILES]
        names += [f"{p}_n_eff", f"{p}_mean_surprisal", f"{p}_present"]
    return names


def _part_vectors(games, moves, label):
    """Vectors of one finished part after checking its per-move scores against the CSV."""
    n = len(games)
    logp, row, t = moves["logp"], moves["game_row"].astype(np.int64), moves["t"]
    if not np.isfinite(logp).all():
        raise ValueError(f"{label}: non-finite log p in the per-move scores")
    if len(row) and (row.min() < 0 or row.max() >= n):
        raise ValueError(f"{label}: game_row outside the {n} games of the CSV")
    ll = logp.astype(np.float64)
    count = np.bincount(row, minlength=n)
    if not np.array_equal(count, games["n_moves"].to_numpy()):
        bad = int(np.flatnonzero(count != games["n_moves"].to_numpy())[0])
        raise ValueError(f"{label}: game {games['game_key'][bad]} has {count[bad]} scored moves, "
                         f"n_moves says {games['n_moves'][bad]}")
    csv_sums = games[SUM_COLS].to_numpy(dtype=float)
    sums = np.zeros((n, len(PROFILES)))
    np.add.at(sums, row, ll)
    off = ~(np.abs(sums - csv_sums) <= CSV_TOL)  # also catches NaN in the CSV
    if off.any():
        g, p = (int(i[0]) for i in np.nonzero(off))
        raise ValueError(f"{label}: {int(off.any(axis=1).sum())} games differ from the CSV sums, e.g. game "
                         f"{games['game_key'][g]} {SUM_COLS[p]}: per-move sum {sums[g, p]:.4f} vs CSV {csv_sums[g, p]:.4f}")
    order = np.argsort(row, kind="stable")
    ll, t = ll[order], t[order]
    bounds = np.searchsorted(row[order], np.arange(n + 1))
    staged = np.zeros((n, SEG_DIM * len(SEGMENTS)), dtype=np.float32)
    whole = np.zeros((n, SEG_DIM * len(WHOLE)), dtype=np.float32)
    for g in range(n):
        sl = slice(bounds[g], bounds[g + 1])
        staged[g] = game_vector(ll[sl], t[sl], SEGMENTS)
        whole[g] = game_vector(ll[sl], t[sl], WHOLE)
    return staged, whole, csv_sums


def fold_vectors(fold_dir):
    """Vectors of every game in the finished parts of one scored fold, in completed-part order.

    Returns game_key (fixed-width unicode), staged [G, 300], whole [G, 75] (float32), sums
    [G, 18] (float64, the CSV values, so question features equal qfeatures.build_matrix bit for
    bit) and n_moves [G]. The per-move scores must reproduce the CSV sums and move counts.
    """
    fold_dir = Path(fold_dir)
    parts = completed_parts(fold_dir)
    if not parts:
        raise FileNotFoundError(f"no finished parts in {fold_dir}")
    keys, staged, whole, sums, n_moves = [], [], [], [], []
    for shard, k in parts:
        csv_path, npz_path = part_paths(fold_dir, shard, k)
        games = pd.read_csv(csv_path)
        moves = dict(np.load(npz_path))  # read every array once
        s, w, csv_sums = _part_vectors(games, moves, csv_path.name)
        keys += games["game_key"].tolist()
        staged.append(s)
        whole.append(w)
        sums.append(csv_sums)
        n_moves.append(games["n_moves"].to_numpy(dtype=np.int64))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{fold_dir}: {len(keys) - len(set(keys))} games scored more than once")
    return {"game_key": np.asarray(keys, dtype=str), "staged": np.concatenate(staged), "whole": np.concatenate(whole),
            "sums": np.concatenate(sums), "n_moves": np.concatenate(n_moves)}
