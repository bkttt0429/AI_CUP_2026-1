"""Per-game move sequences from scored folds (issue #19: is there signal in the order of the moves?).

A scored fold holds, per part, a games table and per-move arrays (logp [M, S], game_row, t). fold_sequences merges, per
game, the target player's moves (18 Human SL log p from the results/scores fold + 10 L1b log p from an l1b_score.py fold)
and the opponent's moves (10 L1b log p from an l1b_score.py --opponent fold) in move order into one step sequence:
STEP_DIM = 18 + 10 + 3 columns -- hsl (zero on the opponent's steps), l1b, is_target, is_opponent, t / T_SCALE. The target's
summed L1b log p (`base`, the l1b_score csv sums) is what the submission's features use today; a sequence encoder adds a
correction to it (aicup.seqmodel).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.humansl import PROFILES
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.scoring import completed_parts, part_paths

N_HSL, N_L1B = len(PROFILES), len(LEVELS)
STEP_DIM = N_HSL + N_L1B + 3
COL_TARGET, COL_OPP, COL_T = N_HSL + N_L1B, N_HSL + N_L1B + 1, N_HSL + N_L1B + 2
T_SCALE = 300.0
STEP_NAMES = [f"hsl_{p}" for p in PROFILES] + [f"l1b_{lv}" for lv in LEVELS] + ["is_target", "is_opponent", "t_norm"]
SUM_TOL = 0.05  # the per-move sums must reproduce the csv sums (bf16 batching noise is ~1e-3 per game)


def read_moves(fold_dir):
    """(games table, {game_key: (t [n] int64, logp [n, S] float32)}) of every finished part of one scored fold."""
    frames, out = [], {}
    parts = completed_parts(fold_dir)
    if not parts:
        raise FileNotFoundError(f"no finished parts in {fold_dir}")
    for shard, k in parts:
        csv, npz = part_paths(fold_dir, shard, k)
        games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
        with np.load(npz, allow_pickle=False) as z:
            logp, row, t = z["logp"].astype(np.float32), z["game_row"].astype(np.int64), z["t"].astype(np.int64)
        if not np.isfinite(logp).all():
            raise ValueError(f"{npz}: non-finite log p")
        order = np.argsort(row, kind="stable")
        logp, row, t = logp[order], row[order], t[order]
        bounds = np.searchsorted(row, np.arange(len(games) + 1))
        for g, key in enumerate(games["game_key"]):
            sl = slice(bounds[g], bounds[g + 1])
            out[key] = (t[sl], logp[sl])
        frames.append(games)
    games = pd.concat(frames, ignore_index=True)
    if not games["game_key"].is_unique:
        raise ValueError(f"{fold_dir}: games scored more than once")
    return games, out


def fold_sequences(hsl_dir, l1b_dir, opp_dir, fold):
    """The step sequences of one fold's games.

    Returns {"game_key" [G], "player_id" [G], "rank" [G] (str), "rank_idx" [G] int64, "color" [G], "n_moves" [G] (the
    target's), "n_steps" [G], "offsets" [G + 1], "X" [sum n_steps, STEP_DIM] float32, "base" [G, 10] float64 (the target's
    summed L1b log p, checked against the csv sums), "games" (the L1b games table)}. The three folds must hold the same
    games; the Human SL and L1b arrays must describe the same target moves. opp_dir None: target steps only."""
    hsl_games, hsl = read_moves(Path(hsl_dir) / f"fold{fold}")
    l1b_games, l1b = read_moves(Path(l1b_dir) / f"fold{fold}")
    opp = read_moves(Path(opp_dir) / f"fold{fold}")[1] if opp_dir else None
    keys = l1b_games["game_key"].tolist()
    if set(keys) != set(hsl_games["game_key"]):
        raise ValueError(f"fold {fold}: {hsl_dir} and {l1b_dir} score different games")
    if opp is not None and set(keys) != set(opp):
        raise ValueError(f"fold {fold}: {opp_dir} covers other games than {l1b_dir}")
    csv_sums = l1b_games[[f"sum_{lv}" for lv in LEVELS]].to_numpy(dtype=float)
    blocks, n_steps, base = [], np.zeros(len(keys), dtype=np.int64), np.zeros((len(keys), N_L1B))
    for g, key in enumerate(keys):
        th, lh = hsl[key]
        tl, ll = l1b[key]
        if len(th) != len(tl) or not np.array_equal(th, tl):
            raise ValueError(f"fold {fold}: game {key}: Human SL and L1b arrays describe different moves")
        base[g] = ll.astype(np.float64).sum(axis=0)
        if not np.all(np.abs(base[g] - csv_sums[g]) <= SUM_TOL):
            raise ValueError(f"fold {fold}: game {key}: per-move L1b sums differ from the csv sums by up to {np.abs(base[g] - csv_sums[g]).max():.4f}")
        tgt = np.zeros((len(tl), STEP_DIM), dtype=np.float32)
        tgt[:, :N_HSL], tgt[:, N_HSL:N_HSL + N_L1B], tgt[:, COL_TARGET], tgt[:, COL_T] = lh, ll, 1.0, tl / T_SCALE
        if opp is not None:
            to, lo = opp[key]
            op = np.zeros((len(to), STEP_DIM), dtype=np.float32)
            op[:, N_HSL:N_HSL + N_L1B], op[:, COL_OPP], op[:, COL_T] = lo, 1.0, to / T_SCALE
            steps = np.concatenate([tgt, op])
            steps = steps[np.argsort(np.concatenate([tl, to]), kind="stable")]
        else:
            steps = tgt[np.argsort(tl, kind="stable")]
        blocks.append(steps)
        n_steps[g] = len(steps)
    offsets = np.concatenate([[0], np.cumsum(n_steps)])
    X = np.concatenate(blocks) if blocks else np.zeros((0, STEP_DIM), dtype=np.float32)
    return {"game_key": np.asarray(keys, dtype=str), "player_id": l1b_games["player_id"].to_numpy(dtype=str),
            "rank": l1b_games["rank"].to_numpy(dtype=str), "rank_idx": l1b_games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64),
            "color": l1b_games["color"].to_numpy(dtype=str), "n_moves": l1b_games["n_moves"].to_numpy(dtype=np.int64),
            "n_steps": n_steps, "offsets": offsets, "X": X, "base": base, "games": l1b_games}


def concat_sequences(parts):
    """Several fold_sequences results as one (offsets re-based)."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("game_key", "player_id", "rank", "rank_idx", "color", "n_moves", "n_steps", "X", "base")}
    out["offsets"] = np.concatenate([[0], np.cumsum(out["n_steps"])])
    out["games"] = pd.concat([p["games"] for p in parts], ignore_index=True)
    return out


def pad_batch(seq, idx):
    """Games `idx` of a sequences dict as padded arrays: X [B, T, D] float32, mask [B, T] bool, base [B, 10], y [B]."""
    idx = np.asarray(idx, dtype=np.int64)
    lengths = seq["n_steps"][idx]
    T = int(lengths.max()) if len(idx) else 1
    X = np.zeros((len(idx), T, STEP_DIM), dtype=np.float32)
    mask = np.zeros((len(idx), T), dtype=bool)
    for b, g in enumerate(idx):
        n = seq["n_steps"][g]
        X[b, :n] = seq["X"][seq["offsets"][g]:seq["offsets"][g] + n]
        mask[b, :n] = True
    return X, mask, seq["base"][idx], seq["rank_idx"][idx]
