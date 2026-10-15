"""Shared Human SL scoring machinery: game loading, CPU workers, streaming GPU scoring,
per-fold game selection and resumable part bookkeeping.

Used by the pilot, the full-fold scoring run and the submission pipeline, so every
log-likelihood in the project comes from the same code path.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.pilot import game_positions

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "training"
_FEATURES = None


def init_worker(config):
    """Pool initializer: build KataGo Features once per worker process."""
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def extract(task):
    idx, sgf, color = task
    return idx, game_positions(sgf, color, _FEATURES)


def load_games(keys, train_dir=TRAIN_DIR):
    """sgf and colour for each 'RANK:game_id' key, reading only the rank files needed."""
    by_rank = {}
    for k in keys:
        r, g = k.split(":")
        by_rank.setdefault(r, set()).add(g)
    out = {}
    for r, ids in by_rank.items():
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["game_id", "color", "sgf_content"])
        for row in df[df["game_id"].isin(ids)].itertuples():
            out[f"{r}:{row.game_id}"] = (row.sgf_content, row.color)
    missing = [k for k in keys if k not in out]
    if missing:
        raise KeyError(f"{len(missing)} game keys not found, e.g. {missing[:3]}")
    return out


def score_stream(pool, model, tasks, rows, flush_at=1024, log_every=200, log=print):
    """Score (idx, sgf, colour) tasks under every metadata row.

    Returns per-task {idx: (n_moves, sums[S])}, stacked per-move arrays (logp [M, S],
    task idx [M], move index t [M]) or None, number of positions, wall seconds.
    """
    from aicup.humansl import score_positions

    per_task, move_logp, move_task, move_t = {}, [], [], []
    buf, t0, done, n_pos = [], time.time(), 0, 0

    def flush():
        nonlocal buf
        if not buf:
            return
        cat = {k: np.concatenate([p[k] for _, p in buf]) for k in ("bins", "globs", "legal", "target", "color")}
        logp = score_positions(model, cat["bins"], cat["globs"], cat["legal"], cat["target"], cat["color"], rows)
        o = 0
        for idx, p in buf:
            n = len(p["t"])
            part = logp[o:o + n]
            o += n
            per_task[idx] = (n, part.sum(axis=0))
            move_logp.append(part)
            move_task.append(np.full(n, idx, dtype=np.int32))
            move_t.append(p["t"])
        buf = []

    for idx, pos in pool.imap(extract, tasks, chunksize=1):
        buf.append((idx, pos))
        n_pos += len(pos["t"])
        done += 1
        if sum(len(p["t"]) for _, p in buf) >= flush_at:
            flush()
        if done % log_every == 0:
            log(f"  {done}/{len(tasks)} games, {n_pos} positions, {n_pos / (time.time() - t0):.0f} positions/s")
    flush()
    arrays = (np.concatenate(move_logp), np.concatenate(move_task), np.concatenate(move_t)) if move_logp else None
    return per_task, arrays, n_pos, time.time() - t0


def player_games(train_dir=TRAIN_DIR):
    """All (player_id, rank, game_key) rows of the training set."""
    frames = []
    for p in sorted(Path(train_dir).glob("train_*.csv")):
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank"])
        df["game_key"] = df["rank"] + ":" + df["game_id"]
        frames.append(df[["player_id", "rank", "game_key"]])
    return pd.concat(frames, ignore_index=True)


def fold_players(games, folds, fold, max_games=20, min_games=5, seed=42):
    """Players of one fold with >= min_games games and up to max_games of them each.

    Each player's keys are sorted, then permuted with a seed derived from the player id,
    so a player's selection does not depend on row order or on any other player.
    """
    by_player = games.groupby("player_id")["game_key"].apply(sorted).to_dict()
    rows = []
    for r in folds[folds["fold"] == fold].sort_values(["rank", "player_id"]).itertuples():
        keys = by_player.get(r.player_id, [])
        if len(keys) < min_games:
            continue
        rng = np.random.RandomState((int(r.player_id[:8], 16) + seed) % 2**32)
        rows.append({"rank": r.rank, "player_id": r.player_id,
                     "game_keys": [keys[i] for i in rng.permutation(len(keys))[:max_games]]})
    return pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])


def plan_parts(n_items, part_size):
    return [(i, min(n_items, i + part_size)) for i in range(0, n_items, part_size)]


def part_paths(out_dir, shard, k):
    out_dir = Path(out_dir)
    return out_dir / f"games_s{shard}_p{k:03d}.csv", out_dir / f"moves_s{shard}_p{k:03d}.npz"


def pending_parts(out_dir, shard, parts):
    """Indices of parts whose CSV or npz output is missing."""
    return [k for k in range(len(parts)) if not all(p.exists() for p in part_paths(out_dir, shard, k))]
