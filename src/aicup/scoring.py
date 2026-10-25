"""Shared Human SL scoring machinery: game loading, CPU workers, streaming GPU scoring,
per-fold game selection and resumable part bookkeeping.

Used by the pilot, the full-fold scoring run and the submission pipeline, so every
log-likelihood in the project comes from the same code path.
"""
import re
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


def score_stream(pool, model, tasks, rows, flush_at=1024, log_every=200, log=print, scorer=None):
    """Score (idx, sgf, colour) tasks under every metadata row.

    The default scorer is humansl.score_positions(model, ...) under every row of `rows`; `scorer`
    replaces it: scorer(bins, globs, legal, target, colors, rows) -> [N, S] (the L1b heads pass one
    that ignores `model` and `rows`). Returns per-task {idx: (n_moves, sums[S])}, stacked per-move
    arrays (logp [M, S], task idx [M], move index t [M]) or None, number of positions, wall seconds.
    """
    if scorer is None:
        from aicup import humansl

        def scorer(bins, globs, legal, target, colors, rows_):
            return humansl.score_positions(model, bins, globs, legal, target, colors, rows_)

    per_task, move_logp, move_task, move_t = {}, [], [], []
    buf, t0, done, n_pos = [], time.time(), 0, 0

    def flush():
        nonlocal buf
        if not buf:
            return
        cat = {k: np.concatenate([p[k] for _, p in buf]) for k in ("bins", "globs", "legal", "target", "color")}
        logp = scorer(cat["bins"], cat["globs"], cat["legal"], cat["target"], cat["color"], rows)
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


_PART_RE = re.compile(r"(games|moves)_s(\d+)_p(\d+)\.(csv|npz)")


def _part_files(out_dir, shard=None):
    """{(shard, index): kinds present} for final part files (temporary files never match)."""
    found = {}
    for p in Path(out_dir).glob("*_s*_p*.*"):
        m = _PART_RE.fullmatch(p.name)
        if m and (shard is None or int(m.group(2)) == shard):
            found.setdefault((int(m.group(2)), int(m.group(3))), set()).add(m.group(1))
    return found


def completed_parts(out_dir, shard=None):
    """(shard, index) of parts whose CSV and npz both exist, sorted."""
    return sorted(k for k, kinds in _part_files(out_dir, shard).items() if kinds == {"games", "moves"})


def done_game_keys(out_dir, shard):
    done = set()
    for s, k in completed_parts(out_dir, shard):
        done |= set(pd.read_csv(part_paths(out_dir, s, k)[0], usecols=["game_key"])["game_key"])
    return done


def next_part_index(out_dir, shard):
    """First part index not used by any file of this shard, finished or not."""
    used = [k for _, k in _part_files(out_dir, shard)]
    return max(used) + 1 if used else 0


def resume_plan(keys, out_dir, shard, part_size):
    """[(part_index, keys)] covering every key not already in a finished part, in order.

    Resume is decided per game, not per part number, so the part size may change between
    runs and no game is ever scored twice; new parts are numbered after any existing file.
    """
    done = done_game_keys(out_dir, shard)
    stray = done - set(keys)
    if stray:
        raise ValueError(f"{len(stray)} finished games are not in this selection, e.g. {sorted(stray)[:3]}")
    pending = [k for k in keys if k not in done]
    start = next_part_index(out_dir, shard)
    return [(start + j, pending[i:i + part_size]) for j, i in enumerate(range(0, len(pending), part_size))]


def pause_requested(path):
    return Path(path).exists()


def read_scored_games(fold_dir):
    """All finished parts of one fold (every shard) as one table; each game must appear once."""
    paths = [part_paths(fold_dir, s, k)[0] for s, k in completed_parts(fold_dir)]
    if not paths:
        raise FileNotFoundError(f"no finished parts in {fold_dir}")
    str_cols = {"game_key": str, "player_id": str, "rank": str, "color": str}  # an all-digit hex player id must stay a string
    games = pd.concat([pd.read_csv(p, dtype=str_cols, keep_default_na=False) for p in paths], ignore_index=True)
    if not games["game_key"].is_unique:
        raise ValueError(f"{fold_dir}: {int(games['game_key'].duplicated().sum())} games scored more than once")
    return games


def compare_scored_games(new, ref):
    """Per-game log p sums of `new` against `ref` (same game_key), e.g. one host against another.

    Differences are absolute, per profile; the per-move figures divide by the reference move
    count. best_profile_agreement is the share of games whose highest-scoring profile is the same.
    """
    cols = [c for c in new.columns if c.startswith("sum_")]
    m = new.merge(ref[["game_key", "n_moves", *cols]], on="game_key", how="left", suffixes=("", "_ref"), indicator=True)
    missing = int((m["_merge"] == "left_only").sum())
    m = m[m["_merge"] == "both"]
    if m.empty:
        return {"games": 0, "missing_in_ref": missing}
    a, b = m[cols].to_numpy(dtype=float), m[[f"{c}_ref" for c in cols]].to_numpy(dtype=float)
    diff = np.abs(a - b)
    per_move = diff / m["n_moves_ref"].to_numpy(dtype=float)[:, None]
    return {"games": int(len(m)), "missing_in_ref": missing,
            "n_moves_mismatch": int((m["n_moves"] != m["n_moves_ref"]).sum()),
            "max_abs_sum": float(diff.max()), "max_abs_per_move": float(per_move.max()),
            "mean_abs_per_move": float(per_move.mean()),
            "best_profile_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean())}
