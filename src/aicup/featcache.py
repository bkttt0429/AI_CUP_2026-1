"""Bit-packed cache of KataGo input positions for L1b training (plan 2026-10-25-l1b-rank-heads, Task 2).

Every position is the board before one target-player move, as pilot.game_positions builds it: 22 binary
input planes (packed to 993 bytes), 19 global features (float32), the legal-move mask (362 bits -> 46
bytes), the target move, the move index and the colour to move. Positions are stored in parts, each a
games table (game_key, player_id, rank, color, n_moves) and an npz of concatenated arrays with game_row
pointing at the table; parts are written to .tmp names and renamed (npz first), so a part counts as done
only when both files exist. read_cache concatenates every finished part with global game rows and the
rank index of each position (read_caches: several directories as one cache, optionally without the games
of given players); plan_parts lists the parts a shard still has to write.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.metric import RANK_TO_IDX

N_PLANES, N_GLOBAL, N_MOVES, BOARD = 22, 19, 362, 19 * 19
BIN_BYTES = -(-N_PLANES * BOARD // 8)  # 993
LEGAL_BYTES = -(-N_MOVES // 8)  # 46
ARRAYS = ("bins_packed", "legal_packed", "globs", "target", "t", "color")
_PART_RE = re.compile(r"cache_s(\d+)_p(\d+)\.npz$")


def pack_positions(pos):
    """game_positions' dict -> packed arrays: bins_packed uint8 [n, 993], legal_packed uint8 [n, 46], globs
    float32 [n, 19], target int16, t int16, color uint8."""
    bins = np.asarray(pos["bins"], dtype=np.uint8)
    n = len(bins)
    legal = np.asarray(pos["legal"], dtype=bool).reshape(n, N_MOVES)
    return {"bins_packed": np.packbits(bins.reshape(n, N_PLANES * BOARD), axis=1).reshape(n, BIN_BYTES),
            "legal_packed": np.packbits(legal, axis=1).reshape(n, LEGAL_BYTES),
            "globs": np.asarray(pos["globs"], dtype=np.float32).reshape(n, N_GLOBAL),
            "target": np.asarray(pos["target"], dtype=np.int16).reshape(n),
            "t": np.asarray(pos["t"], dtype=np.int16).reshape(n),
            "color": np.asarray(pos["color"], dtype=np.uint8).reshape(n)}


def unpack_bins(packed):
    """uint8 [n, 993] -> uint8 [n, 22, 19, 19]."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, BIN_BYTES)
    return np.unpackbits(packed, axis=1, count=N_PLANES * BOARD).reshape(len(packed), N_PLANES, BOARD // 19, 19)


def unpack_legal(packed):
    """uint8 [n, 46] -> bool [n, 362]."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1, LEGAL_BYTES)
    return np.unpackbits(packed, axis=1, count=N_MOVES).astype(bool)


def part_paths(cache_dir, shard, k):
    """(npz path, csv path) of part k of a shard."""
    cache_dir = Path(cache_dir)
    return cache_dir / f"cache_s{shard}_p{k:03d}.npz", cache_dir / f"games_s{shard}_p{k:03d}.csv"


def write_part(cache_dir, shard, k, games, parts):
    """Write one part: `games` (game_key, player_id, rank, color, n_moves) and the packed positions of each game
    in the same order. Both files go to .tmp names first, the npz is renamed before the csv."""
    games = pd.DataFrame(games)[["game_key", "player_id", "rank", "color", "n_moves"]].reset_index(drop=True)
    if len(games) != len(parts):
        raise ValueError(f"{len(games)} games but {len(parts)} position blocks")
    counts = [len(p["target"]) for p in parts]
    if games["n_moves"].tolist() != counts:
        raise ValueError(f"n_moves column {games['n_moves'].tolist()} differs from the positions per game {counts}")
    arrays = {key: (np.concatenate([p[key] for p in parts]) if parts else np.zeros((0,), dtype=np.uint8)) for key in ARRAYS}
    arrays["game_row"] = np.repeat(np.arange(len(games), dtype=np.int32), counts) if parts else np.zeros((0,), dtype=np.int32)
    if sum(counts) == 0:
        arrays = _empty_arrays()
    npz, csv = part_paths(cache_dir, shard, k)
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp, csv_tmp = npz.with_name(npz.name + ".tmp"), csv.with_name(csv.name + ".tmp")
    with open(npz_tmp, "wb") as fh:
        np.savez(fh, **arrays)
    games.to_csv(csv_tmp, index=False)
    npz_tmp.rename(npz)
    csv_tmp.rename(csv)


def _empty_arrays():
    return {"bins_packed": np.zeros((0, BIN_BYTES), dtype=np.uint8), "legal_packed": np.zeros((0, LEGAL_BYTES), dtype=np.uint8),
            "globs": np.zeros((0, N_GLOBAL), dtype=np.float32), "target": np.zeros((0,), dtype=np.int16),
            "t": np.zeros((0,), dtype=np.int16), "color": np.zeros((0,), dtype=np.uint8), "game_row": np.zeros((0,), dtype=np.int32)}


def completed_parts(cache_dir):
    """Sorted (shard, k) of every part whose npz and csv both exist (tmp files are not parts)."""
    out = []
    for p in Path(cache_dir).glob("cache_s*_p*.npz"):
        m = _PART_RE.search(p.name)
        if m and part_paths(cache_dir, int(m.group(1)), int(m.group(2)))[1].exists():
            out.append((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def plan_parts(keys, part_size, cache_dir, shard):
    """[(k, keys of part k)] of the parts a shard has not finished; part k holds keys[k*part_size:(k+1)*part_size]."""
    done = {k for s, k in completed_parts(cache_dir) if s == shard}
    return [(k, keys[k * part_size:(k + 1) * part_size]) for k in range(-(-len(keys) // part_size)) if k not in done]


def read_part(cache_dir, shard, k):
    """(games, arrays) of one finished part; game_row is local to the part."""
    npz, csv = part_paths(cache_dir, shard, k)
    games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
    with np.load(npz, allow_pickle=False) as z:
        arrays = {key: z[key] for key in (*ARRAYS, "game_row")}
    return games, arrays


def positions_by_game(arrays, n_games):
    """The part's positions split per game, in table order: [{key: rows of game g} for g in range(n_games)] (the rows of a
    game in their stored order), as write_part takes them."""
    row = np.asarray(arrays["game_row"], dtype=np.int64)
    order = np.argsort(row, kind="stable")
    bounds = np.searchsorted(row[order], np.arange(n_games + 1))
    return [{key: np.asarray(arrays[key])[order[bounds[g]:bounds[g + 1]]] for key in ARRAYS} for g in range(n_games)]


def _rows_sha256(blocks):
    h = hashlib.sha256()
    for b in blocks:
        for key in ARRAYS:
            h.update(np.ascontiguousarray(b[key]).tobytes())
    return h.hexdigest()


def patch_part(cache_dir, shard, k, new_positions):
    """Rewrite part k with the positions of the games in `new_positions` ({game_key: packed arrays of pack_positions})
    replaced and every other game's rows bit-identical (issue #42 (b) / #51: the HD >= 2 games replayed with their
    handicap stones and actual colours). The table keeps its rows and order; n_moves follows the new positions. Written
    as write_part writes (tmp, npz renamed before csv), then read back: the untouched games' rows must hash the same,
    otherwise RuntimeError. Returns {"games": n in the part, "patched": n replaced, "positions_before", "positions_after",
    "untouched_sha256"}; a part without any listed game is left alone (patched = 0, nothing written)."""
    games, arrays = read_part(cache_dir, shard, k)
    hit = [i for i, key in enumerate(games["game_key"]) if key in new_positions]
    before = int(len(arrays["target"]))
    if not hit:
        return {"games": int(len(games)), "patched": 0, "positions_before": before, "positions_after": before, "untouched_sha256": None}
    blocks = positions_by_game(arrays, len(games))
    untouched = [b for i, b in enumerate(blocks) if i not in set(hit)]
    sha = _rows_sha256(untouched)
    for i in hit:
        blocks[i] = {key: np.asarray(new_positions[games.at[i, "game_key"]][key]) for key in ARRAYS}
    games = games.copy()
    games["n_moves"] = [int(len(b["target"])) for b in blocks]
    write_part(cache_dir, shard, k, games, blocks)
    games2, arrays2 = read_part(cache_dir, shard, k)
    blocks2 = positions_by_game(arrays2, len(games2))
    if _rows_sha256([b for i, b in enumerate(blocks2) if i not in set(hit)]) != sha or games2["game_key"].tolist() != games["game_key"].tolist():
        raise RuntimeError(f"part s{shard} p{k}: the untouched games' rows changed on rewrite")
    return {"games": int(len(games)), "patched": len(hit), "positions_before": before, "positions_after": int(len(arrays2["target"])),
            "untouched_sha256": sha}


def read_cache(cache_dir, drop_players=None):
    """(games, arrays) of every finished part in (shard, k) order; game_row is global, rank_idx int8 per position.
    drop_players: player ids whose games (and positions) are left out, e.g. the players of a held-out fold
    (issue #16). FileNotFoundError when the directory has no finished part."""
    return read_caches([cache_dir], drop_players)


def read_caches(cache_dirs, drop_players=None):
    """read_cache over several cache directories in the given order, as one cache: game_row is global over all of
    them (issue #16: a model trained on folds 0-4 minus one reads cache_A, cache_B and cache_0 together). Every
    directory must have a finished part."""
    drop = set(map(str, drop_players)) if drop_players else None
    frames, blocks, offset = [], [], 0
    for cache_dir in cache_dirs:
        parts = completed_parts(cache_dir)
        if not parts:
            raise FileNotFoundError(f"no finished cache parts in {cache_dir}")
        for shard, k in parts:
            npz, csv = part_paths(cache_dir, shard, k)
            games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)
            with np.load(npz, allow_pickle=False) as z:
                arr = {key: z[key] for key in (*ARRAYS, "game_row")}
            arr["game_row"] = arr["game_row"].astype(np.int32)
            if drop:
                keep = ~games["player_id"].isin(drop).to_numpy()
                if not keep.all():
                    pos_keep = keep[arr["game_row"]] if len(arr["game_row"]) else np.zeros((0,), dtype=bool)
                    new_row = np.cumsum(keep) - 1  # new index of every kept game
                    arr = {key: arr[key][pos_keep] for key in arr}
                    arr["game_row"] = new_row[arr["game_row"]].astype(np.int32)
                    games = games[keep].reset_index(drop=True)
            arr["game_row"] = arr["game_row"] + offset
            offset += len(games)
            frames.append(games)
            blocks.append(arr)
    games = pd.concat(frames, ignore_index=True)
    arrays = {key: np.concatenate([b[key] for b in blocks]) for key in (*ARRAYS, "game_row")}
    rank_of_game = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int8)
    arrays["rank_idx"] = rank_of_game[arrays["game_row"]] if len(arrays["game_row"]) else np.zeros((0,), dtype=np.int8)
    return games, arrays
