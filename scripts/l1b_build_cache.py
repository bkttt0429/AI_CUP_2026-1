#!/usr/bin/env python
"""Build the L1b training cache: every target-player position of the chosen folds' players, bit-packed.

Plan 2026-10-25-l1b-rank-heads, Task 3. Players are every player of --folds with >= --min-games games
(default 1: all of them), each with up to --max-games games chosen by scoring.fold_players (a fixed
per-player permutation, seed 42, so the first 20 are the evaluation games of results/scores). The games
are shuffled once (RandomState(42)) so every part mixes ranks; shard i of N takes keys[i::N] and writes
its parts (featcache.write_part, --part-size games each) to --out, skipping parts it already finished.
Positions come from pilot.game_positions in a CPU pool (features need only the model config's input
version; --model-config b18c384nbt is humanv0's and the default, no checkpoint is read).

After its parts a shard writes <out>/meta_s{shard}.json from its finished parts' games tables; a single
process also writes <out>/meta.json. With several shards run `--out DIR --merge-meta` once at the end to
combine them into meta.json (folds, max_games, min_games, players, games, positions, per_rank_positions,
games_sha256, provenance).

  python -I scripts/l1b_build_cache.py --folds 1 2 --out results/l1b/cache_A --workers 48
  bash scripts/nano4/shards.sh 8 python -I scripts/l1b_build_cache.py --folds 3 4 --out results/l1b/cache_B --workers 12
  python -I scripts/l1b_build_cache.py --out results/l1b/cache_B --merge-meta

--opponent (issue #33, the opponent-head route): the positions before the OPPONENT's moves (the other colour) of the same
games, labelled with the target player's rank -- the training data of heads that read the labelled player's rank off the
opponent's play. The games table's color column is the side extracted (as in l1b_score.py --opponent) and the meta
records side = "opponent"; the trainer carries it into the heads' sidecar.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.featcache import completed_parts, pack_positions, part_paths, plan_parts, write_part  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import extract, fold_players, init_worker, load_games, player_games  # noqa: E402

SHUFFLE_SEED = 42
DEFAULT_MODEL_CONFIG = "b18c384nbt"  # humanv0's architecture name: the same input features, no checkpoint needed


def select_games(folds, max_games, min_games, folds_csv=FOLDS_CSV):
    """DataFrame (game_key, player_id, rank) of the chosen games of every player of `folds`, shuffled with
    RandomState(SHUFFLE_SEED) after a (rank, player_id) sort; and the number of players."""
    games, folds_df = player_games(), pd.read_csv(folds_csv)
    players = pd.concat([fold_players(games, folds_df, f, max_games=max_games, min_games=min_games) for f in sorted(folds)],
                        ignore_index=True).sort_values(["rank", "player_id"], kind="stable")
    rows = [{"game_key": k, "player_id": p.player_id, "rank": p.rank} for p in players.itertuples() for k in p.game_keys]
    table = pd.DataFrame(rows, columns=["game_key", "player_id", "rank"])
    perm = np.random.RandomState(SHUFFLE_SEED).permutation(len(table))
    return table.iloc[perm].reset_index(drop=True), int(len(players))


def games_sha256(table):
    h = hashlib.sha256()
    for r in table.itertuples():
        h.update(f"{r.game_key}:{r.player_id}:{r.rank}\n".encode())
    return h.hexdigest()


def other_colour(c):
    return "W" if c == "B" else "B"


def build_parts(table, out, shard, nshards, part_size, workers, config, log=print, opponent=False):
    """Write the unfinished parts of this shard; returns the number of parts written. opponent: extract the positions before
    the other colour's moves instead of the target player's (the table's color column is the side extracted)."""
    mine = table.iloc[shard::nshards].reset_index(drop=True)
    keys = mine["game_key"].tolist()
    info = {r.game_key: (r.player_id, r.rank) for r in mine.itertuples()}
    todo = plan_parts(keys, part_size, out, shard)
    n_parts = -(-len(keys) // part_size)
    log(f"shard {shard} of {nshards}: {len(keys)} games in {n_parts} parts, {len(todo)} to do", flush=True)
    if not todo:
        log("nothing to do", flush=True)
        return 0
    sgfs = load_games([k for _, ks in todo for k in ks])
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=init_worker, initargs=(config,)) as pool:
        for k, part_keys in todo:
            t0 = time.time()
            side = {key: (other_colour(sgfs[key][1]) if opponent else sgfs[key][1]) for key in part_keys}
            tasks = [(i, sgfs[key][0], side[key]) for i, key in enumerate(part_keys)]
            got = dict(pool.imap_unordered(extract, tasks))
            parts = [pack_positions(got[i]) for i in range(len(part_keys))]
            games = pd.DataFrame({"game_key": part_keys, "player_id": [info[k_][0] for k_ in part_keys],
                                  "rank": [info[k_][1] for k_ in part_keys], "color": [side[k_] for k_ in part_keys],
                                  "n_moves": [len(p["target"]) for p in parts]})
            write_part(out, shard, k, games, parts)
            log(f"part {k}: {len(part_keys)} games, {int(games['n_moves'].sum())} positions, {time.time() - t0:.0f} s", flush=True)
    return len(todo)


def shard_meta(out, shard, table, folds, max_games, min_games, nshards, n_players, opponent=False):
    """Counts of this shard's finished parts from their games tables (no npz is read); side = the colour extracted."""
    frames = [pd.read_csv(part_paths(out, s, k)[1], dtype=str, keep_default_na=False) for s, k in completed_parts(out) if s == shard]
    games = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["game_key", "rank", "n_moves"])
    n_moves = games["n_moves"].astype(int) if len(games) else pd.Series(dtype=int)
    per_rank = {lv: int(n_moves[games["rank"] == lv].sum()) if len(games) else 0 for lv in LEVELS}
    return {"folds": [int(f) for f in sorted(folds)], "max_games": int(max_games), "min_games": int(min_games), "nshards": int(nshards),
            "shards": [int(shard)], "players": int(n_players), "games_selected": int(len(table)), "games": int(len(games)),
            "positions": int(n_moves.sum()), "per_rank_positions": per_rank, "games_sha256": games_sha256(table),
            "side": "opponent" if opponent else "target", "model_input_config": None, **environment("cpu")}


def merge_meta(out):
    """meta.json from every meta_s*.json of `out`; ValueError when they disagree on the selection."""
    metas = [json.loads(p.read_text()) for p in sorted(Path(out).glob("meta_s*.json"))]
    if not metas:
        raise FileNotFoundError(f"no meta_s*.json in {out}")
    for key in ("folds", "max_games", "min_games", "nshards", "players", "games_selected", "games_sha256"):
        if len({json.dumps(m[key]) for m in metas}) != 1:
            raise ValueError(f"the shard metas disagree on {key}")
    if len({m.get("side", "target") for m in metas}) != 1:
        raise ValueError("the shard metas disagree on side (target / opponent)")
    merged = {**metas[0], "shards": sorted(s for m in metas for s in m["shards"]), "games": sum(m["games"] for m in metas),
              "positions": sum(m["positions"] for m in metas),
              "per_rank_positions": {lv: sum(m["per_rank_positions"][lv] for m in metas) for lv in LEVELS}}
    if len(merged["shards"]) != merged["nshards"]:
        print(f"WARNING: {len(merged['shards'])} of {merged['nshards']} shard metas present", file=sys.stderr)
    (Path(out) / "meta.json").write_text(to_json(merged, indent=1) + "\n")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=40)
    ap.add_argument("--min-games", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--part-size", type=int, default=200)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="KataGo config name (katago.train.modelconfigs) for the input features")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--merge-meta", action="store_true", help="only combine <out>/meta_s*.json into meta.json")
    ap.add_argument("--opponent", action="store_true", help="the positions before the opponent's moves (the other colour), labelled with the target player's rank")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.merge_meta:
        m = merge_meta(out)
        print(f"wrote {out / 'meta.json'}: {m['games']} games, {m['positions']} positions, shards {m['shards']}")
        return 0
    if not a.folds:
        ap.error("--folds is required (unless --merge-meta)")
    if a.nshards > 1 and a.shard is None:
        ap.error("--nshards > 1 needs --shard i")
    shard = a.shard or 0
    from katago.train import modelconfigs
    config = modelconfigs.config_of_name[a.model_config]
    table, n_players = select_games(a.folds, a.max_games, a.min_games, a.folds_csv)
    print(f"{n_players} players, {len(table)} games over folds {sorted(a.folds)} (<= {a.max_games} per player)"
          + ("; the OPPONENT's positions, labelled with the target player's rank" if a.opponent else ""), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    build_parts(table, out, shard, a.nshards, a.part_size, a.workers, config, opponent=a.opponent)
    meta = {**shard_meta(out, shard, table, a.folds, a.max_games, a.min_games, a.nshards, n_players, opponent=a.opponent), "model_input_config": a.model_config}
    (out / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")
    if a.nshards == 1:
        merged = merge_meta(out)
        print(f"wrote {out / 'meta.json'}: {merged['games']} games, {merged['positions']} positions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
