#!/usr/bin/env python
"""Patch an L1b position cache in place: replay the listed games again and replace their positions (issue #51, R-full (a)).

Dataset v1.1 showed that the HD >= 2 games (3.6%) were cached on boards missing their handicap stones and with the
recorded (reversed) colours; aicup.sgf.setup_and_moves fixes the replay. This script rewrites only the parts of --cache
whose games table holds a game of --game-keys (scripts/hd_table.py's hd2_keys.csv): those games' positions come from
pilot.game_positions on the fixed replay (the table's color column is the side extracted, as the cache builder wrote it),
every other game's rows are carried over bit for bit (featcache.patch_part hashes them before and after), the table's
n_moves follows the new positions, and the part is written as the builder writes (tmp files, npz renamed before csv), so
an interruption leaves the old part and the next run continues. <cache>/patch_meta.json records every part done (the
resume list), the counts and provenance; meta.json / meta_s*.json get their positions and per_rank_positions recomputed
(games_sha256 hashes keys, players and ranks only, so it does not change) and a "patched" summary. Disk: one part's tmp
files (about 25 MB) at a time.

  python -I scripts/l1b_patch_cache.py --cache results/l1b/cache_A --game-keys results/data/hd2_keys.csv --workers 48
  python -I scripts/l1b_patch_cache.py --cache results/l1b/cache_B --game-keys results/data/hd2_keys.csv --dry-run
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.featcache import completed_parts, pack_positions, part_paths, patch_part  # noqa: E402
from aicup.ladder import environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import extract, init_worker, load_games, read_game_keys  # noqa: E402

DEFAULT_MODEL_CONFIG = "b18c384nbt"
META_NAME = "patch_meta.json"


def parts_to_patch(cache_dir, keys):
    """[(shard, k, [game keys of the part that are in `keys`])] over the finished parts, from the games tables only."""
    out = []
    for s, k in completed_parts(cache_dir):
        table = pd.read_csv(part_paths(cache_dir, s, k)[1], usecols=["game_key"], dtype=str, keep_default_na=False)
        hit = [g for g in table["game_key"] if g in keys]
        if hit:
            out.append((s, k, hit))
    return out


def load_patch_meta(cache_dir):
    p = Path(cache_dir) / META_NAME
    return json.loads(p.read_text()) if p.exists() else {"parts": {}}


def save_patch_meta(cache_dir, meta):
    (Path(cache_dir) / META_NAME).write_text(to_json(meta, indent=1) + "\n")


def recount_meta(cache_dir, patched):
    """Recompute positions / per_rank_positions of meta_s*.json (its shards' parts) and meta.json (all parts) from the games
    tables and record the patch; returns the number of meta files rewritten."""
    cache_dir = Path(cache_dir)
    tables = {(s, k): pd.read_csv(part_paths(cache_dir, s, k)[1], dtype={"rank": str}, keep_default_na=False) for s, k in completed_parts(cache_dir)}

    def counts(sel):
        frames = [t for (s, k), t in tables.items() if sel(s)]
        g = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["rank", "n_moves"])
        n = pd.to_numeric(g["n_moves"]) if len(g) else pd.Series(dtype=int)
        return int(n.sum()), {lv: int(n[g["rank"] == lv].sum()) if len(g) else 0 for lv in LEVELS}

    n_written = 0
    for p in sorted(cache_dir.glob("meta*.json")):
        if p.name == META_NAME:
            continue
        meta = json.loads(p.read_text())
        shards = set(int(s) for s in meta.get("shards", [])) if p.name != "meta.json" else None
        total, per_rank = counts((lambda s: True) if shards is None else (lambda s: s in shards))
        meta.update({"positions": total, "per_rank_positions": per_rank, "patched": patched})
        p.write_text(to_json(meta, indent=1) + "\n")
        n_written += 1
    return n_written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="the cache directory (l1b_build_cache.py's layout), patched in place")
    ap.add_argument("--game-keys", required=True, help="the games to replay again (a CSV with a game_key column or one key per line)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, help="KataGo config name for the input features (the cache's)")
    ap.add_argument("--dry-run", action="store_true", help="only report the parts and games that would change")
    a = ap.parse_args(argv)
    cache = Path(a.cache)
    t0 = time.time()
    try:
        keys = set(read_game_keys(a.game_keys))
        todo = parts_to_patch(cache, keys)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    if not completed_parts(cache):
        print(f"no finished parts in {cache}", file=sys.stderr)
        return 2
    meta = load_patch_meta(cache)
    done = {tuple(json.loads(k)) if k.startswith("[") else tuple(int(x) for x in k.split("_")) for k in meta["parts"]}
    pending = [(s, k, hit) for s, k, hit in todo if (s, k) not in done]
    n_games = sum(len(h) for _, _, h in todo)
    print(f"{cache}: {len(completed_parts(cache))} parts, {len(todo)} hold {n_games} of the {len(keys)} listed games; "
          f"{len(done)} parts already patched, {len(pending)} to do", flush=True)
    if a.dry_run or not pending:
        if not pending and not a.dry_run and todo:
            print("nothing to do", flush=True)
        return 0
    from katago.train import modelconfigs
    config = modelconfigs.config_of_name[a.model_config]
    sgfs = load_games([g for _, _, hit in pending for g in hit])
    keys_sha = hashlib.sha256(Path(a.game_keys).read_bytes()).hexdigest()
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=init_worker, initargs=(config,)) as pool:
        for s, k, hit in pending:
            t1 = time.time()
            table = pd.read_csv(part_paths(cache, s, k)[1], dtype=str, keep_default_na=False).set_index("game_key")
            tasks = [(i, sgfs[g][0], table.at[g, "color"]) for i, g in enumerate(hit)]
            got = dict(pool.imap_unordered(extract, tasks))
            new = {g: pack_positions(got[i]) for i, g in enumerate(hit)}
            res = patch_part(cache, s, k, new)
            meta["parts"][f"{s}_{k}"] = {**res, "keys": hit, "seconds": time.time() - t1}
            meta.update({"game_keys_file": str(a.game_keys), "game_keys_sha256": keys_sha, "model_input_config": a.model_config,
                         "updated": datetime.now().isoformat(timespec="seconds"), **environment("cpu")})
            save_patch_meta(cache, meta)
            print(f"  part s{s} p{k}: {res['patched']} of {res['games']} games replaced, positions {res['positions_before']} -> "
                  f"{res['positions_after']} ({time.time() - t1:.0f} s)", flush=True)
    parts = meta["parts"]
    summary = {"game_keys_file": str(a.game_keys), "game_keys_sha256": keys_sha, "parts": len(parts), "games": sum(v["patched"] for v in parts.values()),
               "positions_before": sum(v["positions_before"] for v in parts.values()), "positions_after": sum(v["positions_after"] for v in parts.values()),
               "finished": datetime.now().isoformat(timespec="seconds")}
    meta["summary"] = summary
    save_patch_meta(cache, meta)
    n_meta = recount_meta(cache, summary)
    print(f"done: {summary['games']} games in {summary['parts']} parts, positions {summary['positions_before']} -> {summary['positions_after']}; "
          f"{n_meta} meta files recounted ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
