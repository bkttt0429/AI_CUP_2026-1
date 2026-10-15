#!/usr/bin/env python
"""Score every eligible player of the given folds under the 18 Human SL profiles.

Each player with >= 5 games contributes up to 20 games (scoring.fold_players). Work is
split into parts of --part-size games; a part is written as
  <out>/fold{F}/games_s{shard}_p{k:03d}.csv   game_key, player_id, rank, color, n_moves, sum_<profile> x 18
  <out>/fold{F}/moves_s{shard}_p{k:03d}.npz   logp [moves, 18], game_row (row in the part CSV), t
and skipped on restart once both files exist, so an interrupted run resumes where it
stopped. The Pool is created before CUDA initialises (fork).
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.scoring import (fold_players, init_worker, load_games, part_paths, pending_parts,  # noqa: E402
                           plan_parts, player_games, score_stream)

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part-size", type=int, default=500)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "scores"))
    a = ap.parse_args()

    all_games = player_games()
    folds = pd.read_csv(ROOT / "splits" / "player_folds.csv")

    import torch
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    from aicup.humansl import PROFILES, load_humansl, meta_rows
    model, _ = load_humansl(str(CKPT), "cuda")
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    rows = meta_rows([(p, p) for p in PROFILES])
    log(f"shard {a.shard}/{a.nshards} on {torch.cuda.get_device_name(0)}; folds {a.folds}; cudnn.benchmark={a.cudnn_benchmark}")

    for fold in a.folds:
        players = fold_players(all_games, folds, fold)
        games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][a.shard::a.nshards]
        if a.limit_games:
            games = games[:a.limit_games]
        out = Path(a.out) / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        parts = plan_parts(len(games), a.part_size)
        todo = pending_parts(out, a.shard, parts)
        log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(parts)} parts, {len(todo)} to do")
        if not todo:
            continue
        data = load_games([games[i][2] for k in todo for i in range(*parts[k])])
        t_fold, pos_fold = time.time(), 0
        for k in todo:
            lo, hi = parts[k]
            sub = games[lo:hi]
            tasks = [(i, data[key][0], data[key][1]) for i, (_, _, key) in enumerate(sub)]
            per_task, (logp, task_idx, ts), n_pos, wall = score_stream(pool, model, tasks, rows, log_every=10**9)
            recs = [{"game_key": key, "player_id": pid, "rank": rank, "color": data[key][1], "n_moves": per_task[i][0],
                     **{f"sum_{p}": float(s) for p, s in zip(PROFILES, per_task[i][1])}}
                    for i, (rank, pid, key) in enumerate(sub)]
            csv_path, npz_path = part_paths(out, a.shard, k)
            tmp_csv, tmp_npz = csv_path.with_suffix(".tmp.csv"), npz_path.with_name(npz_path.stem + ".tmp.npz")
            np.savez_compressed(tmp_npz, logp=logp, game_row=task_idx, t=ts)
            pd.DataFrame(recs).to_csv(tmp_csv, index=False)
            os.replace(tmp_npz, npz_path)
            os.replace(tmp_csv, csv_path)
            pos_fold += n_pos
            log(f"  fold {fold} part {k + 1}/{len(parts)}: {len(sub)} games, {n_pos} positions, {n_pos / wall:.0f} positions/s")
        (out / f"done_s{a.shard}").write_text(json.dumps({
            "fold": fold, "shard": a.shard, "nshards": a.nshards, "games": len(games), "parts": len(parts),
            "positions_this_run": pos_fold, "wall_s_this_run": time.time() - t_fold, "cudnn_benchmark": a.cudnn_benchmark,
            "finished": datetime.now().isoformat(timespec="seconds")}))
        log(f"fold {fold} done")
    pool.close()


if __name__ == "__main__":
    main()
