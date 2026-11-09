#!/usr/bin/env python
"""Score every eligible player of the given folds under the 18 Human SL profiles.

Each player with >= 5 games contributes up to 20 games (scoring.fold_players). Work is
written in parts of --part-size games:
  <out>/fold{F}/games_s{shard}_p{k:03d}.csv   game_key, player_id, rank, color, n_moves, sum_<profile> x 18
  <out>/fold{F}/moves_s{shard}_p{k:03d}.npz   logp [moves, 18], game_row (row in the part CSV), t
Both files are written to temporary names and renamed, npz first, so a part counts as done
only when both exist.

Breakpoints:
  * resume is decided per game (scoring.resume_plan): rerunning the same command scores only
    games that are in no finished part, even if --part-size changed;
  * pause cleanly with `touch <out>/PAUSE` (or SIGTERM / Ctrl-C to the main process): the run
    stops after the part in progress, i.e. within about a minute at the default part size;
    a second signal stops at once and only the unfinished part is lost;
  * the run refuses to start while the pause file exists: delete it to resume.
The CPU Pool is created before CUDA initialises (fork).
"""
import argparse
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.scoring import (fold_players, init_worker, load_games, part_paths, pause_requested,  # noqa: E402
                           player_games, read_game_keys, restrict_players, resume_plan, score_stream)

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
STOP = {"signal": None}


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def _on_signal(signum, _frame):
    if STOP["signal"] is not None:  # second signal: stop now
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    STOP["signal"] = signum
    log(f"signal {signum}: stopping after the part in progress (send again to stop now)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--max-games", type=int, default=20, help="scoring.fold_players' cap per player (section 4M M8: 40 with --game-keys of positions 21-40)")
    ap.add_argument("--cudnn-benchmark", action="store_true")
    ap.add_argument("--opponent", action="store_true", help="section 4M M14: score the OPPONENT's moves (colour flipped) under the same 18 profiles; the color column stays the target's")
    ap.add_argument("--profiles", nargs="+", default=None, metavar="RANK",
                    help="section 4M M16: score under these humanv0 profiles instead of the 18 (12k-6d), e.g. 7d 8d 9d; the sums columns are sum_<rank>")
    ap.add_argument("--out", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--pause-file", default=None, help="default: <out>/PAUSE")
    ap.add_argument("--game-keys", default=None, help="score only these games of each fold's selection (a CSV with a game_key column or one key "
                    "per line), e.g. the HD >= 2 games again into a patch directory that scripts/patch_scored.py merges (issue #42)")
    a = ap.parse_args()
    pause_file = Path(a.pause_file) if a.pause_file else Path(a.out) / "PAUSE"
    if pause_requested(pause_file):
        log(f"pause file {pause_file} exists; delete it to run")
        return

    all_games = player_games()
    folds = pd.read_csv(ROOT / "splits" / "player_folds.csv")

    import torch
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    for sig in (signal.SIGTERM, signal.SIGINT):  # after the fork: workers keep the default handlers
        signal.signal(sig, _on_signal)
    from aicup.humansl import PROFILES as DEFAULT_PROFILES, inverse_rank, load_humansl, meta_rows
    PROFILES = list(a.profiles) if a.profiles else list(DEFAULT_PROFILES)
    for p_ in PROFILES:
        inverse_rank(p_)  # ValueError for a rank humanv0 does not know
    model, _ = load_humansl(str(CKPT), "cuda")
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    rows = meta_rows([(p, p) for p in PROFILES])
    log(f"shard {a.shard}/{a.nshards} on {torch.cuda.get_device_name(0)}; folds {a.folds}; part size {a.part_size}; "
        f"pause file {pause_file}; {len(PROFILES)} profiles {PROFILES[0]}..{PROFILES[-1]}" + ("; the OPPONENT's moves" if a.opponent else ""))

    def stop_now():
        return STOP["signal"] is not None or pause_requested(pause_file)

    only = read_game_keys(a.game_keys) if a.game_keys else None
    for fold in a.folds:
        players = fold_players(all_games, folds, fold, max_games=a.max_games)
        if only is not None:
            players = restrict_players(players, only)
        games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][a.shard::a.nshards]
        if a.limit_games:
            games = games[:a.limit_games]
        meta = {g[2]: g for g in games}
        out = Path(a.out) / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        plan = resume_plan([g[2] for g in games], out, a.shard, a.part_size)
        pending = sum(len(keys) for _, keys in plan)
        log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(games) - pending} already done, "
            f"{pending} to score in {len(plan)} parts")
        if plan:
            data = load_games([k for _, keys in plan for k in keys])
            t_fold, pos_fold = time.time(), 0
            for j, (k, keys) in enumerate(plan):
                if stop_now():
                    log(f"paused before fold {fold} part index {k} ({len(plan) - j} parts left in this fold)")
                    pool.close()
                    return
                sub = [meta[key] for key in keys]
                tasks = [(i, data[key][0], ("W" if data[key][1] == "B" else "B") if a.opponent else data[key][1]) for i, key in enumerate(keys)]
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
                log(f"  fold {fold} part {j + 1}/{len(plan)} (index {k}): {len(sub)} games, {n_pos} positions, {n_pos / wall:.0f} positions/s")
            log(f"fold {fold}: scored {pos_fold} positions in {(time.time() - t_fold) / 60:.1f} min")
        (out / f"done_s{a.shard}").write_text(json.dumps({
            "fold": fold, "shard": a.shard, "nshards": a.nshards, "games": len(games), "part_size": a.part_size,
            "cudnn_benchmark": a.cudnn_benchmark, "finished": datetime.now().isoformat(timespec="seconds")}))
        log(f"fold {fold} done")
    pool.close()


if __name__ == "__main__":
    main()
