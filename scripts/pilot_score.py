#!/usr/bin/env python
"""Human SL pilot scoring (METHOD_NOTES §2.4): log p of every target move under 18 profiles.

Main mode: 20 players per rank from one fold (players with a 20-game question), their 20
games, profiles rank_12k..rank_6d with both players at the profile rank. Output per shard:
  results/pilot/games_fold{F}_shard{i}.csv  game_key, player_id, rank, color, n_moves, sum_<profile> x 18
  results/pilot/moves_fold{F}_shard{i}.npz  logp [moves, 18], game_row, t
  results/pilot/run_fold{F}_shard{i}.json   timing
--opp-sweep: 2 games per rank (first selected player), own rank fixed at the class centre,
opponent rank swept over the 18 profiles -> results/pilot/opp_sweep_shard{i}.csv.

CPU workers replay games and build features (Pool created before CUDA initialises);
the GPU scores each batch of positions once per profile.
"""
import argparse
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
from aicup.metric import LEVELS  # noqa: E402
from aicup.pilot import game_positions, select_players  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
CENTRE = {"D": "11k", "C": "8k", "B": "5k", "A": "2k", "1D": "1d", "2D": "2d", "3D": "3d", "4D": "4d", "5D": "5d", "6D": "6d"}
_FEATURES = None


def _init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def _extract(task):
    idx, sgf, color = task
    return idx, game_positions(sgf, color, _FEATURES)


def load_games(keys):
    """sgf and colour for each 'RANK:game_id' key, read only from the rank files needed."""
    by_rank = {}
    for k in keys:
        r, g = k.split(":")
        by_rank.setdefault(r, set()).add(g)
    out = {}
    for r, ids in by_rank.items():
        df = pd.read_csv(ROOT / "data" / "training" / f"train_{r}.csv", usecols=["game_id", "color", "sgf_content"])
        for row in df[df["game_id"].isin(ids)].itertuples():
            out[f"{r}:{row.game_id}"] = (row.sgf_content, row.color)
    missing = [k for k in keys if k not in out]
    if missing:
        raise KeyError(f"{len(missing)} game keys not found, e.g. {missing[:3]}")
    return out


def score_stream(pool, model, tasks, rows, flush_at=1024, log_every=200):
    """Score all tasks; returns per-task (n_moves, sums[S]) and stacked per-move arrays."""
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

    for idx, pos in pool.imap(_extract, tasks, chunksize=1):
        buf.append((idx, pos))
        n_pos += len(pos["t"])
        done += 1
        if sum(len(p["t"]) for _, p in buf) >= flush_at:
            flush()
        if done % log_every == 0:
            el = time.time() - t0
            print(f"  {done}/{len(tasks)} games, {n_pos} positions, {n_pos / el:.0f} positions/s", flush=True)
    flush()
    arrays = (np.concatenate(move_logp), np.concatenate(move_task), np.concatenate(move_t)) if move_logp else None
    return per_task, arrays, n_pos, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--per-rank", type=int, default=20)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--opp-sweep", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    questions = pd.read_csv(ROOT / "splits" / f"questions_fold{a.fold}.csv")
    players = select_players(questions, a.fold, a.per_rank)
    if a.opp_sweep:
        players = players.groupby("rank", sort=False).head(1)
        games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys[:2]]
    else:
        games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys]
    games = games[a.shard::a.nshards]
    if a.limit_games:
        games = games[:a.limit_games]
    data = load_games([g[2] for g in games])
    print(f"fold {a.fold} shard {a.shard}/{a.nshards}: {len(players)} players, {len(games)} games", flush=True)

    import torch
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker, initargs=(config,))
    from aicup.humansl import PROFILES, load_humansl, meta_rows
    model, _ = load_humansl(str(CKPT), "cuda")

    if a.opp_sweep:
        recs = []
        for rank in LEVELS:
            sub = [i for i, g in enumerate(games) if g[0] == rank]
            if not sub:
                continue
            own = CENTRE[rank]
            tasks = [(i, data[games[i][2]][0], data[games[i][2]][1]) for i in sub]
            per_task, _, _, _ = score_stream(pool, model, tasks, meta_rows([(own, opp) for opp in PROFILES]))
            for i in sub:
                n, sums = per_task[i]
                recs += [{"rank": rank, "game_key": games[i][2], "own": own, "opp": opp, "n_moves": n, "sum_logp": float(s)}
                         for opp, s in zip(PROFILES, sums)]
            print(f"  opp sweep {rank}: {len(sub)} games", flush=True)
        pd.DataFrame(recs).to_csv(out / f"opp_sweep_shard{a.shard}.csv", index=False)
        pool.close()
        return

    tasks = [(i, data[g[2]][0], data[g[2]][1]) for i, g in enumerate(games)]
    per_task, (logp, task_idx, ts), n_pos, wall = score_stream(pool, model, tasks, meta_rows([(p, p) for p in PROFILES]))
    pool.close()
    rows = []
    for i, (rank, pid, key) in enumerate(games):
        n, sums = per_task[i]
        rows.append({"game_key": key, "player_id": pid, "rank": rank, "color": data[key][1], "n_moves": n,
                     **{f"sum_{p}": float(s) for p, s in zip(PROFILES, sums)}})
    tag = f"fold{a.fold}_shard{a.shard}"
    pd.DataFrame(rows).to_csv(out / f"games_{tag}.csv", index=False)
    np.savez_compressed(out / f"moves_{tag}.npz", logp=logp, game_row=task_idx, t=ts)
    run = {"fold": a.fold, "shard": a.shard, "nshards": a.nshards, "games": len(games), "positions": int(n_pos),
           "wall_s": wall, "positions_per_s": n_pos / wall, "profiles": PROFILES, "workers": a.workers,
           "gpu": torch.cuda.get_device_name(0), "tf32_conv": torch.backends.cudnn.allow_tf32}
    (out / f"run_{tag}.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
