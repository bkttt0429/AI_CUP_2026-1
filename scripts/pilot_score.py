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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS  # noqa: E402
from aicup.pilot import select_players  # noqa: E402
from aicup.scoring import init_worker, load_games, score_stream  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
CENTRE = {"D": "11k", "C": "8k", "B": "5k", "A": "2k", "1D": "1d", "2D": "2d", "3D": "3d", "4D": "4d", "5D": "5d", "6D": "6d"}


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
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
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
