"""Section 4M M18 (2026-11-05): humanv0's metadata axis. Production scores every move under rank profiles with source = KGS, main time
1200 s, no overtime, date 2020-03-01. This server's ranks are inflated relative to KGS (6D ~ KGS 1-2d, 1D-4D span 1.4 profiles), so the
same moves are scored here under other sources (Fox, Tygem, which carry 7d-9d ranks) and under a byo-yomi time setting, the TARGET's
moves, the same fold-1 class-stratified players as M13 / M17.  Per game the summed log p of the target's moves per (setting, profile).
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/meta_pilot_score.py --fold 1 --per-class 60 --out results/meta_pilot
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.humansl import PROFILES, load_humansl, make_profile_meta  # noqa: E402
from aicup.scoring import fold_players, init_worker, load_games, player_games, score_stream  # noqa: E402
from katago.game.board import Board  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402

CKPT = ROOT / "models/katago/b18c384nbt-humanv0.ckpt"
HIGH = ["7d", "8d", "9d"]
SETTINGS = {  # name: (source, main time s, period s, byo-yomi periods, profiles)
    "kgs": (SGFMetadata.SOURCE_KGS, 1200.0, 0.0, 0, list(PROFILES)),  # the production setting (control)
    "fox": (SGFMetadata.SOURCE_FOX, 1200.0, 0.0, 0, list(PROFILES) + HIGH),
    "tygem": (SGFMetadata.SOURCE_TYGEM, 1200.0, 0.0, 0, list(PROFILES) + HIGH),
    "kgs_byo": (SGFMetadata.SOURCE_KGS, 600.0, 30.0, 3, list(PROFILES)),
}
SPECS = [(name, p) for name, (_, _, _, _, profs) in SETTINGS.items() for p in profs]


def rows_for(specs):
    rows = np.zeros((len(specs), 2, 192), dtype=np.float32)
    for s, (name, p) in enumerate(specs):
        src, main, period, periods, _ = SETTINGS[name]
        m = make_profile_meta(p, p)
        m.source, m.mainTimeSeconds, m.periodTimeSeconds, m.byoYomiPeriods = src, main, period, periods
        for c, pla in enumerate((Board.BLACK, Board.WHITE)):
            rows[s, c] = m.get_metadata_row(nextPlayer=pla, boardArea=361)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--per-class", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=64)
    a = ap.parse_args()
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, a.fold, max_games=20)
    players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)  # the M13 / M17 sample
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:a.per_class]] for _, grp in players.groupby("rank")])
    games = [(p.rank, p.player_id, k) for p in pick.itertuples() for k in p.game_keys][a.shard::a.nshards]
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda"); model.eval()
        rows = rows_for(SPECS)
        recs, t0 = [], time.time()
        n_parts = (len(games) + a.part_size - 1) // a.part_size
        for p0 in range(0, len(games), a.part_size):
            part = games[p0:p0 + a.part_size]; data = load_games([g[2] for g in part])
            tasks = [(i, data[k][0], data[k][1]) for i, (_, _, k) in enumerate(part)]  # the target's moves
            per_task, _, n_pos, wall = score_stream(pool, model, tasks, rows, log_every=10 ** 9)
            for i, (rank, pid, k) in enumerate(part):
                n, sums = per_task[i]
                rec = {"game_key": k, "player_id": pid, "rank": rank, "target_color": data[k][1], "n_moves": int(n)}
                rec.update({f"ll_{name}_{p}": float(sums[s]) for s, (name, p) in enumerate(SPECS)})
                recs.append(rec)
            print(f"part {p0 // a.part_size + 1}/{n_parts}: {n_pos} positions in {wall:.0f} s, {time.time() - t0:.0f} s total", flush=True)
        pd.DataFrame(recs).to_csv(out_dir / f"shard{a.shard}.csv", index=False)
        print(f"saved {len(recs)} games, {len(SPECS)} rows per move, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
