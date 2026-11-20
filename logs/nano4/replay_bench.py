#!/usr/bin/env python
"""Where does full-ply replay spend its time? 30 fold-1 games; single process step timings, then pools. Report only."""
import multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from katago.game.board import Board
from katago.game.gamestate import GameState
from aicup.pilot import start_state
from aicup.scoring import load_games
from aicup.sgf import setup_and_moves
CKPT = ROOT / "models/katago/b18c384nbt-humanv0.ckpt"
_F = None
def init(config):
    global _F
    from katago.game.features import Features
    _F = Features(config, 19)
def run(task):
    k, sgf, color = task
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, dict(GameState.RULES_CHINESE, whiteKomi=7.5))
    tp = tf = 0.0
    out = []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        t0 = time.perf_counter(); b, g = gs.get_input_features(_F); tf += time.perf_counter() - t0
        out.append(b[0].astype(np.uint8))
        t0 = time.perf_counter(); gs.play(pla, loc); tp += time.perf_counter() - t0
    return len(moves), tf, tp, len(np.stack(out).tobytes())
print("cpus", os.sched_getaffinity(0).__len__(), "load", os.getloadavg(), "OMP", os.environ.get("OMP_NUM_THREADS"), "torch threads", torch.get_num_threads(), flush=True)
fd = ROOT / "results/l1b/scores3_B_hd/fold1"
games = pd.concat([pd.read_csv(p) for p in sorted(fd.glob("games_s*_p000.csv"))]).sample(n=30, random_state=7)
sgf = load_games(games["game_key"].tolist()); tasks = [(k, *sgf[k]) for k in games["game_key"]]
config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
init(config)
t0 = time.time(); r = [run(t) for t in tasks[:6]]; w = time.time() - t0
n = sum(x[0] for x in r); print(f"single: {n} plies {n/w:.1f} ply/s; features {sum(x[1] for x in r)/n*1e3:.2f} ms/ply, play {sum(x[2] for x in r)/n*1e3:.2f} ms/ply", flush=True)
for workers, chunk in ((1, 1), (5, 1), (10, 1), (10, 4)):
    pool = mp.get_context("fork").Pool(workers, initializer=init, initargs=(config,))
    t0 = time.time(); r = pool.map(run, tasks, chunksize=chunk); w = time.time() - t0; pool.close(); pool.join()
    n = sum(x[0] for x in r); print(f"pool {workers} chunk {chunk}: {n/w:.1f} ply/s; features {sum(x[1] for x in r)/n*1e3:.2f} ms/ply, play {sum(x[2] for x in r)/n*1e3:.2f} ms/ply, load {os.getloadavg()[0]:.1f}", flush=True)
