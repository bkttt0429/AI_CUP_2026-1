#!/usr/bin/env python
"""Step 0B: how fast can we score human moves with humanv0 on this machine?

Two different numbers, reported separately:
  * pure inference  — real positions pre-batched on the GPU, warm-up then timed forwards,
                      FP32 and bf16 autocast, batch 64 / 256 / 1024  -> positions/s
  * end-to-end      — target_move_logps() over N games exactly as the pilot would call it
                      (SGF replay + features + legality mask + one forward per position),
                      plus the replay+features and legality costs on their own.
Writes results/bench_0B.json (another host: pass --out).
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402

from aicup.humansl import load_humansl, make_profile_meta, stack_inputs, target_move_logps  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
OUT = ROOT / "results" / "bench_0B.json"


def read_games(csv_path, n):
    csv.field_size_limit(1 << 30)
    with open(csv_path, newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def target_states(moves, target_color):
    """GameState snapshots right before each move of target_color."""
    gs = GameState(19, GameState.RULES_CHINESE)
    out = []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        if c == target_color:
            out.append(GameState(19, GameState.RULES_CHINESE))
            out[-1].board = gs.board.copy()
            out[-1].moves = list(gs.moves)
            out[-1].boards = list(gs.boards)
        gs.play(pla, Board.PASS_LOC if x is None else gs.board.loc(x, y))
    return out


def bench_pure(model, features, states, meta, batch_sizes=(64, 256, 1024), warmup=10, iters=50):
    device = next(model.parameters()).device
    reps = -(-max(batch_sizes) // len(states))
    b, g, m = stack_inputs(features, (states * reps)[: max(batch_sizes)], meta)
    results = {}
    for bs in batch_sizes:
        bb, gg, mm = b[:bs].to(device), g[:bs].to(device), m[:bs].to(device)
        for name, ctx in (("fp32", torch.autocast("cuda", enabled=False)),
                          ("bf16", torch.autocast("cuda", dtype=torch.bfloat16))):
            with torch.no_grad(), ctx:
                for _ in range(warmup):
                    model(bb, gg, input_meta=mm)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iters):
                    model(bb, gg, input_meta=mm)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            results[f"batch{bs}_{name}"] = {"positions_per_s": bs * iters / dt, "ms_per_batch": 1000 * dt / iters}
            print(f"  pure {name} batch {bs:4d}: {bs * iters / dt:8.0f} positions/s", flush=True)
    return results


def bench_end_to_end(model, features, games, meta):
    t0 = time.perf_counter()
    n_feat = 0
    for g in games:
        for gs in target_states(parse_moves(g["sgf_content"]), g["color"]):
            gs.get_input_features(features)
            n_feat += 1
    t_feat = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_legal = 0
    for g in games[:20]:
        for gs in target_states(parse_moves(g["sgf_content"]), g["color"]):
            for i in range(features.pass_pos):
                gs.board.would_be_legal(gs.board.pla, features.tensor_pos_to_loc(i, gs.board))
            n_legal += 1
    t_legal = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_moves = 0
    for g in games:
        n_moves += len(target_move_logps(model, features, parse_moves(g["sgf_content"]), g["color"], meta))
    torch.cuda.synchronize()
    t_e2e = time.perf_counter() - t0
    res = {
        "games": len(games), "target_moves": n_moves,
        "replay_features_per_s": n_feat / t_feat,
        "legality_mask_per_s": n_legal / t_legal,
        "end_to_end_target_moves_per_s": n_moves / t_e2e, "end_to_end_wall_s": t_e2e,
    }
    print(f"  replay+features {res['replay_features_per_s']:.0f}/s | legality {res['legality_mask_per_s']:.0f}/s | "
          f"end-to-end {res['end_to_end_target_moves_per_s']:.1f} target moves/s over {n_moves} moves", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "training" / "train_1D.csv"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    model, features = load_humansl(str(CKPT), "cuda")
    meta = make_profile_meta("1d")
    games = read_games(a.csv, a.games)
    states = target_states(parse_moves(games[0]["sgf_content"]), games[0]["color"])
    print(f"GPU {torch.cuda.get_device_name(0)} | torch {torch.__version__} | {len(states)} real positions for batching")
    out = {
        "when": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "pure_inference": bench_pure(model, features, states, meta),
        "end_to_end": bench_end_to_end(model, features, games, meta),
    }
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
