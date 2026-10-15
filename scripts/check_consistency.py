#!/usr/bin/env python
"""Step 0A: does PyTorch evaluation of humanv0.ckpt reproduce the analysis engine's log p?

Positions: first 4 games of train_1D.csv (two Black targets, two White targets); in each,
the first target move at or after move 10 / 60 / 150 plus the first target move right
after the opponent captured something. Profile rank_1d everywhere, and game 0 again
under rank_5k. Every run appends an attempt to results/consistency_0A.json.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402

from aicup.humansl import load_humansl, make_profile_meta, target_move_logps  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402
from katago_engine_logp import engine_target_logps  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
OUT = ROOT / "results" / "consistency_0A.json"
TOL = 1e-2


def _stone_count(board):
    return sum(1 for x in range(19) for y in range(19)
               if board.board[board.loc(x, y)] in (Board.BLACK, Board.WHITE))


def select_positions(moves, target_color, thresholds=(10, 60, 150)):
    """Indices of target moves to compare: first at/after each threshold + first after a capture."""
    chosen = set()
    for th in thresholds:
        for t, (c, _, _) in enumerate(moves):
            if t >= th and c == target_color:
                chosen.add(t)
                break
    gs = GameState(19, GameState.RULES_CHINESE)
    counts = []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        gs.play(pla, Board.PASS_LOC if x is None else gs.board.loc(x, y))
        counts.append(_stone_count(gs.board))
    for t in range(1, len(moves)):
        prev_c, px, _ = moves[t - 1]
        before = counts[t - 2] if t >= 2 else 0
        captured = px is not None and counts[t - 1] < before + 1
        if captured and moves[t][0] == target_color and prev_c != target_color:
            chosen.add(t)
            break
    return sorted(chosen)


def load_games(csv_path, n_per_color=2):
    csv.field_size_limit(1 << 30)
    games = {"B": [], "W": []}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if len(games[row["color"]]) < n_per_color:
                games[row["color"]].append(row)
            if all(len(v) >= n_per_color for v in games.values()):
                break
    return games["B"] + games["W"]


def compare(py, eng):
    rows = []
    for a, b in zip(py, eng):
        assert a["t"] == b["t"] and a["gtp"] == b["gtp"]
        fa, fb = np.isfinite(a["policy_logp"]), np.isfinite(b["policy_logp"])
        both = fa & fb
        rows.append({
            "t": a["t"], "gtp": a["gtp"],
            "logp_py": a["logp"], "logp_engine": b["logp"], "abs_diff": abs(a["logp"] - b["logp"]),
            "vec_max_abs_diff": float(np.max(np.abs(a["policy_logp"][both] - b["policy_logp"][both]))),
            "n_legal_py": int(fa.sum()), "n_legal_engine": int(fb.sum()),
            "argmax_py": int(np.argmax(a["policy_logp"])), "argmax_engine": int(np.argmax(b["policy_logp"])),
            "corr": float(np.corrcoef(a["policy_logp"][both], b["policy_logp"][both])[0, 1]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "training" / "train_1D.csv"))
    ap.add_argument("--use-swa", action="store_true")
    a = ap.parse_args()

    model, features = load_humansl(str(CKPT), "cuda", use_swa=a.use_swa)
    games = load_games(a.csv)
    jobs = [(i, g, "1d") for i, g in enumerate(games)] + [(0, games[0], "5k")]
    rows = []
    for gi, g, rank in jobs:
        moves, color = parse_moves(g["sgf_content"]), g["color"]
        pos = select_positions(moves, color)
        py = [r for r in target_move_logps(model, features, moves, color, make_profile_meta(rank)) if r["t"] in pos]
        eng = engine_target_logps(moves, color, f"rank_{rank}", only_t=set(pos))
        new = compare(py, eng)
        for r in new:
            r.update({"game": gi, "game_id": g["game_id"], "color": color, "profile": f"rank_{rank}"})
        rows.extend(new)
        print(f"game {gi} ({color}, rank_{rank}) t={pos}: max|dlogp|={max(r['abs_diff'] for r in new):.2e}", flush=True)

    summary = {
        "when": datetime.now().isoformat(timespec="seconds"), "use_swa": a.use_swa, "mask_illegal": True,
        "n_positions": len(rows),
        "max_abs_diff_target": max(r["abs_diff"] for r in rows),
        "max_abs_diff_vector": max(r["vec_max_abs_diff"] for r in rows),
        "min_corr": min(r["corr"] for r in rows),
        "all_argmax_equal": all(r["argmax_py"] == r["argmax_engine"] for r in rows),
        "all_legal_counts_equal": all(r["n_legal_py"] == r["n_legal_engine"] for r in rows),
    }
    summary["passed"] = summary["max_abs_diff_target"] < TOL and summary["all_argmax_equal"]
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"attempts": []}
    doc["attempts"].append({"summary": summary, "positions": rows})
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1))
    print(json.dumps(summary, indent=1))
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
