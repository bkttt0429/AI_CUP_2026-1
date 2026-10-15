#!/usr/bin/env python
"""Real example data for the architecture report page (not pilot results).

1. For the first game of train_D.csv and of train_6D.csv (both Black targets), the
   log p of every target move under all 18 rank profiles (12k..1k, 1d..6d).
2. One position from the D game: the full masked policy under rank_12k, rank_1k and
   rank_6d, chosen as the target move between t=20 and t=80 where rank_12k and rank_6d
   disagree most (total variation). The page states this selection rule.
Writes results/viz_examples.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402

from aicup.humansl import load_humansl, make_profile_meta, target_move_logps  # noqa: E402
from aicup.sgf import parse_moves  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
OUT = ROOT / "results" / "viz_examples.json"
PROFILES = [f"{n}k" for n in range(12, 0, -1)] + [f"{n}d" for n in range(1, 7)]


def first_row(path):
    csv.field_size_limit(1 << 30)
    with open(path, newline="") as fh:
        return next(csv.DictReader(fh))


def stones_before(moves, t):
    gs = GameState(19, GameState.RULES_CHINESE)
    for c, x, y in moves[:t]:
        gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))
    out = []
    for x in range(19):
        for y in range(19):
            v = gs.board.board[gs.board.loc(x, y)]
            if v in (Board.BLACK, Board.WHITE):
                out.append([x, y, "B" if v == Board.BLACK else "W"])
    return out


def main():
    model, features = load_humansl(str(CKPT), "cuda")
    games = []
    per_game = {}
    for rank in ("D", "6D"):
        row = first_row(ROOT / "data" / "training" / f"train_{rank}.csv")
        moves = parse_moves(row["sgf_content"])
        by_profile = {p: target_move_logps(model, features, moves, row["color"], make_profile_meta(p)) for p in PROFILES}
        ts = [r["t"] for r in by_profile[PROFILES[0]]]
        logp = [[by_profile[p][i]["logp"] for p in PROFILES] for i in range(len(ts))]
        games.append({"rank": rank, "game_id": row["game_id"], "color": row["color"], "n_moves_total": len(moves),
                      "t": ts, "gtp": [r["gtp"] for r in by_profile[PROFILES[0]]], "logp": logp})
        per_game[rank] = (moves, row, by_profile)
        print(f"{rank}: {len(ts)} target moves scored under {len(PROFILES)} profiles", flush=True)

    moves, row, by_profile = per_game["D"]
    best = None
    for i, r in enumerate(by_profile["12k"]):
        if not 20 <= r["t"] <= 80:
            continue
        p_lo, p_hi = np.exp(r["policy_logp"]), np.exp(by_profile["6d"][i]["policy_logp"])
        tv = 0.5 * float(np.abs(p_lo - p_hi).sum())
        if best is None or tv > best[0]:
            best = (tv, i)
    tv, i = best
    t = by_profile["12k"][i]["t"]
    position = {
        "game": "D", "game_id": row["game_id"], "t": t, "to_move": moves[t][0],
        "played": by_profile["12k"][i]["gtp"], "played_xy": [moves[t][1], moves[t][2]],
        "last_move_xy": [moves[t - 1][1], moves[t - 1][2]], "tv_12k_6d": tv,
        "stones": stones_before(moves, t),
        "policy": {p: np.exp(by_profile[p][i]["policy_logp"]).round(6).tolist() for p in ("12k", "1k", "6d")},
        "played_logp": {p: by_profile[p][i]["logp"] for p in PROFILES},
    }
    OUT.write_text(json.dumps({"profiles": PROFILES, "games": games, "position": position}))
    print(f"position t={t} played {position['played']} TV(12k,6d)={tv:.3f}; wrote {OUT}")


if __name__ == "__main__":
    main()
