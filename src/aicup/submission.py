"""Test-file reading, per-game sanity checks and submission writing/validation (layer L6).

Official test format (Tutorial README): question_id, num_games, sgf_1..sgf_20,
color_1..color_20, with only the first num_games sgf/color columns filled. Submission:
question_id, pred_rank, public questions then private, every question exactly once.
"""
from collections import Counter

import pandas as pd
from katago.game.board import Board
from katago.game.gamestate import GameState

from aicup.metric import LEVELS
from aicup.sgf import board_size, parse_moves


def read_test(path):
    """[{question_id, declared_games, games: [(sgf, colour), ...]}] with blank game columns dropped."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = []
    for r in df.to_dict("records"):
        declared = int(float(r["num_games"])) if str(r["num_games"]).strip() else 0
        games = []
        for i in range(1, min(declared, 20) + 1):
            sgf = str(r.get(f"sgf_{i}", "")).strip()
            if sgf:
                games.append((sgf, str(r.get(f"color_{i}", "")).strip().upper()))
        out.append({"question_id": r["question_id"], "declared_games": declared, "games": games})
    return out


def check_game(sgf, color):
    """(ok, info): a game is usable when it is 19x19, colours alternate starting with Black,
    every move replays legally and the target colour has at least one move."""
    info = {"moves": 0, "passes": 0, "target_moves": 0, "error": None}
    if color not in ("B", "W"):
        info["error"] = f"invalid colour {color!r}"
        return False, info
    size = board_size(sgf)
    if size != 19:
        info["error"] = f"board size {size}"
        return False, info
    moves = parse_moves(sgf)
    info["moves"] = len(moves)
    gs = GameState(19, GameState.RULES_CHINESE)
    expected = "B"
    for t, (c, x, y) in enumerate(moves):
        if c != expected:
            info["error"] = f"move {t}: {c} played but {expected} to move"
            return False, info
        expected = "W" if c == "B" else "B"
        info["passes"] += x is None
        info["target_moves"] += c == color
        try:
            gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))
        except Exception as e:  # IllegalMoveError or anything the replay cannot handle
            info["error"] = f"move {t}: {e}"
            return False, info
    if info["target_moves"] == 0:
        info["error"] = "no moves by the target colour"
        return False, info
    return True, info


def write_submission(rows, path):
    pd.DataFrame(rows, columns=["question_id", "pred_rank"]).to_csv(path, index=False)


def validate_submission(path, expected_ids):
    """Problems with a submission file; an empty list means it is valid."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["question_id", "pred_rank"]:
        return [f"header is {list(df.columns)}, expected ['question_id', 'pred_rank']"]
    problems = []
    ids = df["question_id"].tolist()
    dups = sorted(i for i, c in Counter(ids).items() if c > 1)
    if dups:
        problems.append(f"duplicate ids: {len(dups)}, e.g. {dups[:3]}")
    missing = sorted(set(expected_ids) - set(ids))
    if missing:
        problems.append(f"missing ids: {len(missing)}, e.g. {missing[:3]}")
    unknown = sorted(set(ids) - set(expected_ids))
    if unknown:
        problems.append(f"unknown ids: {len(unknown)}, e.g. {unknown[:3]}")
    bad = sorted(set(df.loc[~df["pred_rank"].isin(LEVELS), "pred_rank"]))
    if bad:
        problems.append(f"invalid rank values: {bad[:5]}")
    if len(df) != len(expected_ids):
        problems.append(f"{len(df)} rows, expected {len(expected_ids)}")
    return problems
