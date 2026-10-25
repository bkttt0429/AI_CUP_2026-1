"""Test-file reading, per-game sanity checks and submission writing/validation (layer L6).

Official test format (Tutorial README): question_id, num_games, sgf_1..sgf_20,
color_1..color_20, with only the first num_games sgf/color columns filled. Submission:
question_id, pred_rank, public questions then private, every question exactly once.

read_test refuses a file whose num_games disagrees with the filled sgf columns (2026-10-25, item 1 of
the to-do list): a question is only read when exactly its first num_games sgf columns are non-empty,
1 <= num_games <= MAX_GAMES. read_test_lenient reads such a question from its non-empty columns and
returns the problems as warnings instead -- the deadline-day escape hatch of predict_submission
--lenient-inputs. validate_submission checks the submission against the questions read:
header, every id once, the ids in the inputs' order (public file first), no private id before a public
one, the ranks in LEVELS and, when given, the expected total and per-prefix counts of the official
test (800 = 400 pub_ + 400 priv_), so a truncated or wrongly ordered input is caught before upload.
"""
from collections import Counter
from pathlib import Path

import pandas as pd
from katago.game.board import Board
from katago.game.gamestate import GameState

from aicup.metric import LEVELS
from aicup.sgf import board_size, parse_moves


MAX_GAMES = 20
PREFIXES = ("pub_", "priv_")  # the official question_id prefixes, public before private in the submission


def _read_test(path):
    """([{question_id, declared_games, games: [(sgf, colour), ...]}], problems) of an official-format test
    CSV: every question read from its non-empty sgf columns, and one problem per question whose num_games
    is not in 1..MAX_GAMES or whose non-empty columns are not exactly the first num_games."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out, bad = [], []
    for r in df.to_dict("records"):
        qid = r["question_id"]
        raw = str(r["num_games"]).strip()
        try:
            declared = int(float(raw)) if raw else 0
        except ValueError:
            declared = -1
        filled = [i for i in range(1, MAX_GAMES + 1) if str(r.get(f"sgf_{i}", "")).strip()]
        if not 1 <= declared <= MAX_GAMES:
            bad.append(f"{qid}: num_games {raw!r} is not in 1..{MAX_GAMES}")
        elif filled != list(range(1, declared + 1)):
            bad.append(f"{qid}: num_games {declared} but the non-empty sgf columns are {filled}")
        games = [(str(r[f"sgf_{i}"]).strip(), str(r.get(f"color_{i}", "")).strip().upper()) for i in filled]
        out.append({"question_id": qid, "declared_games": max(declared, 0), "games": games})
    return out, bad


def read_test(path, max_bad=5):
    """[{question_id, declared_games, games: [(sgf, colour), ...]}] of an official-format test CSV.

    Every question must declare 1 <= num_games <= MAX_GAMES and have exactly its first num_games sgf
    columns non-empty (the official format fills only those); otherwise ValueError names the file and
    up to max_bad offending questions, so a malformed file stops the pipeline before any scoring.
    Colours are read as given (upper-cased); check_game judges them."""
    out, bad = _read_test(path)
    if bad:
        more = f" (and {len(bad) - max_bad} more)" if len(bad) > max_bad else ""
        raise ValueError(f"{path}: {len(bad)} malformed questions, e.g. " + "; ".join(bad[:max_bad]) + more)
    return out


def read_test_lenient(path):
    """(questions, warnings): read_test without the refusal. A malformed question is read from whatever sgf
    columns are non-empty (declared_games keeps the declared value, 0 when it is not a number) and described
    in `warnings`, one string per question prefixed with the file name; [] for a well-formed file, whose
    questions equal read_test's."""
    out, bad = _read_test(path)
    return out, [f"{Path(path).name}: {b}" for b in bad]


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


def validate_submission(path, expected_ids, expected_total=None, expected_prefix_counts=None):
    """Problems with a submission file; an empty list means it is valid.

    expected_ids: the question ids of the inputs in the order read (public file first). Checked: the
    header; every expected id exactly once and nothing else; the rows in exactly that order; no priv_ row
    before a pub_ row; every id with a PREFIXES prefix; every pred_rank in LEVELS. expected_total (the
    official 800) and expected_prefix_counts ({"pub_": 400, "priv_": 400}) are independent expectations
    on the questions read, so a truncated input file is caught even though the file matches it."""
    expected_ids = list(expected_ids)
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
    elif ids != expected_ids:
        k = next(i for i, (a, b) in enumerate(zip(ids, expected_ids)) if a != b)
        problems.append(f"question order differs from the inputs: row {k} is {ids[k]!r}, expected {expected_ids[k]!r}")
    no_prefix = [i for i in ids if not i.startswith(PREFIXES)]
    if no_prefix:
        problems.append(f"ids without a {'/'.join(PREFIXES)} prefix: {len(no_prefix)}, e.g. {no_prefix[:3]}")
    first_priv = next((k for k, i in enumerate(ids) if i.startswith("priv_")), None)
    last_pub = max((k for k, i in enumerate(ids) if i.startswith("pub_")), default=None)
    if first_priv is not None and last_pub is not None and first_priv < last_pub:
        problems.append(f"a priv_ question (row {first_priv}) comes before a pub_ question (row {last_pub}): public must come first")
    if expected_total is not None and len(expected_ids) != expected_total:
        problems.append(f"{len(expected_ids)} questions read from the inputs, expected {expected_total}")
    for prefix, want in (expected_prefix_counts or {}).items():
        got = sum(i.startswith(prefix) for i in expected_ids)
        if got != want:
            problems.append(f"{got} {prefix} questions read from the inputs, expected {want}")
    return problems
