import csv
from pathlib import Path

import numpy as np
import pytest
from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train import modelconfigs

from aicup.pilot import game_positions
from aicup.sgf import parse_moves

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
TRAIN_1D = ROOT / "data" / "training" / "train_1D.csv"
SGF = "(;SZ[19];B[pd];W[dp];B[pq];W[];B[dd])"


def _features():
    return Features(modelconfigs.config_of_name["b18c384nbt"], 19)


def test_game_positions_black_target_includes_positions_after_an_opponent_pass():
    pos = game_positions(SGF, "B", _features())
    assert pos["t"].tolist() == [0, 2, 4]
    assert pos["color"].tolist() == [0, 0, 0]
    assert pos["bins"].dtype == np.uint8 and pos["bins"].shape[0] == 3
    assert pos["globs"].dtype == np.float32 and pos["legal"].dtype == bool
    # stones on the board before each target move: 0, 2, 3 -> legal points 362 - stones (pass included)
    assert pos["legal"].sum(axis=1).tolist() == [362, 360, 359]
    assert pos["target"].tolist() == [3 * 19 + 15, 16 * 19 + 15, 3 * 19 + 3]


def test_game_positions_white_target_pass_is_index_361_and_legal():
    pos = game_positions(SGF, "W", _features())
    assert pos["t"].tolist() == [1, 3]
    assert pos["color"].tolist() == [1, 1]
    assert pos["target"].tolist() == [15 * 19 + 3, 361]
    assert pos["legal"][1, 361]


def test_game_positions_features_equal_replay_features():
    features = _features()
    pos = game_positions(SGF, "B", features)
    gs = GameState(19, GameState.RULES_CHINESE)
    for c, x, y in parse_moves(SGF)[:2]:
        gs.play(Board.BLACK if c == "B" else Board.WHITE, Board.PASS_LOC if x is None else gs.board.loc(x, y))
    b, g = gs.get_input_features(features)
    assert np.array_equal(pos["bins"][1], b[0].astype(np.uint8))
    assert np.array_equal(pos["globs"][1], g[0].astype(np.float32))


def _first_games_by_colour():
    csv.field_size_limit(1 << 30)
    out = {}
    with open(TRAIN_1D, newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["color"], row)
            if len(out) == 2:
                break
    return out


@pytest.mark.skipif(not (CKPT.exists() and TRAIN_1D.exists()), reason="checkpoint or data not present")
def test_score_positions_matches_single_position_path_for_both_colours():
    import torch

    from aicup.humansl import load_humansl, make_profile_meta, meta_rows, score_positions, target_move_logps

    model, features = load_humansl(str(CKPT), "cuda" if torch.cuda.is_available() else "cpu")
    specs = [("1d", "1d"), ("5k", "5k")]
    rows = meta_rows(specs)
    # cuDNN's default TF32 convolutions pick batch-size-dependent algorithms (~1e-3 nats apart);
    # exact FP32 isolates the code paths so this test checks indexing and metadata, not rounding.
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        for color, row in _first_games_by_colour().items():
            moves = parse_moves(row["sgf_content"])[:60]
            pos = game_positions(row["sgf_content"], color, features)
            keep = pos["t"] < 60
            got = score_positions(model, pos["bins"][keep], pos["globs"][keep], pos["legal"][keep],
                                  pos["target"][keep], pos["color"][keep], rows)
            assert got.shape == (int(keep.sum()), 2)
            for s, (own, _) in enumerate(specs):
                ref = [r["logp"] for r in target_move_logps(model, features, moves, color, make_profile_meta(own))]
                assert np.max(np.abs(got[:, s] - np.array(ref))) < 1e-4, (color, own)
    finally:
        torch.backends.cudnn.allow_tf32 = prev


def _questions_frame():
    import pandas as pd

    rows = []
    def add(rank, pid, n, j, start):
        keys = ";".join(f"{rank}:g{start + i:04d}" for i in range(n))
        rows.append({"question_id": f"f0_{rank}_{pid[:8]}_n{n}_{j}", "rank": rank, "player_id": pid, "n_games": n, "game_keys": keys})
    for pid, base in (("aaaa0000aaaa0000", 0), ("bbbb1111bbbb1111", 100), ("cccc2222cccc2222", 200)):
        add("A", pid, 5, 0, base); add("A", pid, 10, 0, base + 5); add("A", pid, 20, 0, base + 15)
    add("A", "dddd3333dddd3333", 5, 0, 300); add("A", "dddd3333dddd3333", 10, 0, 305)  # no 20-game question
    add("6D", "eeee4444eeee4444", 5, 0, 400); add("6D", "eeee4444eeee4444", 10, 0, 405); add("6D", "eeee4444eeee4444", 20, 0, 415)
    return pd.DataFrame(rows)


def test_select_players_only_takes_players_with_a_20_game_question():
    from aicup.pilot import select_players

    sel = select_players(_questions_frame(), fold=0, per_rank=5, seed=42)
    assert sorted(sel["player_id"]) == ["aaaa0000aaaa0000", "bbbb1111bbbb1111", "cccc2222cccc2222", "eeee4444eeee4444"]
    row = sel[sel["player_id"] == "aaaa0000aaaa0000"].iloc[0]
    assert row["game_keys"] == [f"A:g{i:04d}" for i in range(15, 35)]


def test_select_players_caps_per_rank_and_is_deterministic():
    from aicup.pilot import select_players

    a = select_players(_questions_frame(), fold=0, per_rank=1, seed=42)
    b = select_players(_questions_frame(), fold=0, per_rank=1, seed=42)
    assert a["rank"].value_counts().to_dict() == {"A": 1, "6D": 1}
    assert a["player_id"].tolist() == b["player_id"].tolist()


def test_pilot_questions_nest_20_10_5_without_overlap_within_a_size():
    import pandas as pd

    from aicup.pilot import pilot_questions

    keys = [f"3D:g{i:04d}" for i in range(20)]
    q = pilot_questions(pd.DataFrame([{"rank": "3D", "player_id": "abcdef0123456789", "game_keys": keys}]))
    assert q["n_games"].value_counts().to_dict() == {20: 1, 10: 2, 5: 4}
    for n in (10, 5):
        used = [k for s in q[q["n_games"] == n]["game_keys"] for k in s]
        assert sorted(used) == sorted(keys)
    assert set(q["question_id"]) >= {"3D_abcdef01_n20_0", "3D_abcdef01_n10_1", "3D_abcdef01_n5_3"}
