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
