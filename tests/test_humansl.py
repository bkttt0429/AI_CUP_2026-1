import datetime
from pathlib import Path

import numpy as np
import pytest
from katago.game.board import Board

from aicup.humansl import inverse_rank, load_humansl, make_profile_meta, target_move_logps
from aicup.sgf import parse_moves

CKPT = Path(__file__).resolve().parents[1] / "models" / "katago" / "b18c384nbt-humanv0.ckpt"


def test_inverse_rank_matches_katago_cpp():
    assert inverse_rank("3d") == 7
    assert inverse_rank("5k") == 14
    assert inverse_rank("1d") == 9
    assert inverse_rank("1k") == 10
    assert inverse_rank("20k") == 29


def test_make_profile_meta_reproduces_engine_rank_profile():
    m = make_profile_meta("3d")
    assert m.inverseBRank == 7 and m.inverseWRank == 7
    assert m.bIsHuman and m.wIsHuman
    assert m.tcIsByoYomi and not m.tcIsUnknown
    assert m.mainTimeSeconds == 1200
    assert m.periodTimeSeconds == 30
    assert m.byoYomiPeriods == 5
    assert m.gameDate == datetime.date(2020, 3, 1)
    assert m.source == 2
    assert m.gameRatednessIsUnknown


def test_make_profile_meta_two_rank_form_and_preaz():
    m = make_profile_meta("3d", "5k")
    assert m.inverseBRank == 7 and m.inverseWRank == 14
    assert make_profile_meta("5k", preaz=True).gameDate == datetime.date(2016, 9, 1)


def test_metadata_row_thermometer_encoding():
    row = make_profile_meta("3d").get_metadata_row(nextPlayer=Board.BLACK, boardArea=361)
    assert row[6:13].sum() == 7 and row[13] == 0
    assert row[40:47].sum() == 7 and row[47] == 0


@pytest.mark.skipif(not CKPT.exists(), reason="human SL checkpoint not present")
def test_target_move_logps_on_three_move_game():
    import torch

    model, features = load_humansl(str(CKPT), "cuda" if torch.cuda.is_available() else "cpu")
    moves = parse_moves("(;SZ[19];B[pd];W[dp];B[pq])")
    res = target_move_logps(model, features, moves, "B", make_profile_meta("3d"))
    assert [r["t"] for r in res] == [0, 2]
    assert [r["gtp"] for r in res] == ["Q16", "Q3"]
    for r in res:
        assert np.log(0.01) < r["logp"] < np.log(0.9)
        finite = r["policy_logp"][np.isfinite(r["policy_logp"])]
        assert np.exp(finite).sum() == pytest.approx(1.0, abs=1e-4)
    # move 2 (W already on the board) must have one fewer legal point than move 0
    assert np.isfinite(res[1]["policy_logp"]).sum() == np.isfinite(res[0]["policy_logp"]).sum() - 2


@pytest.mark.skipif(not CKPT.exists(), reason="human SL checkpoint not present")
def test_target_move_logps_handles_passes_by_either_side():
    import torch

    model, features = load_humansl(str(CKPT), "cuda" if torch.cuda.is_available() else "cpu")
    # W passes at move 1 (opponent pass), B plays, then W passes again as the target's own move.
    moves = parse_moves("(;SZ[19];B[pd];W[];B[pq];W[tt];B[dp])")
    res_w = target_move_logps(model, features, moves, "W", make_profile_meta("3d"))
    assert [(r["t"], r["gtp"]) for r in res_w] == [(1, "pass"), (3, "pass")]
    for r in res_w:
        assert np.isfinite(r["logp"]) and r["logp"] == r["policy_logp"][features.pass_pos]
        finite = r["policy_logp"][np.isfinite(r["policy_logp"])]
        assert np.exp(finite).sum() == pytest.approx(1.0, abs=1e-4)
    # the Black target must still be scored correctly across the opponent's passes
    res_b = target_move_logps(model, features, moves, "B", make_profile_meta("3d"))
    assert [(r["t"], r["gtp"]) for r in res_b] == [(0, "Q16"), (2, "Q3"), (4, "D4")]
    # before move 4 only two stones are on the board (W passed twice): 361 - 2 legal points + pass
    assert np.isfinite(res_b[2]["policy_logp"]).sum() == 361 - 2 + 1
