import numpy as np
from katago.game.board import Board

from aicup.humansl import PROFILES, make_profile_meta, meta_rows, profile_meta_for


def test_profiles_weak_to_strong():
    assert len(PROFILES) == 18
    assert PROFILES[0] == "12k" and PROFILES[11] == "1k" and PROFILES[12] == "1d" and PROFILES[17] == "6d"


def test_profile_meta_for_assigns_own_rank_to_target_colour():
    w = profile_meta_for("3d", "5k", "W")
    assert w.inverseWRank == 7 and w.inverseBRank == 14
    b = profile_meta_for("3d", "5k", "B")
    assert b.inverseBRank == 7 and b.inverseWRank == 14


def test_meta_rows_index_by_target_colour_to_move():
    rows = meta_rows([("3d", "3d"), ("3d", "5k")])
    assert rows.shape == (2, 2, 192) and rows.dtype == np.float32
    m = make_profile_meta("3d")
    assert np.array_equal(rows[0, 0], m.get_metadata_row(nextPlayer=Board.BLACK, boardArea=361))
    assert np.array_equal(rows[0, 1], m.get_metadata_row(nextPlayer=Board.WHITE, boardArea=361))
    # own rank (3d, 7 channels) always sits in the side-to-move block, opponent (5k, 14) in the other
    for c in (0, 1):
        assert rows[1, c, 6:40].sum() == 7 and rows[1, c, 40:74].sum() == 14
