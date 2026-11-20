import pytest

from aicup.sgf import HANDICAP_STONES, board_size, from_gtp, other_colour, parse_moves, setup_and_moves, to_gtp


def test_parse_moves_with_pass_empty_brackets():
    assert parse_moves("(;SZ[19];B[pd];W[dp];B[])") == [
        ("B", 15, 3),
        ("W", 3, 15),
        ("B", None, None),
    ]


def test_parse_moves_tt_is_pass():
    assert parse_moves("(;SZ[19];W[tt])") == [("W", None, None)]


def test_to_gtp_skips_i_and_counts_rows_from_bottom():
    assert to_gtp(15, 3) == "Q16"
    assert to_gtp(3, 15) == "D4"
    assert to_gtp(8, 0) == "J19"


def test_from_gtp_inverts_to_gtp():
    assert from_gtp("Q16") == (15, 3)
    assert from_gtp("D4") == (3, 15)
    assert from_gtp("J19") == (8, 0)


def test_board_size():
    assert board_size("(;SZ[19];B[pd])") == 19


def test_setup_and_moves_places_handicap_stones_and_swaps_the_marks():
    # HD >= 2 (dataset v1.1): the stones are not recorded and the record marks White's first move as B
    stones, moves = setup_and_moves("(;SZ[19]HD[2];B[qf];W[cf];B[])")
    assert stones == [(15, 3), (3, 15)] and moves == [("W", 16, 5), ("B", 2, 5), ("W", None, None)]
    stones9, _ = setup_and_moves("(;SZ[19]HD[9];B[qf])")
    assert len(stones9) == 9 and stones9[-1] == (9, 9) and [chr(97 + x) + chr(97 + y) for x, y in stones9] == HANDICAP_STONES[9]
    # HD[0], HD[1] (讓先) and v1.0 records (no HD) are replayed as written
    for sgf in ("(;SZ[19]HD[0];B[pd];W[dp])", "(;SZ[19]HD[1];B[pd];W[dp])", "(;SZ[19];B[pd];W[dp])"):
        assert setup_and_moves(sgf) == ([], parse_moves(sgf))
    with pytest.raises(ValueError):
        setup_and_moves("(;SZ[19]HD[10];B[pd])")
    assert other_colour("B") == "W" and other_colour("W") == "B"
