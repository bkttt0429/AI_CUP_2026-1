from aicup.sgf import board_size, from_gtp, parse_moves, to_gtp


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
