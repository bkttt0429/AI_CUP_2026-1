from aicup.sgf import parse_moves
from check_consistency import select_positions

# B captures the W stone at A19 with its move at index 4 (A18); W's next move is index 5.
CAPTURE_GAME = "(;SZ[19];B[pd];W[aa];B[ba];W[dp];B[ab];W[pp];B[cc];W[dd])"


def test_select_positions_first_target_move_at_or_after_each_threshold():
    moves = parse_moves(CAPTURE_GAME)
    assert select_positions(moves, "B", thresholds=(2, 5)) == [2, 6]
    assert select_positions(moves, "W", thresholds=(2, 5)) == [3, 5]


def test_select_positions_adds_first_target_move_after_a_capture():
    moves = parse_moves(CAPTURE_GAME)
    # thresholds beyond the game length contribute nothing; only the post-capture move remains
    assert select_positions(moves, "W", thresholds=(100,)) == [5]
    # the capturing side itself has no "opponent just captured" position here
    assert select_positions(moves, "B", thresholds=(100,)) == []


def test_select_positions_deduplicates_and_sorts():
    moves = parse_moves(CAPTURE_GAME)
    assert select_positions(moves, "W", thresholds=(5, 4)) == [5]
