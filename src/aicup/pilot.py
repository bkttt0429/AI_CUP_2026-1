"""Human SL pilot: CPU-side position extraction, player selection and pilot questions.

Workers replay a game once and keep, for every move of the target colour, the board
features, the legality mask and the index of the move actually played. The GPU side
(`aicup.humansl.score_positions`) then scores the same features under every profile.
"""
import numpy as np
from katago.game.board import Board
from katago.game.gamestate import GameState

from aicup.sgf import parse_moves

PASS_POS = 19 * 19


def game_positions(sgf, target_color, features):
    """Inputs for every move of `target_color`, taken from the position before that move."""
    gs = GameState(19, GameState.RULES_CHINESE)
    bins, globs, legal, target, ts = [], [], [], [], []
    for t, (c, x, y) in enumerate(parse_moves(sgf)):
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if c == target_color:
            board = gs.board
            b, g = gs.get_input_features(features)
            b8 = b[0].astype(np.uint8)
            if not np.array_equal(b8, b[0]):
                raise ValueError(f"move {t}: non-binary board feature, cannot pack as uint8")
            mask = np.zeros(PASS_POS + 1, dtype=bool)
            for i in range(PASS_POS):
                mask[i] = board.would_be_legal(pla, features.tensor_pos_to_loc(i, board))
            mask[PASS_POS] = True
            bins.append(b8)
            globs.append(g[0].astype(np.float32))
            legal.append(mask)
            target.append(PASS_POS if x is None else features.loc_to_tensor_pos(loc, board))
            ts.append(t)
        gs.play(pla, loc)
    n = len(ts)
    c_shape = tuple(features.bin_input_shape)
    return {
        "bins": np.stack(bins) if n else np.zeros((0,) + c_shape, dtype=np.uint8),
        "globs": np.stack(globs) if n else np.zeros((0, features.global_input_shape[0]), dtype=np.float32),
        "legal": np.stack(legal) if n else np.zeros((0, PASS_POS + 1), dtype=bool),
        "target": np.asarray(target, dtype=np.int16),
        "t": np.asarray(ts, dtype=np.int16),
        "color": np.full(n, 0 if target_color == "B" else 1, dtype=np.int8),
    }
