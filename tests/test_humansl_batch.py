import numpy as np
import torch
from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train import modelconfigs

from aicup.humansl import make_profile_meta, stack_inputs


def _states():
    """Three positions: empty board (B to move), after B pd (W to move), after W dp (B to move)."""
    g0 = GameState(19, GameState.RULES_CHINESE)
    g1 = GameState(19, GameState.RULES_CHINESE)
    g1.play(Board.BLACK, g1.board.loc(15, 3))
    g2 = GameState(19, GameState.RULES_CHINESE)
    g2.play(Board.BLACK, g2.board.loc(15, 3))
    g2.play(Board.WHITE, g2.board.loc(3, 15))
    return [g0, g1, g2]


def test_stack_inputs_shapes_and_dtype():
    features = Features(modelconfigs.config_of_name["b18c384nbt"], 19)
    b, g, m = stack_inputs(features, _states(), make_profile_meta("3d", "5k"))
    assert b.shape == (3, features.bin_input_shape[0], 19, 19) and b.dtype == torch.float32
    assert g.shape == (3, features.global_input_shape[0]) and g.dtype == torch.float32
    assert m.shape == (3, 192) and m.dtype == torch.float32


def test_stack_inputs_meta_follows_side_to_move():
    features = Features(modelconfigs.config_of_name["b18c384nbt"], 19)
    _, _, m = stack_inputs(features, _states(), make_profile_meta("3d", "5k"))
    m = m.numpy()
    # B to move: own rank 3d (7 channels), opponent 5k (14 channels); W to move: swapped.
    assert m[0, 6:40].sum() == 7 and m[0, 40:74].sum() == 14
    assert m[1, 6:40].sum() == 14 and m[1, 40:74].sum() == 7
    assert np.array_equal(m[0], m[2])
