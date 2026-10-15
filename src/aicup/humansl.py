"""Reproduce KataGo's Human SL rank profiles and score a player's actual moves.

`make_profile_meta` mirrors `SGFMetadata::makeBasicRankProfile` in KataGo's
cpp/neuralnet/sgfmetadata.cpp so that Python evaluation uses exactly the
metadata the analysis engine uses for `humanSLProfile = rank_<r>`.
"""
import datetime

import numpy as np
import torch
from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.game.sgfmetadata import SGFMetadata
from katago.train.load_model import load_model

from aicup.sgf import to_gtp


def inverse_rank(s):
    """KataGo's inverse rank: 9d -> 1 ... 1d -> 9, 1k -> 10 ... 20k -> 29."""
    s = s.lower()
    n = int(s[:-1])
    if s.endswith("d"):
        return 10 - n
    if s.endswith("k"):
        return 9 + n
    raise ValueError(f"unrecognised rank {s!r}")


def make_profile_meta(black_rank, white_rank=None, preaz=False):
    """Metadata for `rank_<b>` (both players) or `rank_<b>_<w>`; `preaz` picks the 2016 profile."""
    meta = SGFMetadata()
    meta.inverseBRank = inverse_rank(black_rank)
    meta.inverseWRank = inverse_rank(white_rank or black_rank)
    meta.bIsHuman = True
    meta.wIsHuman = True
    meta.gameRatednessIsUnknown = True
    meta.tcIsUnknown = False
    meta.tcIsByoYomi = True
    meta.mainTimeSeconds = 1200.0
    meta.periodTimeSeconds = 30.0
    meta.byoYomiPeriods = 5
    meta.gameDate = datetime.date(2016, 9, 1) if preaz else datetime.date(2020, 3, 1)
    meta.source = SGFMetadata.SOURCE_KGS
    return meta


def load_humansl(ckpt_path, device, use_swa=False):
    model, swa_model, _ = load_model(ckpt_path, use_swa, device=device, pos_len=19, verbose=False)
    model = swa_model if use_swa else model
    model.eval()
    return model, Features(model.config, 19)


def _masked_policy_logp(model, features, gs, meta):
    """FP32 log-softmax over policy channel 0 for the side to move, illegal points at -inf."""
    board = gs.board
    bin_in, glob_in = gs.get_input_features(features)
    device = next(model.parameters()).device
    meta_row = meta.get_metadata_row(nextPlayer=board.pla, boardArea=board.x_size * board.y_size)
    with torch.no_grad():
        outputs = model(
            torch.tensor(bin_in, dtype=torch.float32, device=device),
            torch.tensor(glob_in, dtype=torch.float32, device=device),
            input_meta=torch.tensor(meta_row, dtype=torch.float32, device=device).reshape(1, -1),
        )
        # postprocess_output -> per head -> per output tensor; [0][0][0] is policy_logits of
        # sample 0 with shape [num_policy_outputs, pos_len*pos_len + 1], same as GameState.
        logits = model.postprocess_output(outputs)[0][0][0][0].float()
    legal = torch.zeros(features.pass_pos + 1, dtype=torch.bool)
    for i in range(features.pass_pos):
        legal[i] = board.would_be_legal(board.pla, features.tensor_pos_to_loc(i, board))
    legal[features.pass_pos] = True
    logits = logits.masked_fill(~legal.to(logits.device), float("-inf"))
    return torch.log_softmax(logits, dim=0).cpu().numpy()


def target_move_logps(model, features, moves, target_color, meta, rules=None):
    """Replay `moves`; before each move of `target_color`, score it under `meta`.

    Returns one dict per target move: t (index in `moves`), gtp, logp of the move actually
    played, and the full masked policy_logp vector (362) at that position.
    """
    gs = GameState(19, rules or GameState.RULES_CHINESE)
    out = []
    for t, (c, x, y) in enumerate(moves):
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        if c == target_color:
            if gs.board.pla != pla:
                raise ValueError(f"move {t}: {c} to play but board expects {gs.board.pla}")
            logp = _masked_policy_logp(model, features, gs, meta)
            pos = features.pass_pos if x is None else features.loc_to_tensor_pos(loc, gs.board)
            out.append({"t": t, "gtp": "pass" if x is None else to_gtp(x, y),
                        "logp": float(logp[pos]), "policy_logp": logp})
        gs.play(pla, loc)
    return out
