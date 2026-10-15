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

# The 18 Human SL rank profiles the pilot scores, weakest first (index 0 = 12k, 17 = 6d).
PROFILES = [f"{n}k" for n in range(12, 0, -1)] + [f"{n}d" for n in range(1, 7)]


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


def profile_meta_for(own, opp, target_color, preaz=False):
    """Profile metadata with the target player at rank `own` and the opponent at `opp`."""
    if target_color == "B":
        return make_profile_meta(own, opp, preaz)
    return make_profile_meta(opp, own, preaz)


def meta_rows(specs):
    """Metadata input rows for each (own, opp) spec: [S, 2, 192], second axis = target colour B/W.

    Row [s, c] is the row KataGo builds when the target (colour c) is the side to move, which
    is the only situation in which we score a move.
    """
    rows = np.zeros((len(specs), 2, 192), dtype=np.float32)
    for s, (own, opp) in enumerate(specs):
        for c, (color, pla) in enumerate((("B", Board.BLACK), ("W", Board.WHITE))):
            rows[s, c] = profile_meta_for(own, opp, color).get_metadata_row(nextPlayer=pla, boardArea=361)
    return rows


def score_positions(model, bins, globs, legal, target, colors, rows, batch=64):
    """Log p of each target move under every metadata spec: [N, S] float32.

    Board features are uploaded once per batch and reused for all S specs; only the metadata
    row changes. FP32 throughout, illegal points masked before log-softmax, exactly as
    `_masked_policy_logp` (the path verified against the analysis engine).
    """
    device = next(model.parameters()).device
    rows_t = torch.as_tensor(rows, dtype=torch.float32, device=device)
    n, s_count = len(target), rows.shape[0]
    out = np.empty((n, s_count), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b = torch.as_tensor(bins[i:j], device=device).float()
            g = torch.as_tensor(globs[i:j], device=device).float()
            illegal = ~torch.as_tensor(legal[i:j], device=device)
            tgt = torch.as_tensor(target[i:j], device=device).long()[:, None]
            col = torch.as_tensor(colors[i:j], device=device).long()
            for s in range(s_count):
                outputs = model(b, g, input_meta=rows_t[s, col])
                logits = model.postprocess_output(outputs)[0][0][:, 0, :].float()
                logp = torch.log_softmax(logits.masked_fill(illegal, float("-inf")), dim=1)
                out[i:j, s] = logp.gather(1, tgt)[:, 0].cpu().numpy()
    return out


def load_humansl(ckpt_path, device, use_swa=False):
    model, swa_model, _ = load_model(ckpt_path, use_swa, device=device, pos_len=19, verbose=False)
    model = swa_model if use_swa else model
    model.eval()
    return model, Features(model.config, 19)


def stack_inputs(features, states, meta):
    """Batch several GameStates into (bin [B,C,19,19], global [B,G], meta [B,192]) float32 tensors.

    The metadata row is built per state for its own side to move, so self/opponent ranks
    swap automatically between Black-to-move and White-to-move positions.
    """
    bins, globs, metas = [], [], []
    for gs in states:
        b, g = gs.get_input_features(features)
        bins.append(b[0])
        globs.append(g[0])
        metas.append(meta.get_metadata_row(nextPlayer=gs.board.pla, boardArea=gs.board.x_size * gs.board.y_size))
    return (torch.tensor(np.stack(bins), dtype=torch.float32),
            torch.tensor(np.stack(globs), dtype=torch.float32),
            torch.tensor(np.stack(metas), dtype=torch.float32))


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
