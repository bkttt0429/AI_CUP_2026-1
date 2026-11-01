"""Human SL pilot: CPU-side position extraction, player selection and pilot questions.

Workers replay a game once and keep, for every move of the target colour, the board
features, the legality mask and the index of the move actually played. The GPU side
(`aicup.humansl.score_positions`) then scores the same features under every profile.
"""
import numpy as np
from katago.game.board import Board
from katago.game.gamestate import GameState

from aicup.sgf import parse_moves, setup_and_moves

PASS_POS = 19 * 19


def start_state(stones=(), rules=None):
    """A GameState at the start of a game: empty, Black to play; with `stones` (the Black setup stones (x, y) of a handicap
    game) placed outside the move history and White to play, as KataGo treats an SGF's AB[] stones."""
    gs = GameState(19, rules or GameState.RULES_CHINESE)
    if stones:
        for x, y in stones:
            gs.board.set_stone(Board.BLACK, gs.board.loc(x, y))
        gs.board.set_pla(Board.WHITE)
        gs.boards = [gs.board.copy()]
    return gs


def game_positions(sgf, target_color, features, handicap=True):
    """Inputs for every move of `target_color`, taken from the position before that move. handicap: replay HD >= 2 records
    with their handicap stones on the board and the actual colours (aicup.sgf.setup_and_moves); False replays the record as
    written (the behaviour before dataset v1.1 made the handicap known)."""
    stones, moves = setup_and_moves(sgf) if handicap else ([], parse_moves(sgf))
    gs = start_state(stones)
    bins, globs, legal, target, ts = [], [], [], [], []
    for t, (c, x, y) in enumerate(moves):
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


def select_players(questions, fold, per_rank=20, seed=42):
    """Up to `per_rank` players per rank that have a 20-game question in this fold's file.

    Returns DataFrame[rank, player_id, game_keys] where game_keys is the list of 20 keys
    from the player's `_n20_0` question. Ranks are processed in a fixed order and players
    sorted by id before sampling, so the result depends only on (questions, seed).
    """
    import pandas as pd

    q = questions[questions["question_id"].str.startswith(f"f{fold}_") & questions["question_id"].str.endswith("_n20_0")]
    rng = np.random.RandomState(seed)
    rows = []
    for rank in sorted(q["rank"].unique(), key=str):
        cand = q[q["rank"] == rank].sort_values("player_id")
        take = cand.iloc[sorted(rng.choice(len(cand), size=min(per_rank, len(cand)), replace=False))]
        for r in take.itertuples():
            rows.append({"rank": r.rank, "player_id": r.player_id, "game_keys": r.game_keys.split(";")})
    return pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])


def pilot_questions(players):
    """Per player: one 20-game question, two disjoint 10-game halves, four disjoint 5-game quarters."""
    import pandas as pd

    rows = []
    for p in players.itertuples():
        keys = list(p.game_keys)
        for n in (20, 10, 5):
            for j in range(len(keys) // n):
                rows.append({"question_id": f"{p.rank}_{p.player_id[:8]}_n{n}_{j}", "rank": p.rank,
                             "player_id": p.player_id, "n_games": n, "game_keys": keys[j * n:(j + 1) * n]})
    return pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])
