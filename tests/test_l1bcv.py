"""aicup.l1bcv: the ladder's question carving reproduced from a scored games table."""
import numpy as np
import pandas as pd

from aicup import l1bcv
from aicup.scoring import fold_players


def test_restore_selection_order_matches_fold_players():
    rng = np.random.RandomState(0)
    rows, folds = [], []
    for i, (n, rank) in enumerate([(30, "A"), (20, "3D"), (7, "D"), (123, "6D")]):
        pid = f"{i:08x}deadbeef"
        rows += [(pid, rank, f"{rank}:g{rng.randint(1e6):06d}") for _ in range(n)]
        folds.append((rank, pid, n, 1))
    games = pd.DataFrame(rows, columns=["player_id", "rank", "game_key"]).drop_duplicates("game_key")
    folds_df = pd.DataFrame(folds, columns=["rank", "player_id", "n_games", "fold"])
    players = fold_players(games, folds_df, 1)  # <= 20 games each, in fold_players' order
    scored = pd.DataFrame([{"game_key": k, "player_id": p.player_id, "rank": p.rank} for p in players.itertuples() for k in p.game_keys])
    scored = scored.sample(frac=1.0, random_state=1).reset_index(drop=True)  # any order, as several shards would leave it
    restored = l1bcv.players_from_scores(scored, folds_df)
    assert restored["player_id"].tolist() == players["player_id"].tolist()
    for a, b in zip(restored.itertuples(), players.itertuples()):
        assert a.rank == b.rank and list(a.game_keys) == list(b.game_keys)
