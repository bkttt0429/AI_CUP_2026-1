"""Helpers of scripts/l1b_cv.py (plan 2026-10-25-l1b-rank-heads, Task 7): the ladder's questions from scored tables.

The L3 ladder carves its validation questions (pilot.pilot_questions: 1 x 20, 2 x 10, 4 x 5 games) and
samples its training questions (qfeatures.sample_questions) from scoring.fold_players' per-player game
lists, whose order is a seeded permutation of the player's sorted keys. A scored games table (several
shards concatenated) holds those games in another order; players_from_scores restores fold_players'
order from the keys alone plus the player's total game count (splits/player_folds.csv), so the CV here
uses exactly the ladder's questions without reading the training data.
"""
import numpy as np
import pandas as pd


def restore_selection_order(player_id, keys, n_total, max_games=20, seed=42):
    """fold_players' order of a player's selected game keys: the selection was permutation(n_total)[:max_games] of the
    player's sorted keys; the selected keys sorted ascending are those indices sorted ascending."""
    keys = sorted(keys)
    rng = np.random.RandomState((int(player_id[:8], 16) + seed) % 2**32)
    sel = rng.permutation(int(n_total))[:max_games]
    if len(sel) != len(keys):
        raise ValueError(f"player {player_id}: {len(keys)} scored games but the selection of {n_total} games has {len(sel)}")
    ranks = np.argsort(np.argsort(sel))  # rank of each selected index among the selected -> position in the sorted keys
    return [keys[r] for r in ranks]


def players_from_scores(games, folds_df, max_games=20, seed=42):
    """DataFrame (rank, player_id, game_keys) in fold_players' order (sorted by rank, player_id; keys in selection
    order) from a scored games table and splits/player_folds.csv (n_games per player)."""
    n_total = folds_df.set_index("player_id")["n_games"].to_dict()
    rows = []
    for (rank, pid), idx in games.groupby(["rank", "player_id"], sort=True).indices.items():
        if pid not in n_total:
            raise KeyError(f"player {pid} is not in the folds table")
        rows.append({"rank": rank, "player_id": pid, "game_keys": restore_selection_order(pid, games["game_key"].to_numpy()[idx], n_total[pid], max_games, seed)})
    return pd.DataFrame(rows, columns=["rank", "player_id", "game_keys"])
