#!/usr/bin/env python
"""M135: where the games of illegal_games.csv sit in the scoring passes' shard lists (16 shards for Human SL, 8 for L1b)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.scoring import fold_players, player_games  # noqa: E402

E = Path("/work/u2507115/ext_rank")
players = fold_players(player_games(), pd.read_csv(E / "player_folds.csv"), 9, max_games=20)
games = [k for p in players.itertuples() for k in p.game_keys]
pos = {k: i for i, k in enumerate(games)}
for k in pd.read_csv(E / "illegal_games.csv")["game_key"]:
    i = pos[k]
    print(f"{k}: position {i} of {len(games)}; Human SL shard {i % 16} of 16, part {(i // 16) // 100 + 1} of its shard; L1b shard {i % 8} of 8, part {(i // 8) // 100 + 1}")
