"""Section 4M M8: the games at positions 21-40 of every player's seeded permutation (scoring.fold_players with max_games 40 minus the
20-game selection every score directory holds), folds 0-4, as a --game-keys CSV; plus the per-player game-count distribution.

  python -I logs/nano4/x40_keys.py --out results/data/x40_keys.csv --report results/data/x40_keys.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-games", type=int, default=40)
    ap.add_argument("--base-games", type=int, default=20, help="the games already scored (positions 1..base of the permutation); the extra keys are base+1..max")
    a = ap.parse_args()
    games, folds = player_games(), pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    counts = games.groupby("player_id").size()
    rep = {"players": int(len(counts)), "games": int(len(games)), "games_per_player_quantiles": {str(q): float(counts.quantile(q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
           "players_with_le_base_games": int((counts <= a.base_games).sum()), "players_with_ge_max_games": int((counts >= a.max_games).sum()), "base_games": a.base_games, "max_games": a.max_games, "folds": {}}
    rows = []
    for f in (0, 1, 2, 3, 4):
        base = fold_players(games, folds, f, max_games=a.base_games)
        wide = fold_players(games, folds, f, max_games=a.max_games)
        b = {r.player_id: list(r.game_keys) for r in base.itertuples()}
        n_new, n_players = 0, 0
        for r in wide.itertuples():
            old = b.get(r.player_id, [])
            assert list(r.game_keys)[:len(old)] == old, r.player_id  # the same permutation: the first 20 are the scored selection
            extra = list(r.game_keys)[len(old):]
            if extra:
                n_players += 1
            for k in extra:
                rows.append({"fold": f, "player_id": r.player_id, "rank": r.rank, "game_key": k})
            n_new += len(extra)
        rep["folds"][str(f)] = {"players": int(len(wide)), "players_with_extra": n_players, "extra_games": n_new}
    pd.DataFrame(rows).to_csv(a.out, index=False)
    rep["extra_games_total"] = len(rows)
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
