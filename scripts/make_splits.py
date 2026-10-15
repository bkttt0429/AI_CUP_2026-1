#!/usr/bin/env python
"""Player-grouped 5-fold split and 5/10/20-game pseudo-questions (METHOD_NOTES §2.1).

Within each rank, players are sorted by game count (ties shuffled with the seed) and dealt
round-robin into folds so every fold gets a similar share of heavy players. For a held-out
player, games are shuffled once and carved into questions round-robin over the sizes
5 → 10 → 20, at most `max_q` per size, never reusing a game.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS  # noqa: E402

SEED = 42
SIZES = (5, 10, 20)


def assign_folds(players, n_folds=5, seed=SEED):
    """players: DataFrame[rank, player_id, n_games] -> same + fold (0..n_folds-1)."""
    rng = np.random.RandomState(seed)
    out = players.copy()
    out["_tie"] = rng.rand(len(out))
    out = out.sort_values(["rank", "n_games", "_tie"], ascending=[True, False, True])
    out["fold"] = out.groupby("rank").cumcount() % n_folds
    return out.drop(columns="_tie").sort_values(["rank", "fold", "player_id"]).reset_index(drop=True)


def make_questions(keys, fold, rank, player_id, sizes=SIZES, max_q=2):
    """Carve a player's (already shuffled) game keys into disjoint questions."""
    qs, pos, counts = [], 0, {n: 0 for n in sizes}
    progressed = True
    while progressed:
        progressed = False
        for n in sizes:
            if counts[n] < max_q and pos + n <= len(keys):
                qs.append({
                    "question_id": f"f{fold}_{rank}_{player_id[:8]}_n{n}_{counts[n]}",
                    "rank": rank, "player_id": player_id, "n_games": n,
                    "game_keys": ";".join(keys[pos:pos + n]),
                })
                pos += n
                counts[n] += 1
                progressed = True
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(ROOT / "data" / "training"))
    ap.add_argument("--out-dir", default=str(ROOT / "splits"))
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank"])
                    for p in sorted(glob.glob(f"{a.train_dir}/train_*.csv"))], ignore_index=True)
    df["game_key"] = df["rank"] + ":" + df["game_id"]
    players = df.groupby(["rank", "player_id"]).size().rename("n_games").reset_index()
    folds = assign_folds(players, a.folds)
    folds.to_csv(out_dir / "player_folds.csv", index=False)
    assert len(folds) == players["player_id"].nunique() == folds["player_id"].nunique()
    per = folds.groupby(["rank", "fold"]).size().unstack("fold")
    assert ((per.max(axis=1) - per.min(axis=1)) <= 1).all()
    print("players per rank x fold:\n", per.reindex(LEVELS).to_string())

    games_by_player = df.groupby("player_id")["game_key"].apply(lambda s: sorted(s))
    rng = np.random.RandomState(SEED)
    summary = []
    for f in range(a.folds):
        rows = []
        for r in folds[folds["fold"] == f].itertuples():
            keys = list(rng.permutation(games_by_player[r.player_id]))
            rows.extend(make_questions(keys, f, r.rank, r.player_id))
        q = pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])
        used = [k for s in q["game_keys"] for k in s.split(";")]
        assert len(used) == len(set(used)), f"fold {f}: a game appears in two questions"
        q.to_csv(out_dir / f"questions_fold{f}.csv", index=False)
        tab = q.groupby(["rank", "n_games"]).size().unstack("n_games").reindex(LEVELS)
        tab["fold"] = f
        summary.append(tab)
    print("questions per fold x rank x n_games:\n", pd.concat(summary).reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
