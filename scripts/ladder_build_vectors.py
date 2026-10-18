#!/usr/bin/env python
"""Build the per-game staged evidence vectors of the scored folds (L3 aggregator ladder input).

Reads <scores>/fold{F}/ (the finished parts written by score_folds.py) and writes
<out>/fold{F}.npz with game_key (fixed-width unicode), staged [G, 300], whole [G, 75],
sums [G, 18] (the CSV values) and n_moves [G]; see aicup.gamevec. Before writing, the vector
games must equal the scoring.fold_players selection of the fold exactly. Read the files back
with dict(np.load(path)) so every array is loaded once.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.gamevec import fold_vectors  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402


def check_selection(keys, players, fold):
    """Raise unless `keys` are exactly the games the fold's players are meant to contribute."""
    want = {k for ks in players["game_keys"] for k in ks}
    have = set(keys)
    if want != have:
        raise ValueError(f"fold {fold}: {len(have)} games scored but {len(want)} selected "
                         f"(missing {len(want - have)}, extra {len(have - want)})")


def build_fold(fold, all_games, folds_df, scores_dir, out_dir):
    vec = fold_vectors(Path(scores_dir) / f"fold{fold}")
    check_selection(vec["game_key"], fold_players(all_games, folds_df, fold), fold)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"fold{fold}.tmp.npz"
    np.savez_compressed(tmp, **vec)
    os.replace(tmp, out / f"fold{fold}.npz")
    print(f"fold {fold}: {len(vec['game_key'])} games, staged {vec['staged'].shape[1]} / whole {vec['whole'].shape[1]}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--scores", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ladder" / "vectors"))
    a = ap.parse_args()
    all_games = player_games()
    folds_df = pd.read_csv(ROOT / "splits" / "player_folds.csv")
    for fold in a.folds:
        build_fold(fold, all_games, folds_df, a.scores, a.out)


if __name__ == "__main__":
    main()
