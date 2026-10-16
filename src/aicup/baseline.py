"""The official AI CUP 2026 rank baseline (Tutorial rank-prediction.ipynb) under the project protocol.

The parser and the network are the Tutorial's own files, imported by path and never edited,
so features and model are the official ones. Only the data split is ours (METHOD_NOTES §2.1):
training and validation players come from folds 1-4, whole players at a time, and fold 0 is
used for evaluation only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from aicup.metric import LEVELS

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = ROOT / "repo" / "AICup-2026-Tutorial"
_TUTORIAL = {}
_PARSER = None


def load_tutorial(tutorial_dir=TUTORIAL_DIR):
    """(SGFParseRankPrediction, GoRankResNet) from the Tutorial checkout, loaded once per process."""
    key = str(Path(tutorial_dir).resolve())
    if key not in _TUTORIAL:
        mods = []
        for name in ("utils", "network"):
            spec = importlib.util.spec_from_file_location(f"aicup_tutorial_{name}", Path(tutorial_dir) / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            mods.append(mod)
        _TUTORIAL[key] = (mods[0].SGFParseRankPrediction, mods[1].GoRankResNet)
    return _TUTORIAL[key]


def game_features(sgf):
    """The notebook's sgf_to_features: uint8 [500, 19, 19] for one game, None if it does not parse."""
    global _PARSER
    if _PARSER is None:
        _PARSER = load_tutorial()[0]()
    feat, err = _PARSER.features_from_sgf_content(sgf)
    return None if err is not None else feat


def split_players(folds, train_folds=(1, 2, 3, 4), val_games_per_rank=10_000, seed=42):
    """(train_ids, val_ids), both sorted: whole players of `train_folds`, validation >= target games per rank.

    Ranks go in LEVELS order; a rank's players are sorted by id, permuted with one shared
    RandomState(seed) and taken in that order until their games reach `val_games_per_rank`.
    Players of other folds are in neither list.
    """
    pool = folds[folds["fold"].isin(list(train_folds))]
    rng = np.random.RandomState(seed)
    val = []
    for rank in [r for r in LEVELS if r in set(pool["rank"])]:
        cand = pool[pool["rank"] == rank].sort_values("player_id")
        ids, games = cand["player_id"].to_numpy(), cand["n_games"].to_numpy()
        taken = 0
        for i in rng.permutation(len(cand)):
            if taken >= val_games_per_rank:
                break
            val.append(ids[i])
            taken += int(games[i])
        if taken >= games.sum():
            raise ValueError(f"rank {rank}: validation would take all {int(games.sum())} training games")
    val_set = set(val)
    return sorted(p for p in pool["player_id"] if p not in val_set), sorted(val)
