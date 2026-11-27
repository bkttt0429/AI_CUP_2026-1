"""Section 4M M36: per-game log-softmax of the cross-fitted spatial-attention heads (experiments/nano4/e2e_pilot.py score, heads A then B)
as question-level R columns. Head A trained on folds 1-2, head B on folds 3-4: a game of fold 1 or 2 takes B, of fold 3 or 4 takes A,
of fold 0 (and the test set) their mean, so no game is scored by a head that saw its player."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

N_COLS = 20  # mean and std over the question's games of the 10 log-softmax values


def load_scores(score_dir, games_csv):
    """{game_key: chosen [10] float32} from the score parts and the games CSV (game_key, fold)."""
    files = sorted(glob.glob(str(Path(score_dir) / "part_*.npz")))
    if not files:
        raise FileNotFoundError(f"{score_dir}: no score parts")
    fold = pd.read_csv(games_csv, dtype={"game_key": str}).set_index("game_key")["fold"].to_dict()
    out = {}
    for f in files:
        z = np.load(f)
        a, b = z["logp0"], z["logp1"]
        for k, la, lb in zip(z["keys"], a, b):
            fo = fold[str(k)]
            out[str(k)] = (lb if fo in (1, 2) else la if fo in (3, 4) else 0.5 * (la + lb)).astype(np.float32)
    return out


def question_features(game_keys, scores):
    """[Q, 20]: per question the mean and the std over its games of the chosen log-softmax; KeyError for an unscored game."""
    rows = []
    for ks in game_keys:
        m = np.stack([scores[str(k)] for k in ks])
        rows.append(np.r_[m.mean(axis=0), m.std(axis=0)])
    return np.asarray(rows, dtype=np.float32)
