"""The handicap information of dataset v1.1 (HD[n], issue #42) as question-level features (proposal (a)).

The server assigns the handicap from the two players' ranks: HD[1] (讓先, about 16% of games) when they are about one
rank apart, HD[n] stones (3.6%) when n ranks apart, HD[0] (80%) when equal. It is label-side information, like the
opponent channel of fused_opp_diff but more direct. The `color` column is the ACTUAL colour (issue #42's census), so the
labelled player is the upper hand of a handicap game when their colour is W.

Per game: hd_even (HD = 0), hd_sente (HD = 1), hd_stones (HD >= 2), hd_gap_up (the gap when the labelled player is the
upper hand, else 0), hd_gap_down (the gap when they are the lower hand, else 0); the signed gap is up - down. A record
without HD (dataset v1.0) gives zeros in every column. Per question: the mean over its games.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.sgf import handicap

HD_FEATURE_NAMES = ["hd_even", "hd_sente", "hd_stones", "hd_gap_up", "hd_gap_down"]
HD_COLUMNS = ["game_key", "player_id", "rank", "color", "hd"]


def hd_table(train_dir):
    """game_key, player_id, rank, color, hd (HD[n] of the record as float; NaN when absent) of every game of the training
    CSVs in `train_dir` (reads the sgf_content column once; a few CPU minutes over a million rows)."""
    frames = []
    for p in sorted(Path(train_dir).glob("train_*.csv")):
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False)
        frames.append(pd.DataFrame({"game_key": df["rank"] + ":" + df["game_id"], "player_id": df["player_id"], "rank": df["rank"], "color": df["color"],
                                    "hd": df["sgf_content"].map(handicap).astype(float)}))
    if not frames:
        raise FileNotFoundError(f"no train_*.csv in {train_dir}")
    return pd.concat(frames, ignore_index=True)


def read_hd_table(path):
    """An hd_table written with to_csv (hd NaN when absent)."""
    t = pd.read_csv(path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False, na_values={"hd": [""]})
    t["hd"] = pd.to_numeric(t["hd"], errors="coerce").astype(float)
    return t


def signed_gap(hd, color):
    """The server's rank gap seen from the labelled player: +HD when they are the upper hand (actual colour W), -HD when
    the lower hand (B); 0 for an even game or an absent HD. Arrays in, array out."""
    h = np.nan_to_num(np.asarray(hd, dtype=float), nan=0.0)
    sign = np.where(np.asarray(color, dtype=str) == "W", 1.0, -1.0)
    return h * sign


def game_hd_features(hd, color):
    """[m, 5]: HD_FEATURE_NAMES of every game from its HD (NaN = absent) and the labelled player's actual colour."""
    h = np.asarray(hd, dtype=float)
    known = np.isfinite(h)
    gap = signed_gap(h, color)
    return np.column_stack([(known & (h == 0)).astype(float), (known & (h == 1)).astype(float), (known & (h >= 2)).astype(float),
                            np.maximum(gap, 0.0), np.maximum(-gap, 0.0)])


def question_hd_features(table, questions):
    """X [Q, 5]: the mean of game_hd_features over each question's game_keys, looked up in an hd_table (KeyError for a game
    the table lacks)."""
    g = table.drop_duplicates("game_key").set_index("game_key")
    F = game_hd_features(g["hd"].to_numpy(), g["color"].to_numpy())
    row = {k: i for i, k in enumerate(g.index)}
    rows = [F[[row[k] for k in ks]].mean(axis=0) for ks in questions["game_keys"]]
    return np.stack(rows) if rows else np.zeros((0, len(HD_FEATURE_NAMES)))


def question_hd_features_from_games(games):
    """[5]: the mean of game_hd_features over (sgf, colour) pairs, for the submission's questions (the test files' HD[])."""
    if not games:
        return np.zeros(len(HD_FEATURE_NAMES))
    hd = [np.nan if (h := handicap(sgf)) is None else float(h) for sgf, _ in games]
    return game_hd_features(hd, [c for _, c in games]).mean(axis=0)


def hd_summary(table):
    """Counts of HD values (\"absent\" for NaN) and the shares of games with HD >= 1 and HD >= 2."""
    h = table["hd"].to_numpy(dtype=float)
    known = np.isfinite(h)
    counts = {"absent": int((~known).sum())} if (~known).any() else {}
    for v in sorted(set(h[known].astype(int).tolist())):
        counts[str(v)] = int((h == v).sum())
    return {"games": int(len(h)), "hd": counts, "share_hd1": float((known & (h >= 1)).mean()) if len(h) else 0.0,
            "share_hd2": float((known & (h >= 2)).mean()) if len(h) else 0.0}
