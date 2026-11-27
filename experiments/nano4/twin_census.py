#!/usr/bin/env python
"""Report only (issue #23 leak check): does a game of folds 1-4 also appear under its opponent's player_id, and was that row in
the scoring head's training folds? Reads the training CSVs and splits/player_folds.csv; scores nothing, evaluates nothing."""
import glob
import json
import sys

import numpy as np
import pandas as pd

TRAIN = {1: json.load(open("results/l1b/heads3_B.json")), 3: json.load(open("results/l1b/heads3_A.json"))}


def train_folds(meta):
    for k in ("train_folds", "folds"):
        if k in meta:
            return [int(f) for f in meta[k]]
    return None


tf_b, tf_a = train_folds(TRAIN[1]), train_folds(TRAIN[3])
print("head B train folds", tf_b, "head A train folds", tf_a, "top-level keys", sorted(TRAIN[1])[:20], flush=True)
if tf_a is None or tf_b is None:
    sys.exit("no train folds in the head side files")
scorer_train = {1: set(tf_b), 2: set(tf_b), 3: set(tf_a), 4: set(tf_a)}
folds = pd.read_csv("splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
frames = [pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False) for p in sorted(glob.glob("data/training/train_*.csv"))]
g = pd.concat(frames, ignore_index=True)
g["fold"] = g["player_id"].map(folds).astype(int)
g["sgf_id"] = pd.factorize(g["sgf_content"])[0]
g = g.drop(columns=["sgf_content"])
ORDER = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
g["y"] = g["rank"].map({r: i for i, r in enumerate(ORDER)})
print(f"rows {len(g)}, distinct move sequences {g['sgf_id'].nunique()}, distinct game_id {g['game_id'].nunique()}", flush=True)
size = g.groupby("sgf_id")["player_id"].transform("nunique")
print(f"rows whose sequence appears under another player_id: {(size > 1).mean():.4f}; group sizes {g.groupby('sgf_id').size().value_counts().head(6).to_dict()}", flush=True)
pairs = g[size > 1].merge(g[size > 1], on="sgf_id", suffixes=("", "_t"))
pairs = pairs[pairs["player_id"] != pairs["player_id_t"]]
print("pair colours (row, twin):", pairs.groupby(["color", "color_t"]).size().to_dict(), flush=True)
pairs["gap"] = (pairs["y"] - pairs["y_t"]).abs()
print("twin rank gap:", pairs["gap"].value_counts(normalize=True).sort_index().round(4).to_dict(), flush=True)
out = {}
for f in (1, 2, 3, 4):
    rows = g[g["fold"] == f]
    p = pairs[pairs["fold"] == f]
    in_train = p[p["fold_t"].isin(scorer_train[f])]
    out[f] = {"rows": int(len(rows)), "with_twin": float(p["sgf_id"].nunique() / rows["sgf_id"].nunique()), "twin_in_scoring_head_train": float(in_train["sgf_id"].nunique() / rows["sgf_id"].nunique()),
              "twin_same_fold": float(p[p["fold_t"] == f]["sgf_id"].nunique() / rows["sgf_id"].nunique()), "twin_fold0": float(p[p["fold_t"] == 0]["sgf_id"].nunique() / rows["sgf_id"].nunique())}
    print(f"fold {f}: {out[f]}", flush=True)
same = g[size == 1].groupby("sgf_id").size()
print(f"same player, same sequence twice: {(same > 1).sum()} sequences", flush=True)
