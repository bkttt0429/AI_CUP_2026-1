#!/usr/bin/env python
"""Report only (issue #23 leak check): games sharing their first 40 moves under different players, in case a game was stored truncated differently."""
import glob
import re

import pandas as pd

MOVE = re.compile(r";[BW]\[[a-t]{0,2}\]")
frames = [pd.read_csv(p, usecols=["player_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False) for p in sorted(glob.glob("data/training/train_*.csv"))]
g = pd.concat(frames, ignore_index=True)
folds = pd.read_csv("splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]
g["fold"] = g["player_id"].map(folds).astype(int)
g["prefix"] = [("".join(MOVE.findall(s)[:40])) for s in g["sgf_content"]]
g["n"] = g["prefix"].str.count(";")
long = g[g["n"] == 40].drop(columns=["sgf_content"])
print(f"games with at least 40 moves: {len(long)} of {len(g)}; distinct prefixes {long['prefix'].nunique()}", flush=True)
sz = long.groupby("prefix")["player_id"].transform("nunique")
d = long[sz > 1]
print(f"rows sharing the first 40 moves with another player: {len(d)} ({len(d) / len(long):.6f}); groups {d['prefix'].nunique()}", flush=True)
p = d.merge(d, on="prefix", suffixes=("", "_t"))
p = p[p["player_id"] != p["player_id_t"]]
print("colours (row, other):", p.groupby(["color", "color_t"]).size().to_dict(), flush=True)
print("same rank share:", float((p["rank"] == p["rank_t"]).mean()) if len(p) else None, flush=True)
print("other-colour pairs per fold:", p[p["color"] != p["color_t"]].groupby("fold").size().to_dict(), flush=True)
