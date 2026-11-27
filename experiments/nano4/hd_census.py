#!/usr/bin/env python
"""Report only: dataset v1.1 against v1.0 -- is sgf_content identical once HD[n] is removed, and how are the handicaps distributed?"""
import re
import sys

import numpy as np
import pandas as pd

HD = re.compile(r"HD\[(-?\d+)\]")
ORDER = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
rows, bad = [], 0
for r in ORDER:
    a = pd.read_csv(f"data/training/train_{r}.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(f"data/dl_v1.1/training/train_{r}.csv", dtype=str, keep_default_na=False)
    same_meta = (a[["player_id", "game_id", "rank", "color"]].to_numpy() == b[["player_id", "game_id", "rank", "color"]].to_numpy()).all()
    hd = b["sgf_content"].str.extract(r"HD\[(-?\d+)\]", expand=False)
    stripped = b["sgf_content"].str.replace(r"HD\[-?\d+\]", "", regex=True)
    identical = (stripped.to_numpy() == a["sgf_content"].to_numpy())
    other = b["sgf_content"].str.count(r"[A-Z]{2}\[") - b["sgf_content"].str.count(r"(?:SZ|HD)\[") - b["sgf_content"].str.count(r";[BW]\[")
    n_hd = hd.notna().sum()
    print(f"{r}: rows {len(b)} meta same {same_meta} sgf identical minus HD {identical.sum()}/{len(b)} HD present {n_hd} other-property rows {(other > 0).sum()}", flush=True)
    bad += (~identical).sum()
    b["hd"] = hd.astype(float)
    rows.append(b[["rank", "color", "hd", "sgf_content"]].assign(n_moves=b["sgf_content"].str.count(r";[BW]\[")))
    if r == "1D":
        ex = b[b["hd"].astype(float) > 0].head(2)
        for s in ex["sgf_content"]:
            print("  example handicap game:", s[:160], flush=True)
t = pd.concat(rows, ignore_index=True)
print(f"total rows with sgf differing beyond HD: {bad}")
print("HD value counts:", t["hd"].value_counts(dropna=False).sort_index().to_dict())
print("share HD>0 by rank:", t.groupby("rank")["hd"].apply(lambda x: float((x > 0).mean())).reindex(ORDER).round(4).to_dict())
print("share HD>0 by rank and target colour:")
print(t.assign(h=t["hd"] > 0).pivot_table(index="rank", columns="color", values="h", aggfunc="mean").reindex(ORDER).round(4).to_string())
print("mean HD among HD>0 by rank and colour:")
print(t[t["hd"] > 0].pivot_table(index="rank", columns="color", values="hd", aggfunc="mean").reindex(ORDER).round(2).to_string())
h = t[t["hd"] > 0]
first = h["sgf_content"].str.extract(r"HD\[\d+\](;[BW]\[[a-t]{0,2}\])(;[BW]\[[a-t]{0,2}\])", expand=True)
print("first two move colours in handicap games:", (first[0].str[1] + first[1].str[1]).value_counts().head(4).to_dict())
print("moves in handicap vs even games (median):", float(h["n_moves"].median()), float(t[t["hd"] == 0]["n_moves"].median()))
