#!/usr/bin/env python
"""Report only: in HD>=2 games, which side is the labelled player? If `color` is the real colour, targets labelled W (the stronger side,
who gives the handicap) should be higher-ranked than targets labelled B; if `color` follows the reversed move labels, the opposite."""
import re
import numpy as np
import pandas as pd
ORDER = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
IDX = {r: i for i, r in enumerate(ORDER)}
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")
STAR9 = ["pd", "dp", "pp", "dd", "jj", "dj", "pj", "jd", "jp"]
frames = []
for r in ORDER:
    b = pd.read_csv(f"data/dl_v1.1/training/train_{r}.csv", dtype=str, keep_default_na=False, usecols=["player_id", "rank", "color", "sgf_content"])
    b["hd"] = b["sgf_content"].str.extract(r"HD\[(\d+)\]", expand=False).astype(int)
    frames.append(b)
t = pd.concat(frames, ignore_index=True)
t["y"] = t["rank"].map(IDX)
print("rows", len(t), "players", t["player_id"].nunique(), "players with any HD>=2 game", t[t["hd"] >= 2]["player_id"].nunique(), flush=True)
for lo, hi, name in ((0, 0, "HD=0"), (1, 1, "HD=1"), (2, 9, "HD>=2")):
    s = t[(t["hd"] >= lo) & (t["hd"] <= hi)]
    g = s.groupby("color")["y"].agg(["count", "mean"])
    print(f"{name}: mean labelled rank index by `color`: " + " ".join(f"{c}: n={int(g.loc[c, 'count'])} mean={g.loc[c, 'mean']:.3f}" for c in g.index), flush=True)
h = t[t["hd"] >= 2]
print("HD>=2 by hd and color, mean rank index:")
print(h.pivot_table(index="hd", columns="color", values="y", aggfunc=["mean", "count"]).round(2).to_string())
# are the handicap points free of the first 2*hd recorded moves? which star points does the first recorded move avoid
occ = {k: 0 for k in STAR9}
n = 0
for hd, s in zip(h["hd"], h["sgf_content"]):
    mv = MOVE.findall(s)[:40]
    pts = {x for _, x in mv if x}
    for k in STAR9:
        occ[k] += k in pts
    n += 1
print("share of HD>=2 games whose first 40 recorded moves touch each star point:", {k: round(v / n, 3) for k, v in occ.items()})
# HD=0 reference
occ0 = {k: 0 for k in STAR9}; n0 = 0
for s in t[t["hd"] == 0]["sgf_content"].head(20000):
    pts = {x for _, x in MOVE.findall(s)[:40] if x}
    for k in STAR9:
        occ0[k] += k in pts
    n0 += 1
print("same for 20k HD=0 games:", {k: round(v / n0, 3) for k, v in occ0.items()})
