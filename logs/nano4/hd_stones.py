#!/usr/bin/env python
"""Report only: in HD>=2 games, do Black's first moves sit on star points (handicap stones recorded as moves) or not (stones missing from the record)?"""
import re
import pandas as pd
STAR = {"dd", "pd", "dp", "pp", "jj", "dj", "pj", "jd", "jp"}
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")
rows = []
for r in ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]:
    b = pd.read_csv(f"data/dl_v1.1/training/train_{r}.csv", dtype=str, keep_default_na=False, usecols=["sgf_content", "color"])
    hd = b["sgf_content"].str.extract(r"HD\[(\d+)\]", expand=False).astype(int)
    for h in (0, 1, 2, 3, 4, 9):
        sub = b[hd == h]["sgf_content"]
        if len(sub) == 0:
            continue
        mv = [MOVE.findall(s)[:2 * max(h, 1)] for s in sub]
        first_black_star = [sum(1 for c, x in m if c == "B" and x in STAR) for m in mv]
        first_black = [m[0][1] in STAR if m else False for m in mv]
        alt = [all(m[i][0] == ("B" if i % 2 == 0 else "W") for i in range(len(m))) for m in mv]
        rows.append({"rank": r, "HD": h, "games": len(sub), "first_move_black_star": round(sum(first_black) / len(sub), 3),
                     "black_star_moves_in_first_2h": round(sum(first_black_star) / len(sub), 2), "strict_BW_alternation": round(sum(alt) / len(sub), 3)})
t = pd.DataFrame(rows)
print(t.groupby("HD")[["games", "first_move_black_star", "black_star_moves_in_first_2h", "strict_BW_alternation"]].agg({"games": "sum", "first_move_black_star": "mean", "black_star_moves_in_first_2h": "mean", "strict_BW_alternation": "mean"}).round(3).to_string())
