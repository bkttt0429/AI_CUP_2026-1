#!/usr/bin/env python
"""Section 4M M160b (2026-11-15, report only): held-out identification accuracy by the length of the game (design doc M160b).

The identity networks read the first 256 plies of a game (28.5% of the games are longer). Held-out own games, all-players
networks (seed 7 = 40,000 steps; the average of seeds 0, 1, 3, 7): top-1 / top-5 by the number of plies of the game.

  python -I logs/nano4/m160b_top1_by_length.py
"""
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; LB = [(0, 100), (100, 150), (150, 200), (200, 256), (256, 300), (300, 10 ** 6)]


def main():
    csv.field_size_limit(10 ** 9); q = np.load(ROOT / "results/m84_cache/queries.npz"); Q, H = int(q["Q"]), int(q["H"]); truth = q["held_truth"].astype(int); hkeys = q["qkey"][2 * Q:2 * Q + H].astype(str); want = set(hkeys); n_ply = {}; pat = re.compile(r";[BW]\[")
    for f in sorted((ROOT / "data/training").glob("train_*.csv")):
        with open(f, newline="") as fh:
            r = csv.reader(fh); next(r)
            for pid, gid, rank, color, sgf in r:
                k = f"{rank}:{gid}"
                if k in want:
                    n_ply[k] = len(pat.findall(sgf))
    L = np.array([n_ply[k] for k in hkeys]); band = np.array([next(i for i, (lo, hi) in enumerate(LB) if lo <= n < hi) for n in L]); print(f"held-out games {H}; plies: quartiles " + " / ".join(f"{x:.0f}" for x in np.quantile(L, [0.1, 0.25, 0.5, 0.75, 0.9])) + f"; 256 or more {float((L >= 256).mean()):.3f}")
    S_all = {}
    for name, tag in (("seed 7, 40,000 steps", "_s7"), ("seed 0", ""), ("seed 1", "_s1"), ("seed 3", "_s3")):
        z = np.load(ROOT / f"results/m120/out_-1{tag}.npz"); pos = {int(g): i for i, g in enumerate(z["qidx"])}; S_all[name] = z["qvec"][[pos[2 * Q + i] for i in range(H)]].astype(np.float64) @ z["protos"].astype(np.float64).T
    S_all["average of the four"] = sum(S_all.values()) / 4; rep = {"bands": LB, "n": [int((band == i).sum()) for i in range(len(LB))]}
    print("plies:              " + "  ".join(f"{lo}-{'' if hi > 10 ** 5 else hi - 1}".ljust(9) for lo, hi in LB)); print("held-out games:     " + "  ".join(f"{int((band == i).sum()):<9d}" for i in range(len(LB))))
    for name in ("seed 7, 40,000 steps", "average of the four"):
        S = S_all[name]; r = (S > S[np.arange(H), truth][:, None]).sum(1); t1 = [float((r[band == i] == 0).mean()) for i in range(len(LB))]; t5 = [float((r[band == i] < 5).mean()) for i in range(len(LB))]; rep[name] = {"top1": t1, "top5": t5}
        print(f"{name}: top-1 " + "  ".join(f"{x:.3f}".ljust(9) for x in t1)); print(" " * len(name) + "  top-5 " + "  ".join(f"{x:.3f}".ljust(9) for x in t5))
    (RV / "m160b_top1_by_length.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m160b_top1_by_length.json")


if __name__ == "__main__":
    main()
