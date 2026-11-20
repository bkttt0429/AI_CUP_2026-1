#!/usr/bin/env python
"""Section 4M M183 part G (2026-11-16, report only): do the game numbers carry time?

M96 / M97 found no time structure in the anchors (label differences are flat along a player's game numbers), and read it as
"the displayed rank moves fast and the label is the centre of the period". That reading needs the game numbers to be in
time order. Check with something that must cluster in time: repeated games against the SAME opponent. For every validation
player's 20 games (a random sample of the games under that player's name), pairs of games whose opponent side is recognised
with a clear margin (m15 above the target side's q95 / q99) - is the pair more often the same recognised opponent when the
two game numbers are close?
  rematch pairs cluster at small distances  -> the numbers are in time order (the flat anchors then mean fast movement);
  no dependence on the distance             -> the numbers carry no time: M96 (2)(3) and M97 say nothing about time.

  python -I logs/nano4/m183_time.py [identifier tag, default SQ4]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "SQ4"; m3 = mb7.m3; q = np.load(mb7.CACHE / "queries.npz"); Q = int(q["Q"]); row = {k: i for i, k in enumerate(q["qkey"][:Q].astype(str))}
    z = np.load(ROOT / f"results/r_variants/m127_{tag}.npz"); TV = z["TV"]; st = TV[:, 0] - TV[:, 1:5].mean(1); best = z["best"]; lab = z["lab"]; null = st[Q:2 * Q]
    edges = [0, 300, 1000, 3000, 10000, 30000, 100001]; rng = np.random.default_rng(0)
    for qq in (0.95, 0.99):
        thr = float(np.quantile(null, qq)); same = np.zeros(len(edges) - 1); tot = np.zeros(len(edges) - 1); dsame, dall, nq = [], [], 0; perm_same = np.zeros(len(edges) - 1)
        for v in mb7.FOLDS:
            y, n, pid, qd = m3.load_fold(v); gk = list(qd["game_keys"])
            for j in np.flatnonzero(np.asarray(n) == 20):
                ks = [k for k in gk[j] if k in row]; r = np.array([row[k] for k in ks]); ok = st[r] >= thr
                if ok.sum() < 2:
                    continue
                ids = np.array([int(k.split("g_")[1]) for k in ks])[ok]; b_ = best[r][ok]; nq += 1; i, jj = np.triu_indices(len(ids), 1); d = np.abs(ids[i] - ids[jj]); s_ = b_[i] == b_[jj]
                h = np.digitize(d, edges) - 1; np.add.at(tot, h, 1); np.add.at(same, h[s_], 1); dsame += d[s_].tolist(); dall += d.tolist()
                p = rng.permutation(len(ids)); dp = np.abs(ids[p][i] - ids[p][jj]); np.add.at(perm_same, (np.digitize(dp, edges) - 1)[s_], 1)  # the same pairs with the game numbers shuffled inside the question
        dsame, dall = np.array(dsame), np.array(dall)
        print(f"margin above the target side's q{int(qq * 100)} ({thr:.2f}): {nq} questions with >= 2 such games, {int(tot.sum())} pairs, {int(same.sum())} of them the same recognised opponent ({same.sum() / max(tot.sum(), 1):.4f})")
        print("  distance between the two game numbers | pairs | share that is the same opponent | the same after shuffling the numbers inside the question")
        for k in range(len(edges) - 1):
            print(f"    {edges[k]:6d}-{edges[k + 1] - 1:6d} | {int(tot[k]):7d} | {same[k] / max(tot[k], 1):.4f} ({int(same[k])}) | {perm_same[k] / max(tot[k], 1):.4f}")
        if len(dsame):
            print(f"  median distance: same-opponent pairs {np.median(dsame):.0f}, all pairs {np.median(dall):.0f}; share of pairs closer than 1,000 numbers: same-opponent {float((dsame < 1000).mean()):.3f}, all {float((dall < 1000).mean()):.3f}")
    # a second look that needs no recognition: are a player's game numbers spread over the whole file or bunched?
    import pandas as pd
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", usecols=["game_key", "player_id"], dtype=str); hd["id"] = hd.game_key.str.split("g_").str[1].astype(int); g = hd.groupby("player_id").id.agg(["count", "min", "max", "std"]); g = g[g["count"] >= 30]
    print(f"\nplayers with >= 30 games ({len(g)}): span of their game numbers (max - min) median {float((g['max'] - g['min']).median()):.0f} of 100,000; sd of the numbers median {float(g['std'].median()):.0f} (uniform over the whole file would be 28,868); share of players whose span is below 50,000: {float(((g['max'] - g['min']) < 50000).mean()):.3f}")


if __name__ == "__main__":
    main()
