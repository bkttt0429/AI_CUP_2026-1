#!/usr/bin/env python
"""Section 4M M241 (2026-11-19), report only: do the accounts with fewer than 20 games in the data set carry the same
"moves -> level" relation as the other accounts? Validation questions of 5 and 10 games, base ensemble (seeds 0-2 and 6-8):
by true level, the base posterior's expected level minus the truth, the share of base decisions below / at / above the truth
and the official score, for the two cohorts. Also the number of training players of every level in each cohort.

  python -I logs/nano4/m241_cohort.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

m3, b, s2, ES, FOLDS, SM = mb7.m3, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.SM
LEV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; RANK = {r: i for i, r in enumerate(LEV)}; KK = np.arange(10.0)


def main():
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "rank"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)
    act = rows.player_id.value_counts(); lab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].map(RANK)
    print("training players by level: [fewer than 15 games | 15-19 | 20 or more] and the share of the level's games that belong to players with fewer than 20 games")
    for c in range(10):
        p = lab.index[lab == c]; a = act.reindex(p).to_numpy(); print(f"   {LEV[c]:3s}: {int((a < 15).sum()):5d} | {int(((a >= 15) & (a < 20)).sum()):4d} | {int((a >= 20).sum()):5d}   games of the few-game accounts {float(a[a < 20].sum() / a.sum()):.3f}; median games per player {int(np.median(a))}")
    a_all = act.reindex(lab.index).to_numpy(); print(f"   all: {int((a_all < 15).sum())} | {int(((a_all >= 15) & (a_all < 20)).sum())} | {int((a_all >= 20).sum())} players; games of the few-game accounts {float(a_all[a_all < 20].sum() / a_all.sum()):.3f}")
    LF = {v: m3.load_fold(v) for v in FOLDS}; rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        y, n, pid, P = [], [], [], []
        for v in FOLDS:
            yy, nn, pp, _ = LF[v]; y.append(np.asarray(yy)); n.append(np.asarray(nn)); pid.append(np.asarray(pp).astype(str)); P.append(np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0))
        y, n, pid, P = np.concatenate(y), np.concatenate(n), np.concatenate(pid), np.concatenate(P); ng = act.reindex(pid).to_numpy(); d = np.asarray(b.decide(P), int); e = P @ KK; sc = SM[d, y]; few = ng < 20
        print(f"\nseeds {seeds}: validation questions {len(y)}; from accounts with fewer than 20 games: 5-game {int((few & (n == 5)).sum())}, 10-game {int((few & (n == 10)).sum())}")
        for s in (5, 10):
            print(f"  {s}-game questions, by true level: [few-game accounts | accounts with 20 or more games]  n, expected level - truth, share of decisions below / at / above the truth, score")
            for c in range(10):
                out = []
                for m in ((n == s) & (y == c) & few, (n == s) & (y == c) & ~few):
                    out.append(f"{int(m.sum()):5d}  {float((e[m] - c).mean()):+.3f}  {float((d[m] < c).mean()):.2f} / {float((d[m] == c).mean()):.2f} / {float((d[m] > c).mean()):.2f}  {float(sc[m].mean()):.3f}" if m.sum() >= 15 else f"{int(m.sum()):5d}  (too few)")
                print(f"   {LEV[c]:3s}: " + "   |   ".join(out))
            for nm, m in (("few-game accounts", (n == s) & few), ("20 or more games", (n == s) & ~few)):
                cb = float(np.mean([sc[m & (y == c)].mean() for c in range(10) if (m & (y == c)).sum() >= 15])); print(f"   {nm}: plain mean score {float(sc[m].mean()):.4f}, class-balanced {cb:.4f}; base decisions by level: " + " ".join(f"{LEV[c]} {float((d[m] == c).mean()):.3f}" for c in range(10)) + " | truth: " + " ".join(f"{LEV[c]} {float((y[m] == c).mean()):.3f}" for c in range(10)))
        rep[f"seeds_{seeds[0]}"] = {"few_share_5": float((few & (n == 5)).sum() / (n == 5).sum())}
    (ROOT / "results/r_variants/m241_cohort.json").write_text(json.dumps(rep, indent=1)); print("\nREPORT ONLY; written m241_cohort.json")


if __name__ == "__main__":
    sys.exit(main())
