#!/usr/bin/env python
"""Architecture v2 section 4H: does mixing the two arms beat a single-arm ensemble of the same size? Saved probabilities only
(results/trunk_seq/es/val{v}/{C,T}_s{0..5}.npz), folds 1-4, equal-weight probability means, expected-score decisions.

Pre-specified (conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0, (4) >= 0.005): 3C+3T (seeds 0-2) vs 6C (seeds 0-5);
3C+3T (seeds 0-2) vs 6T (seeds 0-5); 6T vs 6C. Report-only: 3C+3T (seeds 3-5) vs 6C and 6T; 12 models vs 6C, 6T and 3C+3T;
1C+1T (seed 0) vs 2C (seeds 0-1) and 2T; the single-model means per arm; 6C vs 3C (0-2) and 6T vs 3T (the size effect).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402

CACHE, ES = ROOT / "results/trunk_seq", ROOT / "results/trunk_seq/es"
FOLDS = (1, 2, 3, 4)
SETS = {"6C": [("C", s) for s in range(6)], "6T": [("T", s) for s in range(6)], "3C+3T": [(a, s) for a in "CT" for s in range(3)],
        "3C+3T'": [(a, s) for a in "CT" for s in range(3, 6)], "12": [(a, s) for a in "CT" for s in range(6)],
        "3C": [("C", s) for s in range(3)], "3T": [("T", s) for s in range(3)], "2C": [("C", 0), ("C", 1)], "2T": [("T", 0), ("T", 1)],
        "1C+1T": [("C", 0), ("T", 0)]}
PRE = [("3C+3T", "6C"), ("3C+3T", "6T"), ("6T", "6C")]
REPORT = [("3C+3T'", "6C"), ("3C+3T'", "6T"), ("12", "6C"), ("12", "6T"), ("12", "3C+3T"), ("1C+1T", "2C"), ("1C+1T", "2T"), ("6C", "3C"), ("6T", "3T")]


def main():
    rows, per_fold, singles = [], {}, {"C": [], "T": []}
    for v in FOLDS:
        z = np.load(CACHE / f"r_val{v}.npz")
        q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
        y = z["y_va"].astype(int)
        P = {(a, s): np.load(ES / f"val{v}" / f"{a}_s{s}.npz")["proba"].astype(float) for a in "CT" for s in range(6)}
        for a in "CT":
            singles[a].append([balanced_mean_over_n(evaluate(P[(a, s)], y, q)[1], y, q["n_games"].to_numpy()) for s in range(6)])
        S = {k: evaluate(np.mean([P[m] for m in members], axis=0), y, q)[1] for k, members in SETS.items()}
        for k, s in S.items():
            per_fold.setdefault(k, []).append(balanced_mean_over_n(s, y, q["n_games"].to_numpy()))
        rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
        print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in SETS) + f"  single C {np.mean(singles['C'][-1]):.4f} T {np.mean(singles['T'][-1]):.4f}", flush=True)
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
    for k in SETS:
        print(f"{k:7s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k])
              + "  by n " + " ".join(f"{s}: {balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)):.4f}" for s in SIZES))
    print(f"single-model means over folds and seeds: C {np.mean(singles['C']):.4f} (seed sd {np.std(np.mean(singles['C'], axis=0)):.4f}), T {np.mean(singles['T']):.4f} (seed sd {np.std(np.mean(singles['T'], axis=0)):.4f})")
    out = {"scores": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k]} for k in SETS}, "singles": singles, "comparisons": {}}
    for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
        out["comparisons"][group] = {}
        for x, w in pairs:
            dd = paired_diff(t[x].to_numpy(), t[w].to_numpy(), y, pid, n, n=5000)
            pf = [a - b for a, b in zip(per_fold[x], per_fold[w])]
            cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
            out["comparisons"][group][f"{x} vs {w}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
            print(f"[{group}] {x} vs {w}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
                  + f"  -> {'PASS' if all(cond.values()) else 'no'}", flush=True)
    (ES / "members.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
