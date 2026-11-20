"""Section 4M M7(a): size-routed ensembles from stored validation members. Questions with n <= --split games take the probability
average of the --low arms' members, the others that of the --high arms'; compared with the --base arms' average on every question
(paired player bootstrap, conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005). Also reports the low arms
alone on every question and, when --high differs from --base, the high arms alone.

  python -I logs/nano4/n_routed.py --low C-s5-10 T-s5-10 --high C T --base C T --seeds 0 1 2 --out results/trunk_seq/es/routed_m7a.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402

ES = ROOT / "results/trunk_seq/es"
RANK = {lv: i for i, lv in enumerate(LEVELS)}
FOLDS = (1, 2, 3, 4)


def load_fold(v):
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
    return np.array([RANK[r] for r in m["rank"]]), np.array(m["n_games"]), np.array(m["player_id"], dtype=str), pd.DataFrame(m)


def proba(v, arms, seeds):
    return np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low", nargs="+", required=True, help="arms for questions with n <= split")
    ap.add_argument("--high", nargs="+", required=True, help="arms for the other questions")
    ap.add_argument("--base", nargs="+", required=True, help="the reference arms, every question")
    ap.add_argument("--split", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    k = len(a.seeds)
    names = {"routed": f"routed(n<={a.split}: {'+'.join(a.low)}; else {'+'.join(a.high)}) x{k}", "base": f"{'+'.join(a.base)} x{k}", "low_all": f"{'+'.join(a.low)} x{k} all n"}
    if a.high != a.base:
        names["high_all"] = f"{'+'.join(a.high)} x{k} all n"
    S, per_fold, ys, ns, pids = {x: [] for x in names}, {x: [] for x in names}, [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        ys.append(y); ns.append(n); pids.append(pid)
        P = {"base": proba(v, a.base, a.seeds), "low_all": proba(v, a.low, a.seeds)}
        P["high_all"] = proba(v, a.high, a.seeds) if a.high != a.base else P["base"]
        P["routed"] = np.where((n <= a.split)[:, None], P["low_all"], P["high_all"])
        for x in names:
            s = evaluate(P[x], y, q)[1]
            S[x].append(s)
            per_fold[x].append(balanced_mean_over_n(s, y, n))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
    rep = {"args": vars(a), "names": names, "ensembles": {}, "pairs": {}}
    for x in names:
        s = np.concatenate(S[x])
        rep["ensembles"][x] = {"score": float(np.mean(per_fold[x])), "per_fold": per_fold[x], "by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES}}
        print(f"{x:9s} {rep['ensembles'][x]['score']:.4f}  folds {[round(e, 4) for e in per_fold[x]]}  by n " + " ".join(f"{m}: {rep['ensembles'][x]['by_n'][str(m)]:.4f}" for m in SIZES) + f"   ({names[x]})")
    pairs = [("routed", "base"), ("low_all", "base"), ("routed", "low_all")] + ([("high_all", "base")] if "high_all" in names else [])
    for x, z in pairs:
        d = paired_diff(np.concatenate(S[x]), np.concatenate(S[z]), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(d["diff"] >= 0.005)}
        rep["pairs"][f"{x} vs {z}"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out)


if __name__ == "__main__":
    main()
