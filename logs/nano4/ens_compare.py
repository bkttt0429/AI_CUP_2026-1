"""Ensemble-level adoption check (sections 4L.4 / 4M): the probability average of the `--new` arms' members against that of the
`--old` arms' members over the given seeds, folds 1-4, expected-score decision, paired player bootstrap; conditions (1) CI lower
bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005. Also each new arm against its old counterpart (same position in the lists).
--sizes restricts the evaluation to questions of those sizes (section 4M M2: 5 10). --new-seeds (default --seeds) are the new
arms' seeds; with unequal arm lists or seed lists only the combined pair is tested (section 4M M9: 12 = C T C-h4 T-h4 vs 6 = C T,
and the same-source control 12 = C T seeds 0-5 vs 6 = C T seeds 0-2).

  python -I logs/nano4/ens_compare.py --old C T --new C-ft T-ft --seeds 0 1 2 --out results/trunk_seq/es/ens_m3.json
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


def ensemble_scores(v, members, y, q, sizes):
    P = np.mean([np.load(ES / f"val{v}/{tag}_s{s}.npz")["proba"].astype(np.float64) for tag, s in members], axis=0)
    return evaluate(P, y, q, sizes=sizes)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="the old arms' seeds")
    ap.add_argument("--new-seeds", type=int, nargs="+", default=None, help="the new arms' seeds (default: --seeds)")
    ap.add_argument("--sizes", type=int, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds", type=int, nargs="+", default=list(FOLDS), help="validation folds (default 1 2 3 4)")
    a = ap.parse_args()
    new_seeds = a.new_seeds or a.seeds
    sizes = tuple(a.sizes) if a.sizes else SIZES
    k, kn = len(a.seeds), len(new_seeds)
    sets = {f"{kn * len(a.new)}_new": [(arm, s) for arm in a.new for s in new_seeds], f"{k * len(a.old)}_old": [(arm, s) for arm in a.old for s in a.seeds]}
    pairs = [(f"{kn * len(a.new)}_new", f"{k * len(a.old)}_old")]
    if len(a.old) == len(a.new) and new_seeds == a.seeds:
        for n_arm, o_arm in zip(a.new, a.old):
            sets[f"{k}_{n_arm}"], sets[f"{k}_{o_arm}"] = [(n_arm, s) for s in a.seeds], [(o_arm, s) for s in a.seeds]
            pairs.append((f"{k}_{n_arm}", f"{k}_{o_arm}"))
    S, per_fold, ys, ns, pids = {x: [] for x in sets}, {x: [] for x in sets}, [], [], []
    for v in a.folds:
        y, n, pid, q = load_fold(v)
        ys.append(y); ns.append(n); pids.append(pid)
        for x, members in sets.items():
            s = ensemble_scores(v, members, y, q, sizes)
            S[x].append(s)
            per_fold[x].append(balanced_mean_over_n(s, y, n, sizes=sizes))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
    rep = {"old": a.old, "new": a.new, "seeds": a.seeds, "new_seeds": new_seeds, "sizes": list(sizes), "ensembles": {}, "pairs": {}}
    for x in sets:
        s = np.concatenate(S[x])
        rep["ensembles"][x] = {"members": sets[x], "score": float(np.mean(per_fold[x])), "per_fold": per_fold[x],
                               "by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in sizes}}
        print(f"{x:16s} {rep['ensembles'][x]['score']:.4f}  folds {[round(e, 4) for e in per_fold[x]]}  by n " + " ".join(f"{m}: {rep['ensembles'][x]['by_n'][str(m)]:.4f}" for m in sizes))
    for x, z in pairs:
        d = paired_diff(np.concatenate(S[x]), np.concatenate(S[z]), y, pid, n, sizes=sizes, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= max(1, len(a.folds) - 1)), "4": bool(d["diff"] >= 0.005)}
        rep["pairs"][f"{x} vs {z}"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    if sizes != SIZES:
        print(f"evaluated on questions of {sizes} games only")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out)


if __name__ == "__main__":
    main()
