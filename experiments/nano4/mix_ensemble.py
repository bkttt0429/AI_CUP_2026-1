"""Section 4L.4 adoption check: the 12-model mixed-player ensemble (C-mix50 + T-mix50, seeds 0-5) against the section-4H 12-model
ensemble (C + T, seeds 0-5), folds 1-4, probabilities averaged within each ensemble, expected-score decision, paired player bootstrap;
conditions (1) CI lower bound > 0, (3) >= 3 of 4 folds > 0, (4) diff >= 0.005. Also 6-vs-6 (seeds 0-2) and each arm's 6-model ensemble.

  python -I logs/nano4/mix_ensemble.py [--mix-tag mix50] [--out results/trunk_seq/es/mix_ensemble.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402

ES = ROOT / "results/trunk_seq/es"
RANK = {lv: i for i, lv in enumerate(LEVELS)}
FOLDS = (1, 2, 3, 4)


def load_fold(v):
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
    import pandas as pd
    q = pd.DataFrame(m)
    return np.array([RANK[r] for r in m["rank"]]), np.array(m["n_games"]), np.array(m["player_id"], dtype=str), q


def ensemble_scores(v, members, y, q):
    P = np.mean([np.load(ES / f"val{v}/{tag}_s{s}.npz")["proba"].astype(np.float64) for tag, s in members], axis=0)
    return evaluate(P, y, q)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-tag", default="mix50")
    ap.add_argument("--out", default=str(ES / "mix_ensemble.json"))
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    sets = {"12_old": [(arm, s) for arm in ("C", "T") for s in range(6)],
            f"12_{a.mix_tag}": [(f"{arm}-{a.mix_tag}", s) for arm in ("C", "T") for s in range(6)],
            "6_old_s012": [(arm, s) for arm in ("C", "T") for s in range(3)],
            f"6_{a.mix_tag}_s012": [(f"{arm}-{a.mix_tag}", s) for arm in ("C", "T") for s in range(3)],
            f"6C_{a.mix_tag}": [(f"C-{a.mix_tag}", s) for s in range(6)], f"6T_{a.mix_tag}": [(f"T-{a.mix_tag}", s) for s in range(6)],
            "6C_old": [("C", s) for s in range(6)], "6T_old": [("T", s) for s in range(6)]}
    S, per_fold, ys, ns, pids = {k: [] for k in sets}, {k: [] for k in sets}, [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        ys.append(y); ns.append(n); pids.append(pid)
        for k, members in sets.items():
            try:
                s = ensemble_scores(v, members, y, q)
            except FileNotFoundError as e:
                print(f"{k}: fold {v} incomplete ({e})")
                s = np.full(len(y), np.nan)
            S[k].append(s)
            per_fold[k].append(balanced_mean_over_n(s, y, n) if np.isfinite(s).all() else float("nan"))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
    rep = {"ensembles": {}, "pairs": {}}
    for k in sets:
        s = np.concatenate(S[k])
        ok = np.isfinite(s).all()
        rep["ensembles"][k] = {"score": float(np.mean(per_fold[k])) if ok else None, "per_fold": per_fold[k],
                               "by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES} if ok else None}
        print(f"{k:14s} {rep['ensembles'][k]['score'] if ok else 'incomplete'}  folds {[round(x, 4) for x in per_fold[k]]}")
    pairs = [(f"12_{a.mix_tag}", "12_old"), (f"6_{a.mix_tag}_s012", "6_old_s012"), (f"6C_{a.mix_tag}", "6C_old"), (f"6T_{a.mix_tag}", "6T_old"), (f"12_{a.mix_tag}", f"6_{a.mix_tag}_s012")]
    for x, z in pairs:
        sx, sz = np.concatenate(S[x]), np.concatenate(S[z])
        if not (np.isfinite(sx).all() and np.isfinite(sz).all()):
            print(f"{x} vs {z}: incomplete")
            continue
        d = paired_diff(sx, sz, y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(d["diff"] >= 0.005)}
        rep["pairs"][f"{x} vs {z}"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print("written", a.out)


if __name__ == "__main__":
    main()
