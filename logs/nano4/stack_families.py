"""Section 4M M31 B: stack the existing member families with a positive earlier point estimate on ct7's M22 members (validation
probabilities of folds 1-4, no training). Each family = the mean probability of its members. Primary: cross-fit greedy family selection
(Caruana, with replacement, ct7 included): start from ct7, add the family that raises the class-balanced score of the selection folds
most, stop below +0.0001 or after 20 steps; select on folds 1-2 and score folds 3-4, and the reverse; pooled paired player bootstrap
against ct7's 6 members. Report only (same questions as the selection, optimistic): all families equally weighted, all members equally weighted.
  python -I logs/nano4/stack_families.py --out results/trunk_seq/es/stack_m31.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402

FAMILIES = {"ct7": ("C-f0-x40-oh", "T-f0-x40-oh"), "jp": ("C-f0-x40-oh-jp", "T-f0-x40-oh-jp"), "x80": ("C-x80", "T-x80"), "h4": ("C-h4", "T-h4"),
            "mix50": ("C-mix50", "T-mix50"), "uq": ("C-uq", "T-uq"), "rs": ("C-rs", "T-rs"), "pl": ("C-pl", "T-pl"), "ohr": ("C-x40-ohr", "T-x40-ohr")}
FOLDS = (1, 2, 3, 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=5000); a = ap.parse_args()
    Y, N, PID, Q, PF, NM = {}, {}, {}, {}, {}, {}
    for v in FOLDS:
        Y[v], N[v], PID[v], Q[v] = load_fold(v)
        PF[v], NM[v] = {}, {}
        for fam, tags in FAMILIES.items():
            files = []
            for t in tags:
                fs = sorted(glob.glob(str(ES / f"val{v}/{t}_s*.npz")))
                if fam == "ct7":
                    fs = [f for f in fs if int(f.rsplit("_s", 1)[1].split(".")[0]) in (0, 1, 2)]
                if len(fs) < 3:
                    raise SystemExit(f"{t}: {len(fs)} members on fold {v}")
                files += fs
            PF[v][fam] = np.mean([np.load(f)["proba"].astype(np.float64) for f in files], axis=0); NM[v][fam] = len(files)
    print("members per family (fold 1): " + ", ".join(f"{k} {m}" for k, m in NM[1].items()), flush=True)

    def qscores(w, v):
        P = sum(c * PF[v][f] for f, c in w.items()) / sum(w.values())
        return evaluate(P, Y[v], Q[v], sizes=SIZES)[1]

    def score(w, folds):
        return float(np.mean([balanced_mean_over_n(qscores(w, v), Y[v], N[v]) for v in folds]))

    def greedy(folds):
        w, cur, path = {"ct7": 1}, score({"ct7": 1}, folds), []
        for _ in range(20):
            cand = sorted(((score({**w, f: w.get(f, 0) + 1}, folds), f) for f in FAMILIES), reverse=True)
            if cand[0][0] - cur < 1e-4:
                break
            w[cand[0][1]] = w.get(cand[0][1], 0) + 1; cur = cand[0][0]; path.append((cand[0][1], cur))
        return w, path

    rep = {"families": {k: list(t) for k, t in FAMILIES.items()}, "members_fold1": NM[1], "halves": {}, "pairs": {}}
    held = {}
    for sel, ev in (((1, 2), (3, 4)), ((3, 4), (1, 2))):
        w, path = greedy(sel)
        rep["halves"][f"select {sel} -> score {ev}"] = {"weights": w, "path": path, "selection_score": path[-1][1] if path else score(w, sel)}
        print(f"select on {sel}: weights {w}  path " + " -> ".join(f"{f} {s:.4f}" for f, s in path), flush=True)
        for v in ev:
            held[v] = qscores(w, v)
    base = {v: qscores({"ct7": 1}, v) for v in FOLDS}
    y, n, pid = (np.concatenate([d[v] for v in FOLDS]) for d in (Y, N, PID))
    arms = {"crossfit greedy": held, "all families equal (in-sample)": {v: qscores({f: 1 for f in FAMILIES}, v) for v in FOLDS},
            "all members equal (in-sample)": {v: qscores(dict(NM[v]), v) for v in FOLDS}, "ct7 + jp (in-sample)": {v: qscores({"ct7": 1, "jp": 1}, v) for v in FOLDS}}
    for name, S in arms.items():
        s, b = np.concatenate([S[v] for v in FOLDS]), np.concatenate([base[v] for v in FOLDS])
        d = paired_diff(s, b, y, pid, n, n=a.n_boot)
        pf = [balanced_mean_over_n(S[v], Y[v], N[v]) - balanced_mean_over_n(base[v], Y[v], N[v]) for v in FOLDS]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(d["diff"] >= 0.005)}
        rep["pairs"][f"{name} vs ct7"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"{name} vs ct7: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}", flush=True)
    rep["ct7_score"] = float(np.mean([balanced_mean_over_n(base[v], Y[v], N[v]) for v in FOLDS]))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
