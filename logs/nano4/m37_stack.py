"""Section 4M M37: cross-fit greedy family stacking on the pc (ct9) members, as stack_families.py (M31 B) but with the pc families and
the new adoption rule (CI lower > 0, >= 3 of 4 folds > 0, diff >= 0.0025), run once per seed set: --seeds 0 1 2 (base pc seeds 0-2,
i.e. ct9) and --seeds 3 4 5 (all fresh). --deploy-seeds 0 1 2 3 4 5 also reports the full-data greedy weights on the pooled seeds
(the deployment weights) and pc 12 vs pc 6.
  python -I logs/nano4/m37_stack.py --seeds 0 1 2 --out results/trunk_seq/es/stack_m37_s012.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402

FAMILIES = {"pc": ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"), "ohr": ("C-pc-f0-x40-ohr", "T-pc-f0-x40-ohr"), "uq": ("C-uq-pc-f0-x40-oh", "T-uq-pc-f0-x40-oh"),
            "jp": ("C-pc-f0-x40-oh-jp", "T-pc-f0-x40-oh-jp"), "h4": ("C-pc-f0-h4-x40-oh", "T-pc-f0-h4-x40-oh")}
FOLDS = (1, 2, 3, 4)
MIN_DIFF = 0.0025


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--deploy-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=5000); a = ap.parse_args()
    Y, N, PID, Q = {}, {}, {}, {}
    for v in FOLDS:
        Y[v], N[v], PID[v], Q[v] = load_fold(v)

    def fam_probs(seeds):
        PF = {v: {} for v in FOLDS}
        for v in FOLDS:
            for fam, tags in FAMILIES.items():
                files = [ES / f"val{v}/{t}_s{s}.npz" for t in tags for s in seeds]
                missing = [str(f) for f in files if not f.exists()]
                if missing:
                    raise SystemExit(f"missing {len(missing)} member files, e.g. {missing[0]}")
                PF[v][fam] = np.mean([np.load(f)["proba"].astype(np.float64) for f in files], axis=0)
        return PF

    def qscores(PF, w, v):
        P = sum(c * PF[v][f] for f, c in w.items()) / sum(w.values())
        return evaluate(P, Y[v], Q[v], sizes=SIZES)[1]

    def score(PF, w, folds):
        return float(np.mean([balanced_mean_over_n(qscores(PF, w, v), Y[v], N[v]) for v in folds]))

    def greedy(PF, folds):
        w, cur, path = {"pc": 1}, score(PF, {"pc": 1}, folds), []
        for _ in range(20):
            cand = sorted(((score(PF, {**w, f: w.get(f, 0) + 1}, folds), f) for f in FAMILIES), reverse=True)
            if cand[0][0] - cur < 1e-4:
                break
            w[cand[0][1]] = w.get(cand[0][1], 0) + 1; cur = cand[0][0]; path.append((cand[0][1], cur))
        return w, path

    y, n, pid = (np.concatenate([d[v] for v in FOLDS]) for d in (Y, N, PID))

    def pair(S, base):
        s, b = np.concatenate([S[v] for v in FOLDS]), np.concatenate([base[v] for v in FOLDS])
        d = paired_diff(s, b, y, pid, n, n=a.n_boot)
        pf = [balanced_mean_over_n(S[v], Y[v], N[v]) - balanced_mean_over_n(base[v], Y[v], N[v]) for v in FOLDS]
        cond = {"ci_lower_gt_0": bool(d["ci95"][0] > 0), "3of4_folds": bool(sum(e > 0 for e in pf) >= 3), f"diff_ge_{MIN_DIFF}": bool(d["diff"] >= MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass_new_rule": all(cond.values())}

    PF = fam_probs(a.seeds)
    rep = {"seeds": a.seeds, "families": {k: list(t) for k, t in FAMILIES.items()}, "halves": {}, "pairs": {}}
    held = {}
    for sel, ev in (((1, 2), (3, 4)), ((3, 4), (1, 2))):
        w, path = greedy(PF, sel)
        rep["halves"][f"select {sel} -> score {ev}"] = {"weights": w, "path": path}
        print(f"seeds {a.seeds}: select on {sel}: weights {w}  path " + " -> ".join(f"{f} {s:.4f}" for f, s in path), flush=True)
        for v in ev:
            held[v] = qscores(PF, w, v)
    base = {v: qscores(PF, {"pc": 1}, v) for v in FOLDS}
    arms = {"crossfit greedy": held, "all families equal (in-sample)": {v: qscores(PF, {f: 1 for f in FAMILIES}, v) for v in FOLDS}}
    arms.update({f"pc + {f} (in-sample)": {v: qscores(PF, {"pc": 1, f: 1}, v) for v in FOLDS} for f in FAMILIES if f != "pc"})
    for name, S in arms.items():
        r = pair(S, base); rep["pairs"][f"{name} vs pc"] = r
        print(f"{name} vs pc: {r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in r["per_fold_diff"])
              + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in r["by_n"].items()) + f"  -> {'PASS (new rule)' if r['pass_new_rule'] else 'no'}", flush=True)
    rep["pc_score"] = float(np.mean([balanced_mean_over_n(base[v], Y[v], N[v]) for v in FOLDS]))
    if a.deploy_seeds:
        PD = fam_probs(a.deploy_seeds)
        w, path = greedy(PD, FOLDS)
        rep["deploy"] = {"seeds": a.deploy_seeds, "weights": w, "path": path}
        print(f"deployment weights (full-data greedy, seeds {a.deploy_seeds}): {w}  path " + " -> ".join(f"{f} {s:.4f}" for f, s in path), flush=True)
        P6 = fam_probs(a.seeds)
        r = pair({v: qscores(PD, {"pc": 1}, v) for v in FOLDS}, {v: qscores(P6, {"pc": 1}, v) for v in FOLDS})
        rep["pairs"][f"pc {len(a.deploy_seeds)} seeds vs pc {len(a.seeds)} seeds"] = r
        print(f"pc {len(a.deploy_seeds)} seeds vs pc seeds {a.seeds}: {r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in r["per_fold_diff"]), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
