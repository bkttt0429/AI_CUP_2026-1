"""Section 4M M31 A: cross-fit selection of the positive R families. The families are picked on one half of the folds from the earlier
per-fold differences (seeds 0-2) and scored on the other half with fresh seeds (3-5): folds 1-2 use the families positive on folds 3-4
(kitchen_nowin = opphsl + implied rank + opphist + oppshift), folds 3-4 those positive on folds 1-2 (opphsl_win = opphsl + winlm);
the composite against opphsl, paired player bootstrap. The selection is re-derived from the earlier JSONs and checked.
  python -I logs/nano4/m31_crossfit.py --scores results/r_variants/m31_r_s345.npz --out results/r_variants/m31_crossfit.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402

R = ROOT / "results/r_variants"
PRIOR = {"implied rank (opphsl_hd)": ("m15_r.json", "opphsl_hd vs opphsl"), "opphist": ("opphist_r.json", "opphist vs opphsl"),
         "oppshift": ("oppshift_r.json", "oppshift vs opphsl"), "winlm": ("winlm_r.json", "winlm vs base")}
REGISTERED = {(1, 2): {"winlm"}, (3, 4): {"implied rank (opphsl_hd)", "opphist", "oppshift"}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    chosen = {}
    for half in ((1, 2), (3, 4)):
        chosen[half] = set()
        for name, (f, key) in PRIOR.items():
            pf = json.load(open(R / f))["pairs"][key]["per_fold_diff"]
            m = float(np.mean([pf[v - 1] for v in half]))
            print(f"selection on folds {half}: {name:26s} half mean {m:+.4f}  -> {'in' if m > 0 else 'out'}")
            if m > 0:
                chosen[half].add(name)
    assert chosen == REGISTERED, f"selection {chosen} differs from the pre-registered {REGISTERED}"
    z = np.load(a.scores)
    y, pid, n, fold = z["y"], z["player_id"], z["n_games"], z["fold"]
    ref = z["S_opphsl"]
    comp = np.where(np.isin(fold, [1, 2]), z["S_kitchen_nowin"], z["S_opphsl_win"])  # folds 1-2 scored by what folds 3-4 chose, and vice versa
    rep = {"selection": {f"{h[0]}{h[1]}": sorted(s) for h, s in chosen.items()}, "pairs": {}}
    for name, s in (("crossfit", comp), ("kitchen (in-sample)", z["S_kitchen"]), ("kitchen_nowin", z["S_kitchen_nowin"]), ("opphsl_win", z["S_opphsl_win"])):
        d = paired_diff(s, ref, y, pid, n, n=a.n_boot)
        pf = [balanced_mean_over_n(s[fold == v], y[fold == v], n[fold == v]) - balanced_mean_over_n(ref[fold == v], y[fold == v], n[fold == v]) for v in (1, 2, 3, 4)]
        rep["pairs"][f"{name} vs opphsl"] = {**d, "per_fold_diff": pf, "pass": bool(d["ci95"][0] > 0 and d["diff"] >= 0.005)}
        print(f"{name} vs opphsl: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + ("  -> continue (member-level retrain)" if name == "crossfit" and rep["pairs"][f"{name} vs opphsl"]["pass"] else ""))
    naive = sum(json.load(open(R / f))["pairs"][k]["diff"] for f, k in PRIOR.values())
    rep["naive_sum_of_single_gains"] = naive
    print(f"naive sum of the four single gains (seeds 0-2, each against its own reference): {naive:+.4f}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
