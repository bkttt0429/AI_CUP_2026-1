#!/usr/bin/env python
"""Architecture v2 section 4C: the label-overlap variants (-rs question redraws, -uq uniqueness weights, -bg player bagging) of C and T.
Pre-specified: each variant's six-model C + T probability average (seeds 0-2) against the plain six-model average on the same seeds,
folds 1-4, conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005. Single-model means are report only."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff
C, ES = ROOT / "results/trunk_seq", ROOT / "results/trunk_seq/es"
VARIANTS, SEEDS = ("", "-rs", "-uq", "-bg"), (0, 1, 2)
rows, per_fold = [], {}
for v in (1, 2, 3, 4):
    z = np.load(C / f"r_val{v}.npz"); q = pd.DataFrame(json.loads((C / f"r_val{v}.json").read_text())["val"]); y = z["y_va"]
    S = {}
    for suf in VARIANTS:
        P = [np.load(ES / f"val{v}" / f"{arm}{suf}_s{s}.npz") for arm in ("C", "T") for s in SEEDS]
        S["CT" + suf] = evaluate(np.mean([p["proba"] for p in P], axis=0), y, q)[1]
        S["single" + suf] = np.mean([p["scores"] for p in P], axis=0)
    for k, s in S.items():
        per_fold.setdefault(k, []).append(balanced_mean_over_n(s, y, q["n_games"].to_numpy()))
    rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
t = pd.concat(rows, ignore_index=True); y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
out = {"scores": {k: {"score": float(np.mean(v)), "per_fold": v} for k, v in per_fold.items()}, "comparisons": {}}
for k in per_fold:
    print(f"{k:10s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k]))
for suf in VARIANTS[1:]:
    for lvl, base in (("CT", "CT"), ("single", "single")):
        x = lvl + suf
        dd = paired_diff(t[x].to_numpy(), t[base].to_numpy(), y, pid, n, n=5000)
        pf = [a - b for a, b in zip(per_fold[x], per_fold[base])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
        grp = "pre_specified" if lvl == "CT" else "report_only"
        out["comparisons"].setdefault(grp, {})[f"{x} vs {base}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"[{grp}] {x} vs {base}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + "  by n " + " ".join(f"{m}: {e:+.4f}" for m, e in dd["by_n"].items()) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
(ES / "variants.json").write_text(json.dumps(out, indent=1, default=float))
