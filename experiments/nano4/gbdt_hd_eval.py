#!/usr/bin/env python
"""Architecture v2 section 4B: the GBDTs of gbdt_hd_fit.py against MLP h32 on the same 64 R features and questions, folds 1-4.

Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): best GBDT vs MLP seed 0, and
MLP seed 0 + best GBDT vs MLP seed 0 + MLP seed 1 (probability averages). The best GBDT of a fold is the library with the higher
score on the held-out training players. Everything else is report only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402

CACHE, OUT = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd"
FOLDS = (1, 2, 3, 4)
PRE = [("GBDT", "MLP0"), ("MLP0+GBDT", "MLP0+MLP1")]
REPORT = [("XGB", "MLP0"), ("LGB", "MLP0"), ("MLP5", "MLP0"), ("MLP5+GBDT", "MLP5")]

rows, per_fold, chosen = [], {}, {}
for v in FOLDS:
    z = np.load(CACHE / f"r_val{v}.npz")
    q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
    y = z["y_va"].astype(int)
    mlp = [MLPModel(hidden=32, seed=s).fit(z["X_tr"], z["y_tr"]).predict_proba(z["X_va"]) for s in range(5)]
    gb = {lib: np.load(OUT / f"{lib}_val{v}.npz")["proba"].astype(float) for lib in ("xgb", "lgb")}
    meta = {lib: json.loads((OUT / f"{lib}_val{v}.json").read_text()) for lib in ("xgb", "lgb")}
    best = max(meta, key=lambda k: meta[k]["chosen"]["inner"])
    chosen[str(v)] = {lib: meta[lib]["chosen"] for lib in meta} | {"best": best}
    P = {"MLP0": mlp[0], "MLP1": mlp[1], "MLP5": np.mean(mlp, axis=0), "XGB": gb["xgb"], "LGB": gb["lgb"], "GBDT": gb[best]}
    P["MLP0+GBDT"] = (mlp[0] + gb[best]) / 2
    P["MLP0+MLP1"] = (mlp[0] + mlp[1]) / 2
    P["MLP5+GBDT"] = (P["MLP5"] * 5 + gb[best]) / 6
    S = {k: evaluate(p, y, q)[1] for k, p in P.items()}
    for k, s in S.items():
        per_fold.setdefault(k, []).append(balanced_mean_over_n(s, y, q["n_games"].to_numpy()))
    rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
    print(f"fold {v}: best GBDT {best}; " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in P), flush=True)
t = pd.concat(rows, ignore_index=True)
y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
models = [c for c in t.columns if c not in ("y", "pid", "n")]
for k in models:
    print(f"{k:10s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k])
          + "  by n " + " ".join(f"{m}: {balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(m,)):.4f}" for m in SIZES))
out = {"chosen": chosen, "scores": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k]} for k in models}, "comparisons": {}}
for group, pairs in (("pre_specified", PRE), ("report_only", REPORT)):
    out["comparisons"][group] = {}
    for x, w in pairs:
        dd = paired_diff(t[x].to_numpy(), t[w].to_numpy(), y, pid, n, n=5000)
        pf = [a - b for a, b in zip(per_fold[x], per_fold[w])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
        out["comparisons"][group][f"{x} vs {w}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
        print(f"[{group}] {x} vs {w}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + f"  -> {'PASS' if all(cond.values()) else 'no'}", flush=True)
(OUT / "gbdt_hd_check.json").write_text(json.dumps(out, indent=1, default=float))
