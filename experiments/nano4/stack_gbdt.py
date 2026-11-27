#!/usr/bin/env python
"""Architecture v2 section 4C: GBDT on the learned representation, as a stacker. Each validation question has out-of-fold C and T
probabilities (seed 0-2 means, fixed semantics, so folds can be pooled); the stacker reads [64 R, C 10, T 10] and is cross-fitted over
folds 1-4 (train on three folds' validation questions, predict the fourth), settings and rounds chosen on 10% held-out players of the
three. Pre-specified: XGBoost stacker against the equal-weight six-model C + T average on the same questions, conditions (1)(3)(4)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd(); sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import balanced_mean_over_n, evaluate, paired_diff
from aicup.rankmodel import MLPModel
from gbdt_hd_fit import XGB_GRID, fit_xgb, official
C, ES = ROOT / "results/trunk_seq", ROOT / "results/trunk_seq/es"
F = {}
for v in (1, 2, 3, 4):
    z = np.load(C / f"r_val{v}.npz"); q = pd.DataFrame(json.loads((C / f"r_val{v}.json").read_text())["val"])
    pc = np.mean([np.load(ES / f"val{v}" / f"C_s{s}.npz")["proba"] for s in (0, 1, 2)], axis=0)
    pt = np.mean([np.load(ES / f"val{v}" / f"T_s{s}.npz")["proba"] for s in (0, 1, 2)], axis=0)
    F[v] = {"X": np.hstack([z["X_va"], np.log(np.clip(pc, 1e-6, 1)), np.log(np.clip(pt, 1e-6, 1))]).astype(np.float32), "y": z["y_va"].astype(int), "q": q,
            "eq": (pc + pt) / 2}
rows, per_fold = [], {}
for v in (1, 2, 3, 4):
    tr = [u for u in (1, 2, 3, 4) if u != v]
    X = np.vstack([F[u]["X"] for u in tr]); y = np.concatenate([F[u]["y"] for u in tr])
    pid = np.concatenate([F[u]["q"]["player_id"].to_numpy(dtype=str) for u in tr])
    players = np.array(sorted(set(pid))); held = set(np.random.RandomState(4242 + v).permutation(players)[:len(players) // 10]); inn = np.array([p in held for p in pid])
    tried = []
    for cfg in XGB_GRID:
        n_r, pred = fit_xgb(cfg, X[~inn], y[~inn], X[inn], y[inn])
        tried.append((official(pred(X[inn]), y[inn]), n_r, cfg))
    best = max(tried, key=lambda r: r[0])
    _, pred = fit_xgb(best[2], X, y, rounds=best[1])
    Pm = MLPModel(hidden=32).fit(X, y).predict_proba(F[v]["X"])
    d = F[v]; S = {"equal": evaluate(d["eq"], d["y"], d["q"])[1], "xgb_stack": evaluate(pred(d["X"]), d["y"], d["q"])[1], "mlp_stack": evaluate(Pm, d["y"], d["q"])[1]}
    for k, s in S.items():
        per_fold.setdefault(k, []).append(balanced_mean_over_n(s, d["y"], d["q"]["n_games"].to_numpy()))
    rows.append(pd.DataFrame({"y": d["y"], "pid": d["q"]["player_id"].to_numpy(dtype=str), "n": d["q"]["n_games"].to_numpy(), **S}))
    print(f"fold {v}: xgb {best[2]} {best[1]} rounds; " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in S), flush=True)
t = pd.concat(rows, ignore_index=True); y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
out = {"scores": {k: {"score": float(np.mean(v)), "per_fold": v} for k, v in per_fold.items()}, "comparisons": {}}
for x, grp in (("xgb_stack", "pre_specified"), ("mlp_stack", "report_only")):
    dd = paired_diff(t[x].to_numpy(), t["equal"].to_numpy(), y, pid, n, n=5000)
    pf = [a - b for a, b in zip(per_fold[x], per_fold["equal"])]
    cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
    out["comparisons"][f"{x} vs equal"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values()), "group": grp}
    print(f"[{grp}] {x} vs equal: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if all(cond.values()) else 'no'}")
(ROOT / "results/gbdt_hd/stack.json").write_text(json.dumps(out, indent=1, default=float))
