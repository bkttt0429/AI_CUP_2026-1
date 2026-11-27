#!/usr/bin/env python
"""The user's XGBClassifier setting (500 trees, depth 6, lr 0.05, subsample / colsample 0.8, no early stopping) on the 64 R
features of folds 1-4, as given (no sample weights) and with the class-balanced weights of 4B / 4D, against MLP0 and the
tuned GBDT of 4D (report only; fold 0 is never read)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from gbdt_hd_fit import class_weights  # noqa: E402

CACHE, OUT = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd"


def pasted(X, y, Xv, w=None, random_state=0):
    """XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="gpu_hist",
    random_state=...) through xgb.train (the .venv has no sklearn): the classifier's 10-class objective is multi:softprob, and
    gpu_hist is hist on cuda. Its eval_metric is unused without an eval_set."""
    import xgboost as xgb
    params = {"objective": "multi:softprob", "num_class": 10, "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
              "tree_method": "hist", "device": "cuda", "seed": random_state}
    return xgb.train(params, xgb.DMatrix(X, label=y, weight=w), 500).predict(xgb.DMatrix(Xv))


rows, per_fold = [], {}
for v in (1, 2, 3, 4):
    z = np.load(CACHE / f"r_val{v}.npz")
    q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
    y_tr, y = z["y_tr"].astype(int), z["y_va"].astype(int)
    t = np.load(OUT / f"tune_val{v}.npz")
    P = {"MLP0": t["mlp0"], "TUNED": t["a"],
         "PASTED": pasted(z["X_tr"], y_tr, z["X_va"]),
         "PASTED+W": pasted(z["X_tr"], y_tr, z["X_va"], w=class_weights(y_tr))}
    S, M = {}, {}
    for k, p in P.items():
        M[k], S[k] = evaluate(np.asarray(p, dtype=float), y, q)
        per_fold.setdefault(k, []).append(M[k]["balanced_mean_over_n"])
    rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
    print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f} (exact {M[k]['exact']:.3f}, within1 {M[k]['within1']:.3f})" for k in P), flush=True)
t = pd.concat(rows, ignore_index=True)
y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
out = {"scores": {}, "comparisons": {}}
for k in per_fold:
    by_n = {int(s): balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)) for s in SIZES}
    out["scores"][k] = {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": by_n}
    print(f"{k:9s} {np.mean(per_fold[k]):.4f}  by n " + " ".join(f"{s}: {x:.4f}" for s, x in by_n.items()))
for a, b in (("PASTED", "MLP0"), ("PASTED", "TUNED"), ("PASTED+W", "PASTED"), ("PASTED+W", "TUNED")):
    dd = paired_diff(t[a].to_numpy(), t[b].to_numpy(), y, pid, n, n=5000)
    pf = [x - w for x, w in zip(per_fold[a], per_fold[b])]
    out["comparisons"][f"{a} vs {b}"] = {**dd, "per_fold_diff": pf}
    print(f"{a} vs {b}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True)
(OUT / "pasted.json").write_text(json.dumps(out, indent=1, default=float))
