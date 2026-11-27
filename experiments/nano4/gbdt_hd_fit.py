#!/usr/bin/env python
"""Report-only GBDT fit on the 64 R features (fused_opp_diff_hd on the _hd scores) of one validation fold (architecture v2 section 4B).

Settings and rounds are chosen on the 10% held-out training players of section 4A, by the official class-balanced score; the
chosen setting is then refit on every training question with those rounds, and the validation fold is predicted once.
Needs numpy and the library only (xgboost from .venv, lightgbm from ~/.cache/lgbm_venv). Fold 0 is refused.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

CACHE = Path("results/trunk_seq")
OUT = Path("results/gbdt_hd")
E1 = math.exp(-1)
XGB_GRID = [{"max_depth": d, "min_child_weight": m} for d in (3, 4, 6) for m in (1, 10)]
LGB_GRID = [{"num_leaves": n, "min_data_in_leaf": m} for n in (15, 63) for m in (50, 400)]


def official(P, y):
    """Class-balanced mean official score of the expected-score decisions of P [Q, 10]."""
    q = P.copy()
    q[:, 1:] += E1 * P[:, :-1]
    q[:, :-1] += E1 * P[:, 1:]
    d = np.abs(q.argmax(axis=1) - y)
    s = np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))
    return float(np.mean([s[y == c].mean() for c in np.unique(y)]))


def class_weights(y):
    counts = np.bincount(y, minlength=10).astype(float)
    w = 1.0 / counts[y]
    return w / w.mean()


def fit_xgb(cfg, Xa, ya, Xb=None, yb=None, rounds=None):
    import xgboost as xgb
    params = {"objective": "multi:softprob", "num_class": 10, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.5, "tree_method": "hist",
              "device": "cuda", "eval_metric": "mlogloss", "seed": 0, **cfg}
    da = xgb.DMatrix(Xa, label=ya, weight=class_weights(ya))
    if rounds is None:
        db = xgb.DMatrix(Xb, label=yb, weight=class_weights(yb))
        b = xgb.train(params, da, 3000, evals=[(db, "inner")], early_stopping_rounds=100, verbose_eval=False)
        n = b.best_iteration + 1
        return n, lambda X: b.predict(xgb.DMatrix(X), iteration_range=(0, n))
    b = xgb.train(params, da, rounds)
    return rounds, lambda X: b.predict(xgb.DMatrix(X))


def fit_lgb(cfg, Xa, ya, Xb=None, yb=None, rounds=None):
    import lightgbm as lgb
    params = {"objective": "multiclass", "num_class": 10, "learning_rate": 0.05, "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
              "lambda_l2": 1.0, "verbose": -1, "num_threads": 12, "seed": 0, **cfg}
    da = lgb.Dataset(Xa, label=ya, weight=class_weights(ya))
    if rounds is None:
        db = lgb.Dataset(Xb, label=yb, weight=class_weights(yb), reference=da)
        b = lgb.train(params, da, 3000, valid_sets=[db], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = b.best_iteration
        return n, lambda X: b.predict(X, num_iteration=n)
    b = lgb.train(params, da, rounds)
    return rounds, lambda X: b.predict(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=("xgb", "lgb"), required=True)
    ap.add_argument("--val-fold", type=int, required=True)
    a = ap.parse_args()
    if a.val_fold == 0:
        ap.error("fold 0 is not used here")
    t0 = time.time()
    z = np.load(CACHE / f"r_val{a.val_fold}.npz")
    meta = json.loads((CACHE / f"r_val{a.val_fold}.json").read_text())
    X_tr, y_tr, X_va = z["X_tr"].astype(np.float32), z["y_tr"].astype(int), z["X_va"].astype(np.float32)
    pid = np.array(meta["train"]["player_id"], dtype=str)
    players = np.array(sorted(set(pid)))
    held = set(np.random.RandomState(12345 + a.val_fold).permutation(players)[:int(0.1 * len(players))])
    inner = np.array([p in held for p in pid])
    fit, grid = (fit_xgb, XGB_GRID) if a.lib == "xgb" else (fit_lgb, LGB_GRID)
    tried = []
    for cfg in grid:
        n, predict = fit(cfg, X_tr[~inner], y_tr[~inner], X_tr[inner], y_tr[inner])
        tried.append({**cfg, "rounds": int(n), "inner": official(predict(X_tr[inner]), y_tr[inner])})
        print(f"{a.lib} fold {a.val_fold} {cfg}: {n} rounds, inner {tried[-1]['inner']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    best = max(tried, key=lambda r: r["inner"])
    cfg = {k: v for k, v in best.items() if k not in ("rounds", "inner")}
    _, predict = fit(cfg, X_tr, y_tr, rounds=best["rounds"])
    P = predict(X_va)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{a.lib}_val{a.val_fold}.npz", proba=P.astype(np.float32))
    (OUT / f"{a.lib}_val{a.val_fold}.json").write_text(json.dumps({"lib": a.lib, "val_fold": a.val_fold, "tried": tried, "chosen": best,
                                                                   "held_players": len(held), "wall_s": time.time() - t0}, indent=1))
    print(f"{a.lib} fold {a.val_fold}: chose {cfg}, {best['rounds']} rounds, inner {best['inner']:.4f}; validation predicted ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
