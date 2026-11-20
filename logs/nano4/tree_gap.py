#!/usr/bin/env python
"""Why the GBDTs of sections 4B / 4D trail MLP h32 on the 64 R features (report only; folds 1-4, fold 0 is never read).

Each check removes one suspected cause and asks whether the gap to the MLP closes:
  LIN       multinomial logistic regression: how much of the MLP is a linear function of R
  TREE+LIN  XGBoost given the logistic model's 10 log probabilities as extra columns (oblique directions handed to the
            axis-aligned splits)
  TREE-REG  XGBoost regression on the level index, turned into class probabilities by a discretised Gaussian whose width
            per question-size bucket is chosen on the held-out players (one shared ordinal score instead of 10 class scores)
  curve     XGBoost and MLP on 25 / 50 / 100 % of the non-held-out training players, the trees early-stopped on the
            held-out players (does the gap shrink with more data)
Settings and rounds use the 4B grid and held-out players. MLP0 and the 4B XGBoost are read from results/gbdt_hd.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import MLPModel, SoftmaxModel  # noqa: E402
from gbdt_hd_fit import XGB_GRID, class_weights, fit_xgb, official  # noqa: E402

CACHE, OUT = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd"
FOLDS = (1, 2, 3, 4)
SIGMAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
FRACS = (0.25, 0.5, 1.0)
PAIRS = [("LIN", "MLP0"), ("TREE", "MLP0"), ("TREE+LIN", "MLP0"), ("TREE+LIN", "TREE"), ("TREE-REG", "MLP0"), ("TREE-REG", "TREE")] + \
        [(f"TREE@{f}", f"MLP@{f}") for f in FRACS]


def bucket(n):
    return np.digitize(n, (8, 15))  # 5-7, 8-14, 15-20


def gauss(mu, sig):
    P = np.exp(-(np.arange(10)[None, :] - mu[:, None]) ** 2 / (2 * sig[:, None] ** 2))
    return P / P.sum(axis=1, keepdims=True)


def fit_reg(cfg, Xa, ya, Xb=None, yb=None, rounds=None):
    import xgboost as xgb
    params = {"objective": "reg:squarederror", "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.5, "tree_method": "hist",
              "device": "cuda", "seed": 0, **cfg}
    da = xgb.DMatrix(Xa, label=ya.astype(float), weight=class_weights(ya))
    if rounds is None:
        db = xgb.DMatrix(Xb, label=yb.astype(float), weight=class_weights(yb))
        b = xgb.train(params, da, 3000, evals=[(db, "inner")], early_stopping_rounds=100, verbose_eval=False)
        n = b.best_iteration + 1
        return n, lambda X: b.predict(xgb.DMatrix(X), iteration_range=(0, n))
    b = xgb.train(params, da, rounds)
    return rounds, lambda X: b.predict(xgb.DMatrix(X))


def widths(mu, y, n):
    """Per size bucket, the Gaussian width with the best official score on these questions."""
    bk, sig = bucket(n), np.ones(3)
    for k in range(3):
        m = bk == k
        sig[k] = max(SIGMAS, key=lambda s: official(gauss(mu[m], np.full(m.sum(), s)), y[m]))
    return sig


def select(fit, X, y, inn, n_tr=None):
    """The 4B grid on the held-out players: (inner score, rounds, cfg, predictor, widths or None) of the best setting."""
    best = None
    for cfg in XGB_GRID:
        r, pred = fit(cfg, X[~inn], y[~inn], X[inn], y[inn])
        out = pred(X[inn])
        if n_tr is None:
            s, sig = official(out, y[inn]), None
        else:
            sig = widths(out, y[inn], n_tr[inn])
            s = official(gauss(out, sig[bucket(n_tr[inn])]), y[inn])
        if best is None or s > best[0]:
            best = (s, r, cfg, pred, sig)
    return best


def fold(v):
    z = np.load(CACHE / f"r_val{v}.npz")
    meta = json.loads((CACHE / f"r_val{v}.json").read_text())
    q = pd.DataFrame(meta["val"])
    pid, n_tr = np.array(meta["train"]["player_id"], dtype=str), np.array(meta["train"]["n_games"])
    y, X, Xv = z["y_tr"].astype(int), z["X_tr"].astype(np.float32), z["X_va"].astype(np.float32)
    players = np.array(sorted(set(pid)))
    held = set(np.random.RandomState(12345 + v).permutation(players)[:int(0.1 * len(players))])
    inn = np.array([p in held for p in pid])
    P = {"MLP0": np.load(OUT / f"tune_val{v}.npz")["mlp0"], "TREE": np.load(OUT / f"xgb_val{v}.npz")["proba"]}
    P["LIN"] = SoftmaxModel().fit(z["X_tr"], y).predict_proba(z["X_va"])
    lin_out = SoftmaxModel().fit(z["X_tr"][~inn], y[~inn])
    Xl = np.hstack([X, np.log(np.clip(lin_out.predict_proba(z["X_tr"]), 1e-7, 1))]).astype(np.float32)
    s, r, cfg, _, _ = select(fit_xgb, Xl, y, inn)
    lin_all = SoftmaxModel().fit(z["X_tr"], y)
    add = lambda A, Z: np.hstack([A, np.log(np.clip(lin_all.predict_proba(Z), 1e-7, 1))]).astype(np.float32)  # noqa: E731
    _, pred = fit_xgb(cfg, add(X, z["X_tr"]), y, rounds=r)
    P["TREE+LIN"] = pred(add(Xv, z["X_va"]))
    log = {"TREE+LIN": {**cfg, "rounds": r, "inner": s}}
    s, r, cfg, _, sig = select(fit_reg, X, y, inn, n_tr)
    _, pred = fit_reg(cfg, X, y, rounds=r)
    P["TREE-REG"] = gauss(pred(Xv), sig[bucket(q["n_games"].to_numpy())])
    log["TREE-REG"] = {**cfg, "rounds": r, "inner": s, "widths": sig.tolist()}
    outer = np.random.RandomState(999 + v).permutation(np.array(sorted(set(pid[~inn]))))
    for f in FRACS:
        keep = set(outer[:int(round(f * len(outer)))])
        rows = np.array([p in keep for p in pid]) & ~inn
        sub = rows | inn
        s, r, cfg, pred, _ = select(fit_xgb, X[sub], y[sub], inn[sub])
        P[f"TREE@{f}"] = pred(Xv)
        P[f"MLP@{f}"] = MLPModel(hidden=32, seed=0).fit(z["X_tr"][rows], y[rows]).predict_proba(z["X_va"])
        log[f"TREE@{f}"] = {**cfg, "rounds": r, "inner": s, "questions": int(rows.sum())}
    print(f"fold {v}: {json.dumps(log)}", flush=True)
    return P, z["y_va"].astype(int), q, log


def main():
    rows, per_fold, metrics, logs = [], {}, {}, {}
    for v in FOLDS:
        P, y, q, logs[str(v)] = fold(v)
        S = {}
        for k, p in P.items():
            m, S[k] = evaluate(np.asarray(p, dtype=float), y, q)
            per_fold.setdefault(k, []).append(m["balanced_mean_over_n"])
            for key in ("exact", "within1", "nll_balanced"):
                metrics.setdefault(k, {}).setdefault(key, []).append(m[key])
        rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
        print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in P), flush=True)
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
    out = {"settings": logs, "scores": {}, "comparisons": {}}
    for k in per_fold:
        by_n = {int(s): balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)) for s in SIZES}
        out["scores"][k] = {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k], "by_n": by_n,
                            **{key: float(np.mean(vals)) for key, vals in metrics[k].items()}}
        print(f"{k:10s} {np.mean(per_fold[k]):.4f}  by n " + " ".join(f"{s}: {x:.4f}" for s, x in by_n.items())
              + "  " + "  ".join(f"{key} {np.mean(vals):.4f}" for key, vals in metrics[k].items()))
    for a, b in PAIRS:
        dd = paired_diff(t[a].to_numpy(), t[b].to_numpy(), y, pid, n, n=5000)
        pf = [x - w for x, w in zip(per_fold[a], per_fold[b])]
        out["comparisons"][f"{a} vs {b}"] = {**dd, "per_fold_diff": pf}
        print(f"{a} vs {b}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True)
    (OUT / "tree_gap.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
