#!/usr/bin/env python
"""Architecture v2 section 4D: a 40-setting XGBoost random search, and XGBoost that starts from a model's log probabilities
(base_margin) and fits only the residual. Folds 1-4; fold 0 is refused.

  fit --val-fold v  (a) the search on the 64 R features; (b) the residual on MLP h32 seed 0: training questions start from
                    out-of-fold MLPs cross-fitted over 5 player groups, validation questions from the MLP fitted on every
                    training question (MLP0 of gbdt_hd_eval.py)
  eval              (c) the residual on the C + T six-model mean, cross-fitted over folds 1-4 as in stack_gbdt.py; then the
                    pre-specified comparisons, written to results/gbdt_hd/tune.json

Settings and rounds are chosen on held-out training players by the official class-balanced score, then refitted on every
training question with those rounds. For (b) and (c), 0 rounds (the starting model itself) is a candidate too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from gbdt_hd_fit import class_weights, official  # noqa: E402

CACHE, ES, OUT = ROOT / "results/trunk_seq", ROOT / "results/trunk_seq/es", ROOT / "results/gbdt_hd"
FOLDS = (1, 2, 3, 4)
PRE = [("A", "MLP0"), ("B", "MLP0"), ("CTres", "CT6")]
REPORT = [("A", "GB4B"), ("MLP0+A", "MLP0+MLP1")]
T0 = time.time()


def space(n=40, seed=20261103):
    r = np.random.RandomState(seed)
    lu = lambda a, b: float(np.exp(r.uniform(np.log(a), np.log(b))))  # noqa: E731
    return [{"eta": float(r.choice([0.02, 0.05, 0.1])), "max_depth": int(r.randint(2, 9)), "min_child_weight": lu(1, 300),
             "subsample": float(r.uniform(0.5, 1)), "colsample_bytree": float(r.uniform(0.3, 1)), "lambda": lu(0.1, 100),
             "alpha": lu(1e-3, 10), "gamma": float(r.uniform(0, 1))} for _ in range(n)]


def log_p(P):
    return np.log(np.clip(P, 1e-7, 1)).astype(np.float32)


def softmax(M):
    e = np.exp(M - M.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def dmat(X, y=None, m=None):
    import xgboost as xgb
    return xgb.DMatrix(X, label=y, weight=None if y is None else class_weights(y), base_margin=m)


def params(cfg):
    return {"objective": "multi:softprob", "num_class": 10, "tree_method": "hist", "device": "cuda", "eval_metric": "mlogloss", "seed": 0,
            **{k: v for k, v in cfg.items() if k not in ("rounds", "inner")}}


def check_margin(X, y, m):
    """A near-zero step must return softmax(margin): fails if base_margin were ignored or misread."""
    import xgboost as xgb
    b = xgb.train(params({"eta": 1e-9, "max_depth": 2}), dmat(X[:2000], y[:2000], m[:2000]), 1)
    assert np.allclose(b.predict(dmat(X[:50], m=m[:50])), softmax(m[:50].astype(float)), atol=1e-4), "base_margin not applied"


def search(Xa, ya, ma, Xb, yb, mb, label):
    import xgboost as xgb
    da, db = dmat(Xa, ya, ma), dmat(Xb, yb, mb)
    tried = [] if ma is None else [{"rounds": 0, "inner": official(softmax(mb.astype(float)), yb)}]
    for cfg in space():
        b = xgb.train(params(cfg), da, 5000, evals=[(db, "inner")], early_stopping_rounds=100, verbose_eval=False)
        n = b.best_iteration + 1
        tried.append({**cfg, "rounds": n, "inner": official(b.predict(db, iteration_range=(0, n)), yb)})
    best = max(tried, key=lambda r: r["inner"])
    base = f"; 0 rounds {tried[0]['inner']:.4f}" if ma is not None else ""
    print(f"{label}: best inner {best['inner']:.4f} at {best['rounds']} rounds{base} ({time.time() - T0:.0f} s)", flush=True)
    return tried, best


def refit(best, X, y, m, Xv, mv):
    import xgboost as xgb
    if best["rounds"] == 0:
        return softmax(mv.astype(float))
    b = xgb.train(params(best), dmat(X, y, m), best["rounds"])
    return b.predict(dmat(Xv, m=mv))


def held_mask(pid, seed, frac):
    players = np.array(sorted(set(pid)))
    held = set(np.random.RandomState(seed).permutation(players)[:frac(len(players))])
    return np.array([p in held for p in pid])


def cmd_fit(v):
    z = np.load(CACHE / f"r_val{v}.npz")
    pid = np.array(json.loads((CACHE / f"r_val{v}.json").read_text())["train"]["player_id"], dtype=str)
    y = z["y_tr"].astype(int)
    X, Xv = z["X_tr"].astype(np.float32), z["X_va"].astype(np.float32)
    inn = held_mask(pid, 12345 + v, lambda n: int(0.1 * n))
    tried_a, best_a = search(X[~inn], y[~inn], None, X[inn], y[inn], None, f"fold {v} (a) search")
    Pa = refit(best_a, X, y, None, Xv, None)
    players = np.random.RandomState(5150 + v).permutation(np.array(sorted(set(pid))))
    g = pd.Series(np.arange(len(players)) % 5, index=players)[pid].to_numpy()
    oof = np.zeros((len(y), 10))
    for k in range(5):
        oof[g == k] = MLPModel(hidden=32, seed=0).fit(z["X_tr"][g != k], y[g != k]).predict_proba(z["X_tr"][g == k])
    mlp0 = MLPModel(hidden=32, seed=0).fit(z["X_tr"], y).predict_proba(z["X_va"])
    print(f"fold {v}: out-of-fold MLPs done, inner {official(oof[inn], y[inn]):.4f} ({time.time() - T0:.0f} s)", flush=True)
    m, mv = log_p(oof), log_p(mlp0)
    check_margin(X, y, m)
    tried_b, best_b = search(X[~inn], y[~inn], m[~inn], X[inn], y[inn], m[inn], f"fold {v} (b) MLP residual")
    Pb = refit(best_b, X, y, m, Xv, mv)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"tune_val{v}.npz", a=Pa.astype(np.float32), b=Pb.astype(np.float32), mlp0=mlp0.astype(np.float32))
    (OUT / f"tune_val{v}.json").write_text(json.dumps({"a": {"tried": tried_a, "chosen": best_a}, "b": {"tried": tried_b, "chosen": best_b},
                                                       "wall_s": time.time() - T0}, indent=1))


def cmd_eval():
    F = {}
    for v in FOLDS:
        z = np.load(CACHE / f"r_val{v}.npz")
        q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
        pc = np.mean([np.load(ES / f"val{v}" / f"C_s{s}.npz")["proba"] for s in (0, 1, 2)], axis=0)
        pt = np.mean([np.load(ES / f"val{v}" / f"T_s{s}.npz")["proba"] for s in (0, 1, 2)], axis=0)
        F[v] = {"z": z, "X": z["X_va"].astype(np.float32), "y": z["y_va"].astype(int), "q": q, "ct": (pc + pt) / 2}
    chosen, rows, per_fold = {}, [], {}
    gb4 = json.loads((OUT / "gbdt_hd_check.json").read_text())["chosen"]
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        X, y = np.vstack([F[u]["X"] for u in tr]), np.concatenate([F[u]["y"] for u in tr])
        m = np.vstack([log_p(F[u]["ct"]) for u in tr])
        pid = np.concatenate([F[u]["q"]["player_id"].to_numpy(dtype=str) for u in tr])
        inn = held_mask(pid, 4242 + v, lambda n: n // 10)
        if v == FOLDS[0]:
            check_margin(X, y, m)
        tried_c, best_c = search(X[~inn], y[~inn], m[~inn], X[inn], y[inn], m[inn], f"fold {v} (c) C + T residual")
        d, t = F[v], np.load(OUT / f"tune_val{v}.npz")
        meta = json.loads((OUT / f"tune_val{v}.json").read_text())
        chosen[str(v)] = {"a": meta["a"]["chosen"], "b": meta["b"]["chosen"], "c": best_c, "c_tried": tried_c}
        mlp1 = MLPModel(hidden=32, seed=1).fit(d["z"]["X_tr"], d["z"]["y_tr"]).predict_proba(d["z"]["X_va"])
        P = {"MLP0": t["mlp0"], "MLP1": mlp1, "A": t["a"], "B": t["b"], "GB4B": np.load(OUT / f"{gb4[str(v)]['best']}_val{v}.npz")["proba"],
             "CT6": d["ct"], "CTres": refit(best_c, X, y, m, d["X"], log_p(d["ct"]))}
        P["MLP0+A"] = (P["MLP0"] + P["A"]) / 2
        P["MLP0+MLP1"] = (P["MLP0"] + P["MLP1"]) / 2
        S = {k: evaluate(np.asarray(p, dtype=float), d["y"], d["q"])[1] for k, p in P.items()}
        for k, s in S.items():
            per_fold.setdefault(k, []).append(balanced_mean_over_n(s, d["y"], d["q"]["n_games"].to_numpy()))
        rows.append(pd.DataFrame({"y": d["y"], "pid": d["q"]["player_id"].to_numpy(dtype=str), "n": d["q"]["n_games"].to_numpy(), **S}))
        print(f"fold {v}: rounds a {meta['a']['chosen']['rounds']}, b {meta['b']['chosen']['rounds']}, c {best_c['rounds']}; "
              + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in P), flush=True)
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
    models = [c for c in t.columns if c not in ("y", "pid", "n")]
    for k in models:
        print(f"{k:10s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k])
              + "  by n " + " ".join(f"{s}: {balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)):.4f}" for s in SIZES))
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
    (OUT / "tune.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "eval"))
    ap.add_argument("--val-fold", type=int)
    a = ap.parse_args()
    if a.cmd == "fit":
        if a.val_fold not in FOLDS:
            ap.error("--val-fold must be one of 1-4 (fold 0 is not used here)")
        cmd_fit(a.val_fold)
    else:
        cmd_eval()


if __name__ == "__main__":
    main()
