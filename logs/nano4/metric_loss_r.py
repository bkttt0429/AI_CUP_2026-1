#!/usr/bin/env python
"""Architecture v2 section 4F on the 64 R features: the kernel-target loss (K) and the expected-score fine-tune (ES) for
XGBoost and for the MLP h32, against the cross-entropy models of 4D (tree (a), MLP0). Folds 1-4; fold 0 is refused.

  smoke             the custom cross-entropy objective reproduces multi:softprob; continued training, the metrics and the
                    MLP stages run on a small sample
  fit --val-fold v  tree K (4D's 40-setting space with the kernel objective), tree ES (4D(a)'s setting, continued with the
                    expected-score objective), MLP K, MLP ES -> results/metric_loss/r_val{v}.npz / .json
  eval              the pre-specified pairs (K vs CE and ES vs CE for the tree, the MLP, C and T) and the report-only ones
                    -> results/metric_loss/summary.json

Held-out training players (RandomState(12345 + v), 10%, as in 4B/4D) choose settings, rounds and epochs; the final models
are refitted on every training question. K models are decoded by the plain argmax, CE and ES models by the expected-score
decision; the other decoder is reported (@alt).
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
from aicup.metric import score  # noqa: E402
from aicup.metric_loss import expected_score_grad_hess, expected_score_np, kernel_grad_hess, kernel_targets, softmax  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from gbdt_hd_fit import class_weights, official  # noqa: E402
from gbdt_tune import held_mask, params, space  # noqa: E402

CACHE, ES_DIR, TUNE, OUT = ROOT / "results/trunk_seq", ROOT / "results/trunk_seq/es", ROOT / "results/gbdt_hd", ROOT / "results/metric_loss"
FOLDS = (1, 2, 3, 4)
BASES = ("TREE", "MLP", "C", "T")
T0 = time.time()


def official_argmax(P, y):
    """Class-balanced mean official score of the argmax decisions of P [Q, 10]."""
    s = np.array([score(k, t) for k, t in zip(np.asarray(P).argmax(axis=1), y)])
    return float(np.mean([s[y == c].mean() for c in np.unique(y)]))


def dmat(X, y=None):
    import xgboost as xgb
    return xgb.DMatrix(X, label=y, weight=None if y is None else class_weights(y))


def custom_params(cfg):
    """params(cfg) without the built-in objective and metric: the objective callback supplies the gradients."""
    p = {k: v for k, v in params(cfg).items() if k not in ("objective", "eval_metric")}
    return {**p, "disable_default_eval_metric": 1}


def ce_grad_hess(margin, y, weight):
    """multi:softprob itself (gradient p - 1[y], hessian 2 p (1 - p)): the smoke check against the built-in objective."""
    p = softmax(margin)
    return (p - np.eye(p.shape[1])[y]) * weight[:, None], np.maximum(2.0 * p * (1.0 - p), 1e-6) * weight[:, None]


GH = {"ce": ce_grad_hess, "kernel": kernel_grad_hess, "es": expected_score_grad_hess}


def objective(kind):
    def obj(predt, d):
        return GH[kind](np.asarray(predt, dtype=float).reshape(d.num_row(), -1), d.get_label().astype(int), d.get_weight())
    return obj


def proba_of(predt, n):
    """A custom metric receives raw margins when an objective callback is in use, probabilities otherwise."""
    p = np.asarray(predt, dtype=float).reshape(n, -1)
    return p if (p.min() >= 0 and np.allclose(p.sum(axis=1), 1, atol=1e-3)) else softmax(p)


def inner_loss(kind, P, y):
    """Class-balanced kernel cross-entropy ('kernel') or negative expected score ('es') of probabilities P on labels y."""
    w = class_weights(y)
    v = -(kernel_targets(y) * np.log(np.clip(P, 1e-12, 1))).sum(axis=1) if kind == "kernel" else -expected_score_np(P, y)
    return float((v * w).sum() / w.sum())


def metric(kind):
    def m(predt, d):
        return kind, inner_loss(kind, proba_of(predt, d.num_row()), d.get_label().astype(int))
    return m


def margin_proba(b, d, rounds=None):
    """softmax of the booster's raw margins (the first `rounds` trees; all when None)."""
    return softmax(b.predict(d, output_margin=True, iteration_range=(0, rounds or 0)))


def search_kernel(Xa, ya, Xb, yb, label):
    """4D's 40 settings with the kernel objective: early stopping on the held-out kernel cross-entropy, the setting chosen by
    the held-out argmax score."""
    import xgboost as xgb
    da, db = dmat(Xa, ya), dmat(Xb, yb)
    tried = []
    for cfg in space():
        b = xgb.train(custom_params(cfg), da, 5000, obj=objective("kernel"), custom_metric=metric("kernel"), evals=[(db, "inner")],
                      early_stopping_rounds=100, maximize=False, verbose_eval=False)
        n = b.best_iteration + 1
        P = margin_proba(b, db, n)
        tried.append({**cfg, "rounds": n, "inner": official_argmax(P, yb), "inner_es_decoder": official(P, yb)})
    best = max(tried, key=lambda r: r["inner"])
    print(f"{label}: best inner {best['inner']:.4f} at {best['rounds']} rounds ({time.time() - T0:.0f} s)", flush=True)
    return tried, best


def tree_es(chosen, Xa, ya, Xb, yb, X, y, Xv, label):
    """4D(a)'s setting: the CE booster on the non-held-out questions, continued with the expected-score objective until the
    held-out negative expected score stops improving (0 extra rounds when no continued round beats the start); then the refit
    on every question. best_iteration is the global tree index (the smoke check: 30 + 20 continued rounds -> 49)."""
    import xgboost as xgb
    da, db = dmat(Xa, ya), dmat(Xb, yb)
    rounds = int(chosen["rounds"])
    b1 = xgb.train(params(chosen), da, rounds)
    start = inner_loss("es", margin_proba(b1, db), yb)
    b2 = xgb.train(custom_params(chosen), da, 5000, obj=objective("es"), custom_metric=metric("es"), evals=[(db, "inner")],
                   early_stopping_rounds=100, maximize=False, verbose_eval=False, xgb_model=b1)
    n2 = max(0, int(b2.best_iteration) + 1 - rounds) if float(b2.best_score) < start else 0  # the continued rounds are indexed from `rounds`
    assert n2 >= 1 or float(b2.best_score) >= start, (b2.best_iteration, rounds)  # an improvement must come from a continued round
    assert b2.num_boosted_rounds() >= rounds + n2
    inner = official(margin_proba(b2, db, rounds + n2), yb)
    b = xgb.train(params(chosen), dmat(X, y), rounds)
    if n2 > 0:
        b = xgb.train(custom_params(chosen), dmat(X, y), n2, obj=objective("es"), xgb_model=b)
    print(f"{label}: start {start:.4f}, best {float(b2.best_score):.4f} after {n2} rounds; inner {inner:.4f} ({time.time() - T0:.0f} s)", flush=True)
    return margin_proba(b, dmat(Xv)), {"rounds_ce": rounds, "rounds_es": n2, "inner_start": start, "inner_best": float(b2.best_score), "inner": inner}


def mlp_variants(X, y, Xv, inn, label="MLP"):
    """MLP K (MLP0's recipe with the kernel targets) and MLP ES (MLP0's recipe, then the expected-score stage: its epochs chosen
    on the held-out players, the whole refitted on every question)."""
    Pk = MLPModel(hidden=32, seed=0, metric_kernel=True).fit(X, y).predict_proba(Xv)
    m = MLPModel(hidden=32, seed=0).fit(X[~inn], y[~inn])
    start, inner_ce = inner_loss("es", m.predict_proba(X[inn]), y[inn]), official(m.predict_proba(X[inn]), y[inn])
    m.finetune_expected_score(X[~inn], y[~inn], val=(X[inn], y[inn]), lr=1e-3, epochs=300)
    info = {"epochs_es": m.ft_epoch, "inner_start": start, "inner_best": m.val_es, "inner_ce_model": inner_ce, "inner": official(m.predict_proba(X[inn]), y[inn])}
    m_all = MLPModel(hidden=32, seed=0).fit(X, y)
    if m.ft_epoch > 0:
        m_all.finetune_expected_score(X, y, lr=1e-3, epochs=m.ft_epoch)
    print(f"{label}: K done; ES {m.ft_epoch} epochs, inner {inner_ce:.4f} -> {info['inner']:.4f} ({time.time() - T0:.0f} s)", flush=True)
    return Pk, m_all.predict_proba(Xv), info


def cmd_fit(v):
    import xgboost as xgb
    z = np.load(CACHE / f"r_val{v}.npz")
    pid = np.array(json.loads((CACHE / f"r_val{v}.json").read_text())["train"]["player_id"], dtype=str)
    y = z["y_tr"].astype(int)
    X, Xv = z["X_tr"].astype(np.float32), z["X_va"].astype(np.float32)
    inn = held_mask(pid, 12345 + v, lambda n: int(0.1 * n))
    chosen = json.loads((TUNE / f"tune_val{v}.json").read_text())["a"]["chosen"]
    tried_k, best_k = search_kernel(X[~inn], y[~inn], X[inn], y[inn], f"fold {v} tree K")
    cfg_k = {k: w for k, w in best_k.items() if k != "inner_es_decoder"}
    bk = xgb.train(custom_params(cfg_k), dmat(X, y), int(best_k["rounds"]), obj=objective("kernel"))
    P_tk = margin_proba(bk, dmat(Xv))
    P_tes, es_info = tree_es(chosen, X[~inn], y[~inn], X[inn], y[inn], X, y, Xv, f"fold {v} tree ES")
    P_mk, P_mes, mlp_info = mlp_variants(z["X_tr"], y, z["X_va"], inn, f"fold {v} MLP")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"r_val{v}.npz", tree_k=P_tk.astype(np.float32), tree_es=P_tes.astype(np.float32), mlp_k=P_mk.astype(np.float32), mlp_es=P_mes.astype(np.float32))
    (OUT / f"r_val{v}.json").write_text(json.dumps({"tree_k": {"tried": tried_k, "chosen": best_k}, "tree_es": es_info, "mlp": mlp_info,
                                                    "wall_s": time.time() - T0}, indent=1, default=float))


def load_ct(v, arm, suffix):
    f = ES_DIR / f"val{v}" / f"{arm}{suffix}_s0.npz"
    return np.load(f)["proba"] if f.exists() else None


def cmd_eval():
    rows, per_fold, metrics, info = [], {}, {}, {}
    for v in FOLDS:
        z = np.load(CACHE / f"r_val{v}.npz")
        q = pd.DataFrame(json.loads((CACHE / f"r_val{v}.json").read_text())["val"])
        y = z["y_va"].astype(int)
        t, r = np.load(TUNE / f"tune_val{v}.npz"), np.load(OUT / f"r_val{v}.npz")
        meta = json.loads((OUT / f"r_val{v}.json").read_text())
        info[str(v)] = {"tree_k_chosen": meta["tree_k"]["chosen"], "tree_es": meta["tree_es"], "mlp": meta["mlp"]}
        P = {"TREE": t["a"], "TREE-K": r["tree_k"], "TREE-ES": r["tree_es"], "MLP": t["mlp0"], "MLP-K": r["mlp_k"], "MLP-ES": r["mlp_es"]}
        for arm in ("C", "T"):
            for name, suffix in ((arm, ""), (f"{arm}-K", "-k"), (f"{arm}-ES", "-es")):
                p = load_ct(v, arm, suffix)
                if p is None:
                    print(f"fold {v}: {name} missing", flush=True)
                else:
                    P[name] = p
        for suffix in ("", "-K", "-ES"):
            if f"C{suffix}" in P and f"T{suffix}" in P:
                P[f"CT{suffix}"] = (P[f"C{suffix}"] + P[f"T{suffix}"]) / 2
        S = {}
        for k, p in P.items():
            p = np.asarray(p, dtype=float)
            m, s_exp = evaluate(p, y, q)
            s_arg = np.array([score(c, u) for c, u in zip(p.argmax(axis=1), y)])
            S[k], S[k + "@alt"] = (s_arg, s_exp) if k.endswith("-K") else (s_exp, s_arg)
            metrics.setdefault(k, {})[str(v)] = {"by_n": m["balanced_by_n"], "per_class": m["per_class"], "nll": m["nll_balanced"], "brier": m["brier"],
                                                 "exact": m["exact"], "within1": m["within1"]}
        for k, s in S.items():
            per_fold.setdefault(k, []).append(balanced_mean_over_n(s, y, q["n_games"].to_numpy()))
        rows.append(pd.DataFrame({"y": y, "pid": q["player_id"].to_numpy(dtype=str), "n": q["n_games"].to_numpy(), **S}))
        print(f"fold {v}: " + "  ".join(f"{k} {per_fold[k][-1]:.4f}" for k in P), flush=True)
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["pid"].to_numpy(), t["n"].to_numpy()
    models = [c for c in t.columns if c not in ("y", "pid", "n") and len(per_fold[c]) == len(FOLDS) and not t[c].isna().any()]
    for k in models:
        print(f"{k:12s} {np.mean(per_fold[k]):.4f}  folds " + " ".join(f"{x:.4f}" for x in per_fold[k])
              + "  by n " + " ".join(f"{s}: {balanced_mean_over_n(t[k].to_numpy(), y, n, sizes=(s,)):.4f}" for s in SIZES))
    pre = [(f"{b}-{loss}", b) for loss in ("K", "ES") for b in BASES]
    report = [("TREE-K", "MLP"), ("CT-K", "CT"), ("CT-ES", "CT")] + [(f"{k}@alt", k) for k in models if not k.endswith("@alt")]
    out = {"scores": {k: {"score": float(np.mean(per_fold[k])), "per_fold": per_fold[k]} for k in models}, "metrics": metrics, "info": info, "comparisons": {}}
    for group, pairs in (("pre_specified", pre), ("report_only", report)):
        out["comparisons"][group] = {}
        for x, w in pairs:
            if x not in models or w not in models:
                print(f"[{group}] {x} vs {w}: missing", flush=True)
                continue
            dd = paired_diff(t[x].to_numpy(), t[w].to_numpy(), y, pid, n, n=5000)
            pf = [a - b for a, b in zip(per_fold[x], per_fold[w])]
            cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= 3), "4": bool(dd["diff"] >= 0.005)}
            out["comparisons"][group][f"{x} vs {w}"] = {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
            print(f"[{group}] {x} vs {w}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
                  + "  by n " + " ".join(f"{s}: {dd['by_n'][s]:+.4f}" for s in SIZES) + f"  -> {'PASS' if all(cond.values()) else 'no'}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))


def cmd_smoke():
    import xgboost as xgb
    z = np.load(CACHE / "r_val1.npz")
    X, y = z["X_tr"][:8000].astype(np.float32), z["y_tr"][:8000].astype(int)
    cfg = {"eta": 0.1, "max_depth": 3}
    d = dmat(X, y)
    b0 = xgb.train(params(cfg), d, 30)
    b1 = xgb.train(custom_params(cfg), d, 30, obj=objective("ce"))
    gap = float(np.abs(b0.predict(d) - margin_proba(b1, d)).max())
    print(f"custom CE vs multi:softprob: max |dp| {gap:.2e}")
    assert gap < 1e-3, gap
    bk = xgb.train(custom_params(cfg), d, 30, obj=objective("kernel"))
    k_ce, k_k = inner_loss("kernel", b0.predict(d), y), inner_loss("kernel", margin_proba(bk, d), y)
    print(f"kernel CE on the training rows: CE model {k_ce:.4f}, K model {k_k:.4f}")
    assert k_k < k_ce
    start = inner_loss("es", b0.predict(d), y)
    b2 = xgb.train(custom_params(cfg), d, 20, obj=objective("es"), custom_metric=metric("es"), evals=[(d, "inner")], early_stopping_rounds=100,
                   maximize=False, verbose_eval=False, xgb_model=b0)
    print(f"continued booster: {b2.num_boosted_rounds()} rounds (30 + 20), best_iteration {b2.best_iteration}, ES loss start {start:.4f} best {float(b2.best_score):.4f}")
    assert b2.num_boosted_rounds() == 50 and 30 <= b2.best_iteration < 50, b2.best_iteration
    m1, m2 = metric("es")(b2.predict(d, output_margin=True), d)[1], metric("es")(margin_proba(b2, d), d)[1]
    assert abs(m1 - m2) < 1e-9, (m1, m2)
    assert np.allclose(margin_proba(b2, d), b2.predict(d), atol=1e-6)
    inn = np.zeros(len(y), dtype=bool)
    inn[:1500] = True
    Pk, Pes, info = mlp_variants(z["X_tr"][:8000], y, z["X_va"][:500], inn, "smoke MLP")
    print(info)
    assert Pk.shape == Pes.shape == (500, 10) and np.allclose(Pk.sum(axis=1), 1) and np.allclose(Pes.sum(axis=1), 1)
    print("SMOKE OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("smoke", "fit", "eval"))
    ap.add_argument("--val-fold", type=int)
    a = ap.parse_args()
    if a.cmd == "fit":
        if a.val_fold not in FOLDS:
            ap.error("--val-fold must be one of 1-4 (fold 0 is not used here)")
        cmd_fit(a.val_fold)
    elif a.cmd == "smoke":
        cmd_smoke()
    else:
        cmd_eval()


if __name__ == "__main__":
    main()
