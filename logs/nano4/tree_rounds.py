#!/usr/bin/env python
"""Is the GBDT of section 4D under-trained? (report only; folds 1-4, fold 0 is never read)

Per fold, the setting 4D chose is trained for 4000 rounds on the non-held-out training players without early stopping;
every 50 rounds the official score and the mlogloss are recorded on the held-out players and on 20000 of the training
questions it was fitted on. MLP h32 seed 0 fitted on the same questions gives the same two scores for reference. Reads:
where the official score peaks against where the mlogloss stops, and the training-minus-held-out gap of each model.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.rankmodel import MLPModel  # noqa: E402
from gbdt_hd_fit import class_weights, official  # noqa: E402
from gbdt_tune import params  # noqa: E402

CACHE, OUT = ROOT / "results/trunk_seq", ROOT / "results/gbdt_hd"
ROUNDS, EVERY = 4000, 50


def mlogloss(P, y):
    w = class_weights(y)
    return float((w * -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1))).sum() / w.sum())


def main():
    import xgboost as xgb
    out = {}
    for v in (1, 2, 3, 4):
        z = np.load(CACHE / f"r_val{v}.npz")
        pid = np.array(json.loads((CACHE / f"r_val{v}.json").read_text())["train"]["player_id"], dtype=str)
        y, X = z["y_tr"].astype(int), z["X_tr"].astype(np.float32)
        players = np.array(sorted(set(pid)))
        held = set(np.random.RandomState(12345 + v).permutation(players)[:int(0.1 * len(players))])
        inn = np.array([p in held for p in pid])
        fit_rows = np.flatnonzero(~inn)
        seen = np.random.RandomState(7 + v).choice(fit_rows, 20000, replace=False)
        cfg = json.loads((OUT / f"tune_val{v}.json").read_text())["a"]["chosen"]
        b = xgb.train(params(cfg), xgb.DMatrix(X[~inn], label=y[~inn], weight=class_weights(y[~inn])), ROUNDS)
        d_in, d_seen = xgb.DMatrix(X[inn]), xgb.DMatrix(X[seen])
        curve = []
        for r in range(EVERY, ROUNDS + 1, EVERY):
            Pi, Ps = b.predict(d_in, iteration_range=(0, r)), b.predict(d_seen, iteration_range=(0, r))
            curve.append({"rounds": r, "inner": official(Pi, y[inn]), "inner_mlogloss": mlogloss(Pi, y[inn]), "train": official(Ps, y[seen])})
        mlp = MLPModel(hidden=32, seed=0).fit(z["X_tr"][~inn], y[~inn])
        ref = {"inner": official(mlp.predict_proba(z["X_tr"][inn]), y[inn]), "train": official(mlp.predict_proba(z["X_tr"][seen]), y[seen])}
        peak = max(curve, key=lambda c: c["inner"])
        ll = min(curve, key=lambda c: c["inner_mlogloss"])
        at_es = min(curve, key=lambda c: abs(c["rounds"] - cfg["rounds"]))
        out[str(v)] = {"setting": cfg, "curve": curve, "mlp": ref, "peak_official": peak, "min_mlogloss": ll}
        print(f"fold {v}: early stop {cfg['rounds']} rounds (inner {at_es['inner']:.4f}, train {at_es['train']:.4f}); "
              f"official peaks at {peak['rounds']} ({peak['inner']:.4f}, train {peak['train']:.4f}); mlogloss lowest at {ll['rounds']}; "
              f"at {ROUNDS}: inner {curve[-1]['inner']:.4f}, train {curve[-1]['train']:.4f}; MLP inner {ref['inner']:.4f}, train {ref['train']:.4f}", flush=True)
    (OUT / "tree_rounds.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
