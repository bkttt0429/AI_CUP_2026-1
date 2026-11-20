"""M27 early signal (report only, not the pre-registered test): the 64-dim R question features with the Japanese-rule Human SL columns
against the Chinese-rule ones, on the cached M22-protocol R (r_val{v}_f0_x40[_jp], fold 0 + 3 folds + x40 training questions), the R-layer
MLP of scripts/hd_check.py (h32, seeds 0-2 averaged), class-balanced official score, paired player bootstrap.
  python -I logs/nano4/m27_r_signal.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); T = ROOT / "results/trunk_seq"
sys.path[:0] = [str(ROOT / "src")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402

S, per_fold, ys, pids, ns = {"cn": [], "jp": []}, {"cn": [], "jp": []}, [], [], []
for v in (1, 2, 3, 4):
    va = pd.DataFrame(json.loads((T / f"r_val{v}_f0_x40.json").read_text())["val"])
    for arm, name in (("cn", f"r_val{v}_f0_x40"), ("jp", f"r_val{v}_f0_x40_jp")):
        z = np.load(T / f"{name}.npz")
        P = np.mean([MLPModel(hidden=32, seed=s).fit(z["X_tr"], z["y_tr"]).predict_proba(z["X_va"]) for s in (0, 1, 2)], axis=0)
        res, sc = evaluate(P, z["y_va"], va)
        S[arm].append(sc); per_fold[arm].append(res["balanced_mean_over_n"])
    ys.append(z["y_va"]); pids.append(va["player_id"].to_numpy(dtype=str)); ns.append(va["game_keys"].map(len).to_numpy() if "game_keys" in va else va["n_games"].to_numpy())
    print(f"fold {v}: cn {per_fold['cn'][-1]:.4f}  jp {per_fold['jp'][-1]:.4f}  diff {per_fold['jp'][-1] - per_fold['cn'][-1]:+.4f}", flush=True)
y, pid, n = np.concatenate(ys), np.concatenate(pids), np.concatenate(ns)
Sc, Sj = np.concatenate(S["cn"]), np.concatenate(S["jp"])
d = paired_diff(Sj, Sc, y, pid, n)
print(f"R layer (64 cols, MLP h32 x 3 seeds, 4 folds): cn {np.mean(per_fold['cn']):.4f}  jp {np.mean(per_fold['jp']):.4f}  jp - cn {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
      f"folds {[round(a - b, 4) for a, b in zip(per_fold['jp'], per_fold['cn'])]}  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d["by_n"].items()))
(ROOT / "results/r_variants/m27_r_signal.json").write_text(json.dumps({"per_fold": per_fold, "diff": d}, indent=1, default=float))
