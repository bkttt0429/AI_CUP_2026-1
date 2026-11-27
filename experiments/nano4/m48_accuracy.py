"""M48 follow-up (report only, 2026-11-07; user: 顯示這專家模型預測的具體準確度): the 'bal' adjacent-pair specialists of M48 (seeds 0-2,
cross-fitted as there): per boundary, pairwise accuracy and balanced accuracy (mean of the two levels' recalls) of the specialist and of
ct9's own log-odds; and the 10-level outcome (exact, off by one, two or more, official score; per-level exact) of ct9, chain and blend.
  python -I logs/nano4/m48_accuracy.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m46_a1d import xprobs  # noqa: E402
from m48_chain import wlogit  # noqa: E402

D = {}
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    D[v] = {"y": y, "P": P, "X": xprobs(P, n)}
L = {v: np.zeros((len(D[v]["y"]), 9)) for v in FOLDS}
pair = {k: {"s": [], "b": [], "y": []} for k in range(9)}
for v in FOLDS:
    tr = [u for u in FOLDS if u != v]
    for k in range(9):
        sel = lambda u: (D[u]["y"] == k) | (D[u]["y"] == k + 1)  # noqa: E731
        X = np.concatenate([D[u]["X"][sel(u)] for u in tr]); lab = np.concatenate([(D[u]["y"][sel(u)] == k + 1).astype(float) for u in tr])
        w = np.where(lab == 1, 0.5 / lab.mean(), 0.5 / (1 - lab.mean()))
        L[v][:, k] = wlogit(X, lab, w, D[v]["X"])
        m = sel(v); lp = np.log(np.clip(D[v]["P"][m], 1e-9, 1))
        pair[k]["s"].append(L[v][m, k]); pair[k]["b"].append(lp[:, k + 1] - lp[:, k]); pair[k]["y"].append((D[v]["y"][m] == k + 1).astype(int))
rows = []
for k in range(9):
    s, b, yy = (np.concatenate(pair[k][x]) for x in ("s", "b", "y"))
    def accs(score):
        pr = (score > 0).astype(int); rec_hi = (pr[yy == 1] == 1).mean(); rec_lo = (pr[yy == 0] == 0).mean()
        return (pr == yy).mean(), 0.5 * (rec_hi + rec_lo), rec_lo, rec_hi
    a_s, a_b = accs(s), accs(b)
    rows.append({"boundary": f"{LEVELS[k]}|{LEVELS[k + 1]}", "questions": len(yy), "specialist acc": a_s[0], "ct9 acc": a_b[0], "specialist balanced": a_s[1], "ct9 balanced": a_b[1],
                 "specialist recall lo / hi": f"{a_s[2]:.3f} / {a_s[3]:.3f}", "ct9 recall lo / hi": f"{a_b[2]:.3f} / {a_b[3]:.3f}"})
pd.set_option("display.width", 220)
print("== each specialist on its own two levels (threshold at even odds)"); print(pd.DataFrame(rows).round(3).to_string(index=False))
res, per = {}, {}
for name in ("ct9", "chain", "blend"):
    ds, ys = [], []
    for v in FOLDS:
        lq = np.hstack([np.zeros((len(L[v]), 1)), np.cumsum(L[v], 1)]); lp = np.log(np.clip(D[v]["P"], 1e-9, 1))
        z = lp if name == "ct9" else (lq if name == "chain" else 0.5 * (lq + lp))
        ds.append(decide(np.exp(z - z.max(1, keepdims=True)))); ys.append(D[v]["y"])
    d, y = np.concatenate(ds), np.concatenate(ys); k = np.abs(d - y)
    res[name] = {"exact": (k == 0).mean(), "off by one": (k == 1).mean(), "two or more": (k >= 2).mean(), "official score (plain)": qscore(d, y).mean()}
    per[name] = [(d[y == c] == c).mean() for c in range(10)]
print("\n== 10-level outcome on all 39,327 validation questions"); print(pd.DataFrame(res).T.round(3).to_string())
print("\n== exact rate per true level"); print(pd.DataFrame(per, index=LEVELS).T.round(3).to_string())
