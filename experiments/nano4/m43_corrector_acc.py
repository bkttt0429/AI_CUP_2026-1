"""Section 4M M43/M44 follow-up (report only, 2026-11-07; user: 修正模型的準確率是多少): on the questions the ct9 decision gets wrong, how
often does the error-only corrector name the right level, against two baselines drawn from the main posterior itself: its runner-up
(argmax over the levels other than y-hat) and the more probable neighbour of y-hat. Cross-fitted as M44 (natural frequencies) and as
M43 (class-balanced), seeds 0-2.
  python -I logs/nano4/m43_corrector_acc.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import E1, FOLDS, decide, features, train_mlp  # noqa: E402
from m44_calib import ce_plain, disagreement, fit_net, mlp  # noqa: E402

D = {}
for v in FOLDS:
    y, n, pid, q = load_fold(v)
    PmC = [np.load(ES / f"val{v}/C-pc-f0-x40-oh_s{s}.npz")["proba"].astype(np.float64) for s in (0, 1, 2)]
    PmT = [np.load(ES / f"val{v}/T-pc-f0-x40-oh_s{s}.npz")["proba"].astype(np.float64) for s in (0, 1, 2)]
    P = np.mean(PmC + PmT, axis=0); d = decide(P)
    R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float32)
    D[v] = {"y": y, "pid": pid, "P": P, "d": d, "X": np.hstack([features(P, n, R), disagreement(PmC, PmT, d)])}


def stats(pred, y):
    k = np.abs(pred - y)
    return (k == 0).mean(), (k <= 1).mean(), np.where(k == 0, 1.0, np.where(k == 1, E1, 0.0)).mean()


rows = {"corrector (natural)": [], "corrector (class-balanced, M43)": [], "main runner-up": [], "better neighbour of y-hat": [], "y": [], "k": []}
for v in FOLDS:
    tr = [u for u in FOLDS if u != v]
    X = np.concatenate([D[u]["X"] for u in tr]); y = np.concatenate([D[u]["y"] for u in tr]); d = np.concatenate([D[u]["d"] for u in tr]); g = np.concatenate([D[u]["pid"] for u in tr])
    w = d != y
    net = mlp(X.shape[1], 10); mu, sd = fit_net(net, ce_plain, X[w], y[w], g[w], seed=10 + v)
    bal = train_mlp(X[w], y[w], 10, g[w], seed=10 + v)
    Xv, yv, dv, Pv = D[v]["X"], D[v]["y"], D[v]["d"], D[v]["P"]; wv = dv != yv
    with torch.no_grad():
        qn = torch.softmax(net(torch.as_tensor((Xv[wv] - mu) / sd)), 1).numpy()
    qb = bal(Xv[wv])
    for q in (qn, qb):
        q[np.arange(wv.sum()), dv[wv]] = 0.0
    Pr = Pv[wv].copy(); Pr[np.arange(wv.sum()), dv[wv]] = 0.0
    lo, hi = np.clip(dv[wv] - 1, 0, 9), np.clip(dv[wv] + 1, 0, 9)
    nb = np.where(Pv[wv][np.arange(wv.sum()), lo] >= Pv[wv][np.arange(wv.sum()), hi], lo, hi); nb = np.where(lo == dv[wv], hi, np.where(hi == dv[wv], lo, nb))
    rows["corrector (natural)"].append(qn.argmax(1)); rows["corrector (class-balanced, M43)"].append(qb.argmax(1)); rows["main runner-up"].append(Pr.argmax(1)); rows["better neighbour of y-hat"].append(nb)
    rows["y"].append(yv[wv]); rows["k"].append(np.abs(dv[wv] - yv[wv]))
y, k = np.concatenate(rows.pop("y")), np.concatenate(rows.pop("k"))
print(f"{len(y)} wrong questions (of the four folds' validation questions); off by one {(k == 1).mean():.3f}, by two or more {(k >= 2).mean():.3f}")
for name, p in rows.items():
    acc, adj, sc = stats(np.concatenate(p), y)
    print(f"{name:32s} exact {acc:.3f}   within one {adj:.3f}   official score on these questions {sc:.3f}")
