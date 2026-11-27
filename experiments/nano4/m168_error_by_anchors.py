#!/usr/bin/env python
"""Section 4M M168 diagnostic (2026-11-15, report only): where do the ct106 rule's remaining errors live - in questions without
confident anchors, or in questions that have them?

Validation questions (seeds 0-2; decisions dumped by m165_pool.py --dump): by question size and by the number of games whose
opponent was recognised with a margin above the impostor q95 / q99 (identifier given, even games and one-stone games), the
number of questions, the exact-level accuracy and the class-balanced score of the ct106 rule and of the pool with the new source.

  python -I logs/nano4/m168_error_by_anchors.py m168_dump.npz SX4L
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

m3, FOLDS, SM = mb7.m3, mb7.FOLDS, mb7.SM; RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
D = dict(np.load(RV / sys.argv[1], allow_pickle=True)); tag = sys.argv[2]; z = dict(np.load(RV / f"m84_{tag}~m15_val.npz", allow_pickle=True)); tb = z["target_black"]
edges = {True: np.quantile(z["f_imp_m"][tb], QS), False: np.quantile(z["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], z["f_opp_m"]), np.searchsorted(edges[True], z["f_opp_m"])); band[z["hd"] > 1] = -1
row = {k: i for i, k in enumerate(z["key"].astype(str))}; c95, c99, gk_n = [], [], []
for v in FOLDS:
    y, n, pid, qd = m3.load_fold(v)
    for g_ in qd["game_keys"]:
        b_ = np.array([band[row[k]] for k in g_ if k in row]); c95.append(int((b_ >= 4).sum())); c99.append(int((b_ >= 6).sum())); gk_n.append(len(g_))
c95, c99 = np.array(c95), np.array(c99); y, n = D["y"].astype(int), D["n"].astype(int); assert len(c95) == len(y) and np.array_equal(np.array(gk_n), n), "questions are not aligned"
new = next(k for k in D if k not in ("y", "n", "pid", "fold", "ct106_rule", "ct104b_rule")); print(f"identifier {tag}; new decisions = pool + {new}; questions {len(y)}")
for nm, c in (("games above the impostor q95", c95), ("games above the impostor q99", c99)):
    print(f"\n{nm}: per size and count - questions (share) | exact-level accuracy ct106 rule -> + new | within one ct106 rule | mean score ct106 rule -> + new")
    for s in (5, 10, 20):
        for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 99)):
            m = (n == s) & (c >= lo) & (c <= hi)
            if m.sum() < 30:
                continue
            a0 = float((D["ct106_rule"][m] == y[m]).mean()); a1 = float((D[new][m] == y[m]).mean()); w0 = float((np.abs(D["ct106_rule"][m] - y[m]) <= 1).mean())
            print(f"  n={s:2d} anchors {lo}{'' if hi == lo else ('+' if hi > 50 else f'-{hi}')}: {int(m.sum()):5d} ({m.sum() / (n == s).sum():.3f}) | {a0:.3f} -> {a1:.3f} | {w0:.3f} | {float(SM[D['ct106_rule'][m], y[m]].mean()):.4f} -> {float(SM[D[new][m], y[m]].mean()):.4f}")
