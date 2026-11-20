#!/usr/bin/env python
"""M157b (2026-11-15, report only): which prior over the levels do the recorded Public readings support? A scan.
Ledger solved jointly with prior = pool posterior x w (normalised per question); cost of the best assignment (lower = better).
  (1) only the A : 1D ratio tilted (A x r);  (2) one level at a time multiplied by 1.5 and by 1 / 1.5 (which single levels the
  readings want more or less of);  (3) the by-players composition with A and 1D put back to balanced, and the reverse.
  python -I logs/nano4/m157b_ledger_scan.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

LV = led.LEVELS; z = np.load(ROOT / "results/r_variants/m137_test_posterior.npz", allow_pickle=True); qids = z["qid"].astype(str); Pl = z["learned"].astype(float)
ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}
pfc = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str); rk = {r: i for i, r in enumerate(LV)}; lv = pfc["rank"].map(rk).to_numpy().astype(int); byp = np.bincount(lv, minlength=10) / len(lv)


def cost(w):
    Pp = Pl * np.asarray(w, float)[None, :]; Pp /= Pp.sum(1, keepdims=True); prior = {str(q_): np.maximum(p_, 1e-6) for q_, p_ in zip(qids, Pp)}; qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior, time_limit=600)
    if mapa in (None, "timeout"):
        return float("nan"), None
    return float(sum(-math.log(max(prior[q_][c], 1e-6)) for q_, c in mapa.items())), np.bincount(list(mapa.values()), minlength=10)


c0, n0 = cost(np.ones(10)); print(f"balanced prior: cost {c0:.2f}; truths of the best assignment " + " ".join(f"{LV[c]} {n0[c]}" for c in range(10)), flush=True)
print("(1) A : 1D ratio only (A multiplied by r):")
for r in (0.5, 0.75, 1.5, 2, 3, 4, 6, 8, 12, 20):
    w = np.ones(10); w[3] = r; c, n = cost(w); print(f"    r = {r:5}: cost {c:.2f} ({c - c0:+.2f}); A {n[3]}, 1D {n[4]}", flush=True)
print("(2) one level at a time (x 1.5 | x 1/1.5), change of the cost against balanced:")
for k in range(10):
    w = np.ones(10); w[k] = 1.5; a, _ = cost(w); w[k] = 1 / 1.5; b_, _ = cost(w); print(f"    {LV[k]:3s}: {a - c0:+.2f} | {b_ - c0:+.2f}", flush=True)
print("(3) mixtures:")
for name, w in (("by players", byp * 10), ("by players, A and 1D balanced", np.where(np.isin(np.arange(10), [3, 4]), 1.0, byp * 10)), ("balanced, A and 1D by players", np.where(np.isin(np.arange(10), [3, 4]), byp * 10, 1.0)),
                ("kyu x 1.38 (M68's two-group ratio)", np.where(np.arange(10) <= 3, 1.38, 1.0)), ("A x 4, 6D x 0.5", np.array([1, 1, 1, 4, 1, 1, 1, 1, 1, 0.5])), ("A x 4, 6D x 0.5, 5D x 1.5", np.array([1, 1, 1, 4, 1, 1, 1, 1, 1.5, 0.5]))):
    c, n = cost(w); print(f"    {name:34s}: cost {c:.2f} ({c - c0:+.2f}); truths " + " ".join(f"{LV[k]} {n[k]}" for k in range(10)), flush=True)
