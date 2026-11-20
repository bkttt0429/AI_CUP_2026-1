#!/usr/bin/env python
"""Section 4M M148 (2026-11-15, report only): which KINDS of changes made since ct28 paid on the Public board? (design doc M148)

ct106 differs from ct28 (the last file before the opponent anchors) on a few hundred questions. The Public ledger (every recorded
upload, solved jointly; prior = the learned pool posterior) gives for every Public question that an upload moved the weight of
each possible truth (cheapest joint assignment with the truth fixed, weights exp(-cost)). Per kind of change (ct28's answer ->
ct106's answer): number of Public questions, expected points of the change under the ledger, share of the weight on "the new
answer is exact" / "the old answer was exact", and the number of Private questions with the same kind. Also by boundary and
direction (up / down), and the same table for the last step ct104b -> ct106.

  python -I logs/nano4/m148_cum_kinds.py
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

LV = led.LEVELS
z = np.load(ROOT / "results/r_variants/m137_test_posterior.npz", allow_pickle=True); prior = {str(q_): np.maximum(p_, 1e-6) for q_, p_ in zip(z["qid"], z["learned"])}
ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}; qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior, time_limit=600)
print(f"ledger: {len(obs)} uploads, {len(qs)} Public questions constrained; jointly feasible: {mapa not in (None, 'timeout')}", flush=True); assert mapa not in (None, "timeout")
D = {k: led.decisions(f"submission_rank_{k}.csv") for k in ("ct9", "ct28", "ct44", "ct75u", "ct104b", "ct106")}; allq = list(D["ct106"].index); pubq = [q for q in allq if q.startswith("pub")]; privq = [q for q in allq if not q.startswith("pub")]
cost_of = lambda sol: float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))  # noqa: E731
W = {}


def weights(q):
    if q not in W:
        if q in cand:
            w = {}
            for c in cand[q]:
                sol = led.solve(qs, cand, obs, prior, fix=(q, c), time_limit=60)
                if sol not in (None, "timeout"):
                    w[c] = cost_of(sol)
            m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; t = sum(w.values()); W[q] = ({c: v / t for c, v in w.items()}, True)
        else:
            p = np.asarray(prior[q], float); p = p / p.sum(); W[q] = ({c: float(p[c]) for c in range(10)}, False)
    return W[q]


def table(old, new, title):
    ch = [q for q in pubq if D[new][q] != D[old][q]]; rows = []
    for q in ch:
        a, b_ = int(D[new][q]), int(D[old][q]); w, con = weights(q)
        rows.append({"q": q, "old": b_, "new": a, "kind": f"{LV[b_]}>{LV[a]}", "e": sum(v * (led.s(a, c) - led.s(b_, c)) for c, v in w.items()), "p_new": w.get(a, 0.0), "p_old": w.get(b_, 0.0), "con": con})
    R = pd.DataFrame(rows); priv = pd.Series([f"{LV[int(D[old][q])]}>{LV[int(D[new][q])]}" for q in privq if D[new][q] != D[old][q]]).value_counts()
    print(f"\n=== {title}: {len(R)} Public questions differ ({int(R.con.sum())} constrained by the ledger), expected {R.e.sum():+.2f} points; Private questions that differ: {int(priv.sum())}")
    g = R.groupby("kind").agg(n=("e", "size"), pts=("e", "sum"), per=("e", "mean"), p_new=("p_new", "mean"), p_old=("p_old", "mean"), con=("con", "sum")).sort_values("pts")
    for k, r in g.iterrows():
        print(f"   {k:8s} Public {int(r.n):3d} ({int(r.con)} constrained): {r.pts:+6.2f} points, per question {r.per:+.3f}; weight on new exact {r.p_new:.2f} / old exact {r.p_old:.2f}; Private {int(priv.get(k, 0)):3d}")
    R["bd"] = [f"{LV[min(o, n_)]}|{LV[min(o, n_) + 1]} {'up' if n_ > o else 'down'}" if abs(n_ - o) == 1 else "two or more levels" for o, n_ in zip(R.old, R.new)]
    print("   by boundary and direction (one-level changes): " + "  ".join(f"[{k}: {int(r.n)} q {r.pts:+.2f}]" for k, r in R.groupby("bd").agg(n=("e", "size"), pts=("e", "sum")).iterrows()))
    return {k: {"n": int(r.n), "points": float(r.pts), "private": int(priv.get(k, 0))} for k, r in g.iterrows()}


rep = {"ct28>ct106": table("ct28", "ct106", "all changes since ct28 (ct28 -> ct106)"), "ct28>ct44": table("ct28", "ct44", "first anchor step (ct28 -> ct44)"), "ct44>ct106": table("ct44", "ct106", "everything after ct44 (ct44 -> ct106)"), "ct104b>ct106": table("ct104b", "ct106", "last step (ct104b -> ct106)")}
(ROOT / "results/r_variants/m148_cum_kinds.json").write_text(json.dumps(rep, indent=1)); print("\nwritten results/r_variants/m148_cum_kinds.json")
