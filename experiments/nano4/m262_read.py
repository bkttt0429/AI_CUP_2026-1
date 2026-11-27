#!/usr/bin/env python
"""M262 step 2, the READING of the Public probe (2026-11-20 23:06: upload_20261120/2_probe_winshare.csv = 0.8483167 against the live
file ct118e = 0.85871825). The registered reading (design doc M262 step 2): net points and the ratio "probability of the reading under
the shifted posterior / under the current posterior" over the changed Public questions.

  python -I logs/nano4/m262_read.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m229_like as ml  # noqa: E402

LEV = led.LEVELS; K = np.arange(10.0); RV = ROOT / "results/r_variants"; GAMMA = -0.4; S_REF, S_PROBE = 0.85871825, 0.8483167


def soft(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def main():
    z = np.load(RV / "m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P0 = z["P"][names.index("current")].astype(np.float64); qid = z["qid"].astype(str)
    oc = np.load(RV / "m262_outcomes.npz"); op = {str(x): i for i, x in enumerate(oc["qid"])}; oi = np.array([op[x] for x in qid]); ne = oc["n_even"][oi].astype(float); sh = np.where(ne > 0, oc["wins_even"][oi] / np.maximum(ne, 1), 0.5) - 0.5; x = (sh - sh.mean()) / sh.std()
    P1 = soft(np.log(np.maximum(P0, 1e-300)) + GAMMA * x[:, None] * K[None, :]); rd = lambda f: {r["question_id"]: LEV.index(r["pred_rank"]) for r in csv.DictReader(open(ROOT / f))}  # noqa: E731
    ref, pr = rd("submission_rank_ct118e.csv"), rd("upload_20261120/used/2_probe_winshare.csv"); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); a0 = np.array([ref[q] for q in qid[pi]]); a1 = np.array([pr[q] for q in qid[pi]]); idx = np.flatnonzero(a0 != a1)
    assert all(ref[q] == pr[q] for q in qid[~pub]); d = 400 * (S_PROBE - S_REF); sols = led.decompose(d, len(idx)); print(f"probe against ct118e: {len(idx)} Public answers changed, {d:+.4f} points; (exact units, within-one units): {sols}"); assert len(sols) == 1
    xo, yo = sols[0]; rep = {"points": d, "dE": xo, "dW": yo, "changed": int(len(idx))}; pts = lambda D, n: sum(D[n + a, n + b_] * ((1 - led.E1) * a + led.E1 * b_) for a in range(-n, n + 1) for b_ in range(-n, n + 1))  # noqa: E731
    out = {}
    for nm, P in (("shifted (gamma -0.4)", P1), ("current", P0)):
        D, n = ml.pair_dist(P[pi], a0, a1, idx); p = float(D[n + xo, n + yo]); grid = np.array([[(1 - led.E1) * a + led.E1 * b_ for b_ in range(-n, n + 1)] for a in range(-n, n + 1)]); tail = float(D[grid <= d + 1e-9].sum()); mean = float((D * grid).sum()); sd = float(np.sqrt((D * (grid - mean) ** 2).sum()))
        out[nm] = p; print(f"   under the {nm} posterior: expected {mean:+.2f} points (sd {sd:.2f}); probability of exactly this reading {p:.2e}; of a reading this low or lower {tail:.4f}"); rep[nm] = {"expected": mean, "sd": sd, "p_reading": p, "p_tail": tail}
    lr = out["shifted (gamma -0.4)"] / out["current"]; verdict = "PASS" if d > 0 and lr >= 10 else ("FAIL: the line is closed, the restore file stays the live upload" if d <= 0 or lr <= 1 else "in between")
    print(f"REGISTERED: net {d:+.2f} points, likelihood ratio shifted / current = {lr:.3g}  ->  {verdict}")
    print("   the changed questions (ct118e -> probe | P(old), P(new) under the current posterior | under the shifted posterior):")
    for j in idx:
        i = pi[j]; print(f"      {qid[i]}: {LEV[a0[j]]} -> {LEV[a1[j]]} | {P0[i, a0[j]]:.2f}, {P0[i, a1[j]]:.2f} | {P1[i, a0[j]]:.2f}, {P1[i, a1[j]]:.2f}")
    rep.update(likelihood_ratio=lr, verdict=verdict); (RV / "m262_read.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m262_read.json")


if __name__ == "__main__":
    main()
