#!/usr/bin/env python
"""M208 (2026-11-17), REPORT ONLY: the probe readings of M202 / M206 solved jointly with every recorded upload.

The ledger (m58_decode.py): every upload against ct9 gives two exact counts over its changed Public questions; the truths
are the cheapest joint assignment under a prior. Added here:
  the two small probes of 2026-11-17 (three 4D -> 5D locked questions back to 4D; seven questions A -> B) as two more uploads;
  the exact level composition of the Public half (M202 / M206) as ten count constraints over ALL 400 Public questions
  (the questions no upload ever changed become variables too, with every level as a candidate).
Prior: the pooled posterior behind ct112. For the questions of the two probes: the weight of each truth
(exp(-cost) of the best joint assignment with that truth fixed), with and without the composition constraints.

  python -I scripts/m_series/m208_ledger_probe.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m58_decode as led  # noqa: E402

LEV = led.LEVELS; PROBES = {"upload_probe/probe_4D5D_locked.csv": 0.8347546, "upload_probe/probe_A2B.csv": 0.8363349}


def solve(qs, cand, obs, prior, counts=None, fix=None, time_limit=120):
    """(assignment, cost) of the cheapest joint truth assignment; counts: {level index: exact number of questions} over qs."""
    var = [(q, c) for q in qs for c in cand[q]]; idx = {v: i for i, v in enumerate(var)}; cost = np.array([-math.log(max(prior[q][c], 1e-6)) for q, c in var])
    n_rows = len(qs) + 2 * len(obs) + (10 if counts else 0); A = lil_matrix((n_rows, len(var))); lo = np.zeros(n_rows); hi = np.zeros(n_rows)
    for r, q in enumerate(qs):
        for c in cand[q]:
            A[r, idx[(q, c)]] = 1
        lo[r] = hi[r] = 1
    for k, o in enumerate(obs):
        rx, ry = len(qs) + 2 * k, len(qs) + 2 * k + 1
        for q, (d0, d1) in o["moves"].items():
            for c in cand[q]:
                x, y = led.xy(d0, d1, c)
                if x:
                    A[rx, idx[(q, c)]] = x
                if y:
                    A[ry, idx[(q, c)]] = y
        lo[rx] = hi[rx] = o["X"]; lo[ry] = hi[ry] = o["Y"]
    if counts:
        base = len(qs) + 2 * len(obs)
        for (q, c), i in idx.items():
            A[base + c, i] = 1
        for c in range(10):
            lo[base + c] = hi[base + c] = counts[c]
    lb = np.zeros(len(var)); ub = np.ones(len(var))
    if fix:
        if fix not in idx:
            return None, None
        lb[idx[fix]] = 1
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(len(var)), bounds=Bounds(lb, ub), options={"time_limit": time_limit})
    if res.status != 0 or res.x is None:
        return ("timeout" if res.status == 1 else None), None
    return {q: c for (q, c), v in zip(var, res.x) if v > 0.5}, float(cost @ np.round(res.x))


def main():
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}; qs0, cand0, obs0 = led.build(up, prior); m0, c0 = solve(qs0, cand0, obs0, prior); assert isinstance(m0, dict), "the ledger without the probes is not feasible"
    qs1, cand1, obs1 = led.build({**up, **PROBES}, prior); m1, c1 = solve(qs1, cand1, obs1, prior); print(f"ledger: {len(obs0)} uploads, {len(qs0)} constrained Public questions, cost {c0:.2f}; with the two probes: {len(obs1)} uploads, {len(qs1)} questions, " + ("INFEASIBLE" if not isinstance(m1, dict) else f"cost {c1:.2f}"), flush=True)
    for o in obs1[-2:]:
        print(f"  {o['name']}: against ct9 {len(o['moves'])} changed questions, exact {o['X']:+d}, within one {o['Y']:+d}")
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}; assert sum(counts.values()) == 400
    c112 = led.decisions("submission_rank_ct112.csv"); pub = [q for q in c112.index if q.startswith("pub")]; cand2 = {q: (cand1[q] if q in cand1 else list(range(10))) for q in pub}
    m2, c2 = solve(pub, cand2, obs1, prior, counts); print("with the composition as well (all 400 Public questions): " + ("INFEASIBLE or timeout: " + str(m2) if not isinstance(m2, dict) else f"cost {c2:.2f}"), flush=True)
    if not isinstance(m2, dict):  # the candidate sets of the constrained questions (within two levels of an answer given) may be too narrow for the exact counts
        cand2 = {q: list(range(10)) for q in pub}; m2, c2 = solve(pub, cand2, obs1, prior, counts); print("  retried with every level as a candidate everywhere: " + (str(m2) if not isinstance(m2, dict) else f"cost {c2:.2f}"), flush=True)
    pa = led.decisions("upload_probe/probe_A2B.csv"); pl = led.decisions("upload_probe/probe_4D5D_locked.csv"); c9, c28 = led.decisions("submission_rank_ct9.csv"), led.decisions("submission_rank_ct28.csv")
    groups = {"A -> B probe (7)": [q for q in pub if pa[q] != c112[q]], "4D -> 5D push, pool disagrees (3)": [q for q in pub if pl[q] != c112[q]], "4D -> 5D push, pool agrees (6)": [q for q in pub if c9[q] == 7 and c28[q] == 8 and pl[q] == c112[q]],
              "1D -> A push (8)": [q for q in pub if c9[q] == 4 and c28[q] == 3]}

    def profile(q, qs, cand, obs, cnt):
        w = {}
        for c in cand[q]:
            a_, cost = solve(qs, cand, obs, prior, cnt, fix=(q, c), time_limit=90)
            if isinstance(a_, dict):
                w[c] = cost
        if not w:
            return {}
        m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; t = sum(w.values()); return {c: v / t for c, v in w.items()}
    rep = {}
    for gname, qg in groups.items():
        print(f"\n== {gname}: truth weights (ledger + probes | + composition); the answer of ct112; the pooled posterior alone")
        for q in qg:
            w1 = profile(q, qs1, cand1, obs1, None) if q in cand1 else {}; w2 = profile(q, pub, cand2, obs1, counts) if isinstance(m2, dict) else {}
            fmt = lambda w: " ".join(f"{LEV[c]} {v:.2f}" for c, v in sorted(w.items(), key=lambda t: -t[1])[:3] if v >= 0.005) or "-"  # noqa: E731
            pr = prior[q]; top = np.argsort(-pr)[:3]; print(f"  {q}: {fmt(w1):28s} | {fmt(w2):28s} | ct112 {LEV[int(c112[q])]:>2} | posterior " + " ".join(f"{LEV[c]} {pr[c]:.2f}" for c in top), flush=True); rep[q] = {"ledger": {LEV[c]: v for c, v in w1.items()}, "with_composition": {LEV[c]: v for c, v in w2.items()}}
    if isinstance(m2, dict):  # where the cheapest assignment under everything disagrees with ct112's answers
        truth = np.array([m2[q] for q in pub]); ans = np.array([int(c112[q]) for q in pub]); S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])
        print(f"\ncheapest joint assignment with everything: ct112 would score {float(S[ans, truth].mean()):.4f} on it (obtained 0.8332); it puts the truth away from ct112's answer on {int((truth != ans).sum())} questions ({int((np.abs(truth - ans) == 1).sum())} adjacent)")
    (ROOT / "results/r_variants/m208_ledger_probe.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m208_ledger_probe.json")


if __name__ == "__main__":
    main()
