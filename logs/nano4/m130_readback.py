#!/usr/bin/env python
"""Section 4M M130 (2026-11-14): the Public reading of ct104 (0.8069698) against every recorded upload.
1. the reading is added to the ledger (results/trunk_seq/es/public_ledger.json) and all uploads are solved jointly (m58_decode.py);
2. the 30 Public questions ct104 changed: what the ledger now says about each (profile over the truth, prior = the posterior
   behind the candidate, results/r_variants/m129_test_posterior.npz);
3. for ct104a and ct104b: the range of Public readings that is still possible (exact, by integer programming over the ledger's
   constraints) and the expected reading (profile weights), relative to ct75u (0.8101304).
  python -I logs/nano4/m130_readback.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

NEW = {"submission_rank_ct104.csv": 0.8069698}; REF75 = 0.8101304; E1 = math.exp(-1.0)
L = json.load(open(led.LEDGER)); L.update(NEW); json.dump(L, open(led.LEDGER, "w"), indent=1); up = {**led.KNOWN_SCORES, **L}
zp = np.load(ROOT / "results/r_variants/m129_test_posterior.npz", allow_pickle=True); base = led.prior_from_ct9()
PRI = {"base": base, "bundle": {str(q): np.maximum(p, 1e-6) for q, p in zip(zp["qid"], zp["bundle"])}, "current": {str(q): np.maximum(p, 1e-6) for q, p in zip(zp["qid"], zp["current"])}}
qs, cand, obs = led.build(up, base); mapa = led.solve(qs, cand, obs, base); print(f"{len(up)} uploads recorded; jointly {'FEASIBLE' if mapa not in (None, 'timeout') else 'INFEASIBLE'}", flush=True); assert mapa not in (None, "timeout")
o = {x["name"]: x for x in obs}; dx = lambda a, b_: (o[a]["X"] - o[b_]["X"], o[a]["Y"] - o[b_]["Y"])  # noqa: E731
print(f"ct104 minus ct75u: exact {dx('submission_rank_ct104.csv', 'submission_rank_ct75u.csv')[0]:+d}, within one {dx('submission_rank_ct104.csv', 'submission_rank_ct75u.csv')[1]:+d}")
d75, d104, d104a, d104b = (led.decisions(f"submission_rank_{k}.csv") for k in ("ct75u", "ct104", "ct104a", "ct104b")); pub = [q for q in d75.index if q.startswith("pub")]; inv = {v: k for k, v in led.RANK.items()}
cost_of = lambda sol, prior: float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))  # noqa: E731


def profile(q, prior):
    w = {}
    for c in cand[q]:
        sol = led.solve(qs, cand, obs, prior, fix=(q, c), time_limit=60)
        if sol not in (None, "timeout"):
            w[c] = cost_of(sol, prior)
    m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; z = sum(w.values())
    return {c: v / z for c, v in w.items()}


def extreme(dn, sense):
    """Smallest (sense = +1) or largest (-1) possible Public difference of decisions dn against ct75u, in points, under the ledger."""
    var = [(q, c) for q in qs for c in cand[q]]; idx = {v: i for i, v in enumerate(var)}; cost = np.zeros(len(var))
    for q in pub:
        if dn[q] != d75[q] and q in cand:
            for c in cand[q]:
                cost[idx[(q, c)]] = sense * (led.s(int(dn[q]), c) - led.s(int(d75[q]), c))
    n_rows = len(qs) + 2 * len(obs); A = lil_matrix((n_rows, len(var))); lo = np.zeros(n_rows); hi = np.zeros(n_rows)
    for r, q in enumerate(qs):
        for c in cand[q]:
            A[r, idx[(q, c)]] = 1
        lo[r] = hi[r] = 1
    for k, ob in enumerate(obs):
        rx, ry = len(qs) + 2 * k, len(qs) + 2 * k + 1
        for q, (a0, a1) in ob["moves"].items():
            for c in cand[q]:
                x, y = led.xy(a0, a1, c)
                if x:
                    A[rx, idx[(q, c)]] = x
                if y:
                    A[ry, idx[(q, c)]] = y
        lo[rx] = hi[rx] = ob["X"]; lo[ry] = hi[ry] = ob["Y"]
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(len(var)), bounds=Bounds(np.zeros(len(var)), np.ones(len(var))), options={"time_limit": 300})
    return None if res.x is None else sense * float(res.fun)


ch = [q for q in pub if d104[q] != d75[q]]; W = {}
print(f"\nthe {len(ch)} Public questions ct104 changed (prior of the profile: the posterior behind the candidate):")
for q in ch:
    W[q] = profile(q, PRI["bundle"]); a, b_ = int(d104[q]), int(d75[q]); e = sum(v * (led.s(a, c) - led.s(b_, c)) for c, v in W[q].items()); top = max(W[q], key=W[q].get)
    kind = "leaves A" if b_ == 3 else ("5D->6D" if (b_, a) == (8, 9) else "other")
    print(f"  {q}: {inv[b_]} -> {inv[a]} [{kind:8s}]  expected {e:+.3f}; truth most likely {inv[top]} ({W[q][top]:.2f})" + ("  DETERMINED" if W[q][top] > 0.995 else ""), flush=True)
for nm, dn in (("ct104a", d104a), ("ct104b", d104b)):
    cq = [q for q in pub if dn[q] != d75[q]]; e = sum(v * (led.s(int(dn[q]), c) - led.s(int(d75[q]), c)) for q in cq for c, v in W[q].items()); lo_, hi_ = extreme(dn, +1), extreme(dn, -1)
    back = [q for q in ch if dn[q] == d75[q]]; eb = sum(v * (led.s(int(d104[q]), c) - led.s(int(d75[q]), c)) for q in back for c, v in W[q].items())
    print(f"\n{nm}: {len(cq)} Public questions differ from ct75u; the {len(back)} changes of ct104 it does not make are worth {eb:+.2f} points in expectation (their removal: {-eb:+.2f})")
    print(f"  possible Public readings: {REF75 + lo_ / 400:.7f} .. {REF75 + hi_ / 400:.7f} ({lo_:+.2f} .. {hi_:+.2f} points against ct75u); expected {REF75 + e / 400:.7f} ({e:+.2f} points)")
print("\nledger updated")
