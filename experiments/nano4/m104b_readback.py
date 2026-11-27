"""Section 4M M104 (2026-11-13): back-test of ct75u's Public reading against every recorded upload, and how the reading can split
between its two parts: (a) the 7 unanimous reversals of ct44 answers, (b) the 2 unanimous changes outside the boundary region.
  python -I logs/nano4/m104b_readback.py 0.8101304
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as m  # noqa: E402

read = float(sys.argv[1]); led = json.load(open(m.LEDGER)); led["submission_rank_ct75u.csv"] = read; up = {**m.KNOWN_SCORES, **led}; prior = m.prior_from_ct9()
qs, cand, obs = m.build(up, prior); mapa = m.solve(qs, cand, obs, prior); print(f"{len(up)} uploads; jointly {'FEASIBLE' if mapa is not None else 'INFEASIBLE'}", flush=True); assert mapa is not None
json.dump(led, open(m.LEDGER, "w"), indent=1); o = {x["name"]: x for x in obs}; u, s_ = o["submission_rank_ct75u.csv"], o["submission_rank_ct75s.csv"]; print(f"ct75u minus ct75s: exact {u['X'] - s_['X']:+d}, within one {u['Y'] - s_['Y']:+d}")
c75s, c75u, c44 = (m.decisions(f"submission_rank_{k}.csv") for k in ("ct75s", "ct75u", "ct44")); pub = [q for q in c44.index if q.startswith("pub")]; inv = {v: k for k, v in m.RANK.items()}
ga = [q for q in pub if c75u[q] != c75s[q] and c44[q] != m.decisions("submission_rank_ct28.csv")[q]]; gb = [q for q in pub if c75u[q] != c75s[q] and q not in ga]; print(f"(a) reversals {ga}\n(b) outside the region {gb}")
d_b = c75s.copy(); d_b[gb] = c75u[gb]; rows = []
for a in range(-2, 3):
    for b_ in range(-2, 3):
        qs2, cand2, obs2 = m.build(up, prior, [(d_b, s_["X"] + a, s_["Y"] + b_)]); sol = m.solve(qs2, cand2, obs2, prior, time_limit=20)
        if sol is not None and sol != "timeout":
            rows.append((a, b_, float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))))
cmin = min(r[2] for r in rows); z = sum(math.exp(-(r[2] - cmin)) for r in rows); tot = (u["X"] - s_["X"], u["Y"] - s_["Y"])
print("feasible splits (weight = rough plausibility):")
for a, b_, c in sorted(rows, key=lambda r: r[2]):
    print(f"   (b) outside: exact {a:+d}, within one {b_:+d}  |  (a) reversals: exact {tot[0] - a:+d}, within one {tot[1] - b_:+d}  |  weight {math.exp(-(c - cmin)) / z:.3f}")
fs = m.feasible_sets(qs, cand, obs, prior, mapa, only=ga + gb, time_limit=20)
for q in ga + gb:
    print(f"   {q} {'a' if q in ga else 'b'}: {inv[int(c75s[q])]} -> {inv[int(c75u[q])]} | possible {[inv[c] for c in fs[q]]} | most probable {inv[mapa[q]]}{'  DETERMINED' if len(fs[q]) == 1 else ''}")
print("ledger updated")
