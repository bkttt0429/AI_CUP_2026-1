#!/usr/bin/env python
"""Section 4M M130 (2026-11-14): the Public reading of ct104b (0.8126304) against every recorded upload: the reading is added to
the ledger, all uploads are solved jointly (m58_decode.py), and the two groups are decoded - the 11 changes ct104 makes and
ct104b does not (9 leave A, 2 are 5D->6D) and the 19 changes of ct104b - with the labels that are still possible per question.
  python -I logs/nano4/m130b_readback.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

NEW = {"submission_rank_ct104b.csv": 0.8126304}
L = json.load(open(led.LEDGER)); L.update(NEW); json.dump(L, open(led.LEDGER, "w"), indent=1); up = {**led.KNOWN_SCORES, **L}; prior = led.prior_from_ct9()
qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior); print(f"{len(up)} uploads recorded; jointly {'FEASIBLE' if mapa not in (None, 'timeout') else 'INFEASIBLE'}", flush=True); assert mapa not in (None, "timeout")
o = {x["name"]: x for x in obs}; dx = lambda a, b_: (o[a]["X"] - o[b_]["X"], o[a]["Y"] - o[b_]["Y"])  # noqa: E731
for a, b_ in (("ct104b", "ct75u"), ("ct104", "ct75u"), ("ct104", "ct104b")):
    x, y = dx(f"submission_rank_{a}.csv", f"submission_rank_{b_}.csv"); print(f"{a} minus {b_}: exact {x:+d}, within one {y:+d} ({(1 - led.E1) * x + led.E1 * y:+.3f} points)")
d75, d104, d104b = (led.decisions(f"submission_rank_{k}.csv") for k in ("ct75u", "ct104", "ct104b")); pub = [q for q in d75.index if q.startswith("pub")]; inv = {v: k for k, v in led.RANK.items()}
G = {"dropped by ct104b (leaves A / 5D->6D)": [q for q in pub if d104[q] != d75[q] and d104b[q] == d75[q]], "kept in ct104b": [q for q in pub if d104b[q] != d75[q]]}
fs = led.feasible_sets(qs, cand, obs, prior, mapa, only=[q for g in G.values() for q in g], time_limit=30)
for g, ql in G.items():
    print(f"\n{g}: {len(ql)} questions")
    for q in ql:
        new, old = int(d104[q]), int(d75[q]); poss = fs[q]; sn, so = [led.s(new, c) for c in poss], [led.s(old, c) for c in poss]
        verdict = "the change is certainly better" if all(a > b_ for a, b_ in zip(sn, so)) else "certainly worse" if all(a < b_ for a, b_ in zip(sn, so)) else "certainly no worse" if all(a >= b_ for a, b_ in zip(sn, so)) else "certainly no better" if all(a <= b_ for a, b_ in zip(sn, so)) else "open"
        print(f"  {q}: {inv[old]} -> {inv[new]} | possible truths {[inv[c] for c in poss]} | most probable {inv[mapa[q]]}{'  DETERMINED' if len(poss) == 1 else ''} | {verdict}", flush=True)
print("\nledger updated")
