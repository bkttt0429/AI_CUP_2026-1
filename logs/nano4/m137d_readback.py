#!/usr/bin/env python
"""Section 4M M137d (2026-11-14): the Public reading of ct105 (0.8101304) against every recorded upload. The reading is added to
the ledger, all uploads are solved jointly (m58_decode.py), and ct105's 29 Public changes against ct104b are decoded by group:
changes that leave D, changes of two or more levels, changes that undo a change ct104b made to ct75u, the rest - with the labels
still possible per question.
  python -I logs/nano4/m137d_readback.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

NEW = {"submission_rank_ct105.csv": 0.8101304}
L = json.load(open(led.LEDGER)); L.update(NEW); json.dump(L, open(led.LEDGER, "w"), indent=1); up = {**led.KNOWN_SCORES, **L}; prior = led.prior_from_ct9()
qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior); print(f"{len(up)} uploads recorded; jointly {'FEASIBLE' if mapa not in (None, 'timeout') else 'INFEASIBLE'}", flush=True); assert mapa not in (None, "timeout")
o = {x["name"]: x for x in obs}; dx = lambda a, b_: (o[a]["X"] - o[b_]["X"], o[a]["Y"] - o[b_]["Y"])  # noqa: E731
for a, b_ in (("ct105", "ct104b"), ("ct105", "ct75u"), ("ct104b", "ct75u")):
    x, y = dx(f"submission_rank_{a}.csv", f"submission_rank_{b_}.csv"); print(f"{a} minus {b_}: exact {x:+d}, within one {y:+d} ({(1 - led.E1) * x + led.E1 * y:+.3f} points)")
d75, d104b, d105 = (led.decisions(f"submission_rank_{k}.csv") for k in ("ct75u", "ct104b", "ct105")); pub = [q for q in d75.index if q.startswith("pub")]; inv = {v: k for k, v in led.RANK.items()}
ch = [q for q in pub if d105[q] != d104b[q]]
grp = lambda q: ("leaves D" if d104b[q] == 0 else "two or more levels" if abs(d105[q] - d104b[q]) >= 2 else "undoes a ct104b change (back to ct75u)" if d105[q] == d75[q] else "other one-level change")  # noqa: E731
G = {}
for q in ch:
    G.setdefault(grp(q), []).append(q)
fs = led.feasible_sets(qs, cand, obs, prior, mapa, only=ch, time_limit=30)
tot = {}
for g, ql in G.items():
    print(f"\n{g}: {len(ql)} questions")
    for q in ql:
        a, b_ = int(d105[q]), int(d104b[q]); f = sorted(fs.get(q, [])) if q in fs else None; m = mapa.get(q)
        gains = sorted({round(led.s(a, c) - led.s(b_, c), 3) for c in f}) if f else None
        print(f"  {q}: ct75u {inv[int(d75[q])]}, ct104b {inv[b_]} -> ct105 {inv[a]}; " + (f"labels still possible {[inv[c] for c in f]}; gain of the change over them {gains}; most likely {inv[m] if m is not None else '-'}" if f else "not constrained by the ledger"))
        if f and len(gains) == 1:
            tot[g] = tot.get(g, 0.0) + gains[0]
print("\nchanges whose outcome is already fixed by the ledger, by group (points): " + ", ".join(f"{g}: {v:+.3f}" for g, v in tot.items()))
