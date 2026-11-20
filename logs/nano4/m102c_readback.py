"""Section 4M M102 (2026-11-13): back-test of a new Public reading against every recorded one. Adds the uploads after ct40 to the
ledger (ct42, ct44, ct64, ct71, ct72, ct73, ct75n, ct75s), solves all of them jointly (m58_decode.py) and prints, for the questions
of the groups A / B / C, the labels that are still possible.
  python -I logs/nano4/m102c_readback.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as m  # noqa: E402

NEW = {"submission_rank_ct42.csv": 0.7880062, "submission_rank_ct44.csv": 0.8067107, "submission_rank_ct64.csv": 0.8038092, "submission_rank_ct71.csv": 0.8053895,
       "submission_rank_ct72.csv": 0.8038092, "submission_rank_ct73.csv": 0.8060501, "submission_rank_ct75n.csv": 0.8044698, "submission_rank_ct75s.csv": 0.8092107}
led = json.load(open(m.LEDGER)); led.update(NEW); json.dump(led, open(m.LEDGER, "w"), indent=1); up = {**m.KNOWN_SCORES, **led}; prior = m.prior_from_ct9()
qs, cand, obs = m.build(up, prior); mapa = m.solve(qs, cand, obs, prior); print(f"{len(up)} uploads recorded; jointly {'FEASIBLE' if mapa is not None else 'INFEASIBLE'}", flush=True); assert mapa is not None
o = {x["name"]: x for x in obs}; d = lambda a, b: (o[a]["X"] - o[b]["X"], o[a]["Y"] - o[b]["Y"])  # noqa: E731
for a, b in (("submission_rank_ct75s.csv", "submission_rank_ct44.csv"), ("submission_rank_ct73.csv", "submission_rank_ct75s.csv"), ("submission_rank_ct75n.csv", "submission_rank_ct75s.csv")):
    print(f"{a[16:-4]} minus {b[16:-4]}: exact {d(a, b)[0]:+d}, within one {d(a, b)[1]:+d}")
c44, c73, c75n = (m.decisions(f"submission_rank_{k}.csv") for k in ("ct44", "ct73", "ct75n")); pub = [q for q in c44.index if q.startswith("pub")]; inv = {v: k for k, v in m.RANK.items()}
G = {"A": [q for q in pub if c73[q] != c44[q] and c75n[q] == c73[q]], "B": [q for q in pub if c73[q] != c44[q] and c75n[q] == c44[q]], "C": [q for q in pub if c75n[q] != c44[q] and c73[q] == c44[q]]}
fs = m.feasible_sets(qs, cand, obs, prior, mapa, only=[q for g in G.values() for q in g], time_limit=30)
for g, ql in G.items():
    for q in ql:
        new = c73[q] if g != "C" else c75n[q]; poss = fs[q]; sc = lambda dd: [m.s(dd, c) for c in poss]  # noqa: E731
        better = all(a_ > b_ for a_, b_ in zip(sc(int(new)), sc(int(c44[q])))); worse = all(a_ < b_ for a_, b_ in zip(sc(int(new)), sc(int(c44[q]))))
        print(f"  {q} {g}: {inv[int(c44[q])]} -> {inv[int(new)]} | possible {[inv[c] for c in poss]} | most probable {inv[mapa[q]]}{'  DETERMINED' if len(poss) == 1 else ''}{'  (the change is certainly better)' if better else '  (the change is certainly worse)' if worse else ''}")
print("ledger updated")
