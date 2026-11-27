"""Section 4M M108 (2026-11-14): the two split uploads of 11/13 into the Public ledger, joint feasibility, and what the nine Public
questions of ct75u's extra changes can be.
  python -I logs/nano4/m108_readback.py
"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as m  # noqa: E402

NEW = {"submission_rank_ct75s_out.csv": 0.8092107, "submission_rank_ct75s_out_up.csv": 0.8076304}
led = json.load(open(m.LEDGER)); led.update(NEW); up = {**m.KNOWN_SCORES, **led}; prior = m.prior_from_ct9()
qs, cand, obs = m.build(up, prior); mapa = m.solve(qs, cand, obs, prior); print(f"{len(up)} uploads; jointly {'FEASIBLE' if mapa is not None else 'INFEASIBLE'}", flush=True); assert mapa is not None
json.dump(led, open(m.LEDGER, "w"), indent=1); o = {x["name"]: x for x in obs}; names = ["submission_rank_ct75s.csv", "submission_rank_ct75s_out.csv", "submission_rank_ct75s_out_up.csv", "submission_rank_ct75u.csv"]
for a, b in zip(names[:-1], names[1:]):
    print(f"{b[16:-4]} minus {a[16:-4]}: exact {o[b]['X'] - o[a]['X']:+d}, within one {o[b]['Y'] - o[a]['Y']:+d}")
dec = {k: m.decisions(k) for k in names}; c44 = m.decisions("submission_rank_ct44.csv"); inv = {v: k for k, v in m.RANK.items()}; pub = [q for q in c44.index if q.startswith("pub") and dec[names[0]][q] != dec[names[3]][q]]
fs = m.feasible_sets(qs, cand, obs, prior, mapa, only=pub, time_limit=20)
for q in pub:
    grp = "outside" if dec[names[1]][q] != dec[names[0]][q] else ("ct44 up, reverted" if dec[names[2]][q] != dec[names[1]][q] else "ct44 down, reverted")
    print(f"   {q} [{grp}]: ct44 {inv[int(c44[q])]}, ct75s {inv[int(dec[names[0]][q])]} -> ct75u {inv[int(dec[names[3]][q])]} | possible truths {[inv[c] for c in fs[q]]} | most probable {inv[mapa[q]]}{'  DETERMINED' if len(fs[q]) == 1 else ''}")
print("ledger updated")
