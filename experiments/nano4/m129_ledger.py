#!/usr/bin/env python
"""M129: back-test of the candidate against the Public ledger (all recorded uploads; logs/nano4/m58_decode.py), report only.
For every Public question whose answer differs between the candidate and ct75u: if earlier uploads already moved it, the profile
of the ledger over its truth (the cheapest joint assignment with the truth fixed to each class; weights exp(-cost)); otherwise
the base probabilities (ct9). Expected Public difference in points and the questions the ledger pins down."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith("--")]; new = args[0] if args else "submission_rank_ct104.csv"; old = args[1] if len(args) > 1 else "submission_rank_ct75u.csv"
ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}; prior = led.prior_from_ct9()
which = next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--prior=")), "base")
if which != "base":  # the posterior behind the candidate's decisions (or the current recipe's) as the prior of every question, instead of the base probabilities
    post = next((x.split("=")[1] for x in sys.argv[1:] if x.startswith("--post=")), "results/r_variants/m129_test_posterior.npz")  # M137b: another posterior file (keys qid, <which>)
    zp = np.load(ROOT / post, allow_pickle=True); prior = {str(q_): np.maximum(p_, 1e-6) for q_, p_ in zip(zp["qid"], zp[which])}
print(f"prior of the ledger: {which}")
qs, cand, obs = led.build(up, prior)
mapa = led.solve(qs, cand, obs, prior); assert mapa not in (None, "timeout"), "the ledger is not jointly feasible"
dn, do = led.decisions(new), led.decisions(old); ch = [q for q in dn.index if q.startswith("pub") and dn[q] != do[q]]
cost_of = lambda sol: float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))  # noqa: E731
tot_led = tot_pri = 0.0; rows = []
for q in ch:
    a, b_ = int(dn[q]), int(do[q]); p = np.asarray(prior[q], float); p = p / p.sum(); e_pri = float(sum(p[c] * (led.s(a, c) - led.s(b_, c)) for c in range(10)))
    if q in cand:
        w = {}
        for c in cand[q]:
            sol = led.solve(qs, cand, obs, prior, fix=(q, c), time_limit=60)
            if sol not in (None, "timeout"):
                w[c] = cost_of(sol)
        m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; z = sum(w.values()); w = {c: v / z for c, v in w.items()}
        e_led = float(sum(v * (led.s(a, c) - led.s(b_, c)) for c, v in w.items())); src = "ledger"; det = max(w.values())
    else:
        e_led, src, det = e_pri, "base only", float(p.max())
    tot_led += e_led; tot_pri += e_pri
    rows.append((q, led.LEVELS[b_], led.LEVELS[a], src, e_led, e_pri, det))
    print(f"  {q}: {led.LEVELS[b_]} -> {led.LEVELS[a]}  [{src}]  expected {e_led:+.3f} (base alone {e_pri:+.3f}); most likely truth has weight {det:.2f}", flush=True)
n_led = sum(r[3] == "ledger" for r in rows)
print(f"{new} against {old}: {len(ch)} Public questions differ, {n_led} of them constrained by earlier uploads")
print(f"expected Public difference: {tot_led:+.2f} points = {tot_led / 400:+.5f} (with the ledger); {tot_pri:+.2f} points = {tot_pri / 400:+.5f} (base probabilities alone)")
print(f"  the constrained questions alone: {sum(r[4] for r in rows if r[3] == 'ledger'):+.2f} points (base alone {sum(r[5] for r in rows if r[3] == 'ledger'):+.2f})")
(ROOT / "results/r_variants" / f"m129_ledger_{Path(new).stem.replace('submission_rank_', '')}_{which}.json").write_text(json.dumps([dict(zip(("q", "old", "new", "src", "exp_ledger", "exp_base", "top_weight"), r)) for r in rows], indent=1))
