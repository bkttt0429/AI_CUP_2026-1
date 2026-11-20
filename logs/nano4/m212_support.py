#!/usr/bin/env python
"""M212 (2026-11-17), Public only: how well supported are the strongest soft truth weights of M210?

For the questions where M210 gives a truth other than ct116's answer a weight >= 0.95: the number of recorded uploads that
changed the question (each such upload is one exact count that involves it), and the truth weights of the ledger WITHOUT the
composition constraints (uploads + the three readings of 2026-11-17 only), for comparison with M210's weights.

  python -I logs/nano4/m212_support.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m208_ledger_probe as m8  # noqa: E402

LEV = led.LEVELS
z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}
up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **m8.PROBES, "submission_rank_ct116.csv": 0.8394955}; qs, cand, obs = led.build(up, prior); m0, c0 = m8.solve(qs, cand, obs, prior); assert isinstance(m0, dict)
P = json.load(open(ROOT / "results/r_variants/m210_profiles.json")); a = led.decisions("submission_rank_ct116.csv")
tg = [q for q, w in P.items() if max(w.values()) >= 0.95 and LEV.index(max(w, key=w.get)) != int(a[q])] + ["pub_q_0157", "pub_q_0103", "pub_q_0266"]
print(f"ledger with the three readings of 11-17: {len(obs)} uploads, {len(qs)} questions, cost {c0:.2f}")
for q in dict.fromkeys(tg):
    n_obs = sum(q in o["moves"] for o in obs); moves = sorted({(LEV[d0], LEV[d1]) for o in obs if q in o["moves"] for d0, d1 in [o["moves"][q]]}); w = {}
    for c in cand[q]:
        s_, cost = m8.solve(qs, cand, obs, prior, fix=(q, c), time_limit=60)
        if isinstance(s_, dict):
            w[c] = cost
    m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; t = sum(w.values()); w = {c: v / t for c, v in w.items()}
    print(f"  {q}: ct116 answers {LEV[int(a[q])]}; changed by {n_obs} uploads, answers tried {sorted({m_[1] for m_ in moves})} (base answer {moves[0][0] if moves else '-'}); feasible truths {len(w)}; weights without the composition " + " ".join(f"{LEV[c]} {v:.2f}" for c, v in sorted(w.items(), key=lambda t_: -t_[1])[:3])
          + " | M210 (with the composition, before the ct116 reading) " + " ".join(f"{k} {v:.2f}" for k, v in sorted(P.get(q, {}).items(), key=lambda t_: -t_[1])[:2]), flush=True)
