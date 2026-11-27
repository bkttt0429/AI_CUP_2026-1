#!/usr/bin/env python
"""M120 diagnostic (report only): how much of the opponent side does each identifier really recognise?
Per identifier (the m84_<tag>_val.npz files): the estimated share of true identifications among the opponent-side queries
(sum over the margin bands of max(share_b - nominal_b, 0), bands = quantiles of the impostor margins, m71_step2.fit), by group,
the share of opponent-side queries in the bands >= q90 / >= q98, and how concentrated the readings of the top bands are
(label of the identified candidate minus the target's label: same level, within one). Next to it the fused held-out top-1
of the target-side own games (the number the networks were selected on)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m86_bundle as b  # noqa: E402
s2 = b.s2
RV = ROOT / "results/r_variants"
ids = sys.argv[1:] or ["5e", "S2", "SP2", "S2e", "SP2e", "S2y", "SP2y", "S3c", "S4", "SP4"]
V, T, key_pid, used = b.sources(ids)
nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
print(f"{'id':7s} held-out  mass(all)  even   HD1-B  HD1-W   >=q90   >=q98   top bands: same / within one   test: >=q90  >=q98")
out = {}
for x in used:
    z = V[x]; ones = np.ones(len(z["y"]), bool); edges, cal, band, g = s2.fit(z, ones); ok = g >= 0
    mass = {k: float(np.maximum(cal[k]["share"] - nominal, 0).sum()) for k in (0, 1, 2)}; w = np.array([(g == k).sum() for k in (0, 1, 2)], float); w /= w.sum()
    top = ok & (band >= 5); d = (z["opp_lab"] - z["y"])[top & (g == 0)]
    t = T[x]; tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"]); tok = tg >= 0
    ev = RV / f"m84_{x}_eval.json"; h = None
    if ev.exists():
        e = json.loads(ev.read_text()); h = (e.get("heldout", {}).get("fused") or e.get("heldout_fused") or {}).get("top1")
    row = {"heldout": h, "mass": float(sum(w[k] * mass[k] for k in (0, 1, 2))), "mass_by_group": mass, "q90": float((band[ok] >= 3).mean()), "q98": float((band[ok] >= 5).mean()), "top_same": float((d == 0).mean()), "top_within1": float((np.abs(d) <= 1).mean()),
           "test_q90": float((tb[tok] >= 3).mean()), "test_q98": float((tb[tok] >= 5).mean())}
    out[x] = row
    print(f"{x:7s} {h if h is None else round(h, 3)!s:8s}  {row['mass']:.4f}     {mass[0]:.4f} {mass[1]:.4f} {mass[2]:.4f}  {row['q90']:.4f}  {row['q98']:.4f}   {row['top_same']:.3f} / {row['top_within1']:.3f}                 {row['test_q90']:.4f}  {row['test_q98']:.4f}")
(RV / "m120_diag_mass.json").write_text(json.dumps(out, indent=1))
