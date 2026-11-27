"""Section 4M M102 (2026-11-13, report only): what the cross-validation expects of the three groups the split file separates.
In CV under the test rules (as m90_testmimic.py), with the no-reversal composition, the questions that ct44's recipe leaves at the
base decision and that the later recipes change fall into: A = changed the same way by the ct73-style recipe (SP, bundle v3) and the
ct75n-style recipe (SP2e, bundle v5); B = changed only by the ct73-style recipe; C = changed only by the ct75n-style recipe.
Per group: share of the questions, how often the change is better / worse than ct44's answer, mean gain, and the mean change of the
exact count and of the within-one count per changed question - the same two numbers a Public reading decodes into, so the split
file's reading can be compared with what CV expects (8 / 4 / 8 Public questions).
  python -I logs/nano4/m102b_groups_cv.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m90_testmimic as t  # noqa: E402  (runs its own report first; reuses its fold data and recipes)

b, s2, mb, ES, FOLDS = t.b, t.s2, t.mb, t.ES, t.FOLDS
tot = {k: {"n": 0, "up": 0, "dn": 0, "gain": 0.0, "dx": 0, "dy": 0} for k in ("A shared", "B ct73-style only", "C ct75n-style only", "changed differently")}; nq = 0
for seeds in ((0, 1, 2), (6, 7, 8)):
    for v in FOLDS:
        y, n, _, Ls = t.Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = {}
        for name in ("ct44", "ct72", "ct75"):
            d0, _, dec[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
        keep = dec["ct44"] != d0; n72 = np.where(keep, dec["ct44"], dec["ct72"]); n75 = np.where(keep, dec["ct44"], dec["ct75"]); d44 = dec["ct44"]; nq += len(y)
        for k, sel, new in (("A shared", (n72 != d44) & (n75 == n72), n72), ("B ct73-style only", (n72 != d44) & (n75 == d44), n72), ("C ct75n-style only", (n75 != d44) & (n72 == d44), n75), ("changed differently", (n72 != d44) & (n75 != d44) & (n72 != n75), n75)):
            g = b.SM[new[sel], y[sel]] - b.SM[d44[sel], y[sel]]; ex = (new[sel] == y[sel]).astype(int) - (d44[sel] == y[sel]).astype(int); w1 = (np.abs(new[sel] - y[sel]) <= 1).astype(int) - (np.abs(d44[sel] - y[sel]) <= 1).astype(int)
            r = tot[k]; r["n"] += int(sel.sum()); r["up"] += int((g > 0).sum()); r["dn"] += int((g < 0).sum()); r["gain"] += float(g.sum()); r["dx"] += int(ex.sum()); r["dy"] += int(w1.sum())
print("\nCV under the test rules, no-reversal composition, seeds 0-2 and 6-8 pooled: new changes on top of ct44's recipe by group")
out = {}
for k, r in tot.items():
    m_ = max(r["n"], 1); out[k] = {"share": r["n"] / nq, "better": r["up"] / m_, "worse": r["dn"] / m_, "gain": r["gain"] / m_, "exact": r["dx"] / m_, "within_one": r["dy"] / m_}
    print(f"  {k:22s} share of questions {r['n'] / nq:.4f} ({r['n']}); better {r['up'] / m_:.3f}, worse {r['dn'] / m_:.3f}; mean gain per changed question {r['gain'] / m_:+.3f}; exact count {r['dx'] / m_:+.3f}, within-one count {r['dy'] / m_:+.3f} per changed question")
for k, npub in (("A shared", 8), ("B ct73-style only", 4), ("C ct75n-style only", 8)):
    o = out[k]; print(f"  expected for the {npub} Public questions of {k[0]}: exact {npub * o['exact']:+.2f}, within one {npub * o['within_one']:+.2f}, points {npub * o['gain']:+.2f} (sd of the points about {np.sqrt(npub) * 0.62:.1f})")
(ROOT / "results/r_variants/m102b_groups_cv.json").write_text(json.dumps(out, indent=1))
