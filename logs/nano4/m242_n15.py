#!/usr/bin/env python
"""Section 4M M242 (2026-11-19), report only, no labels: are the 15-game test questions (a quarter of the test set, a size the
cross-validation never sees; their pool coefficients are the mean of the 10- and 20-game ones) in line with the other sizes?
Per question size: the mean largest probability and the mean entropy of the base posterior and of the pooled posterior, the
share of questions whose answer the pool changes against the base decision, and the anchor / look-alike evidence strength;
the 15-game value is compared with the straight line between the 10- and 20-game values (bootstrap interval over questions).
Validation values (5 / 10 / 20) are printed beside for the trend.

  python -I logs/nano4/m242_n15.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402

b = mb7.b


def ent(P):
    return -(P * np.log(np.maximum(P, 1e-12))).sum(1)


def main():
    rng = np.random.default_rng(242); z = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P = z["P"][names.index("current")]; Pn = z["P"][names.index("no look-alike reading")]; B = z["base"].astype(float); B = B / B.sum(1, keepdims=True)
    nq = z["nq"].astype(int); qid = z["qid"].astype(str); pub = np.char.startswith(qid, "pub"); db = np.asarray(b.decide(B), int); dp = np.asarray(b.decide(P), int)
    Q = {"base: largest probability": B.max(1), "base: entropy": ent(B), "pooled: largest probability": P.max(1), "pooled: entropy": ent(P), "pool changes the base decision": (dp != db).astype(float), "pooled without look-alike: largest probability": Pn.max(1),
         "sharpening by the evidence (base entropy - pooled entropy)": ent(B) - ent(P), "look-alike reading: absolute tilt": np.abs(z["R_cur"]).mean(1), "shift of the expected level by the pool": P @ np.arange(10.0) - B @ np.arange(10.0)}
    print("test questions by size: " + ", ".join(f"n={s}: {int((nq == s).sum())}" for s in (5, 10, 15, 20)))
    for k, x in Q.items():
        m = {s: float(x[nq == s].mean()) for s in (5, 10, 15, 20)}; bs = []
        for _ in range(2000):
            v = {s: (lambda a: a[rng.integers(0, len(a), len(a))].mean())(x[nq == s]) for s in (10, 15, 20)}; bs.append(v[15] - (v[10] + v[20]) / 2)
        lo, hi = np.percentile(bs, [2.5, 97.5]); d = m[15] - (m[10] + m[20]) / 2
        print(f"   {k:58s}: " + "  ".join(f"n={s}: {m[s]:+.4f}" for s in (5, 10, 15, 20)) + f"   | 15 minus the mean of 10 and 20: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]   (Public {float(x[(nq == 15) & pub].mean()):+.4f}, Private {float(x[(nq == 15) & ~pub].mean()):+.4f})")
    print("REPORT ONLY")


if __name__ == "__main__":
    sys.exit(main())
