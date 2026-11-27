#!/usr/bin/env python
"""M200 reading: the networks of m200_prod.py against the production networks of the same seed at the same steps.

Reads results/m200/train_<net>_s<seed>.json (written at every evaluation) and results/m161q/train_<net>_s<seed>.json;
row-weighted held-out top-1 of the prototypes and of the learned centres, the bands 30-100 and 100-300 games of the
prototypes, per network and the mean over the networks. Only reads small JSON files.

  python -I logs/nano4/m200_read.py [--seed 30] [--new results/m200] [--ref results/m161q]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=30); ap.add_argument("--new", default="results/m200"); ap.add_argument("--ref", default="results/m161q"); a_ = ap.parse_args()
nets = (1, 2, 3, 4, -1); H = {}
for v in nets:
    fn, fr = Path(a_.new) / f"train_{v}_s{a_.seed}.json", Path(a_.ref) / f"train_{v}_s{a_.seed}.json"
    if fn.exists() and fr.exists():
        H[v] = ({h["step"]: h for h in json.load(open(fn))["history"]}, {h["step"]: h for h in json.load(open(fr))["history"]})
steps = sorted(set().union(*[set(n) & set(r) for n, r in H.values()])) if H else []
g = lambda h, k: (h.get(k) or {}).get("top1w", float("nan"))  # noqa: E731
for s in steps:
    print(f"step {s}: new | production (same seed) | difference; prototypes, centres, prototypes of the bands 30-100 and 100-300 games")
    acc = []
    for v in nets:
        if v in H and s in H[v][0] and s in H[v][1]:
            n, r = H[v][0][s], H[v][1][s]; row = [(g(n, "seen"), g(r, "seen")), (g(n, "centres"), g(r, "centres")), (n["seen"]["by_band"][1], r["seen"]["by_band"][1]), (n["seen"]["by_band"][2], r["seen"]["by_band"][2])]; acc.append([a - b for a, b in row])
            print(f"  net {v:>2}: " + "   ".join(f"{a:.3f} | {b:.3f} | {a - b:+.3f}" for a, b in row))
    if acc:
        print(f"  mean of {len(acc)} network(s): " + "   ".join(f"{sum(x[i] for x in acc) / len(acc):+.4f}" for i in range(4)) + "   (prototypes, centres, band 30-100, band 100-300)")
if not steps:
    print("no common evaluation step yet")
