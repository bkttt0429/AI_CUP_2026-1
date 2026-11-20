#!/usr/bin/env python
"""M219 reading: the arms of m219_prod.py (window input) against the production networks of the same seed, network and step.
Only reads small JSON files (results/m219/<arm>/train_<net>_s<seed>.json against results/m161q/train_<net>_s<seed>.json).

Per arm and evaluation step: the mean paired difference of the row-weighted held-out top-1 (prototypes, learned centres), of the
bands 30-100 and 100-300 games, and of the unseen-player top-1 (fold networks). Gate (design doc M219, step 40,000, all ten
networks of the two seeds): prototypes >= +0.02 or centres >= +0.02, and that reading above production in both middle bands.
w9p replaces w9 for step 2 only if the same reading is >= +0.005 above w9's.

  python -I logs/nano4/m219_read.py [w9 w9p] [--seeds 30 31]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("arms", nargs="*", default=["w9", "w9p"]); ap.add_argument("--seeds", type=int, nargs="+", default=[30, 31]); a_ = ap.parse_args()
R, NETS = Path("results"), (1, 2, 3, 4, -1); mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731


def hist(path):
    return {h["step"]: h for h in json.load(open(path))["history"]} if path.exists() else None


def diff(H, P, ks, st, f):
    v = [(f(H[k][st]), f(P[k][st])) for k in ks]; v = [x - y for x, y in v if x is not None and y is not None]
    return mean(v)


prod = {(v, s): hist(R / f"m161q/train_{v}_s{s}.json") for v in NETS for s in a_.seeds}; final = {}
for arm in a_.arms:
    H = {(v, s): hist(R / f"m219/{arm}/train_{v}_s{s}.json") for v in NETS for s in a_.seeds}; H = {k: h for k, h in H.items() if h and prod[k]}
    if not H:
        print(f"== {arm}: no reading yet"); continue
    done = sum("final" in json.load(open(R / f"m219/{arm}/train_{v}_s{s}.json")) for v, s in H); steps = sorted(set.union(*[set(h) for h in H.values()]))
    print(f"== {arm}: {len(H)} network(s) with a reading, {done} finished")
    for st in steps:
        ks = [k for k in H if st in H[k] and st in prod[k]]; kc = [k for k in ks if "centres" in H[k][st] and "centres" in prod[k][st]]; ku = [k for k in ks if "unseen" in H[k][st] and "unseen" in prod[k][st]]
        if not ks:
            continue
        p = [diff(H, prod, ks, st, lambda h: h["seen"]["top1w"]), diff(H, prod, ks, st, lambda h: h["seen"]["by_band"][1]), diff(H, prod, ks, st, lambda h: h["seen"]["by_band"][2])]
        c = [diff(H, prod, kc, st, lambda h: h["centres"]["top1w"]), diff(H, prod, kc, st, lambda h: h["centres"]["by_band"][1]), diff(H, prod, kc, st, lambda h: h["centres"]["by_band"][2])] if kc else None
        print(f"  step {st:5d} ({len(ks):2d} nets): prototypes {p[0]:+.4f} (30-100 {p[1]:+.4f}, 100-300 {p[2]:+.4f})" + (f" | centres {c[0]:+.4f} (30-100 {c[1]:+.4f}, 100-300 {c[2]:+.4f})" if c else "")
              + (f" | unseen players ({len(ku)} fold nets) {diff(H, prod, ku, st, lambda h: h['unseen']['top1']):+.4f}" if ku else "") + f" | levels: prototypes {mean([H[k][st]['seen']['top1w'] for k in ks]):.4f}" + (f", centres {mean([H[k][st]['centres']['top1w'] for k in kc]):.4f}" if kc else ""))
        if st == 40000:
            for s in a_.seeds:
                kk = [k for k in ks if k[1] == s]
                if kk:
                    print(f"      seed {s} ({len(kk)} nets): prototypes " + " ".join(f"{H[k][st]['seen']['top1w'] - prod[k][st]['seen']['top1w']:+.3f}" for k in kk) + ("; centres " + " ".join(f"{H[k][st]['centres']['top1w'] - prod[k][st]['centres']['top1w']:+.3f}" for k in kk if k in kc) if kc else ""))
            if len(ks) == len(NETS) * len(a_.seeds) and len(kc) == len(ks):
                final[arm] = (p, c); ok_p = p[0] >= 0.02 and p[1] > 0 and p[2] > 0; ok_c = c[0] >= 0.02 and c[1] > 0 and c[2] > 0
                print(f"  GATE {arm} (step 40,000, {len(ks)} nets): prototypes {'PASS' if ok_p else 'no'}, centres {'PASS' if ok_c else 'no'} -> {'PASSED' if ok_p or ok_c else 'NOT passed'}")
if "w9" in final and "w9p" in final:
    print(f"w9p against w9 at step 40,000: prototypes {final['w9p'][0][0] - final['w9'][0][0]:+.4f}, centres {final['w9p'][1][0] - final['w9'][1][0]:+.4f} (w9p replaces w9 for step 2 only at >= +0.005 on the reading that passes)")
