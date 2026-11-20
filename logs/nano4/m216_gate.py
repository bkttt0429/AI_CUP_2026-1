#!/usr/bin/env python
"""M216: the registered component gate at 40,000 steps (design doc M216), read from the finished all-players network.

Pass = (prototypes >= production + 0.02 OR centres >= production + 0.02) AND the prototypes of the bands 30-100 and 100-300
games both above the production network's (same seed, step 40,000). Exit status 0 = pass, 3 = not passed.
"""
import json
import sys

n = json.load(open("results/m216/train_-1_s30.json")); r = json.load(open("results/m161q/train_-1_s30.json"))
if n.get("partial"):
    sys.exit("the all-players network has not finished")
hn = {h["step"]: h for h in n["history"]}[40000]; hr = {h["step"]: h for h in r["history"]}[40000]
p, c, b1, b2 = hn["seen"]["top1w"], hn["centres"]["top1w"], hn["seen"]["by_band"][1], hn["seen"]["by_band"][2]; P, C, B1, B2 = hr["seen"]["top1w"], hr["centres"]["top1w"], hr["seen"]["by_band"][1], hr["seen"]["by_band"][2]
main = (p >= P + 0.02) or (c >= C + 0.02); bands = (b1 > B1) and (b2 > B2)
print(f"M216 component gate at 40,000 steps (all-players network, seed 30): prototypes {p:.4f} vs {P:.4f} ({p - P:+.4f}), centres {c:.4f} vs {C:.4f} ({c - C:+.4f}) -> {'ok' if main else 'not reached'}; band 30-100 {b1:.4f} vs {B1:.4f}, band 100-300 {b2:.4f} vs {B2:.4f} -> {'ok' if bands else 'not both above'}")
print("GATE " + ("PASSED" if main and bands else "NOT PASSED")); sys.exit(0 if main and bands else 3)
