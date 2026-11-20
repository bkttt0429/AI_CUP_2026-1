#!/usr/bin/env python
"""M221 gate (design doc, registered 2026-11-18 01:54), seed 30, five networks - only reads small JSON files.
(a) M218's gate against the production control: open-set (prototypes or centres, best state) >= +0.02 or top-1 at step 40,000
    (prototypes or centres) >= +0.02, with the same reading above production in the bands 30-100 and 100-300 games;
(b) the combination adds something to g25 (external strangers alone): open-set (prototypes or centres) above g25's and top-1
    (prototypes, step 40,000) not below g25's.
Exit code 0 = (a) and (b) hold, 1 = not, 2 = a network of the arm has not finished.

  python -I logs/nano4/m221_gate.py [arm=wg25]
"""
import json
import sys
from pathlib import Path

R, NETS, SEED = Path("results"), (1, 2, 3, 4, -1), 30; arm = sys.argv[1] if len(sys.argv) > 1 else "wg25"; mean = lambda x: sum(x) / len(x)  # noqa: E731
ld = lambda p: json.load(open(p))  # noqa: E731
try:
    J = {v: ld(R / f"m221/{arm}/train_{v}_s{SEED}.json") for v in NETS}; G = {v: ld(R / f"m218/g25/train_{v}_s{SEED}.json") for v in NETS}
except FileNotFoundError as e:
    print(f"missing: {e}"); sys.exit(2)
if not all("final" in j for j in J.values()):
    print(f"{arm}: {sum('final' in j for j in J.values())} of 5 networks finished"); sys.exit(2)
ctl = {v: ld(R / f"m218/control/eval_{v}_s{SEED}.json")["final"] for v in NETS}; prod = {v: {h["step"]: h for h in ld(R / f"m161q/train_{v}_s{SEED}.json")["history"]} for v in NETS}
H = {v: {h["step"]: h for h in j["history"]} for v, j in J.items()}; HG = {v: {h["step"]: h for h in j["history"]} for v, j in G.items()}; F = {v: j["final"] for v, j in J.items()}; FG = {v: j["final"] for v, j in G.items()}; a_ok = False
for nm, key in (("open-set, prototypes", "open"), ("open-set, centres", "open_centres")):
    d = [mean([F[v][key]["clear_w"] - ctl[v][key]["clear_w"] for v in NETS])] + [mean([F[v][key]["clear_by_band"][b] - ctl[v][key]["clear_by_band"][b] for v in NETS]) for b in (1, 2)]
    ok = d[0] >= 0.02 and d[1] > 0 and d[2] > 0; a_ok |= ok; print(f"(a) {nm}: {d[0]:+.4f}; bands 30-100 {d[1]:+.4f}, 100-300 {d[2]:+.4f} -> {'PASS' if ok else 'no'}")
for nm, key in (("top-1, prototypes", "seen"), ("top-1, centres", "centres")):
    d = [mean([H[v][40000][key]["top1w"] - prod[v][40000][key]["top1w"] for v in NETS])] + [mean([H[v][40000][key]["by_band"][b] - prod[v][40000][key]["by_band"][b] for v in NETS]) for b in (1, 2)]
    ok = d[0] >= 0.02 and d[1] > 0 and d[2] > 0; a_ok |= ok; print(f"(a) {nm}: {d[0]:+.4f}; bands 30-100 {d[1]:+.4f}, 100-300 {d[2]:+.4f} -> {'PASS' if ok else 'no'}")
op = [mean([F[v][k]["clear_w"] for v in NETS]) - mean([FG[v][k]["clear_w"] for v in NETS]) for k in ("open", "open_centres")]; t1 = mean([H[v][40000]["seen"]["top1w"] - HG[v][40000]["seen"]["top1w"] for v in NETS])
un = mean([F[v]["unseen"]["top1"] for v in NETS if "unseen" in F[v]]) - mean([FG[v]["unseen"]["top1"] for v in NETS if "unseen" in FG[v]])
b_ok = (op[0] > 0 or op[1] > 0) and t1 >= 0
print(f"(b) against g25: open-set prototypes {op[0]:+.4f}, centres {op[1]:+.4f}; top-1 (prototypes, step 40,000) {t1:+.4f}; unseen-player top-1 {un:+.4f} -> {'PASS' if b_ok else 'no'}")
print(f"GATE {arm}: (a) {'passed' if a_ok else 'NOT passed'}, (b) {'passed' if b_ok else 'NOT passed'} -> {'step 2' if a_ok and b_ok else 'no step 2'}"); sys.exit(0 if a_ok and b_ok else 1)
