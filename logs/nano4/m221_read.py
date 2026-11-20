#!/usr/bin/env python
"""M221 reading (copy of m218_read.py; the arm folders are results/m221/<arm>, the control is M218's): the arms of m221_prod.py (external accounts as strangers, with / without the outlier term) against the production
networks of the same seed. Only reads small JSON files.

Per arm and common evaluation step: row-weighted held-out top-1 (prototypes, learned centres) against results/m161q at the same
step; the arm's open-set reading (owner first and clear of the strangers; prototypes / centres) and its unseen-player top-1.
Gate (design doc M218; the production control exists only at its stored best state, so the open-set comparison uses the best
state on both sides and top-1 uses step 40,000 on both sides): mean over the five networks of
  open-set (prototypes or centres) >= +0.02  or  top-1 (prototypes or centres) >= +0.02,
  and the same reading above production in the bands 30-100 and 100-300 games.

  python -I logs/nano4/m218_read.py [g25 g25o g50 g50o] [--seed 30]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("arms", nargs="*", default=["wg25"]); ap.add_argument("--seed", type=int, default=30); a_ = ap.parse_args()
R, NETS = Path("results"), (1, 2, 3, 4, -1); mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731
ctl = {v: json.load(open(R / f"m218/control/eval_{v}_s{a_.seed}.json"))["final"] for v in NETS if (R / f"m218/control/eval_{v}_s{a_.seed}.json").exists()}
prod = {v: {h["step"]: h for h in json.load(open(R / f"m161q/train_{v}_s{a_.seed}.json"))["history"]} for v in NETS}
print("production control at its best state (5 nets): open-set prototypes " + f"{mean([c['open']['clear_w'] for c in ctl.values()]):.4f}, centres {mean([c['open_centres']['clear_w'] for c in ctl.values()]):.4f}; unseen-player top-1 (4 fold nets) {mean([c['unseen']['top1'] for v, c in ctl.items() if 'unseen' in c]):.4f}")
for arm in a_.arms:
    J = {v: json.load(open(R / f"m221/{arm}/train_{v}_s{a_.seed}.json")) for v in NETS if (R / f"m221/{arm}/train_{v}_s{a_.seed}.json").exists()}
    if not J:
        print(f"== {arm}: no reading yet"); continue
    H = {v: {h["step"]: h for h in j["history"]} for v, j in J.items()}; steps = sorted(set.intersection(*[set(h) for h in H.values()])); print(f"== {arm}: {len(J)} network(s), evaluation steps {steps}, finished {sum('final' in j for j in J.values())}")
    for s in steps:
        d = lambda f: mean([f(H[v][s]) - f(prod[v][s]) for v in H if s in prod[v]])  # noqa: E731
        u = [H[v][s]["unseen"]["top1"] for v in H if "unseen" in H[v][s]]
        print(f"  step {s:5d}: top-1 against production at the same step: prototypes {d(lambda h: h['seen']['top1w']):+.4f}, centres {d(lambda h: h['centres']['top1w']) if all('centres' in H[v][s] for v in H) else float('nan'):+.4f}, bands 30-100 / 100-300 (prototypes) {d(lambda h: h['seen']['by_band'][1]):+.4f} / {d(lambda h: h['seen']['by_band'][2]):+.4f}"
              + f" | open-set (this arm) prototypes {mean([H[v][s]['open']['clear_w'] for v in H]):.4f}" + (f", centres {mean([H[v][s]['open_centres']['clear_w'] for v in H]):.4f}" if all('open_centres' in H[v][s] for v in H) else "") + (f" | unseen players {mean(u):.4f}" if u else ""))
    if len(J) == len(NETS) and all("final" in j for j in J.values()) and len(ctl) == len(NETS) and 40000 in steps:
        F = {v: j["final"] for v, j in J.items()}; res = {}
        for nm, key, kb in (("open-set, prototypes", "open", "clear_by_band"), ("open-set, centres", "open_centres", "clear_by_band")):
            res[nm] = (mean([F[v][key]["clear_w"] - ctl[v][key]["clear_w"] for v in NETS]), mean([F[v][key][kb][1] - ctl[v][key][kb][1] for v in NETS]), mean([F[v][key][kb][2] - ctl[v][key][kb][2] for v in NETS]))
        for nm, key in (("top-1, prototypes", "seen"), ("top-1, centres", "centres")):
            res[nm] = (mean([H[v][40000][key]["top1w"] - prod[v][40000][key]["top1w"] for v in NETS]), mean([H[v][40000][key]["by_band"][1] - prod[v][40000][key]["by_band"][1] for v in NETS]), mean([H[v][40000][key]["by_band"][2] - prod[v][40000][key]["by_band"][2] for v in NETS]))
        ok = False
        for nm, (dall, b1, b2) in res.items():
            hit = dall >= 0.02 and b1 > 0 and b2 > 0; ok |= hit; print(f"  gate reading {nm:22s}: {dall:+.4f}; band 30-100 {b1:+.4f}, band 100-300 {b2:+.4f} -> {'PASS' if hit else 'no'}")
        print(f"  unseen-player top-1 at the best state (4 fold nets): {mean([F[v]['unseen']['top1'] for v in NETS if 'unseen' in F[v]]):.4f} against {mean([c['unseen']['top1'] for v, c in ctl.items() if 'unseen' in c]):.4f}; wrong player above the stranger threshold (prototypes) {mean([F[v]['open']['wrong_above'] for v in NETS]):.4f} against {mean([c['open']['wrong_above'] for c in ctl.values()]):.4f}")
        print(f"  GATE {arm}: {'PASSED' if ok else 'NOT PASSED'}")
