"""How games end, per level: share ending with two passes, with one pass, without a pass; move counts. Report only."""
import csv, re, sys, collections
import numpy as np
csv.field_size_limit(10**9)
MOVE = re.compile(r";([BW])\[([a-s]{0,2}|tt)\]")
rows = []
for lv in ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"):
    with open(f"data/training/train_{lv}.csv", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= 20000:
                break
            mv = MOVE.findall(r["sgf_content"])
            passes = [m[1] in ("", "tt") for m in mv]
            last2 = passes[-2:] if len(passes) >= 2 else passes
            end = "two passes" if len(last2) == 2 and all(last2) else ("one pass" if passes and passes[-1] else "no pass")
            rows.append((lv, end, len(mv), sum(passes), r["color"], mv[-1][0] if mv else "", r.get("game_id", "")))
print("sample: first 20000 games of each level CSV")
by = collections.defaultdict(collections.Counter); n = collections.defaultdict(list)
for lv, end, m, p, c, last, _ in rows:
    by[lv][end] += 1; n[lv].append(m)
for lv in by:
    t = sum(by[lv].values())
    print(f"{lv:3s} two passes {by[lv]['two passes']/t:.3f}  one pass {by[lv]['one pass']/t:.3f}  no pass {by[lv]['no pass']/t:.3f}  moves median {np.median(n[lv]):.0f} p10 {np.percentile(n[lv],10):.0f} p90 {np.percentile(n[lv],90):.0f}")
lv, end, m, p, c, last, _ = zip(*rows)
nopass = [i for i in range(len(rows)) if end[i] == "no pass"]
tgt_last = np.mean([("B" if c[i] == "B" else "W") == last[i] for i in nopass])
print(f"games ending without a pass: the target made the last move in {tgt_last:.3f}")
print(f"mid-game passes per game (all): {np.mean([p[i] for i in range(len(rows))]):.2f}")
