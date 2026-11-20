import csv, re, collections
import numpy as np
csv.field_size_limit(10**9)
MOVE = re.compile(r";[BW]\[")
for lv in ("D", "A", "1D", "3D", "6D"):
    n = []
    with open(f"data/training/train_{lv}.csv", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= 20000: break
            n.append(len(MOVE.findall(r["sgf_content"])))
    n = np.array(n); top = collections.Counter(n).most_common(3)
    print(f"{lv:3s} p95 {np.percentile(n,95):.0f} p99 {np.percentile(n,99):.0f} max {n.max()}  share >= 280: {np.mean(n>=280):.3f}  >= 300: {np.mean(n>=300):.3f}  < 100: {np.mean(n<100):.3f}  most common lengths {top}")
