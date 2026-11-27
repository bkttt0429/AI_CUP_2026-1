#!/usr/bin/env python
"""Section 4M M179(b) (2026-11-16, report only): gains and losses of the new rule by the true level (design doc M179).

Input: the per-question decisions dumped by m165_pool.py --dump (seeds 0-2): truth, size, player, the ct106 rule, the pool with
each source set. For the first source set: the class-balanced score difference per true level with a player bootstrap, and for
the true levels given (default C and 6D) the transition table old answer -> new answer.
  python -I logs/nano4/m180_by_class.py m179_dump.npz [SET_NAME]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402

SM = mb7.SM; D = dict(np.load(ROOT / "results/r_variants" / sys.argv[1], allow_pickle=True)); keys = [k for k in D if k not in ("y", "n", "pid", "fold", "ct106_rule", "ct104b_rule")]; name = sys.argv[2] if len(sys.argv) > 2 else keys[0]
y, n, pid = D["y"].astype(int), D["n"].astype(int), D["pid"].astype(str); old, new = D["ct106_rule"].astype(int), D[name].astype(int); ds = SM[new, y] - SM[old, y]; rng = np.random.default_rng(180)
up, inv = np.unique(pid, return_inverse=True); print(f"new rule = pool + {name}; questions {len(y)}, players {len(up)}; mean difference by size " + " ".join(f"{s}: {ds[n == s].mean():+.4f}" for s in (5, 10, 20)))
print("true level | questions | changed | mean difference (sizes weighted equally) [95% player bootstrap] | exact old -> new | moved down / up")
for c in range(10):
    m = y == c; est = float(np.mean([ds[m & (n == s)].mean() for s in (5, 10, 20)])); bs = []
    idx_by_p = {}
    for i in np.flatnonzero(m):
        idx_by_p.setdefault(inv[i], []).append(i)
    ps = np.array(list(idx_by_p)); 
    for _ in range(1000):
        take = np.concatenate([idx_by_p[p] for p in rng.choice(ps, len(ps))]); bs.append(np.mean([ds[take][n[take] == s].mean() for s in (5, 10, 20)]))
    ch = m & (new != old)
    print(f"  {LEVELS[c]:3s} | {int(m.sum()):5d} | {float(ch.sum() / m.sum()):.3f} | {est:+.4f} [{np.quantile(bs, 0.025):+.4f}, {np.quantile(bs, 0.975):+.4f}] | {float((old[m] == c).mean()):.3f} -> {float((new[m] == c).mean()):.3f} | {int((new[ch] < old[ch]).sum())} / {int((new[ch] > old[ch]).sum())}")
for c in (LEVELS.index("C"), LEVELS.index("6D"), LEVELS.index("A")):
    m = (y == c) & (new != old); tab = {}
    for a, b_ in zip(old[m], new[m]):
        tab[(a, b_)] = tab.get((a, b_), 0) + 1
    print(f"true {LEVELS[c]}: changed answers old -> new (count, score change each): " + ", ".join(f"{LEVELS[a]}>{LEVELS[b_]} {k} ({SM[b_, c] - SM[a, c]:+.2f})" for (a, b_), k in sorted(tab.items(), key=lambda t: -t[1])[:8]))
