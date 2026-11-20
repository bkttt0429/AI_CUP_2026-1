#!/usr/bin/env python
"""M224 (2026-11-18), diagnostic only: why did the recognised share go up (SQ4 0.319 -> SZ3 0.362) while the pool did not move?
Per validation game (results/r_variants/m84_<id>~m15_val.npz: the best candidate's level label f_opp_lab, the statistic
f_opp_m, the question player's true level y, handicap group) for two identifiers:
 A. anchor quality by depth: the games sorted by each identifier's own statistic; in the top 10 / 20 / ... %: share whose
    candidate label equals the player's level, share within one level;
 B. overlap at equal depth: games in both top sets / only in one; the hit rates of the two "only" sets;
 C. agreement of the best candidate over all games, and who is right when they differ;
 D. information: plug-in mutual information between the true level and the candidate label inside 20 bands of the statistic
    (nats per game), even games and all games; per question size (5 / 10 / 15 / 20 games): share of questions without any
    game in the top 30 %.

  python -I logs/nano4/m224_anchor_diag.py [SQ4 SZ3]
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RV = Path("results/r_variants"); ids = sys.argv[1:3] if len(sys.argv) >= 3 else ["SQ4", "SZ3"]
Z = {i: dict(np.load(RV / f"m84_{i}~m15_val.npz", allow_pickle=True)) for i in ids}; a, b = (Z[i] for i in ids); assert (a["key"] == b["key"]).all() and (a["y"] == b["y"]).all()
y, hd, key, pid = a["y"], a["hd"], a["key"], a["pid"]; n = len(y); print(f"{n} validation games of {len(set(pid))} questions; handicap groups {dict(Counter(hd.tolist()))}; target black {a['target_black'].mean():.3f}")
for i in ids:
    z = Z[i]; print(f"  {i}: games with a candidate label {int((z['f_opp_lab'] >= 0).sum())}; statistic min / median / max {z['f_opp_m'].min():.3f} / {np.median(z['f_opp_m']):.3f} / {z['f_opp_m'].max():.3f}; same candidate as {ids[0]}: {(z['f_opp_best'] == a['f_opp_best']).mean():.3f}")
ev = hd == 0; DEPTHS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)


def top(z, d, mask):
    m = np.where(mask & (z["f_opp_lab"] >= 0), z["f_opp_m"], -np.inf); k = int(round(d * mask.sum())); idx = np.argsort(-m, kind="stable")[:k]; s = np.zeros(n, bool); s[idx] = True; return s & np.isfinite(m)


for nm, mask in (("even games (handicap group 0)", ev), ("all games", np.ones(n, bool))):
    print(f"\n== A. anchor quality by depth, {nm}: {int(mask.sum())} games; label equals the player's level / within one level")
    for d in DEPTHS:
        row = []
        for i in ids:
            s = top(Z[i], d, mask); df = Z[i]["f_opp_lab"][s] - y[s]; row.append(f"{i} {np.mean(df == 0):.4f} / {np.mean(np.abs(df) <= 1):.4f} (mean diff {df.mean():+.3f})")
        print(f"  top {d:4.0%}: " + " | ".join(row))
    print(f"== B. overlap at equal depth, {nm}")
    for d in (0.1, 0.2, 0.3, 0.4):
        sa, sb = top(a, d, mask), top(b, d, mask); both, oa, ob = sa & sb, sa & ~sb, sb & ~sa; h = lambda z, s: (np.mean(z["f_opp_lab"][s] == y[s]) if s.any() else float("nan"), np.mean(np.abs(z["f_opp_lab"][s] - y[s]) <= 1) if s.any() else float("nan"))  # noqa: E731
        print(f"  top {d:4.0%}: in both {int(both.sum())} (same candidate {np.mean(a['f_opp_best'][both] == b['f_opp_best'][both]):.3f}; hit {h(a, both)[0]:.4f} / {h(a, both)[1]:.4f}) | only {ids[0]} {int(oa.sum())}: {h(a, oa)[0]:.4f} / {h(a, oa)[1]:.4f} | only {ids[1]} {int(ob.sum())}: {h(b, ob)[0]:.4f} / {h(b, ob)[1]:.4f}"
              f" | the other identifier's own label on those games: {ids[1]} on only-{ids[0]} {h(b, oa)[0]:.4f}, {ids[0]} on only-{ids[1]} {h(a, ob)[0]:.4f}")
same = a["f_opp_best"] == b["f_opp_best"]; dif = ~same & (a["f_opp_lab"] >= 0) & (b["f_opp_lab"] >= 0)
print(f"\n== C. best candidate: the same in {same.mean():.4f} of all games; where they differ ({int(dif.sum())} games): {ids[0]} label right {np.mean(a['f_opp_lab'][dif] == y[dif]):.4f}, {ids[1]} label right {np.mean(b['f_opp_lab'][dif] == y[dif]):.4f}; where the same: right {np.mean(a['f_opp_lab'][same] == y[same]):.4f}")


def info(z, mask, bands=20):
    ok = mask & (z["f_opp_lab"] >= 0); m, lab, yy = z["f_opp_m"][ok], z["f_opp_lab"][ok], y[ok]; q = np.quantile(m, np.linspace(0, 1, bands + 1)[1:-1]); bd = np.searchsorted(q, m); tot = 0.0; per = []
    for k in range(bands):
        s = bd == k
        if not s.any():
            per.append(0.0); continue
        J = np.zeros((10, 10)); np.add.at(J, (yy[s], lab[s]), 1); J /= J.sum(); py, pl = J.sum(1, keepdims=True), J.sum(0, keepdims=True); nz = J > 0; mi = float((J[nz] * np.log(J[nz] / (py @ pl)[nz])).sum()); per.append(mi); tot += s.mean() * mi
    return tot, per


print("\n== D. information between the true level and the candidate label inside 20 bands of the statistic (plug-in, nats per game)")
for nm, mask in (("even games", ev), ("all games", np.ones(n, bool))):
    r = {i: info(Z[i], mask) for i in ids}; print(f"  {nm}: " + " | ".join(f"{i} {r[i][0]:.4f}" for i in ids) + f"  ({ids[1]} / {ids[0]} = {r[ids[1]][0] / r[ids[0]][0]:.3f}); by band, weakest to strongest statistic:")
    for i in ids:
        print(f"      {i}: " + " ".join(f"{v:.2f}" for v in r[i][1]))
qs, inv = np.unique(pid, return_inverse=True); size = np.bincount(inv); print("\n== share of questions without any game in an identifier's top 30 % (all games), by question size")
for i in ids:
    s = top(Z[i], 0.3, np.ones(n, bool)); cnt = np.bincount(inv, weights=s.astype(float)); print(f"  {i}: " + " | ".join(f"{k} games: none {np.mean(cnt[size == k] == 0):.3f}, mean {cnt[size == k].mean():.2f}" for k in sorted(set(size.tolist())) if (size == k).sum() >= 200))
