#!/usr/bin/env python
"""Section 4M M170/M171 follow-up diagnostic (2026-11-15, report only): the candidate's offset used as a pure location shift.

Validation even games of an identifier: mean log-likelihood of the target's true level given the first candidate under
  (i)  the pooled reading pbar[label of the candidate, .]            (ii) the candidate's own count-based reading (M171)
  (iii) the pooled reading of a fractional label lab_c + s * u_c: (1 - f) pbar[lab + a] + f pbar[lab + a + 1], u_c = the shrunk
        offset of M170 (fold-excluded), for scales s = 0.5 / 0.75 / 1.0.
  python -I logs/nano4/m172_shift_diag.py SX4L~m15
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]; FOLDS = (1, 2, 3, 4); tag = sys.argv[1]
O = np.load(RV / "m170_offsets.npz", allow_pickle=True); Cn = np.load(RV / "m171_cand.npz", allow_pickle=True); players = O["players"].astype(str); q = np.load(ROOT / "results/m84_cache/queries.npz"); plab = q["plab"].astype(int)
val = dict(np.load(RV / f"m84_{tag}_val.npz", allow_pickle=True)); pidx = {p: i for i, p in enumerate(players)}; ci = np.array([pidx[p] for p in val["f_opp_best"].astype(str)]); fo = val["fold"].astype(int); tb = val["target_black"]; y = val["y"].astype(int)
edges = {True: np.quantile(val["f_imp_m"][tb], QS), False: np.quantile(val["f_imp_m"][~tb], QS)}; band = np.where(tb, np.searchsorted(edges[False], val["f_opp_m"]), np.searchsorted(edges[True], val["f_opp_m"])); e0 = val["hd"] == 0
res = {k: np.zeros(len(y)) for k in ("pooled", "counts", "shift 0.5", "shift 0.75", "shift 1.0")}
for j, v in enumerate(FOLDS):
    m = np.flatnonzero(fo == v); pb = Cn["pbar_fold"][j]; M = Cn["M_fold"][j]; c = ci[m]; lab = plab[c]; u = O["u_fold"][j][c]; r = np.arange(len(m)); base = pb[lab]; res["pooled"][m] = np.log(base[r, y[m]])
    cand = base * M[c]; cand /= cand.sum(1, keepdims=True); res["counts"][m] = np.log(cand[r, y[m]])
    for s in (0.5, 0.75, 1.0):
        x = np.clip(lab + s * u, 0, 9); a = np.floor(x).astype(int); f = x - a; b_ = np.minimum(a + 1, 9); p = (1 - f)[:, None] * pb[a] + f[:, None] * pb[b_]; res[f"shift {s}"][m] = np.log(p[r, y[m]])
print(f"{tag}, validation even games: mean log-likelihood of the target's true level given the first candidate")
for nm, lo, hi in (("bands < q50 (control)", 0, 0), ("q80-q95", 2, 3), ("q95-q99", 4, 5), (">= q99", 6, 8), (">= q95", 4, 8)):
    m = e0 & (band >= lo) & (band <= hi); print(f"  {nm:22s} n {int(m.sum())}: " + "  ".join(f"{k} {v[m].mean():.4f}" + ("" if k == "pooled" else f" ({v[m].mean() - res['pooled'][m].mean():+.4f})") for k, v in res.items()))
