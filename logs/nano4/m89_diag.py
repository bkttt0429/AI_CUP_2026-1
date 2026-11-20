"""Section 4M M89 diagnostics (2026-11-11, report only): two candidate NEW signals, both built from how the model reads OTHER players.
D1 relative anchors. Today an identified opponent X contributes only his label l_X: e(c) uses K(l_X - c). The model also has its own
   out-of-fold estimate m_X of X (X's 20-game validation question) and m_T of the target. The pool offset is shared by both, so
   m_T - m_X should predict y_T - l_X. Among the opponent queries of 5e by margin band: correlation and slope of (y_T - l_X) on
   (m_T - m_X), and the spread of y_T - l_X before / after.
D2 neighbour residuals. For the target's OWN side, the stylistically nearest training players (top-20 of the fused identifier, own
   player excluded) and their residuals r_X = l_X - m_X: does their weighted mean predict the target's residual y_T - m_T?
  python -I logs/nano4/m89_diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

RV, CACHE, M82 = ROOT / "results/r_variants", ROOT / "results/m84_cache", ROOT / "results/m82"
m, lab = {}, {}
for v in FOLDS:
    y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); E = P @ np.arange(10)
    for i in np.flatnonzero(np.asarray(n) == 20):
        m[pid[i]] = float(E[i]); lab[pid[i]] = int(y[i])
print(f"out-of-fold model estimate available for {len(m)} training players (folds 1-4); residual label - estimate: mean {np.mean([lab[p] - m[p] for p in m]):+.3f}, sd {np.std([lab[p] - m[p] for p in m]):.3f}")
d = dict(np.load(RV / "m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}
edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); mT = np.array([m.get(p, np.nan) for p in z["pid"]]); mX = np.array([m.get(p, np.nan) for p in d["f5e_opp_best"]]); dy = (z["y"] - z["opp_lab"]).astype(float); dm = mT - mX
print("D1 relative anchors, HD0 opponent queries (y_T - l_X against m_T - m_X):")
print("  band            n      with m_X   sd(y-l)   corr    slope   sd after   exact by l_X   exact by round(l_X + slope (m_T - m_X))")
for name, sel in (("< q50", band == 0), ("q50-q90", (band >= 1) & (band <= 2)), ("q90-q98", (band >= 3) & (band <= 4)), ("q98-q99.5", (band >= 5) & (band <= 6)), (">= q99.5", band >= 7)):
    s = sel & (g == 0); ok = s & np.isfinite(dm); x, yv = dm[ok], dy[ok]; b = float((x * yv).sum() / (x * x).sum()); pred = np.clip(np.rint(z["opp_lab"][ok] + b * x), 0, 9)
    print(f"  {name:12s} {int(s.sum()):6d}   {ok.sum() / max(s.sum(), 1):.3f}     {yv.std():.3f}    {np.corrcoef(x, yv)[0, 1]:+.3f}  {b:+.3f}   {(yv - b * x).std():.3f}      {float((yv == 0).mean()):.3f}          {float((pred == z['y'][ok]).mean()):.3f}")
s = (g == 0) & (band >= 5) & np.isfinite(dm); x, yv = dm[s], dy[s]; print("  top bands (>= q98), y_T - l_X by tercile of m_T - m_X: " + "; ".join(f"[{lo:+.2f}, {hi:+.2f}]: mean {yv[(x >= lo) & (x <= hi)].mean():+.3f}, share of 0 {float((yv[(x >= lo) & (x <= hi)] == 0).mean()):.3f}, n {int(((x >= lo) & (x <= hi)).sum())}" for lo, hi in zip(np.quantile(x, [0, 1 / 3, 2 / 3]), np.quantile(x, [1 / 3, 2 / 3, 1]))))
# D2: neighbours of the target's own side
import torch  # noqa: E402
dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H = int(q["Q"]), int(q["H"]); C = len(players)
V = np.load(CACHE / "var_ref.npz"); protos = torch.as_tensor(V["protos"][0], device=dev); qvec = V["qvec"][0]; beta = json.loads((M82 / "report.json").read_text())["weights"]["5e"]["matcher"]
idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); pos = {int(i): int(o) for i, o in zip(idx[order], order)}
res_c = torch.as_tensor(np.array([lab[p] - m[p] if p in m else np.nan for p in players]), device=dev); lab_c = torch.as_tensor(plab.astype(np.float64), device=dev); has = torch.isfinite(res_c)
ids = np.arange(Q, 2 * Q); K = 20; nres = np.zeros(Q); nlab = np.zeros(Q); nw = np.zeros(Q)
for i in range(0, Q, 2048):
    b_ = ids[i:i + 2048]; ex = torch.as_tensor(excl[b_], device=dev); valid = torch.ones(len(b_), C, device=dev, dtype=torch.bool); valid[torch.arange(len(b_), device=dev), ex] = False; nj = torch.as_tensor(net[b_], device=dev).long(); S = torch.zeros(len(b_), C, device=dev); qv = torch.as_tensor(qvec[b_].astype(np.float32), device=dev)
    for j in range(5):
        s_ = nj == j
        if s_.any():
            S[s_] = qv[s_] @ protos[j].T
    cnt = valid.sum(1, keepdim=True); mu = (S * valid).sum(1, keepdim=True) / cnt; zc = (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)
    F = (torch.as_tensor(blk[[pos[int(x)] for x in b_]].astype(np.float32), device=dev) + beta * zc).masked_fill(~valid, -float("inf")); top = F.topk(K, dim=1); w = torch.softmax(top.values.double(), dim=1) * has[top.indices]
    r = torch.where(has[top.indices], res_c[top.indices], torch.zeros_like(w)); nres[i:i + 2048] = (w * r).sum(1).cpu().numpy(); nlab[i:i + 2048] = (w * lab_c[top.indices]).sum(1).cpu().numpy(); nw[i:i + 2048] = w.sum(1).cpu().numpy()
pidq = d["pid"]; up = np.unique(pidq); agg = {p: [0.0, 0.0, 0.0] for p in up}
for p, a_, b__, c_ in zip(pidq, nres, nlab, nw):
    t = agg[p]; t[0] += a_; t[1] += b__; t[2] += c_
P_ = [p for p in up if p in m and agg[p][2] > 0]; rn = np.array([agg[p][0] / agg[p][2] for p in P_]); ln = np.array([agg[p][1] / agg[p][2] for p in P_]); rt = np.array([lab[p] - m[p] for p in P_]); mt = np.array([m[p] for p in P_])
A = np.column_stack([np.ones(len(P_)), mt, ln - mt]); res_t = rt - A @ np.linalg.lstsq(A, rt, rcond=None)[0]; res_n = rn - A @ np.linalg.lstsq(A, rn, rcond=None)[0]
rng = np.random.default_rng(0); boot = [np.corrcoef(res_t[k], res_n[k])[0, 1] for k in (rng.integers(0, len(P_), len(P_)) for _ in range(1000))]
print(f"D2 neighbour residuals ({len(P_)} validation players; top-{K} neighbours per game, softmax-weighted, averaged over the player's 20 games):")
print(f"  correlation of the target's residual with the neighbours' mean residual: {np.corrcoef(rt, rn)[0, 1]:+.3f}; partial (given the target's estimate and the neighbours' mean label minus it): {np.corrcoef(res_t, res_n)[0, 1]:+.3f} [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]; regression slope of the residual on the neighbours' residual: {float((res_t * res_n).sum() / (res_n * res_n).sum()):+.3f}; sd of the neighbours' mean residual {rn.std():.3f}")
print(f"  for reference: correlation of the target's residual with (neighbours' mean label - target's estimate): {np.corrcoef(rt, ln - mt)[0, 1]:+.3f}")
