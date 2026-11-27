"""Section 4M M89 diagnostic D3 (2026-11-11, report only): is "the candidate's model-estimated level is close to the target's" a
verification signal on top of the margin? For HD0 queries of 5e: joint distribution of (margin band group, |m_T - m_X| bin) for the
opponent queries and for the impostor queries (target's own side: certainly false matches, their top candidate's estimate m_X');
implied share of real matches per cell a = 1 - (1 - pi) * impostor share / opponent share (pi from the lowest band, as M87), and the
residual coupling corr(y_T - m_T, l_X - m_X) per cell.
  python -I logs/nano4/m89_diag3.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

RV = ROOT / "results/r_variants"; m, lab = {}, {}
for v in FOLDS:
    y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); E = P @ np.arange(10)
    for i in np.flatnonzero(np.asarray(n) == 20):
        m[pid[i]] = float(E[i]); lab[pid[i]] = int(y[i])
d = dict(np.load(RV / "m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}
edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"]))
mT = np.array([m.get(p, np.nan) for p in z["pid"]]); mX = np.array([m.get(p, np.nan) for p in d["f5e_opp_best"]])
# the impostor's candidate id is not stored; its label is. Approximate its estimate by the label's mean estimate is useless here, so the null over the strength gap is taken from SHUFFLED pairs within the same margin band group: opponent candidates of other targets
rng = np.random.default_rng(0); s0 = g == 0; grp = np.select([band == 0, band <= 2, band <= 4, band <= 8], [0, 1, 2, 3]); names = ("< q50", "q50-q90", "q90-q98", ">= q98"); bins = (0.5, 1.5); bl = ("|gap| <= 0.5", "0.5 - 1.5", "> 1.5")
gap = np.abs(mT - mX); ok = s0 & np.isfinite(gap); kb = np.digitize(gap, bins); perm = rng.permutation(np.flatnonzero(ok)); gap_null = np.full(len(gap), np.nan); gap_null[np.flatnonzero(ok)] = np.abs(mT[np.flatnonzero(ok)] - mX[perm]); kn = np.digitize(gap_null, bins)
print("HD0 opponent queries with an estimate for the candidate; share of each strength-gap bin, observed vs random pairing (candidates shuffled across targets), and the residual coupling:")
print("  band group    gap bin        n     share   share under random pairing   exact-label rate   corr(target residual, candidate residual)")
rT = z["y"] - mT; rX = z["opp_lab"] - mX
for gi, gn in enumerate(names):
    s = ok & (grp == gi)
    for k in range(3):
        c = s & (kb == k); c0 = s & (kn == k)
        print(f"  {gn:10s}   {bl[k]:12s} {int(c.sum()):6d}   {c.sum() / s.sum():.3f}   {c0.sum() / s.sum():.3f}                        {float((z['opp_lab'][c] == z['y'][c]).mean()):.3f}              {np.corrcoef(rT[c], rX[c])[0, 1]:+.3f}")
    print(f"  {gn:10s}   all gaps: corr(target residual, candidate residual) {np.corrcoef(rT[s], rX[s])[0, 1]:+.3f}; sd of y_T - l_X {np.std((z['y'] - z['opp_lab'])[s]):.3f}; sd of m_T - m_X {np.std((mT - mX)[s]):.3f}")
top = ok & (grp == 3); A = np.column_stack([np.ones(top.sum()), rX[top], (mX - mT)[top]]); coef = np.linalg.lstsq(A, rT[top], rcond=None)[0]; fit = A @ coef
print(f">= q98: regression of the target's residual on the candidate's residual and on (m_X - m_T): coefficients {coef[1]:+.3f}, {coef[2]:+.3f}; R^2 {1 - ((rT[top] - fit) ** 2).sum() / ((rT[top] - rT[top].mean()) ** 2).sum():.3f}; sd of the target's residual {rT[top].std():.3f} -> {(rT[top] - fit).std():.3f}")
low = ok & (grp == 0); A = np.column_stack([np.ones(low.sum()), rX[low], (mX - mT)[low]]); coef = np.linalg.lstsq(A, rT[low], rcond=None)[0]; fit = A @ coef
print(f"<  q50: the same regression (false matches): coefficients {coef[1]:+.3f}, {coef[2]:+.3f}; R^2 {1 - ((rT[low] - fit) ** 2).sum() / ((rT[low] - rT[low].mean()) ** 2).sum():.3f}")
