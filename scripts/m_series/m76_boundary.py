"""Section 4M M76 (2026-11-10; user: Jev 只想用在分不開的邊界上，參考它的機率設計方法). Report only.
A boundary-only decision module in Jev's shape, on the opponent-anchor evidence (bands >= q90, as ct41):
  region  = the posterior's two most likely levels are adjacent, hold >= MASS of the probability, and their ratio is <= RMAX
  Noul    = pairwise posterior of the second level given the pair, from the model's two probabilities x the anchors' evidence for those two levels
  act     = switch to the second level only when its pairwise posterior >= TAU (0.5 = Bayes within the pair; > 0.5 = act only when confident);
            outside the region the base decision stands
against A = the anchors on all ten levels of every question (the current composition), and "A inside the region only".
Scenarios: CV density with matched weights; test-like density 0.74 with weights x0.74; weights too high (density 0.74, x1.0);
density lower than assumed (0.5, x0.74). Seeds 0-2 and 6-8. Gain vs the base ensemble, share of changed decisions, share of changes of >= 2 levels.
  python -I logs/nano4/m76_boundary.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_cv_density as cd  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

K = np.arange(10); S = np.where(K[:, None] == K[None, :], 1.0, np.where(np.abs(K[:, None] - K[None, :]) == 1, np.exp(-1), 0.0))
KMIN = 3


def fold_data(z_cal, z_eval, scale, seeds):
    out = []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        edges, cal, _, _ = s2.fit(z_cal, z_cal["fold"] != v)
        te = np.flatnonzero(z_eval["fold"] == v); tb = z_eval["target_black"][te]
        band = np.where(tb, np.searchsorted(edges[False], z_eval["opp_m"][te]), np.searchsorted(edges[True], z_eval["opp_m"][te]))
        g = s2.group_of(z_eval["hd"][te], tb); lab = z_eval["opp_lab"][te]; logE = np.zeros((len(te), 10))
        for k in (0, 1, 2):
            s = g == k; a = scale * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= KMIN); dd = lab[s][:, None] - K[None, :]
            logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])
        key_row = {k: i for i, k in enumerate(z_eval["key"][te])}; L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        out.append((y, n, pid, P, L))
    return out


def decisions(P, L, mass, rmax, tau):
    d0 = (P @ S.T).argmax(1); Pa = P * np.exp(L - L.max(1, keepdims=True)); dA = (Pa @ S.T).argmax(1)
    top = np.argsort(-P, axis=1)[:, :2]; c1, c2 = top[:, 0], top[:, 1]; r = np.arange(len(P))
    region = (np.abs(c1 - c2) == 1) & (P[r, c1] + P[r, c2] >= mass) & (P[r, c1] / np.maximum(P[r, c2], 1e-12) <= rmax)
    w1, w2 = P[r, c1] * np.exp(L[r, c1]), P[r, c2] * np.exp(L[r, c2]); p2 = w2 / (w1 + w2)
    dB = np.where(region & (p2 >= tau), c2, np.where(region & (1 - p2 >= tau), c1, d0))
    return d0, dA, np.where(region, dA, d0), dB, region


def main():
    z = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); rep = {}
    variants = [("A all levels, all questions", None), ("A inside the region only", None)] + [(f"pair-only, act at >= {t:.2f}", t) for t in (0.5, 0.6, 0.7)]
    for sc_name, rho, scale in (("CV density, matched weights", 1.0, 1.0), ("test-like density 0.74, weights x0.74", 0.74, 0.74), ("weights too high (0.74, x1.0)", 0.74, 1.0), ("density lower than assumed (0.5, x0.74)", 0.5, 0.74)):
        print(f"== {sc_name}"); rep[sc_name] = {}
        for seeds in ((0, 1, 2), (6, 7, 8)):
            acc = {nm: [] for nm, _ in variants}; reg_share = []
            for draw in ((1, 2, 3) if rho < 1 else (1,)):
                data = fold_data(z, cd.thin(z, rho, draw) if rho < 1 else z, scale, seeds)
                y, n, pid = (np.concatenate([d[k] for d in data]) for k in (0, 1, 2))
                per = {nm: [] for nm, _ in variants}; d0s, regs = [], []
                for yy, nn, pp, P, L in data:
                    for (nm, tau) in variants:
                        d0, dA, dAr, dB, region = decisions(P, L, 0.5, 3.0, tau if tau else 0.5)
                        per[nm].append({"A all levels, all questions": dA, "A inside the region only": dAr}.get(nm, dB))
                    d0s.append(d0); regs.append(region)
                d0 = np.concatenate(d0s); reg_share.append(float(np.concatenate(regs).mean()))
                for nm, _ in variants:
                    d = np.concatenate(per[nm]); dd = paired_diff(S[d, y], S[d0, y], y, pid, n, sizes=SIZES, n=2000); ch = d != d0
                    acc[nm].append((dd["diff"], dd["ci95"][0], float(ch.mean()), float((np.abs(d - d0)[ch] >= 2).mean()) if ch.any() else 0.0))
            for nm, _ in variants:
                a = np.mean(acc[nm], axis=0); rep[sc_name].setdefault(nm, []).append(a.tolist())
            print(f"  seeds {seeds} (region = {np.mean(reg_share):.3f} of questions): " + " | ".join(f"{nm}: {np.mean(acc[nm], axis=0)[0]:+.4f} (CI lower {np.mean(acc[nm], axis=0)[1]:+.4f}; changed {np.mean(acc[nm], axis=0)[2]:.3f}, >=2 levels {np.mean(acc[nm], axis=0)[3]:.2f})" for nm, _ in variants), flush=True)
    Path(ROOT / "results/r_variants/m76_boundary.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
