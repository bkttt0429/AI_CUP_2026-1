"""Section 4M M111b (2026-11-14, report only, CPU): the readings in label space with the PERSONAL OFFSET in the model.
M111's simulation with independent readings predicted +0.078 for the current evidence where the real gain is +0.028: a player's
readings share an offset (M96: sd 0.50 levels between players, rounding of the label included). Here:
 1  measured on the validation players: the mean of a player's confident readings (bands >= q98) minus the own label, against the
    base ensemble's error on the same player (correlation), by number of readings.
 2  hierarchical simulation: reading = round(own level + u + eps), u ~ N(0, su^2) per player, eps a two-normal mixture fitted so
    that the marginal equals the measured K of each group. (a) the current evidence emulated on such data (independent-product
    formula, with and without the tempering by question size) against the real evidence; (b) the ceiling: a share rho of the games
    read with certainty, combined by the correct random-effects likelihood, and by the independent product.
 3  real data: each opponent's own offset (the labels of the players he meets minus his label, from the identified pairs of M96,
    cross-fitted by the owners' folds, shrunk) added to his label before the reading is used; against the standard evidence.
 4  real data: random-effects combination of a question's readings (shared offset integrated out) against the independent product
    with and without tempering.
  python -I logs/nano4/m111b_hier.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m111_label_readings as m1  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM; RV = ROOT / "results/r_variants"; rng = np.random.default_rng(1112); SU = 0.50; UG = np.linspace(-2.0, 2.0, 81); LAB = np.arange(10)


def ncdf(x):
    return 0.5 * (1.0 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / math.sqrt(2.0)).numpy())


def kw(r, par):
    """P(reading = l | continuous own position r) [len(r), 10] for one group: eps = two-normal mixture, shift mu; labels 0 and 9 absorb the tails."""
    mu, s1, s2_, w2 = par; r = np.asarray(r, float)[:, None] + mu; hi = np.concatenate([LAB[:-1] + 0.5, [np.inf]])[None, :]; lo = np.concatenate([[-np.inf], LAB[1:] - 0.5])[None, :]
    return (1 - w2) * (ncdf((hi - r) / s1) - ncdf((lo - r) / s1)) + w2 * (ncdf((hi - r) / s2_) - ncdf((lo - r) / s2_))


def fit_group(K, su):
    """(mu, s1, s2, w2) so that round(u + mu + eps) - 0 has the measured difference distribution K (19 bins), u ~ N(0, su^2)."""
    pu = np.exp(-0.5 * (UG / su) ** 2); pu /= pu.sum(); D = np.arange(-9, 10); best = None; mu0 = float((K * D).sum())
    for mu in (mu0 - 0.1, mu0, mu0 + 0.1):
        for s1 in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
            for s2_ in (1.2, 1.6, 2.0, 2.6):
                for w2 in (0.05, 0.1, 0.15, 0.2, 0.3):
                    r = UG[:, None] + mu; hi, lo = D[None, :] + 0.5, D[None, :] - 0.5; pm = (1 - w2) * (ncdf((hi - r) / s1) - ncdf((lo - r) / s1)) + w2 * (ncdf((hi - r) / s2_) - ncdf((lo - r) / s2_)); m = pu @ pm; e = float(((m - K) ** 2).sum())
                    if best is None or e < best[0]:
                        best = (e, (mu, s1, s2_, w2), m)
    return best


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; dv = np.load(RV / "m84_SP2e_val.npz"); best = dv["f_opp_best"].astype(str); row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool)
    edges, cal, om = m87.corrected(z, ones); _, _, band, g = s2.fit(z, ones); pu = np.exp(-0.5 * (UG / SU) ** 2); pu /= pu.sum(); PAR = {}
    for k in (0, 1, 2):
        e, par, m = fit_group(cal[k]["K"], SU); PAR[k] = par; K = cal[k]["K"]; print(f"group {m1.GN[k]}: within-player reading fitted (shared offset sd {SU}): shift {par[0]:+.2f}, eps = {1 - par[3]:.2f} N(0, {par[1]}^2) + {par[3]:.2f} N(0, {par[2]}^2); marginal same level {m[9]:.3f} (measured {K[9]:.3f}), within one {m[8:11].sum():.3f} ({K[8:11].sum():.3f})")
    players = np.array(sorted(pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).player_id)); pidx = {p: i for i, p in enumerate(players)}; pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); pfold = pf.fold.astype(int).reindex(players).to_numpy(); plab = pf["rank"].map(b.RANK).reindex(players).to_numpy()
    pr = np.load(RV / "m96_pairs.npz"); cand_i = np.array([pidx.get(p, -1) for p in best])
    # fold data
    F = {}
    for v in FOLDS:
        y, n, pid, q = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(q["game_keys"])]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in q["game_keys"]]).astype(int); ok = g[gi] >= 0
        F[v] = dict(y=y, n=n, pid=pid, qi=qi[ok], gi=gi[ok])
    # ---------------------------------------------------------------- 1: offset of the readings against the base error
    P1 = {v: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0) for v in FOLDS}; rows_ = []
    for v in FOLDS:
        f = F[v]; s20 = f["n"] == 20; conf = (band[f["gi"]] >= 5) & (g[f["gi"]] == 0); cnt = np.bincount(f["qi"][conf], minlength=len(f["y"])); sm = np.bincount(f["qi"][conf], weights=z["opp_lab"][f["gi"]][conf], minlength=len(f["y"])); eb = P1[v] @ LAB - f["y"]
        for j in np.flatnonzero(s20 & (cnt >= 1)):
            rows_.append((cnt[j], sm[j] / cnt[j] - f["y"][j], eb[j], f["y"][j]))
    R_ = np.array(rows_)
    for lo, hi in ((1, 2), (3, 5), (6, 99)):
        s = (R_[:, 0] >= lo) & (R_[:, 0] <= hi); x, e_ = R_[s, 1], R_[s, 2]; xw = x - pd.Series(x).groupby(R_[s, 3]).transform("mean").to_numpy(); ew = e_ - pd.Series(e_).groupby(R_[s, 3]).transform("mean").to_numpy()
        print(f"20-game questions with {lo}-{hi if hi < 99 else '..'} confident even-game readings ({int(s.sum())}): mean reading minus own label: mean {x.mean():+.2f}, sd {x.std():.2f}; base error sd {e_.std():.2f}; correlation of the two {float(np.corrcoef(x, e_)[0, 1]):+.3f} (within the label {float(np.corrcoef(xw, ew)[0, 1]):+.3f})")
    # ---------------------------------------------------------------- 2: hierarchical simulation
    RHO = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0); REPS = 2; KW_U = {k: np.stack([kw(c + UG, PAR[k]) for c in range(10)]) for k in (0, 1, 2)}  # [c, u, l]
    for seeds in ((0, 1, 2), (6, 7, 8)):
        acc = {}; add = lambda k_, x: acc.setdefault(k_, []).append(x)  # noqa: E731
        for v in FOLDS:
            tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); lK = np.log(np.maximum(Kc, 1e-6)); f = F[v]; y, n, qi, gi = f["y"], f["n"], f["qi"], f["gi"]; Q = len(y)
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); kk, yy, bb, lab = g[gi], y[qi], band[gi], z["opp_lab"][gi]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]
            bal = lambda dd_, sz=SIZES: float(balanced_mean_over_n(SM[dd_, y], y, n, sizes=sz))  # noqa: E731

            def sc(L, name):
                d0, dA, dR, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); add((name, "all"), bal(dA)); add((name, "region"), bal(dR))
                for m in SIZES:
                    add((name, m), bal(dA, (m,)))

            def summed(logE):
                L = np.zeros((Q, 10)); np.add.at(L, qi, logE); return L
            sc(np.zeros((Q, 10)), "base"); Lr = summed(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab])); sc(Lr, "real"); sc(Lr * Tn, "real tempered")
            for r_ in range(REPS):
                uq = rng.normal(0, SU, Q); pos = yy + uq[qi]; lt = np.zeros(len(qi), int)
                for k in (0, 1, 2):
                    s = kk == k; cdf = np.cumsum(kw(pos[s], PAR[k]), 1); lt[s] = np.minimum((rng.random(int(s.sum()))[:, None] > cdf).sum(1), 9)
                true = rng.random(len(qi)) < a; ls = np.where(true, lt, m1.draw(P0c, kk, yy)); Le = summed(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, ls])); sc(Le, "emulated"); sc(Le * Tn, "emulated tempered")
                for rho in RHO:
                    hit = rng.random(len(qi)) < rho; sc(summed(np.where(hit[:, None], lK[kk, :, lt], 0.0)), ("independent product", rho))
                    # correct combination: log sum_u p(u) prod_i Kw(l_i | c + u)
                    LU = np.zeros((Q, 10, len(UG)))
                    for k in (0, 1, 2):
                        s = hit & (kk == k); np.add.at(LU, qi[s], np.log(np.maximum(KW_U[k][:, :, lt[s]], 1e-9)).transpose(2, 0, 1))
                    mx = LU.max(2, keepdims=True); sc(np.log((np.exp(LU - mx) * pu[None, None, :]).sum(2)) + mx[:, :, 0], ("random effects", rho))
        mean = {k_: float(np.mean(x)) for k_, x in acc.items()}; base = mean[("base", "all")]
        print(f"\n=== seeds {seeds}: base {base:.4f} ===\ncurrent evidence, REAL: {mean[('real', 'all')] - base:+.4f} (tempered {mean[('real tempered', 'all')] - base:+.4f}); EMULATED on data with the personal offset: {mean[('emulated', 'all')] - base:+.4f} (tempered {mean[('emulated tempered', 'all')] - base:+.4f})   [M111 without the offset: +0.078]")
        print("ceiling with a personal offset of the readings: share of the games read with certainty -> gain over the base: correct random-effects combination | independent product | by size 5 / 10 / 20 (random effects)")
        for rho in RHO:
            print(f"    {rho:4.2f}   {mean[(('random effects', rho), 'all')] - base:+.4f} ({mean[(('random effects', rho), 'all')]:.4f}) | {mean[(('independent product', rho), 'all')] - base:+.4f} | " + " / ".join(f"{mean[(('random effects', rho), m)]:.4f}" for m in SIZES), flush=True)
    # ---------------------------------------------------------------- 3 and 4: real data
    S = {k_: {s_: [] for s_ in (0, 6)} for k_ in ("standard", "opponent offsets", "tempered", "parametric independent", "parametric independent tempered", "random effects", "random effects + opponent offsets")}; ys, ns, ps = [], [], []
    for v in FOLDS:
        tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); f = F[v]; y, n, qi, gi = f["y"], f["n"], f["qi"], f["gi"]; Q = len(y); kk, bb, lab = g[gi], band[gi], z["opp_lab"][gi]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]
        okp = pfold[pr["owner"]] != v; dd = (plab[pr["owner"]] - plab[pr["opp"]]).astype(float); sw = np.bincount(pr["opp"][okp], weights=pr["weight"][okp], minlength=len(players)); sd_ = np.bincount(pr["opp"][okp], weights=(pr["weight"] * dd)[okp], minlength=len(players)); mu0 = sd_.sum() / sw.sum(); off = (sd_ + 5.0 * mu0) / (sw + 5.0) - mu0  # shrunk, centred
        ci = cand_i[gi]; o = np.where(ci >= 0, off[np.maximum(ci, 0)], 0.0); lf = np.clip(lab + o, 0, 9); l0 = np.floor(lf).astype(int); l1 = np.minimum(l0 + 1, 9); fr = (lf - l0)[:, None]
        if v == FOLDS[0]:
            print(f"\nopponents' own offsets (labels of the players they meet minus their label; fold {v}): candidates with >= 5 weighted pairs {int((sw >= 5).sum())}; sd of the shrunk offset over the anchors actually used (bands >= q90) {float(o[bb >= 3].std()):.2f}; share of those anchors whose candidate has >= 5 pairs {float((sw[np.maximum(ci, 0)][bb >= 3] >= 5).mean()):.2f}")
        Rs = R[kk, :, l0] * (1 - fr) + R[kk, :, l1] * fr; E_std = np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab]); E_off = np.log((1 - a)[:, None] + a[:, None] * Rs)

        def summed(logE):
            L = np.zeros((Q, 10)); np.add.at(L, qi, logE); return L

        def rand_eff(lab_f):
            """Random effects: log sum_u p(u) prod_i [(1 - a_i) + a_i Kw(l_i | c + u) / P0c(l_i | c)]; and the independent product with the
            same parametric kernel (every reading marginalised over u on its own), to separate the shared offset from the change of tables."""
            LU = np.zeros((Q, 10, len(UG))); LI = np.zeros((Q, 10)); l0_ = np.floor(lab_f).astype(int); l1_ = np.minimum(l0_ + 1, 9); fr_ = lab_f - l0_
            for k in (0, 1, 2):
                s = kk == k; kwl = KW_U[k][:, :, l0_[s]] * (1 - fr_[s]) + KW_U[k][:, :, l1_[s]] * fr_[s]; p0 = (P0c[k][:, l0_[s]] * (1 - fr_[s]) + P0c[k][:, l1_[s]] * fr_[s])[:, None, :]  # [c, u, G], [c, 1, G]
                rat = np.minimum(kwl / np.maximum(p0, 1e-9), 30.0); np.add.at(LU, qi[s], np.log((1 - a[s])[None, None, :] + a[s][None, None, :] * rat).transpose(2, 0, 1)); np.add.at(LI, qi[s], np.log((1 - a[s])[None, :] + a[s][None, :] * (rat * pu[None, :, None]).sum(1)).T)
            mx = LU.max(2, keepdims=True); return np.log((np.exp(LU - mx) * pu[None, None, :]).sum(2)) + mx[:, :, 0], LI
        re_, pi_ = rand_eff(lab.astype(float)); Ls = {"standard": summed(E_std), "opponent offsets": summed(E_off), "tempered": summed(E_std) * Tn, "parametric independent": pi_, "parametric independent tempered": pi_ * Tn, "random effects": re_, "random effects + opponent offsets": rand_eff(lf)[0]}
        for s_, seeds in ((0, (0, 1, 2)), (6, (6, 7, 8))):
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            for k_, L in Ls.items():
                S[k_][s_].append(SM[mb.decisions(P, L, 0.5, 3.0, 0.6)[1], y])
        ys.append(y); ns.append(n); ps.append(f["pid"])
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(ps); print("\nreal data, SP2e evidence (class-conditional tables, pi0, all bands; unrestricted decision), seeds 0-2 / 6-8:")
    for new, old in (("opponent offsets", "standard"), ("tempered", "standard"), ("parametric independent", "standard"), ("random effects", "parametric independent"), ("random effects", "parametric independent tempered"), ("random effects", "tempered"), ("random effects + opponent offsets", "random effects"), ("random effects + opponent offsets", "tempered")):
        out = []
        for s_ in (0, 6):
            d = paired_diff(np.concatenate(S[new][s_]), np.concatenate(S[old][s_]), y, pid, n, sizes=SIZES, n=3000); pf_ = [float(balanced_mean_over_n(a_, yy, nn, sizes=SIZES) - balanced_mean_over_n(c_, yy, nn, sizes=SIZES)) for a_, c_, yy, nn in zip(S[new][s_], S[old][s_], ys, ns)]; out.append(f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf_))
        print(f"  {new:34s} vs {old:15s} " + "  |  ".join(out), flush=True)
    print("  scores: " + "  ".join(f"{k_} {np.mean([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(S[k_][0], ys, ns)]):.4f}" for k_ in S))


if __name__ == "__main__":
    main()
