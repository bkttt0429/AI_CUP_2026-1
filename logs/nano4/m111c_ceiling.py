"""Section 4M M111c (2026-11-14, report only, CPU): the ceiling of perfect opponent identification, with the dependence found in
M111b (a player's readings deviate from his label together with the base model's error, correlation about 0.5).
 1  real data: validation questions by size and by the number of near-certain readings (bands >= q98, about 90% right): base score
    and score with the current evidence (tempered) - what questions that happen to have many readings actually reach.
 2  simulation calibrated on the real gain: the shared offset of a player's readings u = 0.5 x (r z + sqrt(1 - r^2) noise), z the
    standardised base error of that question (within label and size); for each r: the current evidence emulated (must equal the
    real gain at the right r), then the same pipeline if a share rho of the games were read with certainty (temperature on a small
    grid, best taken) - rho 0.7 = every labelled opponent identified, 1.0 = every opponent.
  python -I logs/nano4/m111c_ceiling.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m87_pi0 as m87  # noqa: E402
import m111_label_readings as m1  # noqa: E402
import m111b_hier as mh  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM; rng = np.random.default_rng(1113); LAB = np.arange(10); RS = (0.0, 0.3, 0.5, 0.7, 0.85); RHO = (0.3, 0.5, 0.7, 1.0); TG = (0.25, 0.4, 0.6, 0.8, 1.0); REPS = 2


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool); edges, cal, om = m87.corrected(z, ones); _, _, band, g = s2.fit(z, ones)
    PAR = {k: mh.fit_group(cal[k]["K"], mh.SU)[1] for k in (0, 1, 2)}; rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        acc = {}; add = lambda k_, x: acc.setdefault(k_, []).append(x); tab = []  # noqa: E731
        for v in FOLDS:
            tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = m1.full_tables(z, tr, cl, bnd, gg); y, n, pid, q = m3.load_fold(v); y, n = np.asarray(y), np.asarray(n); Q = len(y)
            qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(q["game_keys"])]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in q["game_keys"]]).astype(int); ok = g[gi] >= 0; qi, gi = qi[ok], gi[ok]; kk, yy, bb, lab = g[gi], y[qi], band[gi], z["opp_lab"][gi]
            P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]
            bal = lambda dd_: float(balanced_mean_over_n(SM[dd_, y], y, n, sizes=SIZES))  # noqa: E731

            def summed(logE):
                L = np.zeros((Q, 10)); np.add.at(L, qi, logE); return L
            d0 = mb.decisions(P, np.zeros((Q, 10)), 0.5, 3.0, 0.6)[0]; Lr = summed(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab])) * Tn; dr = mb.decisions(P, Lr, 0.5, 3.0, 0.6)[1]; add("base", bal(d0)); add("real", bal(dr))
            cnt = np.bincount(qi[bb >= 5], minlength=Q); tab.append(np.stack([n, cnt, SM[d0, y], SM[dr, y]], 1))
            e = P @ LAB - y; zq = np.zeros(Q)
            for c in range(10):
                for m in np.unique(n):
                    s = (y == c) & (n == m)
                    if s.sum() > 5:
                        zq[s] = (e[s] - e[s].mean()) / (e[s].std() + 1e-9)
            for r in RS:
                for _ in range(REPS):
                    uq = mh.SU * (r * zq + np.sqrt(1 - r * r) * rng.normal(0, 1, Q)); pos = yy + uq[qi]; lt = np.zeros(len(qi), int)
                    for k in (0, 1, 2):
                        s = kk == k; cdf = np.cumsum(mh.kw(pos[s], PAR[k]), 1); lt[s] = np.minimum((rng.random(int(s.sum()))[:, None] > cdf).sum(1), 9)
                    true = rng.random(len(qi)) < a; ls = np.where(true, lt, m1.draw(P0c, kk, yy)); add(("emulated", r), bal(mb.decisions(P, summed(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, ls])) * Tn, 0.5, 3.0, 0.6)[1]))
                    for rho in RHO:
                        hit = rng.random(len(qi)) < rho; Lc = summed(np.where(hit[:, None], np.log(0.02 + 0.98 * R[kk, :, lt]), 0.0))
                        for t_ in TG:
                            dd_ = mb.decisions(P, Lc * Tn * t_, 0.5, 3.0, 0.6)[1]; add(("certain", r, rho, t_), bal(dd_))
                            for m in SIZES:
                                add(("certain", r, rho, t_, m), float(balanced_mean_over_n(SM[dd_, y], y, n, sizes=(m,))))
        mean = {k_: float(np.mean(x)) for k_, x in acc.items()}; base = mean["base"]; print(f"\n=== seeds {seeds}: base {base:.4f}; current evidence (tempered), real: {mean['real'] - base:+.4f} ===")
        if seeds[0] == 0:
            Tb = np.concatenate(tab); print("real data: questions by size and by the number of near-certain readings (bands >= q98): share of the size's questions | base score -> with the current evidence")
            for m in SIZES:
                out = []
                for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99)):
                    s = (Tb[:, 0] == m) & (Tb[:, 1] >= lo) & (Tb[:, 1] <= hi)
                    if s.sum() >= 30:
                        out.append(f"{lo}{'' if lo == hi else ('-' + str(hi) if hi < 99 else '+')}: {s.sum() / (Tb[:, 0] == m).sum():.2f} | {Tb[s, 2].mean():.3f} -> {Tb[s, 3].mean():.3f}")
                print(f"    n={m}: " + "   ".join(out))
        print("simulation: correlation r between the readings' shared offset and the base error -> emulated current evidence | pipeline with certain readings on 0.3 / 0.5 / 0.7 / 1.0 of the games (best temperature)")
        for r in RS:
            best = {rho: max(TG, key=lambda t_: mean[("certain", r, rho, t_)]) for rho in RHO}
            print(f"    r {r:4.2f}: emulated {mean[('emulated', r)] - base:+.4f} | " + "  ".join(f"{mean[('certain', r, rho, best[rho])] - base:+.4f} (T x{best[rho]})" for rho in RHO) + "   | at 0.7 by size: " + " / ".join(f"{mean[('certain', r, 0.7, best[0.7], m)]:.4f}" for m in SIZES), flush=True)
        rep[f"seeds_{seeds[0]}"] = {str(k_): v_ for k_, v_ in mean.items() if not (isinstance(k_, tuple) and len(k_) == 5)}
    (ROOT / "results/r_variants/m111c_ceiling.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
