"""Section 4M M111 (2026-11-14, report only, CPU): the chain of the "readings in label space" (opponent anchors), link by link,
with the SP2e identifier on the validation queries (calibration cross-fitted by fold, validation density, no tempering/encoder).
 A  the chain: by margin band - share of the games, probability that the first candidate is the opponent (quantile excess with
    the pi0 fix; and a label fit a*K + (1-a)*P0), true-identification mass per game; the share of opponents that are labelled
    training players at all = mass / held-out top-1 (0.4035); the reading's noise (K per group); anchors per question.
 B  value of information: validation questions, the base ensemble's probabilities, and SIMULATED anchors: every game with an
    opponent group gets, with probability rho, a certain anchor whose label is drawn from K_c(. | true level); posterior =
    base x product of K_c. Score against rho (0 = base ... 1 = every opponent read with certainty), by question size; the same with
    a sharper reading (K squared). Check of the simulation: the CURRENT evidence emulated (real bands, first candidate true with
    probability a_b, label from K_c or from the null P0_c) against the REAL evidence on the same games.
 C  what the candidates' prior could add: how often each training player is met as an opponent (identified pairs of M96), whether
    the first candidates of the confident bands are the often-met ones (AUC against the null bands), and the evidence with the
    mixture weight conditioned on the candidate's activity quintile (report).
 D  opponents met by more than one test question: expected from the activity weights, and counted among the confident test anchors.
  python -I logs/nano4/m111_label_readings.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

b, s2, mb, ES, FOLDS, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.SM; RV = ROOT / "results/r_variants"; NOM = np.diff(np.concatenate([[0.0], s2.QS, [1.0]])); TOP1 = 0.40351418; rng = np.random.default_rng(111)
BAND = ["<q50", "q50-80", "q80-90", "q90-95", "q95-98", "q98-99", "q99-99.5", "q99.5-99.9", ">=q99.9"]; GN = ["even", "HD1, target Black", "HD1, target White"]


def full_tables(z, tr, cal, band, g, kappa=m83.KAPPA):
    """As m83.tables, but the class-conditional true and null label distributions themselves: Kc, P0c [3, 10, 10] and R = Kc / P0c."""
    y, lab = z["y"], z["opp_lab"]; Kc, P0c = np.zeros((3, 10, 10)), np.zeros((3, 10, 10))
    for k in (0, 1, 2):
        s = tr & (g == k); low, top = s & (band <= 1), s & (band >= 5); a_top, K, P0 = cal[k]["a_top"], cal[k]["K"], cal[k]["P0"]
        for c in range(10):
            pk, p0 = K[np.arange(10) - c + 9], P0[np.arange(10) - c + 9]; pk, p0 = pk / pk.sum(), p0 / p0.sum()
            n0 = np.bincount(lab[low & (y == c)], minlength=10).astype(float); P0c[k, c] = (n0 + kappa * p0) / (n0.sum() + kappa)
            nt = np.bincount(lab[top & (y == c)], minlength=10).astype(float); Ht = (nt + kappa * ((1 - a_top) * p0 + a_top * pk)) / (nt.sum() + kappa)
            x = np.clip(Ht - (1 - a_top) * P0c[k, c], 0, None); Kc[k, c] = x / x.sum()
    return Kc, P0c, np.minimum(Kc / P0c, 30.0)


def draw(tab, k, c):
    """One label per row from tab[k, c] (rows of probabilities over the ten labels)."""
    cdf = np.cumsum(tab[k, c], 1); return np.minimum((rng.random(len(k))[:, None] > cdf).sum(1), 9)


def main():
    V, T, key_pid, used = b.sources(["5e", "SP2e"]); z = V["SP2e"]; t = T["SP2e"]; dv = np.load(RV / "m84_SP2e_val.npz"); best = dv["f_opp_best"].astype(str); row = {k: i for i, k in enumerate(z["key"])}; ones = np.ones(len(z["y"]), bool)
    # ---------------------------------------------------------------- A: the chain
    edges, cal, om = m87.corrected(z, ones); _, cal0, band, g = s2.fit(z, ones); d = z["opp_lab"] - z["y"]; gshare = np.array([(g == k).mean() for k in (0, 1, 2)]); print(f"validation queries: {len(g)} games; groups: " + ", ".join(f"{GN[k]} {gshare[k]:.3f}" for k in (0, 1, 2)) + f"; no evidence (HD >= 2) {float((g < 0).mean()):.3f}")
    grid = np.linspace(0, 0.99, 100); mass = {"quantile excess (pi0 fix)": 0.0, "label fit": 0.0}
    for k in (0, 1, 2):
        c = cal[k]; K, P0 = c["K"], c["P0"]; s = g == k; a_fit = []
        for bb in range(9):
            h = np.array([((d[s & (band == bb)]) == x).sum() for x in s2.D], float); a_fit.append(float(grid[int(np.argmax([(h * np.log(u * K + (1 - u) * P0)).sum() for u in grid]))]))
        a_fit = np.array(a_fit); mass["quantile excess (pi0 fix)"] += gshare[k] * float((c["a"] * c["share"]).sum()); mass["label fit"] += gshare[k] * float((a_fit * c["share"]).sum())
        print(f"  group {GN[k]}: null share used {om[k]:.3f}; reading K: same level {K[9]:.3f}, within one {K[8:11].sum():.3f}, mean {float((K * s2.D).sum()):+.2f}, sd {float(np.sqrt((K * s2.D ** 2).sum() - (K * s2.D).sum() ** 2)):.2f} | null P0: same {P0[9]:.3f}, within one {P0[8:11].sum():.3f}, sd {float(np.sqrt((P0 * s2.D ** 2).sum() - (P0 * s2.D).sum() ** 2)):.2f}")
        print("      band        share   P(first candidate is the opponent): excess / label fit   mass of true identifications (excess)")
        for bb in range(9):
            print(f"      {BAND[bb]:11s} {c['share'][bb]:.4f}   {c['a'][bb]:.2f} / {a_fit[bb]:.2f}                                      {c['a'][bb] * c['share'][bb]:.4f}")
    for k_, m_ in mass.items():
        print(f"true-identification mass per game, {k_}: {m_:.3f}  ->  share of opponents that are labelled training players = mass / held-out top-1 ({TOP1:.3f}) = {m_ / TOP1:.2f}")
    PI = {k_: min(m_ / TOP1, 1.0) for k_, m_ in mass.items()}; a_all = np.zeros(len(g)); a_all[g >= 0] = np.array([cal[k]["a"][bb] for k, bb in zip(g[g >= 0], band[g >= 0])])
    for v in FOLDS[:1]:
        y, n, pid, q = m3.load_fold(v); idx = [[row[k] for k in g_ if k in row] for g_ in q["game_keys"]]; ea = np.array([a_all[i].sum() for i in idx]); c90 = np.array([int((band[i] >= 3).sum()) for i in idx])
        print(f"anchors per question (fold {v}): " + "  ".join(f"n={m}: expected true identifications {ea[n == m].mean():.2f}, questions with none in the bands >= q90 {float((c90[n == m] == 0).mean()):.2f}" for m in SIZES))
    # ---------------------------------------------------------------- B: value of information
    RHO = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0); REPS = 3; rep = {"mass": mass, "pi": PI}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        acc = {}; add = lambda k_, x: acc.setdefault(k_, []).append(x)  # noqa: E731
        for v in FOLDS:
            tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = full_tables(z, tr, cl, bnd, gg); lK = np.log(np.maximum(Kc, 1e-6)); K2 = Kc ** 2; K2 /= K2.sum(2, keepdims=True); lK2 = np.log(np.maximum(K2, 1e-9))
            y, n, pid, q = m3.load_fold(v); y, n = np.asarray(y), np.asarray(n); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); Q = len(y)
            qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(q["game_keys"])]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in q["game_keys"]]).astype(int); ok = g[gi] >= 0; qi, gi = qi[ok], gi[ok]; kk, yy, bb, lab = g[gi], y[qi], band[gi], z["opp_lab"][gi]
            bal = lambda dd_, sz=SIZES: float(balanced_mean_over_n(SM[dd_, y], y, n, sizes=sz))  # noqa: E731

            def score(logE, name):
                L = np.zeros((Q, 10)); np.add.at(L, qi, logE); d0, dA, dR, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); add((name, "all"), bal(dA)); add((name, "region"), bal(dR))
                for m in SIZES:
                    add((name, m), bal(dA, (m,)))
            score(np.zeros((len(qi), 10)), "base")
            a = cl_a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)])
            score(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, lab]), "real evidence")
            for r_ in range(REPS):
                true = rng.random(len(qi)) < cl_a; ls = np.where(true, draw(Kc, kk, yy), draw(P0c, kk, yy)); score(np.log((1 - a)[:, None] + a[:, None] * R[kk, :, ls]), "emulated evidence")
                for rho in RHO[1:]:
                    hit = rng.random(len(qi)) < rho; ls = draw(Kc, kk, yy); score(np.where(hit[:, None], lK[kk, :, ls], 0.0), ("oracle", rho))
                    if rho in (0.3, 0.7, 1.0):
                        ls2 = draw(K2, kk, yy); score(np.where(hit[:, None], lK2[kk, :, ls2], 0.0), ("sharp", rho))
        mean = {k_: float(np.mean(x)) for k_, x in acc.items()}; base = mean[("base", "all")]; print(f"\n=== seeds {seeds}: base ensemble {base:.4f} (by size " + " / ".join(f"{mean[('base', m)]:.4f}" for m in SIZES) + ") ===")
        print(f"check of the simulation: REAL evidence (SP2e, class-conditional, pi0, all bands; unrestricted decision) {mean[('real evidence', 'all')]:.4f} ({mean[('real evidence', 'all')] - base:+.4f}); EMULATED with the same bands and tables {mean[('emulated evidence', 'all')]:.4f} ({mean[('emulated evidence', 'all')] - base:+.4f}); inside the boundary region only: real {mean[('real evidence', 'region')]:.4f}, emulated {mean[('emulated evidence', 'region')]:.4f}")
        print("oracle: share of the games with a certain reading -> score (gain over the base) | by size 5 / 10 / 20 | region-only decision")
        for rho in RHO[1:]:
            k_ = ("oracle", rho); print(f"    {rho:4.2f}   {mean[(k_, 'all')]:.4f} ({mean[(k_, 'all')] - base:+.4f}) | " + " / ".join(f"{mean[(k_, m)]:.4f}" for m in SIZES) + f" | {mean[(k_, 'region')]:.4f}" + (f"   | with a reading twice as sharp: {mean[(('sharp', rho), 'all')]:.4f}" if (("sharp", rho), "all") in mean else ""))
        xs = np.array(RHO); ys_ = np.array([base] + [mean[(("oracle", r_), "all")] for r_ in RHO[1:]]); eff = float(np.interp(mean[("real evidence", "all")], ys_, xs))
        print(f"the real evidence is worth as much as certain readings on {eff:.2f} of the games; labelled opponents are {PI['quantile excess (pi0 fix)']:.2f} - {PI['label fit']:.2f} of the games -> with perfect identification of every labelled opponent: {float(np.interp(PI['quantile excess (pi0 fix)'], xs, ys_)):.4f} - {float(np.interp(PI['label fit'], xs, ys_)):.4f}")
        rep[f"seeds_{seeds[0]}"] = {"base": base, "real": mean[("real evidence", "all")], "emulated": mean[("emulated evidence", "all")], "oracle": {str(r_): mean[(("oracle", r_), "all")] for r_ in RHO[1:]}, "effective_rho": eff}
    # ---------------------------------------------------------------- C: the candidates' prior (activity)
    players = np.array(sorted(pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).player_id)); pidx = {p: i for i, p in enumerate(players)}; pr = np.load(RV / "m96_pairs.npz"); act = np.bincount(pr["opp"], weights=pr["weight"], minlength=len(players))
    o = np.sort(act)[::-1]; print(f"\nhow often a training player is met as an opponent (identified even-game pairs of M96, {act.sum():.0f} weighted pairs): never {float((act == 0).mean()):.3f} of the players; median {np.median(act):.1f}, 90th percentile {np.quantile(act, 0.9):.1f}, maximum {act.max():.0f}; the most-met 10% of the players hold {o[:len(o) // 10].sum() / o.sum():.2f} of the meetings, the most-met 1% {o[:len(o) // 100].sum() / o.sum():.2f}")
    ca = np.array([act[pidx[p]] if p in pidx else 0.0 for p in best]); hi, lo = (g >= 0) & (band >= 5), (g >= 0) & (band <= 1); r_ = pd.Series(np.concatenate([ca[hi], ca[lo]])).rank().to_numpy(); auc = (r_[:hi.sum()].sum() - hi.sum() * (hi.sum() + 1) / 2) / (hi.sum() * lo.sum())
    print(f"first candidates of the confident bands (>= q98) against those of the null bands (< q80), by how often the candidate is met: AUC {auc:.3f}; median meetings {np.median(ca[hi]):.1f} against {np.median(ca[lo]):.1f}")
    qe = np.quantile(act, [0.2, 0.4, 0.6, 0.8]); aq = np.searchsorted(qe, ca); res = {"standard": [], "activity-conditioned": []}
    for v in FOLDS:
        tr = z["fold"] != v; edg, cl, _ = m87.corrected(z, tr); _, _, bnd, gg = s2.fit(z, tr); Kc, P0c, R = full_tables(z, tr, cl, bnd, gg); st, sn = tr & (gg >= 0) & (bnd >= 5), tr & (gg >= 0) & (bnd <= 1)
        lr = (np.bincount(aq[st], minlength=5) / st.sum() + 1e-3) / (np.bincount(aq[sn], minlength=5) / sn.sum() + 1e-3)
        y, n, pid, q = m3.load_fold(v); y, n = np.asarray(y), np.asarray(n); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(q["game_keys"])]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in q["game_keys"]]).astype(int); ok = g[gi] >= 0; qi, gi = qi[ok], gi[ok]; kk, bb, lab = g[gi], band[gi], z["opp_lab"][gi]; a = np.array([cl[k]["a"][x] for k, x in zip(kk, bb)])
        for name, aa in (("standard", a), ("activity-conditioned", np.clip(a * lr[aq[gi]] / (a * lr[aq[gi]] + 1 - a), 0, 0.98))):
            L = np.zeros((len(y), 10)); np.add.at(L, qi, np.log((1 - aa)[:, None] + aa[:, None] * R[kk, :, lab])); res[name].append(float(balanced_mean_over_n(SM[mb.decisions(P, L, 0.5, 3.0, 0.6)[1], y], y, n, sizes=SIZES)))
        if v == FOLDS[0]:
            print("    likelihood ratio of the candidate's activity quintile (confident against null bands, least-met first): " + "  ".join(f"{x:.2f}" for x in lr))
    print(f"evidence with the mixture weight conditioned on the activity quintile (seeds 0-2, unrestricted decision): standard {np.mean(res['standard']):.4f}, conditioned {np.mean(res['activity-conditioned']):.4f} ({np.mean(res['activity-conditioned']) - np.mean(res['standard']):+.4f}; folds " + " ".join(f"{a_ - b_:+.4f}" for a_, b_ in zip(res["activity-conditioned"], res["standard"])) + ")")
    # ---------------------------------------------------------------- D: opponents met by several test questions
    w = act / act.sum(); tb = np.where(~t["target_black"], np.searchsorted(edges[True], t["opp_m"]), np.searchsorted(edges[False], t["opp_m"])); tg = s2.group_of(t["hd"], t["target_black"]); tbest = np.load(RV / "m84_SP2e_test.npz")["f_opp_best"].astype(str); conf = (tg >= 0) & (tb >= 3)
    for name, pi_ in PI.items():
        lam = 10000 * pi_ * w; print(f"expected, labelled opponents = {pi_:.2f} of the 10,000 test games ({name}): share of those games whose opponent is met by the test set at least twice {float((lam * (1 - np.exp(-lam))).sum() / lam.sum()):.2f}; labelled players met at least twice {float((1 - np.exp(-lam) * (1 + lam)).sum()):.0f}")
    tab = pd.DataFrame({"cand": tbest[conf], "q": t["qid"][conf]}).drop_duplicates(); cnt = tab.cand.value_counts(); lam = conf.sum() * w
    print(f"counted: confident test anchors (bands >= q90) {int(conf.sum())} games; first candidates that appear in two or more different questions {int((cnt >= 2).sum())} (covering {int(cnt[cnt >= 2].sum())} question-candidate pairs); expected from the activity weights if every confident anchor were right: {float((1 - np.exp(-lam) * (1 + lam)).sum()):.0f} candidates")
    (RV / "m111_label_readings.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
