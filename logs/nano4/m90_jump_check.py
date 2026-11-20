"""Section 4M M90 follow-up (2026-11-13, report only, decides nothing): ct75 moves one Public question (pub_q_0157) from 4D to A, four
levels away from the base answer; every earlier file only moved answers by one level. (1) CV under the test rules: how often does a
recipe move the decision two or more levels away from the base decision, and how good are those moves? (2) The test questions where
ct74 / ct75 are two or more levels away from ct28: base probabilities, per-game evidence (identified opponent's level, handicap
group, confidence band, weight), summed evidence, posterior.
  python -I logs/nano4/m90_jump_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m90_testmimic as t  # noqa: E402  (runs its own report first; reuses its fold data and recipes)
from aicup.metric import LEVELS  # noqa: E402

b, s2, mb, m83, m87, ES, FOLDS = t.b, t.s2, t.mb, t.m83, t.m87, t.ES, t.FOLDS
print("\n== (1) CV under the test rules: decisions two or more levels away from the base decision, seeds 0-2 and 6-8 pooled")
for name in ("ct44", "ct72", "ct74", "ct75"):
    c = {"n": 0, "j1": 0, "j2": 0, "gain2": 0.0, "up2": 0, "dn2": 0, "exact_new": 0, "exact_base": 0, "near_new": 0, "near_base": 0, "gain1": 0.0}; sizes = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        for v in FOLDS:
            y, n, _, Ls = t.Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            d0, _, d, _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6); j = np.abs(d - d0); g = b.SM[d, y] - b.SM[d0, y]; s = j >= 2; c["n"] += len(y); c["j1"] += int((j == 1).sum()); c["j2"] += int(s.sum()); c["gain1"] += float(g[j == 1].sum())
            c["gain2"] += float(g[s].sum()); c["up2"] += int((g[s] > 0).sum()); c["dn2"] += int((g[s] < 0).sum()); c["exact_new"] += int((d[s] == y[s]).sum()); c["exact_base"] += int((d0[s] == y[s]).sum()); c["near_new"] += int((np.abs(d[s] - y[s]) <= 1).sum()); c["near_base"] += int((np.abs(d0[s] - y[s]) <= 1).sum()); sizes += list(j[s])
    k = max(c["j2"], 1)
    print(f"  {name} recipe: {c['n']} question-evaluations; one-level moves {c['j1']} (mean gain {c['gain1'] / max(c['j1'], 1):+.3f}); moves of two or more levels {c['j2']} ({c['j2'] / c['n']:.5f}), sizes {dict(zip(*np.unique(sizes, return_counts=True))) if sizes else {}}; new better {c['up2'] / k:.3f}, worse {c['dn2'] / k:.3f}, mean gain {c['gain2'] / k:+.3f}; exact: new {c['exact_new'] / k:.3f}, base {c['exact_base'] / k:.3f}; within one level: new {c['near_new'] / k:.3f}, base {c['near_base'] / k:.3f}")

print("\n== (2) test questions two or more levels away from ct28")
V, T, _, used = b.sources(["5e", "SP", "SP2", "SP2e"]); rk = np.vectorize(b.RANK.get)
pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float)
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
F = {k: sub(k) for k in ("ct9", "ct28", "ct44", "ct72", "ct73", "ct74", "ct75", "ct75n")}; nq_t = pd.Series(T["5e"]["qid"]).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]


def test_evidence(name, cc, r, floor=1):
    """Per-game log-evidence of one source on the test, as in m86_bundle.py's test_evidence (pi0 on), with the density ratio r it printed."""
    v, tt = V[name], T[name]; ones = np.ones(len(v["y"]), bool); edges, cal, vband, vg = s2.fit(v, ones); cal_e = m87.corrected(v, ones)[1]
    tb = np.where(~tt["target_black"], np.searchsorted(edges[True], tt["opp_m"]), np.searchsorted(edges[False], tt["opp_m"])); tg = s2.group_of(tt["hd"], tt["target_black"])
    R = m83.tables(v, ones, cal_e, vband, vg, m83.KAPPA) if cc else None; logE = np.zeros((len(tb), 10)); aw_all = np.zeros(len(tb))
    for k in (0, 1, 2):
        s = tg == k; aw = (r * cal_e[k]["a"] * (np.arange(len(cal_e[k]["a"])) >= floor))[tb[s]][:, None]; lab = tt["opp_lab"][s]
        ratio = R[k][:, lab].T if cc else cal_e[k]["ratio"][lab[:, None] - np.arange(10)[None, :] + 9]; logE[s] = np.log((1 - aw) + aw * ratio); aw_all[s] = aw[:, 0]
    return logE, tb, tg, aw_all


out = {}
for tag, idn, r in (("ct74", "SP2", 0.768), ("ct75", "SP2e", 0.852)):
    Lm, tbm, tgm, awm = test_evidence(idn, 1, r); Le, tbe, tge, awe = test_evidence("enc", 0, 0.713)
    L = (pd.DataFrame(Lm).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy() + b.U_ENC * pd.DataFrame(Le).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()) * Tq
    d0, dA, dAr, _, region = mb.decisions(P9, L, 0.5, 3.0, 0.6); lev = np.array(LEVELS); locked = F["ct28"] != F["ct9"]; fin = np.where(locked, F["ct28"], lev[dAr]); print(f"{tag}: rebuilt decisions equal the file: {bool((fin == F[tag]).all())} ({int((fin != F[tag]).sum())} differ)")
    far = np.flatnonzero(np.abs(rk(F[tag]) - rk(F["ct28"])) >= 2); out[tag] = (L, Lm, Le, tbm, tgm, awm, tbe, tge, awe, far, idn)
    for i in far:
        q = qids[i]; Pa = P9[i] * np.exp(L[i] - L[i].max()); Pa /= Pa.sum(); sm = T[idn]["qid"] == q; se = T["enc"]["qid"] == q
        print(f"\n  {tag} {q}: games {int(nq_t[i])}, tempering {float(Tq[i, 0]):.1f}, in the boundary region {bool(region[i])}; answers " + ", ".join(f"{k} {F[k][i]}" for k in F))
        print("    levels            " + " ".join(f"{lv:>6s}" for lv in LEVELS)); print("    base probability  " + " ".join(f"{x:6.3f}" for x in P9[i])); print("    summed evidence   " + " ".join(f"{x:+6.2f}" for x in L[i] - L[i].max())); print("    posterior         " + " ".join(f"{x:6.3f}" for x in Pa))
        print("    expected score    " + " ".join(f"{x:6.3f}" for x in Pa @ mb.S.T) + f"   (base decision {LEVELS[d0[i]]}, with evidence {LEVELS[dA[i]]})")
        print(f"    main identifier {idn}, per game (group 0 = even game, 1 / 2 = handicap 1 by colour, -1 = not used; band 0..8 = confidence; weight; identified opponent's level; log-evidence relative to its maximum):")
        for g_, bd, a_, lab, le in sorted(zip(tgm[sm], tbm[sm], awm[sm], T[idn]["opp_lab"][sm], Lm[sm]), key=lambda x: -x[2]):
            print(f"      group {int(g_):2d} band {int(bd)} weight {a_:.2f} opponent {LEVELS[int(lab)]:>2s} | " + " ".join(f"{x:+6.2f}" for x in le - le.max()))
        print("    whole-game encoder, games with a positive weight:")
        for g_, bd, a_, lab, le in sorted(zip(tge[se], tbe[se], awe[se], T["enc"]["opp_lab"][se], Le[se]), key=lambda x: -x[2]):
            if a_ > 0:
                print(f"      group {int(g_):2d} band {int(bd)} weight {a_:.2f} opponent {LEVELS[int(lab)]:>2s} | " + " ".join(f"{x:+6.2f}" for x in le - le.max()))
# how the same question looks to the other identifiers
i = int(np.flatnonzero(qids == "pub_q_0157")[0]); print("\n  pub_q_0157 under the other identifiers (per game: band / weight / identified opponent's level), sorted by weight:")
for idn, r in (("5e", 0.851), ("SP", 0.875), ("SP2", 0.768), ("SP2e", 0.852)):
    Lm, tbm, tgm, awm = test_evidence(idn, 1 if idn != "5e" else 0, r, 1 if idn != "5e" else 3); sm = T[idn]["qid"] == "pub_q_0157"
    print(f"    {idn:5s}: " + ", ".join(f"{int(bd)}/{a_:.2f}/{LEVELS[int(lab)]}" for bd, a_, lab in sorted(zip(tbm[sm], awm[sm], T[idn]["opp_lab"][sm]), key=lambda x: -x[1])))
