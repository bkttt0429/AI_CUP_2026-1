"""Section 4M M90 follow-up (2026-11-13, report only; written after the pre-registered tests of S2 / SP2 / S2e / SP2e and of bundles
v4 / v5 had been read, it decides nothing): the recipes of ct74 (SP2 + four switches) and ct75 (SP2e + four switches) scored in CV
under the TEST-side rules (weights x density ratio, decisions only inside the boundary region, every band with a positive weight),
against ct44's recipe under its own test rules, and the "no reversal" composition that the files ct73 / ct75n actually use (decision
= ct44 recipe's decision where it differs from the base decision, else the new recipe's decision). Same construction as
m86e_testmimic.py, m86f_breakdown.py and m86g_norev.py. Then checks of the candidate files.
  python -I logs/nano4/m90_testmimic.py
"""
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

R_MAIN = {"5e": 0.851, "SP": 0.875, "SP2": 0.768, "SP2e": 0.852}; R_ENC = 0.713  # density ratios printed by the builders of the test files


def evidence(z, v, cc, pi0, band_min, scale):
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)
    if pi0:
        _, cal, _ = m87.corrected(z, tr)
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out))


RECIPES = {"ct44": ("5e", 0, 0, 0, 0, 3), "ct72": ("SP", 1, 1, 1, 1, 1), "ct74": ("SP2", 1, 1, 1, 1, 1), "ct75": ("SP2e", 1, 1, 1, 1, 1)}
V, _, _, used = b.sources(["5e", "SP", "SP2", "SP2e"]); assert set(used) >= {"5e", "SP", "SP2", "SP2e"}, used; Fd = {}
for v in FOLDS:
    y, n, pid, q = load_fold(v); gk = list(q["game_keys"]); nq = len(y); Tn = np.array([b.TN[int(m)] for m in n])[:, None]

    def qsum(ev):
        L = np.zeros((nq, 10))
        for i, g_ in enumerate(gk):
            for k in g_:
                if k in ev:
                    L[i] += ev[k]
        return L
    Ls = {}
    for name, (idn, enc, cc, temp, pi0, bmin) in RECIPES.items():
        L = qsum(evidence(V[idn], v, cc, pi0, bmin, R_MAIN[idn]))
        if enc:
            L = L + b.U_ENC * qsum(evidence(V["enc"], v, 0, pi0, bmin, R_ENC))
        Ls[name] = L * Tn if temp else L
    Fd[v] = (np.asarray(y), np.asarray(n), np.asarray(pid), Ls)
y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); tot = {k: [0, 0.0, 0, 0] for k in ("reversal", "new change", "other")}; nq_all = 0
print("CV under the test rules (class-balanced paired differences; n72 / n74 / n75 = the no-reversal composition on top of the ct44 recipe):")
for seeds in ((0, 1, 2), (6, 7, 8)):
    S = {k: [] for k in ("ct44", "ct72", "ct74", "ct75", "n72", "n74", "n75")}; D = {k: [] for k in S}
    for v in FOLDS:
        yv, nv, _, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); dec = {}
        for name in RECIPES:
            d0, _, dec[name], _, _ = mb.decisions(P, Ls[name], 0.5, 3.0, 0.6)
        for name in ("ct72", "ct74", "ct75"):
            dec["n" + name[2:]] = np.where(dec["ct44"] != d0, dec["ct44"], dec[name])
        for k in S:
            S[k].append(b.SM[dec[k], yv]); D[k].append(dec[k])
        gain = b.SM[dec["ct75"], yv] - b.SM[dec["ct44"], yv]; dN, dR = dec["ct75"], dec["ct44"]; nq_all += len(yv)
        for k, sel in (("reversal", (dN != dR) & (dN == d0)), ("new change", (dN != dR) & (dR == d0)), ("other", (dN != dR) & (dN != d0) & (dR != d0))):
            tot[k][0] += int(sel.sum()); tot[k][1] += float(gain[sel].sum()); tot[k][2] += int((gain[sel] > 0).sum()); tot[k][3] += int((gain[sel] < 0).sum())
    print(f"seeds {seeds}:")
    for a_, r_ in (("ct72", "ct44"), ("ct74", "ct44"), ("ct75", "ct44"), ("n72", "ct44"), ("n74", "ct44"), ("n75", "ct44"), ("n75", "n72"), ("n74", "n72"), ("n75", "n74"), ("ct75", "ct72"), ("ct75", "n75")):
        dd = paired_diff(np.concatenate(S[a_]), np.concatenate(S[r_]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(x, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(w, Fd[v][0], Fd[v][1], sizes=SIZES) for x, w, v in zip(S[a_], S[r_], FOLDS)]
        print(f"  {a_:5s} vs {r_:5s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  decisions differ {float((np.concatenate(D[a_]) != np.concatenate(D[r_])).mean()):.3f}", flush=True)
print("\nct75 recipe against ct44 recipe, CV under the test rules, seeds 0-2 and 6-8 pooled:")
for k, (c, g, up, dn) in tot.items():
    print(f"  {k:11s} share of questions {c / nq_all:.4f}; new better {up / max(c, 1):.3f}, worse {dn / max(c, 1):.3f}; mean gain per question {g / max(c, 1):+.3f}; contribution to the mean score {g / nq_all:+.4f}")
print(f"  total (question-weighted, not class-balanced) {sum(x[1] for x in tot.values()) / nq_all:+.4f}")

print("\nfiles:")
sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank  # noqa: E731
c44 = sub("ct44"); q = c44.index.to_numpy(); pub = np.char.startswith(q.astype(str), "pub_"); F = {k: sub(k).reindex(q).to_numpy() for k in ("ct28", "ct44", "ct72", "ct73", "ct74", "ct74n", "ct75", "ct75n")}; rk = np.vectorize(b.RANK.get)
for k in ("ct74n", "ct75n", "ct74", "ct75"):
    f = ROOT / f"submission_rank_{k}.csv"; raw = f.read_bytes(); x = F[k]; ok = len(x) == 800 and set(x) <= set(b.RANK) and b"\r" not in raw and raw == (ROOT / "upload_next" / f.name).read_bytes()
    print(f"{k}: md5 {hashlib.md5(raw).hexdigest()[:12]}; 800 rows, valid levels, LF line ends, same bytes in upload_next: {ok}; public / private questions: {int(pub.sum())} / {int((~pub).sum())}")
    for o in ("ct73", "ct44", "ct28", "ct72", "ct75" if k != "ct75" else "ct74"):
        ch = x != F[o]; j = np.abs(rk(x) - rk(F[o])); print(f"   vs {o}: {int(ch.sum())} differ (public {int((ch & pub).sum())}, private {int((ch & ~pub).sum())}); one level {int((j == 1).sum())}, two or more {int((j >= 2).sum())}")
for k, full in (("ct74n", "ct74"), ("ct75n", "ct75")):
    keep = F["ct44"] != F["ct28"]; print(f"{k}: no-reversal property (equals ct44 on the {int(keep.sum())} questions ct44 changed, equals {full} elsewhere): {bool((F[k][keep] == F['ct44'][keep]).all() and (F[k][~keep] == F[full][~keep]).all())}")
ch = F["ct75n"] != F["ct73"]; mv = Counter(f"{a_}->{b_}" for a_, b_ in zip(F["ct73"][ch], F["ct75n"][ch]))
print("ct75n vs ct73, moves (ct73 -> ct75n): " + ", ".join(f"{k} {c}" for k, c in mv.most_common()) + f"; public {int((ch & pub).sum())}: " + ", ".join(f"{a_}->{b_}" for a_, b_ in zip(F["ct73"][ch & pub], F["ct75n"][ch & pub])))
new73 = F["ct73"] != F["ct44"]; new75 = F["ct75n"] != F["ct44"]; print(f"new changes on top of ct44: ct73 {int(new73.sum())} (public {int((new73 & pub).sum())}), ct75n {int(new75.sum())} (public {int((new75 & pub).sum())}); shared with the same answer {int((new73 & new75 & (F['ct73'] == F['ct75n'])).sum())}, only in ct73 {int((new73 & ~new75).sum())}, only in ct75n {int((new75 & ~new73).sum())}, both changed but differently {int((new73 & new75 & (F['ct73'] != F['ct75n'])).sum())}")
