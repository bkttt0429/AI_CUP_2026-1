"""Section 4M M86 follow-up (2026-11-11, report only): how much of the bundles' CV gain survives the TEST-side rules? The CV comparisons
apply the evidence of every margin band at full weight and decide on all levels; the test files use only bands >= q90, weights times the
density ratio, and change decisions only inside the boundary region. Here every recipe is scored in CV under those test rules (no
thinning), against ct44's recipe under the same rules; the band floor is varied for the bundle with corrected weights.
  python -I logs/nano4/m86c_testmimic.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
sys.argv = [sys.argv[0], "--v2"]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

R_MAIN = {"5e": 0.851, "P3": 0.836}; R_ENC = 0.713


def evidence(z, v, cc, pi0, band_min, scale):
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)
    if pi0:
        _, cal, _ = m87.corrected(z, tr)
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out))


RECIPES = {"ct44 recipe (5e, >= q90)": ("5e", 0, 0, 0, 0, 3), "ct64 recipe (four switches, >= q90)": ("P3", 1, 1, 1, 0, 3), "ct70 recipe (five switches, >= q90)": ("P3", 1, 1, 1, 1, 3),
           "five switches, >= q80": ("P3", 1, 1, 1, 1, 2), "five switches, >= q50": ("P3", 1, 1, 1, 1, 1), "ct44 + corrected weights only, >= q90": ("5e", 0, 0, 0, 1, 3), "ct44 + corrected weights only, >= q50": ("5e", 0, 0, 0, 1, 1)}
V, _, _ = b.sources(); Fd = {}
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
for seeds in ((0, 1, 2), (6, 7, 8)):
    S = {name: [] for name in RECIPES}; ch = {name: [] for name in RECIPES}
    for v in FOLDS:
        y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        for name, L in Ls.items():
            d0, _, dAr, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); S[name].append(b.SM[dAr, y]); ch[name].append(dAr != d0)
    y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); ref = next(iter(RECIPES)); print(f"seeds {seeds}: under the test rules (weights x density ratio, region-only decisions), difference to the ct44 recipe:")
    for name in RECIPES:
        if name == ref:
            print(f"  {name:42s} (reference; changes {float(np.concatenate(ch[name]).mean()):.3f} of the questions)"); continue
        dd = paired_diff(np.concatenate(S[name]), np.concatenate(S[ref]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(a_, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(r_, Fd[v][0], Fd[v][1], sizes=SIZES) for a_, r_, v in zip(S[name], S[ref], FOLDS)]
        print(f"  {name:42s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changes {float(np.concatenate(ch[name]).mean()):.3f}")
