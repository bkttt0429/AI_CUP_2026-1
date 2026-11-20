"""Section 4M M86 / M88 follow-up (2026-11-12, report only): the recipes of ct71 and ct72 scored in CV under the TEST-side rules
(weights x density ratio, decisions only inside the boundary region, every band with a positive weight), against ct44's recipe under
its own test rules (bands >= q90). Same construction as m86c_testmimic.py.
  python -I logs/nano4/m86e_testmimic.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m76_boundary as mb  # noqa: E402
import m83_classcond as m83  # noqa: E402
import m86_bundle as b  # noqa: E402
import m87_pi0 as m87  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

R_MAIN = {"5e": 0.851, "P3": 0.836, "SP": 0.875}; R_ENC = 0.713


def evidence(z, v, cc, pi0, band_min, scale):
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr)
    if pi0:
        _, cal, _ = m87.corrected(z, tr)
    R = m83.tables(z, tr, cal, band_all, g_all, m83.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]; out[s_] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out))


RECIPES = {"ct44 recipe (5e, >= q90)": ("5e", 0, 0, 0, 0, 3), "ct71 recipe (P3 + four switches, all bands)": ("P3", 1, 1, 1, 1, 1), "ct72 recipe (SP + four switches, all bands)": ("SP", 1, 1, 1, 1, 1), "ct68 recipe (SP only, >= q90)": ("SP", 0, 0, 0, 0, 3)}
V, _, _, used = b.sources(["5e", "P3", "SP"]); Fd = {}
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
names = list(RECIPES)
for seeds in ((0, 1, 2), (6, 7, 8)):
    S = {name: [] for name in RECIPES}; ch = {name: [] for name in RECIPES}
    for v in FOLDS:
        y, n, pid, Ls = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        for name, L in Ls.items():
            d0, _, dAr, _, _ = mb.decisions(P, L, 0.5, 3.0, 0.6); S[name].append(b.SM[dAr, y]); ch[name].append(dAr != d0)
    y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); print(f"seeds {seeds}: under the test rules")
    for a_, r_ in ((names[1], names[0]), (names[2], names[0]), (names[3], names[0]), (names[2], names[1])):
        dd = paired_diff(np.concatenate(S[a_]), np.concatenate(S[r_]), y, pid, n, sizes=SIZES, n=3000); pf = [balanced_mean_over_n(x, Fd[v][0], Fd[v][1], sizes=SIZES) - balanced_mean_over_n(w, Fd[v][0], Fd[v][1], sizes=SIZES) for x, w, v in zip(S[a_], S[r_], FOLDS)]
        print(f"  {a_.split(' (')[0]:12s} vs {r_.split(' (')[0]:12s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()))
