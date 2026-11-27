#!/usr/bin/env python
"""Section 4M M174 diagnostic (2026-11-15, report only): is the handicap composition of a question (even / one-stone as Black /
one-stone as White) evidence about the level beyond the base ensemble? (design doc M174)

  python -I logs/nano4/m174_hd_composition.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, s2, ES, FOLDS, b, SM = mb7.m3, mb7.s2, mb7.ES, mb7.FOLDS, mb7.b, mb7.SM; RV = ROOT / "results/r_variants"


def main():
    import pandas as pd
    ht = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}); row = {k: i for i, k in enumerate(ht.game_key)}; hd = ht.hd.to_numpy().astype(int); tb = (ht.color == "B").to_numpy(); D = {}
    for v in FOLDS:
        y, n, pid, qd = m3.load_fold(v); y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); K = np.zeros((len(y), 4))
        for i, g_ in enumerate(qd["game_keys"]):
            r = np.array([row[k] for k in g_], dtype=np.int64); K[i] = [(hd[r] == 0).sum(), ((hd[r] == 1) & tb[r]).sum(), ((hd[r] == 1) & ~tb[r]).sum(), (hd[r] > 1).sum()]
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); D[v] = dict(y=y, n=n, pid=pid, K=K, lp=np.log(np.maximum(P, 1e-12)))
    yy = np.concatenate([D[v]["y"] for v in FOLDS]); KK = np.concatenate([D[v]["K"] for v in FOLDS]); print(f"questions {len(yy)}; games per kind (even / one stone Black / one stone White / more stones): " + " / ".join(f"{x:.3f}" for x in KK.sum(0) / KK.sum()))
    print("composition by true level (D..6D), share of one-stone games | of these, as Black: " + "  ".join(f"{(KK[yy == c, 1:3].sum() / KK[yy == c, :3].sum()):.3f}|{(KK[yy == c, 1].sum() / max(KK[yy == c, 1:3].sum(), 1)):.3f}" for c in range(10)))
    out = {"base": [], "new": []}; ll = {"base": 0.0, "new": 0.0}; nq = 0; gam = {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; Kt = np.concatenate([D[u]["K"] for u in tr])[:, :3]; yt = np.concatenate([D[u]["y"] for u in tr]); nt = np.concatenate([D[u]["n"] for u in tr]); lpt = np.concatenate([D[u]["lp"] for u in tr])
        th = np.stack([Kt[yt == c].sum(0) + 1.0 for c in range(10)]); th /= th.sum(1, keepdims=True); Lt = Kt @ np.log(th).T; Lv = D[v]["K"][:, :3] @ np.log(th).T; Pn = np.zeros_like(D[v]["lp"])
        for s in SIZES:
            m = nt == s; cw = np.zeros(m.sum())
            for c in range(10):
                mc = yt[m] == c; cw[mc] = 1.0 / max(mc.sum(), 1)
            best = (-1e18, 0.0)
            for g in np.arange(0.0, 1.51, 0.1):  # class-balanced log-likelihood on the three training folds
                zt = lpt[m] + g * Lt[m]; zt -= zt.max(1, keepdims=True); lpz = zt - np.log(np.exp(zt).sum(1, keepdims=True)); val = float((cw * lpz[np.arange(m.sum()), yt[m]]).sum())
                if val > best[0]:
                    best = (val, float(g))
            gam.setdefault(s, []).append(best[1]); mv = D[v]["n"] == s; zv = D[v]["lp"][mv] + best[1] * Lv[mv]; zv -= zv.max(1, keepdims=True); Pn[mv] = np.exp(zv) / np.exp(zv).sum(1, keepdims=True)
        Pb = np.exp(D[v]["lp"]); Pb /= Pb.sum(1, keepdims=True); r = np.arange(len(D[v]["y"])); ll["base"] += float(np.log(Pb[r, D[v]["y"]]).sum()); ll["new"] += float(np.log(np.maximum(Pn[r, D[v]["y"]], 1e-12)).sum()); nq += len(r)
        out["base"].append(SM[np.asarray(b.decide(Pb), int), D[v]["y"]]); out["new"].append(SM[np.asarray(b.decide(Pn), int), D[v]["y"]])
    ys = [D[v]["y"] for v in FOLDS]; ns = [D[v]["n"] for v in FOLDS]; bal = lambda S_: float(np.mean([balanced_mean_over_n(x, a, c, sizes=SIZES) for x, a, c in zip(S_, ys, ns)]))  # noqa: E731
    d_ = paired_diff(np.concatenate(out["new"]), np.concatenate(out["base"]), yy, np.concatenate([D[v]["pid"] for v in FOLDS]), np.concatenate(ns), sizes=SIZES, n=3000)
    print(f"gamma chosen per size (four fits): " + "  ".join(f"n={s}: {gam[s]}" for s in SIZES))
    print(f"base ensemble decisions {bal(out['base']):.4f} -> with the composition evidence {bal(out['new']):.4f}: {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}]  by n " + " ".join(f"{m_}: {x:+.4f}" for m_, x in d_["by_n"].items()) + f"; mean log-likelihood per question {ll['base'] / nq:.4f} -> {ll['new'] / nq:.4f}")


if __name__ == "__main__":
    main()
