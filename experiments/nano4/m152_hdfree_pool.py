#!/usr/bin/env python
"""Section 4M M152 (2026-11-15): the test set holds no game with a handicap of two stones or more (validation: 4.85% of the
games); does fitting the pool on test-like questions (no such game) change anything? (design doc M152)

Validation questions are split into "clean" (no game with HD >= 2, like every test question) and the rest. Reported: the share of
clean questions by size, the scores of the base and of the ct106 rule on each part, the pool coefficients fitted on all questions
against those fitted on the clean ones. REGISTERED: on the clean questions of the held-out fold, the pool fitted on the clean
questions of the other three folds against the pool fitted on all their questions (both under the four guards relative to the
ct106 rule), M37 on seeds 0-2 and 6-8 -> on a pass the test-side coefficients are refitted on clean questions.

  python -I logs/nano4/m152_hdfree_pool.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m141_opp_lookalike as m41  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); hd = np.asarray(z["hd"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    Lg = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = Lg * Tn if temp else Lg
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        nh = np.zeros(len(y)); np.add.at(nh, qi, (hd[gi] >= 2).astype(float)); found = np.bincount(qi, minlength=len(y)); clean = (nh == 0) & (found == n)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), clean=clean); print(f"fold {v}: inputs ready; clean questions " + " ".join(f"n={s}: {float(clean[n == s].mean()):.3f}" for s in SIZES) + f"; questions with all their games in the query table {float((found == n).mean()):.3f}", flush=True)

    def cwt(y, n, m):
        w = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                k = m & (n == s) & (y == c); w[k] = 1.0 / max(int(k.sum()), 1)
        return w
    rep = {}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}; f["P"] = P
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        S = {"all": [], "clean": []}; cf = {"all": [], "clean": []}; base = []
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; yt = np.concatenate([Fd[u]["y"] for u in tr]); nt = np.concatenate([Fd[u]["n"] for u in tr]); ct = np.concatenate([Fd[u]["clean"] for u in tr]); Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            c_all = m41.fit(Xt, yt, nt, cwt(yt, nt, np.ones(len(yt), bool)), ()); ref = mf7.guard2(np.asarray(b.decide(m41.apply(c_all, X[v], Fd[v]["n"], ())), int), Fd[v]["dc"])
            Xc = {k: x[ct] for k, x in Xt.items()}; c_cl = m41.fit(Xc, yt[ct], nt[ct], cwt(yt[ct], nt[ct], np.ones(int(ct.sum()), bool)), ()); new = mf7.guard2(np.asarray(b.decide(m41.apply(c_cl, X[v], Fd[v]["n"], ())), int), ref)
            S["all"].append(SM[ref, Fd[v]["y"]]); S["clean"].append(SM[new, Fd[v]["y"]]); cf["all"].append(c_all); cf["clean"].append(c_cl); base.append(SM[np.asarray(b.decide(Fd[v]["P"]), int), Fd[v]["y"]])
        cl = [Fd[v]["clean"] for v in FOLDS]; ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; pids = [Fd[v]["pid"] for v in FOLDS]
        bal = lambda S_, sel: float(np.mean([balanced_mean_over_n(x[m], yy[m], nn[m], sizes=SIZES) for x, yy, nn, m in zip(S_, ys, ns, sel)]))  # noqa: E731
        nc = [~m for m in cl]
        print(f"\nseeds {seeds}: base decision on clean / other questions {bal(base, cl):.4f} / {bal(base, nc):.4f}; ct106 rule {bal(S['all'], cl):.4f} / {bal(S['all'], nc):.4f} (all questions {float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S['all'], ys, ns)])):.4f})")
        A_ = np.concatenate([x[m] for x, m in zip(S["clean"], cl)]); B_ = np.concatenate([x[m] for x, m in zip(S["all"], cl)]); yy = np.concatenate([x[m] for x, m in zip(ys, cl)]); nn = np.concatenate([x[m] for x, m in zip(ns, cl)]); pp = np.concatenate([x[m] for x, m in zip(pids, cl)])
        dd = paired_diff(A_, B_, yy, pp, nn, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(x[m], y_[m], n_[m], sizes=SIZES) - balanced_mean_over_n(c[m], y_[m], n_[m], sizes=SIZES)) for x, c, y_, n_, m in zip(S["clean"], S["all"], ys, ns, cl)]
        good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); ok.append(bool(good))
        print(f"  pool fitted on clean questions against the pool fitted on all, judged on the clean questions: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}")
        for s in SIZES:
            a = np.mean([c[s] for c in cf["all"]], 0); c_ = np.mean([c[s] for c in cf["clean"]], 0); print(f"    n={s} coefficients (all | clean): alpha {a[0]:.2f} | {c_[0]:.2f}; v44 {a[1]:.2f} | {c_[1]:.2f}; v5 {a[3]:.2f} | {c_[3]:.2f}; s2 {a[4]:.2f} | {c_[4]:.2f}; look-alike {a[7]:.2f} | {c_[7]:.2f}")
        rep[f"seeds_{seeds[0]}"] = {**dd, "per_fold": pf, "meets_m37": bool(good)}
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (ROOT / "results/r_variants/m152_hdfree_pool.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m152_hdfree_pool.json")


if __name__ == "__main__":
    main()
