#!/usr/bin/env python
"""Section 4M M142 (2026-11-15): two small items without new data (design doc M142).

(a) the base probabilities averaged over more seeds (current 3 -> 15; also 6 and 30) inside the L0 pool, the four guards relative
    to the ct106 rule computed with the current three seeds; two disjoint replications (seeds 0-14 vs 0-2, 15-29 vs 15-17), M37 each;
(b) screening: one more identifier's anchor evidence as an extra pool input (SP4e, S4e, SP2y, P3, SP2x), against the ct106 rule,
    seeds 0-2 and 6-8, report only.

  python -I logs/nano4/m142_bundle3.py
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
import m141_opp_lookalike as m41  # noqa: E402  (fit / apply with extra inputs)
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
MORE = ("SP4e", "S4e", "SP2y", "P3", "SP2x")


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], *MORE]); more = [x for x in MORE if x in used]; print("extra identifiers available:", more)
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}

        def ev_of(spec):
            if spec not in cache:
                idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
            return cache[spec]
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}
        Lx = {x: ev_of((x, 1, 1, 1, 1, 1, 0.85)) for x in more}
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)
        print(f"fold {v}: inputs ready", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    base_of = lambda v, seeds: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # noqa: E731

    def pool_decisions(X, extra=()):
        out = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra)
            out[v] = (np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], extra)), int), coef)
        return out

    def current(seeds):
        X, ref = {}, {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"]}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd = pool_decisions(X)
        for v in FOLDS:
            ref[v] = mf7.guard2(dd[v][0], Fd[v]["dc"])
        return X, ref

    def report(tag, new, old, m37=True):
        A_, B_ = np.concatenate(new), np.concatenate(old); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
        pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new, old, ys, ns)]
        good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
        print(f"  {tag:44s}: {bal(new):.4f} vs {bal(old):.4f}  {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f}" + (f" {'PASS' if good else 'no'}" if m37 else ""), flush=True)
        return {**dd, "per_fold": pf, "meets_m37": bool(good)}
    print("\n(a) more seeds in the base (pool refitted; guards relative to the ct106 rule of the current three seeds)")
    for cur, sets in (((0, 1, 2), {"6 seeds (0-5)": range(6), "15 seeds (0-14)  [registered A]": range(15), "30 seeds (0-29)": range(30)}), ((15, 16, 17), {"6 seeds (15-20)": range(15, 21), "15 seeds (15-29) [registered B]": range(15, 30)})):
        X, ref = current(cur); S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f" current base = seeds {cur}: ct106 rule {bal(S_old):.4f}; base decision {bal([SM[np.asarray(b.decide(np.exp(X[v]['logp'])), int), Fd[v]['y']] for v in FOLDS]):.4f}")
        for tag, seeds in sets.items():
            Xn = {v: {**X[v], "logp": np.log(np.maximum(base_of(v, tuple(seeds)), 1e-12))} for v in FOLDS}; dd = pool_decisions(Xn)
            rep[f"a {cur} -> {tag}"] = report(tag, [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS], S_old)
            print(f"      base decision with these seeds {bal([SM[np.asarray(b.decide(np.exp(Xn[v]['logp'])), int), Fd[v]['y']] for v in FOLDS]):.4f}; alpha " + " ".join(f"n={s}: {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}" for s in SIZES))
    print("\n(b) one more identifier's evidence as an extra pool input (report only)")
    for cur in ((0, 1, 2), (6, 7, 8)):
        X, ref = current(cur); S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f" seeds {cur}: ct106 rule {bal(S_old):.4f}")
        for x in more:
            dd = pool_decisions(X, (x,)); rep[f"b {cur} + {x}"] = report(f"+ {x}", [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS], S_old, m37=False)
            print(f"      coefficient of the new source " + " ".join(f"n={s}: {np.mean([dd[v][1][s][8] for v in FOLDS]):+.2f}" for s in SIZES))
        if more:
            dd = pool_decisions(X, tuple(more)); rep[f"b {cur} + all"] = report("+ all of them", [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS], S_old, m37=False)
    (ROOT / "results/r_variants/m142_bundle3.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m142_bundle3.json")


if __name__ == "__main__":
    main()
