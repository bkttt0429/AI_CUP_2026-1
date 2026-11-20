#!/usr/bin/env python
"""Section 4M M147 (2026-11-15): do the short-question specialist members (M69: C / T-pc-f0-s5-10-x40-oh, trained on 5-10 game
questions only; base level +0.0014 / +0.0019 in test weights, real but below the threshold) survive the layers above the base?

The base probabilities of the questions with n <= 10 become the mean of the six standard members and the six specialists (M69's
deployable form); n = 20 unchanged. The L0 pool is refitted on the new base, the four guards are taken relative to the ct106 rule
computed with the current base. REGISTERED: discovery = standard seeds 0-2 + specialists 0-2 against standard 0-2; replication =
standard 6-8 + specialists 3-5 against standard 6-8; M37 on both. Report: the mix only for n = 5, specialists alone, by size.

  python -I logs/nano4/m147_short_final.py
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
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE; SHORT = ("C-pc-f0-s5-10-x40-oh", "T-pc-f0-s5-10-x40-oh")


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
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
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw); print(f"fold {v}: inputs ready", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; ok = []
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    mem = lambda v, arms, seeds: [np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds]  # noqa: E731

    def pool_decisions(X):
        out = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), ())
            out[v] = np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], ())), int)
        return out
    for tag, std_seeds, sh_seeds in (("discovery", (0, 1, 2), (0, 1, 2)), ("replication", (6, 7, 8), (3, 4, 5))):
        X = {}; Pstd, Psh = {}, {}
        for v in FOLDS:
            f = Fd[v]; S_ = mem(v, s2.STD, std_seeds); H_ = mem(v, SHORT, sh_seeds); assert all(len(p) == len(f["y"]) for p in S_ + H_); Pstd[v] = np.mean(S_, axis=0); Psh[v] = (np.mean(S_ + H_, axis=0), np.mean(H_, axis=0))
            X[v] = {"logp": np.log(np.maximum(Pstd[v], 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(Pstd[v], f["Ls"][0]); _, db = mf.rule75u(la.pooled(Pstd[v], f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; bdec = lambda P: np.asarray(b.decide(P), int)  # noqa: E731
        print(f"\n{tag}: standard seeds {std_seeds}, specialists {sh_seeds}; ct106 rule {bal(S_old):.4f}; base decision {bal([SM[bdec(Pstd[v]), Fd[v]['y']] for v in FOLDS]):.4f}")
        out = {}
        for name, which, sizes_ in (("PRIMARY: n <= 10 use standard + specialists", 0, (5, 10)), ("only n = 5 uses the mix", 0, (5,)), ("n <= 10 use the specialists alone", 1, (5, 10))):
            Pn = {v: np.where(np.isin(Fd[v]["n"], sizes_)[:, None], Psh[v][which], Pstd[v]) for v in FOLDS}; Xn = {v: {**X[v], "logp": np.log(np.maximum(Pn[v], 1e-12))} for v in FOLDS}; dd = pool_decisions(Xn)
            new_ = [SM[mf7.guard2(dd[v], ref[v]), Fd[v]["y"]] for v in FOLDS]; A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]; good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            bb = [SM[bdec(Pn[v]), Fd[v]["y"]] for v in FOLDS]; b0 = [SM[bdec(Pstd[v]), Fd[v]["y"]] for v in FOLDS]; db_ = paired_diff(np.concatenate(bb), np.concatenate(b0), y, pid, n, sizes=SIZES, n=2000)
            tw = 0.25 * d_["by_n"][5] + 0.375 * d_["by_n"][10] + 0.375 * d_["by_n"][20] if 5 in d_["by_n"] else float("nan")
            print(f"  {name:44s}: final {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items()) + f"  in test weights {tw:+.4f}  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'} | base level {db_['diff']:+.4f} by n " + " ".join(f"{m}: {x:+.4f}" for m, x in db_["by_n"].items()), flush=True)
            out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good), "test_weighted": tw, "base_level": db_}
            if name.startswith("PRIMARY"):
                ok.append(bool(good))
        rep[tag] = out
    passed = all(ok); print(f"\nREGISTERED (M37 on the discovery and the replication set): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (ROOT / "results/r_variants/m147_short_final.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m147_short_final.json")


if __name__ == "__main__":
    main()
