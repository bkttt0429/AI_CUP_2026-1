#!/usr/bin/env python
"""Section 4M M137e (2026-11-14): the learned pool's changes by kind, and two more guards (design doc M137e).

Cross-validation, both seed sets, the L0 pool of M137b (no level offset) under ct104b's guards against the ct104b rule:
the changes by group (leaves D; two or more levels; undoes an M129 change back to the ct75u rule's answer; other one-level), by
the old answer's level and by the new answer's level - share of questions and mean gain per changed question. Then the guard
variants: + G_D (no change out of D), + G_2 (no change of two or more levels), both; each against the ct104b rule (M37), and
its cost against the pool without the new guard.

  python -I logs/nano4/m137e_kinds.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402  (fit, apply, the constants)
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
A_, D5, D6, D_ = LEVELS.index("A"), LEVELS.index("5D"), LEVELS.index("6D"), LEVELS.index("D")


def guard(d_new, d_old, out_of=(A_,), no_pair=((D5, D6),), max_jump=None):
    d = d_new.copy()
    for o in out_of:
        m = (d_old == o) & (d_new != d_old); d[m] = d_old[m]
    for o, w_ in no_pair:
        m = (d_old == o) & (d_new == w_); d[m] = d_old[m]
    if max_jump is not None:
        m = np.abs(d_new - d_old) > max_jump; d[m] = d_old[m]
    return d


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])
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
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    GV = {"current guards": {}, "+ G_D": dict(out_of=(A_, D_)), "+ G_2": dict(max_jump=1), "+ G_D + G_2": dict(out_of=(A_, D_), max_jump=1)}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X, S, DD = {}, {}, {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["du"], f["dc"] = du, guard(db, du)
            S.setdefault("ct104b_rule", []).append(SM[f["dc"], f["y"]])
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = mb7.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), "L0")
            ds = np.asarray(b.decide(mb7.apply(coef, X[v], Fd[v]["n"])), int)
            for g, kw in GV.items():
                dg = guard(ds, Fd[v]["du"], **kw); S.setdefault(g, []).append(SM[dg, Fd[v]["y"]]); DD.setdefault(g, []).append(dg)
        dc, du = np.concatenate([Fd[v]["dc"] for v in FOLDS]), np.concatenate([Fd[v]["du"] for v in FOLDS]); dg = np.concatenate(DD["current guards"]); gain = np.concatenate(S["current guards"]) - np.concatenate(S["ct104b_rule"]); ch = dg != dc
        print(f"seeds {seeds}: L0 + current guards changes {ch.mean():.3f} of the questions against the ct104b rule; mean gain per changed question {gain[ch].mean():+.3f}")
        grp = np.where(dc == D_, "leaves D", np.where(np.abs(dg - dc) >= 2, "two or more levels", np.where((dc != du) & (dg == du), "undoes an M129 change", "other one-level")))
        for g in ("leaves D", "two or more levels", "undoes an M129 change", "other one-level"):
            m = ch & (grp == g); print(f"    {g:24s}: {m.mean():.4f} of the questions ({int(m.sum())}), mean gain per changed question {gain[m].mean():+.3f}, by n " + " ".join(f"{s}: {gain[m & (n == s)].mean():+.3f} ({int((m & (n == s)).sum())})" for s in SIZES))
        print("    by old answer: " + "  ".join(f"{LEVELS[c]} {gain[ch & (dc == c)].mean():+.3f} ({int((ch & (dc == c)).sum())})" for c in range(10)))
        print("    by new answer: " + "  ".join(f"{LEVELS[c]} {gain[ch & (dg == c)].mean():+.3f} ({int((ch & (dg == c)).sum())})" for c in range(10)))
        m = ch & (dc == D_); print("    leaving D, by new answer: " + "  ".join(f"{LEVELS[c]} {gain[m & (dg == c)].mean():+.3f} ({int((m & (dg == c)).sum())})" for c in range(1, 10) if (m & (dg == c)).any()))
        out = {}
        for g in GV:
            A1, B1 = np.concatenate(S[g]), np.concatenate(S["ct104b_rule"]); dd = paired_diff(A1, B1, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[g], S["ct104b_rule"], ys, ns)]
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); cost = bal(S[g]) - bal(S["current guards"])
            print(f"  L0 {g:16s}: {bal(S[g]):.4f}; against the ct104b rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" changed {float((A1 != B1).mean()):.3f} {'PASS' if good else 'no'}; cost of the added guard {cost:+.4f}")
            out[g] = {**dd, "per_fold": pf, "meets_m37": bool(good), "cost": cost}
        rep[f"seeds_{seeds[0]}"] = out
    (ROOT / "results/r_variants/m137e_kinds.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m137e_kinds.json")


if __name__ == "__main__":
    main()
