#!/usr/bin/env python
"""Section 4M M157c (2026-11-15, report only): the payoff table rule x true composition, on top of the rule the test file already uses.

Original docstring of M157:
Section 4M M157 (2026-11-15, report only): how much is the test set's level composition worth, and which composition do the
recorded Public readings support? (design doc M157)

M153: the training data are 100,000 games per level; the number of players per level differs a lot (A 1,582 ... 1D 433). If
test players were drawn per player, the test composition follows the players (A 20%, 1D 5%), not the levels (10% each). The
pipeline decides with class-balanced posteriors plus two Public-confirmed pushes.
(a) stakes, in cross-validation: for a composition pi, validation questions are weighted to pi (per question size); the ct106
    rule's answers against the answers of the same pool posterior multiplied by pi (expected-score decision; plain, and under
    the four guards). Player bootstrap. Compositions: balanced, by the number of training players, players with 1-25 games,
    M68's estimate.
(b) the Public ledger: every recorded upload solved jointly with prior = pool posterior x pi; the cost of the best assignment
    (sum of -log prior over the constrained questions) per composition - lower fits the recorded readings better - and the level
    counts of that assignment.

  python -I logs/nano4/m157_prior_stakes.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m141_opp_lookalike as m41  # noqa: E402
import m58_decode as led  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE; TW = {5: 0.25, 10: 0.375, 20: 0.375}


def main():
    pfc = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str); lv = pfc["rank"].map(b.RANK).to_numpy().astype(int); ng = pfc.n_games.astype(int).to_numpy()
    PI = {"balanced": np.full(10, 0.1), "by players": np.bincount(lv, minlength=10) / len(lv), "players with 1-25 games": np.bincount(lv[ng <= 25], minlength=10) / (ng <= 25).sum(),
          "M68 estimate": np.array([.123, .110, .134, .113, .085, .077, .091, .097, .092, .078])}
    PI["M68 estimate"] /= PI["M68 estimate"].sum(); PI["half-way to players"] = (PI["balanced"] + PI["by players"]) / 2
    for k, v in PI.items():
        print(f"composition {k:26s}: " + " ".join(f"{LEVELS[c]} {v[c]:.3f}" for c in range(10)) + f" | kyu {v[:4].sum():.3f}")
    # ---------------------------------------------------------------- (a) cross-validation
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; zz = V["5e"]; row = {k: i for i, k in enumerate(zz["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); hd = np.asarray(zz["hd"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pfc.set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}
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
        nh = np.zeros(len(y)); np.add.at(nh, qi, (hd[gi] >= 2).astype(float)); found = np.bincount(qi, minlength=len(y))
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); R = lr_ - lr_.mean(0, keepdims=True); wq = np.array([W_FIN[int(m)] for m in n]); d0, du = mf.rule75u(P, Ls[0]); _, db = mf.rule75u(la.pooled(P, R, wq), Ls[2])
        Fd[v] = dict(y=y, n=n, pid=pid, cw=cw, clean=(nh == 0) & (found == n), dc=mb7.keep_old(db, du), X={"logp": np.log(np.maximum(P, 1e-12)), "R": R, **{k: Ls[2][k] for k in NAMES}}); print(f"fold {v}: inputs ready", flush=True)
    PV, DV = {}, {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([Fd[u]["X"][k] for u in tr]) for k in Fd[v]["X"]}
        coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), ()); PV[v] = m41.apply(coef, Fd[v]["X"], Fd[v]["n"], ()); DV[v] = mf7.guard2(np.asarray(b.decide(PV[v]), int), Fd[v]["dc"])
    P_, d_, y_, n_, pid_, cl_ = (np.concatenate([x[v] for v in FOLDS]) for x in (PV, DV, {v: Fd[v]["y"] for v in FOLDS}, {v: Fd[v]["n"] for v in FOLDS}, {v: Fd[v]["pid"] for v in FOLDS}, {v: Fd[v]["clean"] for v in FOLDS}))
    upid, pinv = np.unique(pid_, return_inverse=True); rng = np.random.default_rng(0); rep = {}; A_, D1 = 3, 4

    def wmean(s, w):
        return float(sum(TW[m] * (s[n_ == m] * w[n_ == m]).sum() / w[n_ == m].sum() for m in SIZES))

    def push(d_cur, r):
        Q_ = P_.copy(); m = P_[:, D1] + P_[:, A_]; s_ = P_[:, D1] / np.maximum(P_[:, D1] + r * P_[:, A_], 1e-300); Q_[:, D1] = m * s_; Q_[:, A_] = m * (1 - s_); dq = np.asarray(b.decide(Q_), int); d = d_cur.copy(); f = (d_cur == D1) & (dq == A_); d[f] = A_; return d

    def tilt(pi, ref, cap_a=None):
        w_ = pi.copy()
        if cap_a is not None:  # the A : 1D ratio of the composition limited to cap_a
            w_ = w_.copy(); w_[A_] = min(w_[A_], cap_a * w_[D1])
        Pp = P_ * w_[None, :]; Pp /= Pp.sum(1, keepdims=True); return mf7.guard2(np.asarray(b.decide(Pp), int), ref)
    R1 = push(d_, 2.0)
    RULES = {"R0 ct106 rule (no push)": d_, "R1 = R0 + 1D->A push r 2 (what the test file has)": R1, "R2 = R0 + 1D->A push r 3": push(d_, 3.0), "R2b = R0 + 1D->A push r 4": push(d_, 4.0), "R3 posterior x by-players, guarded": tilt(PI["by players"], d_), "R4 posterior x half-way, guarded": tilt(PI["half-way to players"], d_),
             "R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1": tilt(PI["by players"], R1, cap_a=2.0), "R6 = R1 + other levels half-way, guarded against R1": tilt(PI["half-way to players"], R1, cap_a=2.0)}
    print("\npayoff against R1 (score difference, test size weights, clean questions), by the TRUE composition; rows = rules")
    print("   " + " " * 82 + " | ".join(f"{k:>24s}" for k in PI))
    for name, d in RULES.items():
        line = f"   {name:82s}"
        for cn, pi in PI.items():
            w = np.zeros(len(y_))
            for m in SIZES:
                s = (n_ == m) & cl_; fr = np.bincount(y_[s], minlength=10) / s.sum(); w[s] = pi[y_[s]] / fr[y_[s]]
            dlt = SM[d, y_] - SM[R1, y_]; v = wmean(dlt, w); boot = []
            for _ in range(200):
                k = rng.integers(0, len(upid), len(upid)); ww = w * np.bincount(k, minlength=len(upid))[pinv]; boot.append(wmean(dlt, ww))
            line += f" | {v:+.4f} [{np.quantile(boot, .025):+.4f},{np.quantile(boot, .975):+.4f}]"; rep[f"{name} | {cn}"] = v
        ch = float(((d != R1) & cl_).sum() / cl_.sum()); print(line + f" | changed vs R1 {ch:.3f}", flush=True)
    kinds = pd.Series([f"{LEVELS[a]}>{LEVELS[c]}" for a, c in zip(R1[(RULES['R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1'] != R1) & cl_], RULES['R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1'][(RULES['R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1'] != R1) & cl_])]).value_counts()
    gain = SM[RULES['R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1'], y_] - SM[R1, y_]; d5 = RULES['R5 = R1 + the other levels of by-players (A:1D capped at 2), guarded against R1']
    print("R5's changes against R1 by kind (count, mean gain per change with every question weighted equally = by-players composition): " + ", ".join(f"{k} {v} ({float(gain[(d5 != R1) & cl_ & (np.array([LEVELS[a] for a in R1]) == k.split('>')[0]) & (np.array([LEVELS[a] for a in d5]) == k.split('>')[1])].mean()):+.3f})" for k, v in kinds.head(10).items()))
    (ROOT / "results/r_variants/m157c_payoff.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m157c_payoff.json")


if __name__ == "__main__":
    main()
