#!/usr/bin/env python
"""Section 4M M157 (2026-11-15, report only): how much is the test set's level composition worth, and which composition do the
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
    upid, pinv = np.unique(pid_, return_inverse=True); rng = np.random.default_rng(0); rep = {}

    def wmean(s, w):
        return float(sum(TW[m] * (s[n_ == m] * w[n_ == m]).sum() / w[n_ == m].sum() for m in SIZES))
    print("\n(a) cross-validation, clean questions only (no game with HD >= 2, like the test), weighted to each composition; test size weights 0.25 / 0.375 / 0.375")
    for name, pi in PI.items():
        w = np.zeros(len(y_))
        for m in SIZES:
            s = (n_ == m) & cl_; fr = np.bincount(y_[s], minlength=10) / s.sum(); w[s] = pi[y_[s]] / fr[y_[s]]
        Pp = P_ * pi[None, :]; Pp /= Pp.sum(1, keepdims=True); dp = np.asarray(b.decide(Pp), int); dg = mf7.guard2(dp, d_); s0, s1, s2_ = SM[d_, y_], SM[dp, y_], SM[dg, y_]
        base, plain, guard = wmean(s0, w), wmean(s1, w), wmean(s2_, w); boot = []
        for _ in range(400):
            k = rng.integers(0, len(upid), len(upid)); cnt = np.bincount(k, minlength=len(upid))[pinv]; ww = w * cnt; boot.append((wmean(s1, ww) - wmean(s0, ww), wmean(s2_, ww) - wmean(s0, ww)))
        boot = np.array(boot); ch = float(((dp != d_) * w).sum() / w.sum()); kinds = pd.Series([f"{LEVELS[a]}>{LEVELS[c]}" for a, c in zip(d_[(dp != d_) & cl_], dp[(dp != d_) & cl_])]).value_counts().head(6)
        print(f"  {name:26s}: ct106 rule {base:.4f}; posterior x composition {plain:.4f} ({plain - base:+.4f} [{np.quantile(boot[:, 0], .025):+.4f}, {np.quantile(boot[:, 0], .975):+.4f}]); under the guards {guard:.4f} ({guard - base:+.4f} [{np.quantile(boot[:, 1], .025):+.4f}, {np.quantile(boot[:, 1], .975):+.4f}]); answers changed {ch:.3f}; most frequent changes " + ", ".join(f"{k} {v}" for k, v in kinds.items()), flush=True)
        rep[f"cv {name}"] = {"base": base, "plain": plain, "guarded": guard, "ci_plain": np.quantile(boot[:, 0], [.025, .975]).tolist(), "ci_guarded": np.quantile(boot[:, 1], [.025, .975]).tolist()}
    # the answers' composition under each rule (what a file's level counts would look like), weighted to "by players"
    # ---------------------------------------------------------------- (b) ledger
    z = np.load(ROOT / "results/r_variants/m137_test_posterior.npz", allow_pickle=True); qids = z["qid"].astype(str); Pl = z["learned"].astype(float); ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}
    c106 = pd.read_csv(ROOT / "submission_rank_ct106.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).map(b.RANK).to_numpy().astype(int); pub = np.char.startswith(qids, "pub_")
    print("\n(b) Public ledger with prior = pool posterior x composition (normalised per question): cost of the best joint assignment over the constrained questions (lower = the recorded readings fit better); level counts of that assignment")
    for name, pi in PI.items():
        Pp = Pl * pi[None, :]; Pp /= Pp.sum(1, keepdims=True); prior = {str(q_): np.maximum(p_, 1e-6) for q_, p_ in zip(qids, Pp)}; qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior, time_limit=900)
        if mapa in (None, "timeout"):
            print(f"  {name:26s}: no solution within the time limit ({mapa})"); continue
        cost = float(sum(-math.log(max(prior[q_][c], 1e-6)) for q_, c in mapa.items())); cnt = np.bincount(list(mapa.values()), minlength=10)
        # cost of the same assignment under the plain posterior (the part of the cost that is not the composition term)
        soft = Pp[pub].sum(0); dq = np.asarray(b.decide(Pp), int); n_ch = int(((dq != c106) & pub).sum()), int(((dq != c106) & ~pub).sum())
        print(f"  {name:26s}: cost {cost:.2f} over {len(qs)} questions; truths of the best assignment " + " ".join(f"{LEVELS[c]} {cnt[c]}" for c in range(10)) + f" | soft level counts of all 400 Public questions under this prior: " + " ".join(f"{LEVELS[c]} {soft[c]:.0f}" for c in range(10)) + f" | answers that would change against ct106 (no locks, no guards): Public {n_ch[0]}, Private {n_ch[1]}", flush=True)
        rep[f"ledger {name}"] = {"cost": cost, "counts": cnt.tolist(), "soft_public": soft.tolist()}
    (ROOT / "results/r_variants/m157_prior_stakes.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m157_prior_stakes.json")


if __name__ == "__main__":
    main()
