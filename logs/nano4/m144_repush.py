#!/usr/bin/env python
"""Section 4M M144 (2026-11-15): the two boundary pushes the Public board confirmed (1D -> A and 4D -> 5D, r = 2: ct13 / ct28),
re-derived on the CURRENT posterior (the learned pool behind ct106) instead of ct9's. Report only, no file (design doc M144).

ct13 / ct28 pushed the questions that were close to the two boundaries under ct9's base posterior; those 38 answers have been
locked since. The later evidence (anchors, look-alikes, the pool) moved other questions next to the same boundaries; they were
never pushed. Rule (ct13's): on the pool posterior, keep m = P(i) + P(j), multiply the odds of j by r, expected-score decision;
a question flips only if the current answer is i and the adjusted decision is j. Locked questions are left alone.
  (1) test: number of flips by r (Public / Private), the questions, what the locked questions look like under the pool posterior;
  (2) Public ledger (all recorded uploads): the weight of each truth for the flipped Public questions it constrains;
  (3) cross-validation (pool posterior of the ct106 rule, seeds 0-2): precision and score change of the same flips, all players
      and by the player's number of games (the least active group is the one that resembles the test set at A|1D: M61, M116).

  python -I logs/nano4/m144_repush.py
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
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
L = {x: i for i, x in enumerate(LEVELS)}; PAIRS = (("1D", "A"), ("4D", "5D")); RS = (1.5, 2.0, 2.6, 3.5)
EXTRA_PAIRS = (("A", "1D"), ("5D", "4D"), ("B", "A"), ("5D", "6D"), ("2D", "1D"), ("3D", "4D"), ("C", "D"), ("D", "C"))  # reference directions (r = 2): the opposite and the neighbouring pushes


def push(P, i, j, r):
    Q = P.copy(); m = P[:, i] + P[:, j]; s = P[:, i] / np.maximum(P[:, i] + r * P[:, j], 1e-300); Q[:, i] = m * s; Q[:, j] = m * (1 - s)
    return Q


def flips(P, d_cur, i, j, r, free):
    return free & (d_cur == i) & (np.asarray(b.decide(push(P, i, j, r)), int) == j)


def main():
    rep = {}
    # ---------------------------------------------------------------- (1) test
    z = np.load(ROOT / "results/r_variants/m137_test_posterior.npz", allow_pickle=True); qids = z["qid"].astype(str); Pl, P9 = z["learned"].astype(float), z["base"].astype(float)
    rk = np.vectorize(b.RANK.get); sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c13, c28, c104b, c106 = (sub(k) for k in ("ct9", "ct13", "ct28", "ct104b", "ct106")); locked = c28 != c9; d106 = rk(c106).astype(int); pub = np.char.startswith(qids, "pub_")
    T5 = dict(np.load(ROOT / "results/r_variants/m82_test.npz")); nq = pd.Series(T5["qid"].astype(str)).value_counts().reindex(qids).to_numpy()
    d_pool = mf7.guard2(np.where(locked, rk(c28), np.asarray(b.decide(Pl), int)), rk(c104b).astype(int)); assert (d_pool == d106).all(), int((d_pool != d106).sum()); print("check: the stored pool posterior with the locks and the guards reproduces ct106")
    print("ct106 answers by level (Public | Private): " + "  ".join(f"{lv} {int((c106[pub] == lv).sum())}|{int((c106[~pub] == lv).sum())}" for lv in LEVELS))
    print(f"kyu share of the answers: Public {float((d106[pub] <= 3).mean()):.3f}, Private {float((d106[~pub] <= 3).mean()):.3f} (M68's estimate of the truth: 0.493 / 0.467)")
    for src, dst in PAIRS:
        i, j = L[src], L[dst]; lk = locked & (rk(c9) == i) & (rk(c28) == j); odds = Pl[:, j] / np.maximum(Pl[:, i], 1e-12)
        print(f"\n== {src} -> {dst}: the {int(lk.sum())} locked pushes of ct13 / ct28 under the pool posterior: odds {dst}:{src} quartiles " + " / ".join(f"{x:.2f}" for x in np.quantile(odds[lk], [0, 0.25, 0.5, 0.75, 1]))
              + f"; the pool alone would answer {dst} in {int((np.asarray(b.decide(Pl), int)[lk] == j).sum())}, {src} in {int((np.asarray(b.decide(Pl), int)[lk] == i).sum())}, something else in {int((~np.isin(np.asarray(b.decide(Pl), int)[lk], [i, j])).sum())}")
        for r in RS:
            f = flips(Pl, d106, i, j, r, ~locked); rep[f"test {src}>{dst} r{r}"] = {"public": qids[f & pub].tolist(), "private": qids[f & ~pub].tolist()}
            print(f"  r = {r}: {int(f.sum())} flips (Public {int((f & pub).sum())}, Private {int((f & ~pub).sum())}); by size " + " ".join(f"{m}: {int((f & (nq == m)).sum())}" for m in (5, 10, 15, 20)))
        f2 = flips(Pl, d106, i, j, 2.0, ~locked)
        for k in np.flatnonzero(f2):
            print(f"      {qids[k]} ({int(nq[k])} games): ct9 {c9[k]}, ct104b {c104b[k]}, ct106 {c106[k]}; pool P({src}) {Pl[k, i]:.3f} P({dst}) {Pl[k, j]:.3f} odds {odds[k]:.2f}; base P({src}) {P9[k, i]:.3f} P({dst}) {P9[k, j]:.3f}")
    print("\nreference directions at r = 2 (number of flips Public / Private): " + "  ".join(f"{s_}>{t_} {int((flips(Pl, d106, L[s_], L[t_], 2.0, ~locked) & pub).sum())}/{int((flips(Pl, d106, L[s_], L[t_], 2.0, ~locked) & ~pub).sum())}" for s_, t_ in EXTRA_PAIRS))
    # ---------------------------------------------------------------- (2) ledger
    ledger = json.load(open(led.LEDGER)); up = {**led.KNOWN_SCORES, **ledger}; prior = {str(q_): np.maximum(p_, 1e-6) for q_, p_ in zip(qids, Pl)}
    qs, cand, obs = led.build(up, prior); mapa = led.solve(qs, cand, obs, prior, time_limit=300); print(f"\nledger: {len(obs)} uploads, {len(qs)} Public questions constrained; jointly feasible: {mapa not in (None, 'timeout')}")
    cost_of = lambda sol: float(sum(-math.log(max(prior[q][c], 1e-6)) for q, c in sol.items()))  # noqa: E731
    if mapa not in (None, "timeout"):
        for src, dst in PAIRS:
            i, j = L[src], L[dst]; f2 = flips(Pl, d106, i, j, 2.0, ~locked) & pub; tot = 0.0
            for k in np.flatnonzero(f2):
                q = qids[k]; p = Pl[k] / Pl[k].sum(); e_pri = float(sum(p[c] * (led.s(j, c) - led.s(i, c)) for c in range(10)))
                if q in cand:
                    w = {}
                    for c in cand[q]:
                        sol = led.solve(qs, cand, obs, prior, fix=(q, c), time_limit=60)
                        if sol not in (None, "timeout"):
                            w[c] = cost_of(sol)
                    m_ = min(w.values()); w = {c: math.exp(-(v - m_)) for c, v in w.items()}; zt = sum(w.values()); w = {c: v / zt for c, v in w.items()}; e = float(sum(v * (led.s(j, c) - led.s(i, c)) for c, v in w.items()))
                    print(f"  {q} {src}->{dst}: ledger weights " + ", ".join(f"{LEVELS[c]} {v:.2f}" for c, v in sorted(w.items()) if v >= 0.005) + f" -> expected {e:+.3f} (pool posterior alone {e_pri:+.3f})", flush=True)
                else:
                    e = e_pri; print(f"  {q} {src}->{dst}: not constrained by any upload -> expected {e:+.3f} (pool posterior)", flush=True)
                tot += e
            print(f"  {src}->{dst} r = 2, Public: expected {tot:+.2f} points over {int(f2.sum())} flips")
    # ---------------------------------------------------------------- (3) cross-validation
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; zz = V["5e"]; row = {k: i for i, k in enumerate(zz["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); pfold = pf.fold.astype(int).reindex(players).to_numpy()
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
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand_ = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand_].astype(np.float64) for m in range(NM)], plab[cand_], la.BE)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); R = lr_ - lr_.mean(0, keepdims=True); wq = np.array([W_FIN[int(m)] for m in n])
        d0, du = mf.rule75u(P, Ls[0]); _, db = mf.rule75u(la.pooled(P, R, wq), Ls[2])
        Fd[v] = dict(y=y, n=n, pid=pid, cw=cw, dc=mb7.keep_old(db, du), X={"logp": np.log(np.maximum(P, 1e-12)), "R": R, **{k: Ls[2][k] for k in NAMES}}); print(f"fold {v}: inputs ready", flush=True)
    PV, DV = {}, {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([Fd[u]["X"][k] for u in tr]) for k in Fd[v]["X"]}
        coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), ())
        PV[v] = m41.apply(coef, Fd[v]["X"], Fd[v]["n"], ()); DV[v] = mf7.guard2(np.asarray(b.decide(PV[v]), int), Fd[v]["dc"])
    P_, d_, y_, n_, pid_ = (np.concatenate([x[v] for v in FOLDS]) for x in (PV, DV, {v: Fd[v]["y"] for v in FOLDS}, {v: Fd[v]["n"] for v in FOLDS}, {v: Fd[v]["pid"] for v in FOLDS}))
    ng = pf.n_games.astype(int).reindex(pid_.astype(str)).to_numpy(); GR = (("all", ng > 0), ("1-25 games", ng <= 25), ("26-40", (ng > 25) & (ng <= 40)), ("41-70", (ng > 40) & (ng <= 70)), ("71-170", (ng > 70) & (ng <= 170)), ("171+", ng > 170))
    print(f"\ncross-validation (ct106 rule, seeds 0-2): {len(y_)} questions; plain mean score {float(SM[d_, y_].mean()):.4f}")
    print("composition of the questions by group (share of A | share of 1D | share of 4D | share of 5D): " + "  ".join(f"{nm}: {float((y_[g] == L['A']).mean()):.3f} | {float((y_[g] == L['1D']).mean()):.3f} | {float((y_[g] == L['4D']).mean()):.3f} | {float((y_[g] == L['5D']).mean()):.3f}" for nm, g in GR))
    for src, dst in (*PAIRS, *EXTRA_PAIRS):
        i, j = L[src], L[dst]
        for r in (RS if (src, dst) in PAIRS else (2.0,)):
            f = flips(P_, d_, i, j, r, np.ones(len(y_), bool)); gain = SM[j, y_] - SM[i, y_]; out = {}
            line = f"  {src}->{dst} r = {r}: "
            for nm, g in GR:
                s = f & g; k = int(s.sum()); out[nm] = {"flips": k, "share_of_questions": float(s.sum() / max(g.sum(), 1)), "precision": float((y_[s] == j).mean()) if k else None, "truth_is_source": float((y_[s] == i).mean()) if k else None, "gain_per_flip": float(gain[s].mean()) if k else None}
                line += f"{nm}: {k} flips ({100 * out[nm]['share_of_questions']:.2f}% of its questions)" + (f", truth {dst} {out[nm]['precision']:.2f} / {src} {out[nm]['truth_is_source']:.2f}, per flip {out[nm]['gain_per_flip']:+.3f}" if k else "") + " | "
            rep[f"cv {src}>{dst} r{r}"] = out; print(line, flush=True)
    (ROOT / "results/r_variants/m144_repush.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m144_repush.json")


if __name__ == "__main__":
    main()
