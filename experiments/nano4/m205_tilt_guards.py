#!/usr/bin/env python
"""Section 4M M204 (2026-11-16), REPORT ONLY: the decision layer under the MEASURED level composition of the test set.

M203: training and test differ only in the level composition, and M202 measured the Public composition exactly. The current
rule (the SQ4 pool, fitted and decided with every level equally likely, four guards) plus the two hand-made pushes of the
test file (1D -> A and 4D -> 5D, odds x 2 on the base posterior, the flipped questions locked) is compared, on validation
questions weighted to the measured composition, with decisions that use the composition directly (the pooled posterior
multiplied by the composition; the pool fitted with composition weights). Design doc M204, registered before the run.

Data preparation as m165_pool.py (identifier SQ4 as one more source of the L0 pool); evaluation as m157c_payoff.py.

  python -I logs/nano4/m204_prior.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m105_voters as m5  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m141_opp_lookalike as m41  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
TW = {5: 0.25, 10: 0.375, 20: 0.375}; XS = "SQ4"; A_, D1, D4, D5 = (LEVELS.index(x) for x in ("A", "1D", "4D", "5D"))


def main():
    pfc = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str); lvp = pfc["rank"].map(b.RANK).to_numpy().astype(int); ngp = pfc.n_games.astype(int).to_numpy(); c15 = np.bincount(lvp[ngp >= 15], minlength=10); share_d = c15[0] / (c15[0] + c15[1])
    pc = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cd = 400 - sum(pc.values()); PI = np.array([cd * share_d, cd * (1 - share_d)] + [pc[l] for l in LEVELS[2:]], float) / 400.0; assert abs(PI.sum() - 1) < 1e-9
    print("measured composition (C + D split by the players with >= 15 games, D share %.3f): " % share_d + " ".join(f"{LEVELS[c]} {PI[c]:.3f}" for c in range(10)), flush=True); COMPS = {"measured": PI, "equal": np.full(10, 0.1)}
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], f"{XS}~{STAT}"]); assert f"{XS}~{STAT}" in used
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    xscale = round(m5.density_ratio(V[f"{XS}~{STAT}"], T[f"{XS}~{STAT}"], key_pid, dec, ct9s, nominal), 3); print(f"test-side density ratio of {XS}: {xscale}", flush=True)
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    xspec = (f"{XS}~{STAT}", 1, 1, 1, 1, 1, float(xscale))
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"]); hd = np.asarray(z["hd"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pfc.set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}

        def ev_of(spec):
            if spec not in cache:
                idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
            return cache[spec]
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}
        Lx = ev_of(xspec); qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        nh = np.zeros(len(y)); np.add.at(nh, qi, (hd[gi] >= 2).astype(float)); found = np.bincount(qi, minlength=len(y))
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, cw=cw, cwpi=cw * PI[y] * 10.0, wq=np.array([W_FIN[int(m)] for m in n]), R=lr_ - lr_.mean(0, keepdims=True), clean=(nh == 0) & (found == n)); print(f"fold {v}: inputs ready ({len(y)} questions, clean {float(Fd[v]['clean'].mean()):.3f})", flush=True)
    cat = lambda k: np.concatenate([Fd[v][k] for v in FOLDS])  # noqa: E731
    y_, n_, pid_, cl_ = cat("y"), cat("n"), cat("pid"), cat("clean"); upid, pinv = np.unique(pid_, return_inverse=True); dcd = lambda P: np.asarray(b.decide(P), int); rep = {"composition": PI.tolist()}  # noqa: E731
    W = {}
    for cn, pi in COMPS.items():
        w = np.zeros(len(y_))
        for m in SIZES:
            s = (n_ == m) & cl_; fr = np.bincount(y_[s], minlength=10) / s.sum(); w[s] = pi[y_[s]] / fr[y_[s]]
        W[cn] = w

    def wmean(s, w):
        return float(sum(TW[m] * (s[n_ == m] * w[n_ == m]).sum() / w[n_ == m].sum() for m in SIZES))

    def lock(P, i, j, r=2.0):
        Q_ = P.copy(); m = P[:, i] + P[:, j]; s_ = P[:, i] / np.maximum(P[:, i] + r * P[:, j], 1e-300); Q_[:, i] = m * s_; Q_[:, j] = m * (1 - s_)
        return (dcd(P) == i) & (dcd(Q_) == j)

    def tilt(P, lam):
        Pp = P * (PI / 0.1)[None, :] ** lam; return Pp / Pp.sum(1, keepdims=True)
    for seeds in ((0, 1, 2), (6, 7, 8)):
        rng = np.random.default_rng(204 + seeds[0]); X = {}; Pb = {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); Pb[v] = P
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, XS: f["Lx"]}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)

        def pool(extra, wkey):
            out = {}
            for v in FOLDS:
                tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
                coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u][wkey] for u in tr]), extra); out[v] = m41.apply(coef, X[v], Fd[v]["n"], extra)
            return np.concatenate([out[v] for v in FOLDS])
        P0 = pool((), "cw"); ref = mf7.guard2(dcd(P0), np.concatenate([Fd[v]["dc"] for v in FOLDS])); PQ = pool((XS,), "cw"); PW = pool((XS,), "cwpi"); Pb_ = np.concatenate([Pb[v] for v in FOLDS])
        R0 = mf7.guard2(dcd(PQ), ref); lkA = lock(Pb_, D1, A_); lk5 = lock(Pb_, D4, D5); R1a = np.where(lkA, A_, R0); R1b = np.where(lk5, D5, R1a); T1 = dcd(tilt(PQ, 1.0)); T1g = mf7.guard2(T1, ref); Th = dcd(tilt(PQ, 0.5))
        RULES = {"R0 current rule, no push": R0, "R1a = R0 + 1D->A push": R1a, "R1b = R1a + 4D->5D push (= ct112, reference)": R1b, "R1c = R0 + 4D->5D push only": np.where(lk5, D5, R0), "T1 posterior x composition, no guards": T1, "T1g = T1 + guards (PRIMARY)": T1g,
                 "Th posterior x composition^0.5, no guards": Th, "Thg = Th + guards": mf7.guard2(Th, ref), "W1g pool fitted with composition weights, x composition, guards": mf7.guard2(dcd(tilt(PW, 1.0)), ref), "W0g pool fitted with composition weights only, guards": mf7.guard2(dcd(PW), ref),
                 "T1g5 = T1g + 4D->5D push": np.where(lk5, D5, T1g), "T1gA = T1g + 1D->A push": np.where(lkA, A_, T1g), "plain pooled posterior, no guards, no push": dcd(PQ), "base decision": dcd(Pb_),
                 # added after the first reading (exploratory, not registered): the composition used everywhere EXCEPT on the two Public-confirmed pushes, which stay locked as in ct112
                 "T1gAB = T1g + both pushes (exploratory)": np.where(lk5, D5, np.where(lkA, A_, T1g)), "T1AB = T1, no guards, + both pushes (exploratory)": np.where(lk5, D5, np.where(lkA, A_, T1)), "ThgAB = Thg + both pushes (exploratory)": np.where(lk5, D5, np.where(lkA, A_, mf7.guard2(Th, ref))),
                 "R0AB without guards = plain pool + both pushes (exploratory)": np.where(lk5, D5, np.where(lkA, A_, dcd(PQ)))}
        Dl, D6 = LEVELS.index("D"), LEVELS.index("6D"); AB = lambda d: np.where(lk5, D5, np.where(lkA, A_, d))  # noqa: E731
        gm = {"never leave A": lambda d: (ref == A_) & (d != ref), "never leave D": lambda d: (ref == Dl) & (d != ref), "no 5D->6D": lambda d: (ref == D5) & (d == D6), "no two levels": lambda d: np.abs(d - ref) >= 2}

        def guarded(d, names):
            m = np.zeros(len(d), bool)
            for k in names:
                m |= gm[k](d)
            return np.where(m, ref, d)
        RULES = {"R1b (= ct112, reference)": R1b, "tilt + pushes, no guard": AB(T1), "tilt + pushes + all four guards": AB(mf7.guard2(T1, ref)), **{f"tilt + pushes + only '{k}'": AB(guarded(T1, [k])) for k in gm},
                 "tilt + pushes + the two standing rules (never leave A, no 5D->6D) (PRIMARY)": AB(guarded(T1, ["never leave A", "no 5D->6D"])), "tilt + pushes + never leave A + no two levels": AB(guarded(T1, ["never leave A", "no two levels"])), "tilt + pushes + all but 'never leave D'": AB(guarded(T1, ["never leave A", "no 5D->6D", "no two levels"]))}
        print(f"\n== seeds {seeds}: score on clean questions (test size weights) and the difference to R1b with a player-bootstrap 95% interval; columns: measured composition | equal levels")
        out = {}
        for name, d in RULES.items():
            line = f"  {name:66s}"; out[name] = {}
            for cn in COMPS:
                w = W[cn]; sc = wmean(SM[d, y_], w); dlt = SM[d, y_] - SM[R1b, y_]; v = wmean(dlt, w); boot = []
                for _ in range(300):
                    k = rng.integers(0, len(upid), len(upid)); boot.append(wmean(dlt, w * np.bincount(k, minlength=len(upid))[pinv]))
                lo, hi = np.quantile(boot, [0.025, 0.975]); line += f" | {sc:.4f}  {v:+.4f} [{lo:+.4f},{hi:+.4f}]"; out[name][cn] = {"score": sc, "diff": v, "ci95": [float(lo), float(hi)]}
            print(line + f" | differs from R1b on {float(((d != R1b) & cl_).sum() / cl_.sum()):.3f}", flush=True)
        rep[f"seeds_{seeds[0]}"] = out
    (ROOT / "results/r_variants/m205_tilt_guards.json").write_text(json.dumps(rep, indent=1, default=float)); print("\nREPORT ONLY (no file is built); written results/r_variants/m205_tilt_guards.json")


if __name__ == "__main__":
    main()
