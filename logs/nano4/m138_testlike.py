#!/usr/bin/env python
"""Section 4M M138 (2026-11-14, report only): the gain of every layer on validation players grouped by their number of games in
the data set, under the current yardstick (class-balanced, sizes 5 / 10 / 20 equally) and under a test-like one (the group's
natural level mix, question sizes weighted as in the 800 test questions).

Layers: base, ct44 rule, ct75u rule, ct104b rule (M129 + two guards), ct105 rule (learned pool L0 + two guards relative to the
ct75u rule), ct106 rule (L0 + four guards relative to the ct104b rule). Both seed sets; player bootstrap.

  python -I logs/nano4/m138_testlike.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402  (guard2)
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n  # noqa: E402

m3, mf, la, b, s2, mb, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.mb, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
LAYERS = ("base", "ct44 rule", "ct75u rule", "ct104b rule", "ct105 rule", "ct106 rule"); STEPS = (("ct44 rule", "base"), ("ct75u rule", "ct44 rule"), ("ct104b rule", "ct75u rule"), ("ct105 rule", "ct104b rule"), ("ct106 rule", "ct104b rule"))
GROUPS = (("all", 1, 10**9), ("<=40", 1, 40), ("<=25", 1, 25), ("26-40", 26, 40), ("41-110", 41, 110), ("111-260", 111, 260), ("261+", 261, 10**9))


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])
    tsz = pd.Series(T["5e"]["qid"].astype(str)).value_counts().value_counts().sort_index(); tot = float(tsz.sum()); share = {int(k): float(v) / tot for k, v in tsz.items()}
    wsz = {5: share.get(5, 0.0), 10: share.get(10, 0.0) + share.get(15, 0.0) / 2, 20: share.get(20, 0.0) + share.get(15, 0.0) / 2}
    print("test questions by size: " + ", ".join(f"{k} games {int(v)}" for k, v in tsz.items()) + " -> size weights of the test-like yardstick " + ", ".join(f"{k}: {v:.3f}" for k, v in wsz.items()))
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); pfold = pf.fold.astype(int).reindex(players).to_numpy()
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
    y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); ng = pf.n_games.astype(int).reindex(pid.astype(str)).to_numpy(); rep = {}
    upid, pinv = np.unique(pid, return_inverse=True)

    def natural(sc, m):
        return float(sum(wsz[s] * sc[m & (n == s)].mean() for s in SIZES if (m & (n == s)).any()) / sum(wsz[s] for s in SIZES if (m & (n == s)).any()))

    def boot(d, m, natural_mix, B=1000, seed=0):
        """95% interval of the mean of d (a per-question difference) over the questions m, players resampled."""
        rng = np.random.default_rng(seed); pl = np.unique(pinv[m]); out = []
        num = {s: np.bincount(pinv[m & (n == s)], weights=d[m & (n == s)], minlength=len(upid)) for s in SIZES}; den = {s: np.bincount(pinv[m & (n == s)], minlength=len(upid)).astype(float) for s in SIZES}
        if not natural_mix:  # class-balanced: weights 1 / (questions of the class and size in the group)
            w = np.zeros(len(d))
            for s in SIZES:
                for c in range(10):
                    k = m & (n == s) & (y == c); w[k] = 1.0 / max(int(k.sum()), 1)
            num = {s: np.bincount(pinv[m & (n == s)], weights=(d * w)[m & (n == s)], minlength=len(upid)) for s in SIZES}; den = {s: np.bincount(pinv[m & (n == s)], weights=w[m & (n == s)], minlength=len(upid)) for s in SIZES}
        ws = wsz if natural_mix else {s: 1.0 for s in SIZES}
        for _ in range(B):
            pick = rng.choice(pl, len(pl)); tot_w = sum(ws[s] for s in SIZES if den[s][pick].sum() > 0)
            out.append(sum(ws[s] * num[s][pick].sum() / den[s][pick].sum() for s in SIZES if den[s][pick].sum() > 0) / tot_w)
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X, D = {}, {k: [] for k in LAYERS}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); d44 = mb.decisions(P, f["Ls"][0]["v44"], 0.5, 3.0, 0.6)[2]
            f["du"], f["dc"] = du, mb7.keep_old(db, du); D["base"].append(d0); D["ct44 rule"].append(d44); D["ct75u rule"].append(du); D["ct104b rule"].append(f["dc"])
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = mb7.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), "L0")
            ds = np.asarray(b.decide(mb7.apply(coef, X[v], Fd[v]["n"])), int); D["ct105 rule"].append(mb7.keep_old(ds, Fd[v]["du"])); D["ct106 rule"].append(mf7.guard2(ds, Fd[v]["dc"]))
        S = {k: SM[np.concatenate(D[k]), y] for k in LAYERS}; out = {}
        print(f"\nseeds {seeds}")
        for g, lo, hi in GROUPS:
            m = (ng >= lo) & (ng <= hi); npl = len(np.unique(pid[m])); comp = np.bincount(y[m & (n == 5)], minlength=10) / max(int((m & (n == 5)).sum()), 1)
            print(f"  players with {g} games: {npl} players, {int(m.sum())} questions (5 / 10 / 20 games: {int((m & (n == 5)).sum())} / {int((m & (n == 10)).sum())} / {int((m & (n == 20)).sum())}); level mix of their 5-game questions: kyu {comp[:4].sum():.2f}, 1D {comp[4]:.3f}, 2D-6D {comp[5:].sum():.2f}")
            bal = {k: float(balanced_mean_over_n(S[k][m], y[m], n[m], sizes=SIZES)) for k in LAYERS}; nat = {k: natural(S[k], m) for k in LAYERS}
            print("      class-balanced: " + "  ".join(f"{k} {bal[k]:.4f}" for k in LAYERS)); print("      test-like     : " + "  ".join(f"{k} {nat[k]:.4f}" for k in LAYERS))
            o = {"players": npl, "questions": int(m.sum()), "balanced": bal, "testlike": nat, "steps": {}}
            for new, old in STEPS:
                dd = S[new] - S[old]; cb, ct = boot(dd, m, False), boot(dd, m, True)
                print(f"      {new:11s} - {old:11s}: class-balanced {bal[new] - bal[old]:+.4f} [{cb[0]:+.4f}, {cb[1]:+.4f}] | test-like {nat[new] - nat[old]:+.4f} [{ct[0]:+.4f}, {ct[1]:+.4f}] = {400 * (nat[new] - nat[old]):+.1f} points per 400")
                o["steps"][f"{new} - {old}"] = {"balanced": bal[new] - bal[old], "balanced_ci": cb, "testlike": nat[new] - nat[old], "testlike_ci": ct}
            out[g] = o
        rep[f"seeds_{seeds[0]}"] = out
    print("\nPublic, for comparison (points per 400): ct28 -> ct44 +9.6; ct44 -> ct75u +1.4; ct75u -> ct104b +1.0; ct104b -> ct105 -1.0; ct104b -> ct106 0.0")
    (ROOT / "results/r_variants/m138_testlike.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m138_testlike.json")


if __name__ == "__main__":
    main()
