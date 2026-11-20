#!/usr/bin/env python
"""Section 4M M155 (2026-11-15): the "twin" reading - a question whose player has a near-identical other id among the training
players (design doc M155; follows M154).

M154: for about a tenth of the players the nearest other prototype is far closer than the second nearest, and then it carries the
same level in 70-80% and an adjacent level in the rest, never two levels away. The look-alike reading R of the pool is a soft
k-nearest average (sharpness 20) with one weight per question size; it cannot say "this one is a twin, trust it".
Reading per question (cross-fitted like R: the fold's own network, candidates = the players of the other folds): the mean
target-side identity vector against the prototypes (three members averaged); L1 = label of the nearest, gap = its similarity
minus the second nearest's. Questions are banded by the gap (quantiles 0.5 / 0.8 / 0.9 / 0.95 / 0.98 / 0.99 of the other folds'
questions of the same size); per size and band the distribution of d = L1 - true level from the other folds (smoothed towards
the lowest band's). New pool input: log P_band(L1 - c) - log P_lowest(L1 - c) for every level c.
REGISTERED: the pool with this input against the ct106 rule (four guards relative to it), M37 on seeds 0-2 and 6-8.
Report: share of questions, exact / within-one of L1 and of the ct106 rule's answers per band; the variant banded by the top
similarity; the variant where the input replaces R.

  python -I logs/nano4/m155_twin_reading.py [--name ct109] [--no-build]
"""
import argparse
import csv
import hashlib
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
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE
QB = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99]); NB = len(QB) + 1; KAP = 20.0


def tables(stat, d, n, sizes=SIZES):
    """{size: (edges, logratio [NB, 19])}: bands of the statistic and, per band, log P_band(d) - log P_lowest(d) (d + 9 indexed)."""
    out = {}
    for s in sizes:
        m = n == s; edges = np.quantile(stat[m], QB); band = np.searchsorted(edges, stat[m]); h = np.zeros((NB, 7)); np.add.at(h, (band, np.clip(d[m], -3, 3) + 3), 1.0)
        p0 = (h[0] + 1.0) / (h[0].sum() + 7.0); pb = (h + 1.0) / (h.sum(1, keepdims=True) + 7.0); out[s] = (edges, np.clip(np.log(pb) - np.log(p0)[None], -3.0, 3.0))  # 2026-11-15 06:05: see m156 (sparse tail cells)
    return out


def reading(tab, stat, L1, n):
    """[N, 10]: log-ratio of every level c given the nearest prototype's label and the question's band."""
    E = np.zeros((len(stat), 10)); band = np.zeros(len(stat), int)
    for s, (edges, lr) in tab.items():
        m = n == s; bd = np.searchsorted(edges, stat[m]); band[m] = bd; E[m] = lr[bd[:, None], np.clip(L1[m][:, None] - np.arange(10)[None, :], -3, 3) + 3]
    return E, band


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct109"); ap.add_argument("--no-build", action="store_true"); a_ = ap.parse_args()
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"].astype(int); pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}

    def nearest(tq, protos, labels):
        S = sum(tq[m] @ protos[m].T for m in range(NM)) / NM; j = S.argmax(1); s1 = S[np.arange(len(S)), j]; S[np.arange(len(S)), j] = -2; return labels[j], s1, s1 - S.max(1), j
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
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); protos = [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)]
        lr_, _, _ = la.reading(tq, protos, plab[cand], la.BE); L1, s1, gap, _ = nearest(tq, protos, plab[cand]); cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw, L1=L1, s1=s1, gap=gap); print(f"fold {v}: inputs ready ({len(y)} questions)", flush=True)
    # cross-fitted twin inputs
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; cat = lambda k: np.concatenate([Fd[u][k] for u in tr])  # noqa: E731
        for key, stat in (("Tw", "gap"), ("Ts", "s1")):
            tab = tables(cat(stat), cat("L1") - cat("y"), cat("n")); Fd[v][key], Fd[v][key + "_band"] = reading(tab, Fd[v][stat], Fd[v]["L1"], Fd[v]["n"])
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); cw_all = np.concatenate([Fd[v]["cw"] for v in FOLDS]); rep = {}; ok = []; keep = {}
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731

    def pool_decisions(X, extra=()):
        out = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra)
            out[v] = (np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], extra)), int), coef)
        return out
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], "Tw": f["Tw"], "Ts": f["Ts"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f"\nseeds {seeds}: ct106 rule {bal(S_old):.4f}")
        if seeds == (0, 1, 2):
            keep = {"X": X, "ref": ref}; bd = np.concatenate([Fd[v]["Tw_band"] for v in FOLDS]); L1 = np.concatenate([Fd[v]["L1"] for v in FOLDS]); rf = np.concatenate([ref[v] for v in FOLDS]); gp = np.concatenate([Fd[v]["gap"] for v in FOLDS])
            print("  by band of the gap (all sizes): share of the questions | nearest prototype: same level / within one | the ct106 rule's answer: exact / within one | mean score of the ct106 rule | of answering the nearest prototype's level")
            for k in range(NB):
                m = bd == k; print(f"    band {k} (gap about {np.median(gp[m]):.3f}): {m.mean():.3f} | {float((L1[m] == y[m]).mean()):.3f} / {float((np.abs(L1[m] - y[m]) <= 1).mean()):.3f} | {float((rf[m] == y[m]).mean()):.3f} / {float((np.abs(rf[m] - y[m]) <= 1).mean()):.3f} | {float(SM[rf[m], y[m]].mean()):.4f} | {float(SM[L1[m], y[m]].mean()):.4f}")
            for s in SIZES:
                m = (n == s) & (bd >= 4); print(f"    n={s}: questions in the bands >= 4: {float(m.sum() / (n == s).sum()):.3f}; nearest same / within one {float((L1[m] == y[m]).mean()):.3f} / {float((np.abs(L1[m] - y[m]) <= 1).mean()):.3f}; ct106 rule exact {float((rf[m] == y[m]).mean()):.3f}")
        out = {}
        for name, extra, rep_r in (("PRIMARY: + twin reading (gap bands)", ("Tw",), False), ("+ reading banded by the top similarity", ("Ts",), False), ("+ both", ("Tw", "Ts"), False), ("twin reading (gap) instead of R", ("Tw",), True)):
            Xv = {v: ({**X[v], "R": np.zeros_like(X[v]["R"])} if rep_r else X[v]) for v in FOLDS}; dd = pool_decisions(Xv, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]
            A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]
            good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {name:40s}: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}; coefficients " + " | ".join(
                f"n={s}: alpha {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}, v5 {np.mean([dd[v][1][s][3] for v in FOLDS]):.2f}, R {np.mean([dd[v][1][s][7] for v in FOLDS]):.2f}, new " + "/".join(f"{np.mean([dd[v][1][s][8 + i] for v in FOLDS]):+.2f}" for i in range(len(extra))) for s in SIZES), flush=True)
            if name.startswith("PRIMARY"):
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)))
                bd = np.concatenate([Fd[v]["Tw_band"] for v in FOLDS]); print("        by band of the gap (plain mean of the score difference): " + "  ".join(f"{k}: {float((A_ - B_)[bd == k].mean()):+.4f}" for k in range(NB)), flush=True)
            out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good)}
        rep[f"seeds_{seeds[0]}"] = out
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets, the pool with the twin reading against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (ROOT / "results/r_variants/m155_twin_reading.json").write_text(json.dumps(rep, indent=1, default=float))
    # ---------------------------------------------------------------- test side (the reading's distribution is always reported)
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy().astype(int); ti = np.flatnonzero(q["qtest"] >= 0); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000
    tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); protos = [PR[m, 4].astype(np.float64) for m in range(NM)]; L1_t, s1_t, gap_t, j_t = nearest(tq, protos, plab)
    allv = lambda k: np.concatenate([Fd[v][k] for v in FOLDS])  # noqa: E731
    tab = tables(allv("gap"), allv("L1") - y, n); tab[15] = ((tab[10][0] + tab[20][0]) / 2, (tab[10][1] + tab[20][1]) / 2); Tw_t, band_t = reading(tab, gap_t, L1_t, nq_t)
    print("\ntest side: share of the questions per band of the gap (validation shares are 0.5 / 0.3 / 0.1 / 0.05 / 0.03 / 0.01 / 0.01 by construction): " + "  ".join(f"{k}: {float((band_t == k).mean()):.3f}" for k in range(NB)))
    for s in (5, 10, 15, 20):
        m = nq_t == s; vm = n == (s if s != 15 else 10); print(f"    n={s}: test gap quartiles " + " / ".join(f"{x:.3f}" for x in np.quantile(gap_t[m], [0.5, 0.9, 0.98])) + f" (validation n={s if s != 15 else 10}: " + " / ".join(f"{x:.3f}" for x in np.quantile(allv('gap')[vm], [0.5, 0.9, 0.98])) + f"); top similarity median {np.median(s1_t[m]):.3f} (validation {np.median(allv('s1')[vm]):.3f}); test questions in the bands >= 4: {int((m & (band_t >= 4)).sum())}")
    c106 = pd.read_csv(ROOT / "submission_rank_ct106.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy(); d106 = rk(c106).astype(int); hi = band_t >= 4
    print(f"test questions in the bands >= 4: {int(hi.sum())} (Public {int((hi & np.char.startswith(qids.astype(str), 'pub_')).sum())}); ct106 already answers the nearest prototype's level in {float((d106[hi] == L1_t[hi]).mean()):.3f} of them, within one in {float((np.abs(d106[hi] - L1_t[hi]) <= 1).mean()):.3f}")
    if not passed or a_.no_build:
        return
    X012 = keep["X"]; Xcur = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]] if k != "Ts"}; coef = m41.fit(Xcur, y, n, cw_all, ("Tw",)); coef[15] = (coef[10] + coef[20]) / 2
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c104b = (sub(k) for k in ("ct9", "ct28", "ct104b")); locked = c28 != c9; l28 = rk(c28); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}
    for name in NAMES:
        idn, enc, cc, temp, pi0, bmin, sc = rec2[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)
    lr, _, _ = la.reading(tq, protos, plab, la.BE); lr = lr - lr.mean(0, keepdims=True)
    X0 = {"logp": np.log(np.maximum(P9, 1e-12)), "R": lr, **{k: LQ[k] for k in NAMES}}; X0fit = {k: v for k, v in Xcur.items() if k != "Tw"}; c0 = mb7.fit(X0fit, y, n, cw_all, "L0"); c0[15] = ((c0[10][0] + c0[20][0]) / 2, (c0[10][1] + c0[20][1]) / 2)
    d0 = mf7.guard2(np.where(locked, l28, np.asarray(b.decide(mb7.apply(c0, X0, nq_t)), int)), rk(c104b).astype(int)); assert (lev[d0] == c106).all(), int((lev[d0] != c106).sum()); print("checks: without the new input the builder reproduces ct106")
    Pl = m41.apply(coef, {**X0, "Tw": Tw_t}, nq_t, ("Tw",)); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin = lev[mf7.guard2(d_plain, d106)]; assert (fin[locked] == c28[locked]).all()
    np.savez(ROOT / f"results/r_variants/m155_test_posterior_{a_.name}.npz", qid=qids, learned=Pl, base=P9)
    print("coefficients on all folds: " + " | ".join(f"n={s}: alpha {coef[s][0]:.2f}, v5 {coef[s][3]:.2f}, R {coef[s][7]:.2f}, twin {coef[s][8]:+.2f}" for s in SIZES))
    for nm, o in (("ct106", c106), ("ct104b", c104b)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    for i in np.flatnonzero(fin != c106):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct106 {c106[i]} -> {fin[i]}; nearest prototype {LEVELS[L1_t[i]]} (similarity {s1_t[i]:.3f}, gap {gap_t[i]:.3f}, band {band_t[i]})")
    f = ROOT / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]
    print(f"{f.name} written (not copied to upload_next), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct106": qids[fin != c106].tolist(), "public": int(((fin != c106) & pub).sum()), "private": int(((fin != c106) & ~pub).sum())}
    (ROOT / "results/r_variants/m155_twin_reading.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
