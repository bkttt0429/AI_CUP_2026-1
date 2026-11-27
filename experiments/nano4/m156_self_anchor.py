#!/usr/bin/env python
"""Section 4M M156 (2026-11-15): target-side identification as evidence - per game, the production identifier's best training
candidate for the TARGET's side (design doc M156; follows M154 / M155).

The identifier files hold, next to the opponent-side match, the best candidate and margin of the target's own side with the
target himself left out (f_imp_*; on the test side f_tgt_*). So far it only served as the null distribution of the margins.
M154 showed that about a tenth of the players have a near-identical other id; then a confident target-side match is that twin
and its label is the target's level in 70-80% of the cases and never two levels away.
Evidence per game (identifier SP2e, statistic m15): bands of the target-side margin (the usual quantiles, per colour, from the
other folds); per band the distribution of d = candidate's label - true level from the other folds' games (smoothed towards the
lowest band's, kappa 50); log P_band(d) - log P_lowest(d) for every level; summed over a question's games, tempered like the
other evidence. One new pool coefficient per question size.
REGISTERED: the pool with this input against the ct106 rule (four guards relative to it), M37 on seeds 0-2 and 6-8.
Report: per band the share of games and the same-level / within-one rates; only bands >= 3 as a variant.

  python -I logs/nano4/m156_self_anchor.py [--name ct109] [--no-build]
"""
import argparse
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
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE; IDN = "SP2e"; KAP = 50.0; QS = np.array(s2.QS); NB = len(QS) + 1


def fit_tables(zc, tr):
    edges = {c: np.quantile(zc["imp_m"][tr & (zc["target_black"] == c)], QS) for c in (True, False)}
    band = np.where(zc["target_black"], np.searchsorted(edges[True], zc["imp_m"]), np.searchsorted(edges[False], zc["imp_m"])); d = np.clip((zc["imp_lab"] - zc["y"]).astype(int), -3, 3) + 3
    h = np.zeros((NB, 7)); np.add.at(h, (band[tr], d[tr]), 1.0); p0 = (h[0] + 1.0) / (h[0].sum() + 7.0); pb = (h + 1.0) / (h.sum(1, keepdims=True) + 7.0)  # 2026-11-15 06:05: differences of three or more levels are one cell, the same add-one smoothing on both sides, the log ratio capped (the first version blew up in the sparse tail cells)
    return edges, np.clip(np.log(pb) - np.log(p0)[None], -3.0, 3.0), h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct109"); ap.add_argument("--no-build", action="store_true"); a_ = ap.parse_args()
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    zc = V[f"{IDN}~{STAT}"]; ones = np.ones(len(zc["y"]), bool); _, _, hh = fit_tables(zc, ones); edges_all, _, _ = fit_tables(zc, ones)
    band_all = np.where(zc["target_black"], np.searchsorted(edges_all[True], zc["imp_m"]), np.searchsorted(edges_all[False], zc["imp_m"])); dd_ = (zc["imp_lab"] - zc["y"]).astype(int)
    print("target-side match by band of its margin (all validation games): share | candidate has the same level / within one / two or more")
    for k in range(NB):
        m = band_all == k; print(f"   band {k}: {m.mean():.4f} | {float((dd_[m] == 0).mean()):.3f} / {float((np.abs(dd_[m]) <= 1).mean()):.3f} / {float((np.abs(dd_[m]) >= 2).mean()):.3f}")
    LF = {v: m3.load_fold(v) for v in FOLDS}; z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(z["key"], zc["key"])
    players = q["players"].astype(str); plab = q["plab"].astype(int); pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
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
        tr = zc["fold"] != v; edges, lr, _ = fit_tables(zc, tr); te = np.flatnonzero(zc["fold"] == v); tb = zc["target_black"][te]; bd = np.where(tb, np.searchsorted(edges[True], zc["imp_m"][te]), np.searchsorted(edges[False], zc["imp_m"][te])); lab = zc["imp_lab"][te].astype(int)
        eg = lr[bd[:, None], np.clip(lab[:, None] - np.arange(10)[None, :], -3, 3) + 3]; eg3 = eg * (bd >= 3)[:, None]; e1 = dict(zip(zc["key"][te], eg)); e3 = dict(zip(zc["key"][te], eg3)); hb = dict(zip(zc["key"][te], bd))
        Ta = np.stack([sum((e1.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]) * Tn; Ta3 = np.stack([sum((e3.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]) * Tn; nhi = np.array([sum(hb.get(k, 0) >= 5 for k in g_) for g_ in gk])
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw, Ta=Ta, Ta3=Ta3, nhi=nhi)
        print(f"fold {v}: inputs ready; questions with at least one near-certain target-side match (band >= 5): " + " ".join(f"n={s}: {float((nhi[n == s] > 0).mean()):.3f}" for s in SIZES) + f"; the reading alone (argmax of the summed evidence) on those questions: exact {float((Ta.argmax(1) == y)[nhi > 0].mean()):.3f}, within one {float((np.abs(Ta.argmax(1) - y) <= 1)[nhi > 0].mean()):.3f}", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; ok = []
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
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], "Ta": f["Ta"], "Ta3": f["Ta3"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f"\nseeds {seeds}: ct106 rule {bal(S_old):.4f}"); out = {}
        nhi = np.concatenate([Fd[v]["nhi"] for v in FOLDS])
        for name, extra in (("PRIMARY: + target-side evidence (all bands)", ("Ta",)), ("+ target-side evidence, bands >= 3 only", ("Ta3",))):
            dd = pool_decisions(X, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]; A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]; good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {name:44s}: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}; coefficients " + " | ".join(
                f"n={s}: alpha {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}, v5 {np.mean([dd[v][1][s][3] for v in FOLDS]):.2f}, R {np.mean([dd[v][1][s][7] for v in FOLDS]):.2f}, new {np.mean([dd[v][1][s][8] for v in FOLDS]):+.2f}" for s in SIZES), flush=True)
            if name.startswith("PRIMARY"):
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)) + f" | questions with a near-certain target-side match: {float((A_ - B_)[nhi > 0].mean()):+.4f} (share {float((nhi > 0).mean()):.3f}), others {float((A_ - B_)[nhi == 0].mean()):+.4f}", flush=True)
            out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good)}
        rep[f"seeds_{seeds[0]}"] = out
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets, the pool with the target-side evidence against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    tt = np.load(ROOT / f"results/r_variants/m84_{IDN}~{STAT}_test.npz"); tbt = np.where(tt["target_black"], np.searchsorted(edges_all[True], tt["f_tgt_m"]), np.searchsorted(edges_all[False], tt["f_tgt_m"]))
    print("test side: share of the games per band of the target-side margin (validation: 0.5 / 0.3 / 0.1 / 0.05 / 0.03 / 0.01 / 0.005 / 0.004 / 0.001): " + " ".join(f"{k}: {float((tbt == k).mean()):.4f}" for k in range(NB)) + f"; test questions with at least one game in the bands >= 5: {int(pd.Series(tbt >= 5).groupby(tt['qid'].astype(str)).any().sum())} of 800")
    (ROOT / "results/r_variants/m156_self_anchor.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m156_self_anchor.json")
    if passed and not a_.no_build:
        print("PASSED: the test-side builder is not part of this script - build it next (same recipe on f_tgt_lab / f_tgt_m, tables from all folds).")


if __name__ == "__main__":
    main()
