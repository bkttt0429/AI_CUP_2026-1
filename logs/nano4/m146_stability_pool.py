#!/usr/bin/env python
"""Section 4M M146 (2026-11-15): anchor tables split by the recognised candidate's own stability, as one more input of the learned
pool (design doc M146; follows the pilot M145).

Stability of a candidate X: among X's own identified even-game pairs (results/r_variants/m96_pairs.npz: every training row's
opponent side matched by the all-players networks, margin >= q90, weight = the tier's true-pair share), the weighted share of
opponents with X's label, shrunk towards the mean of X's level (kappa 5), minus that mean. For a reading (target T, candidate X)
the pairs of X whose opponent is T are left out, and the level means leave out T's fold, so T's label never enters.
Tables: P(label of the candidate | label of the target) from the near-certain even readings (band >= 5) of the other folds,
pooled and per tercile of the candidate's stability (Dirichlet smoothing towards the pooled table, kappa 30).
New input of the pool: per question the sum over its even games with band >= 4 and a known tercile of
a_band x (log T_tercile(L_X | c) - log T_pooled(L_X | c)), tempered like the other evidence; one coefficient per size, start 0.
REGISTERED: the pool with this input against the ct106 rule (four guards relative to it), M37 on seeds 0-2 and 6-8.
Report: share of readings with a known tercile, same-level rate by tercile, held-out log-likelihood gain per reading.

  python -I logs/nano4/m146_stability_pool.py
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
import m87_pi0 as m87  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE; RV = ROOT / "results/r_variants"
IDN = "SP2e"; KAP, KAP2, NMIN = 5.0, 30.0, 3.0


def main():
    pj = json.load(open(RV / "m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    zc = V[f"{IDN}~{STAT}"]; best = np.load(RV / f"m84_{IDN}~{STAT}_val.npz")["f_opp_best"].astype(str); sc_id = float(scale[IDN])
    pfc = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str); players = np.array(sorted(pfc.player_id)); pidx = {p: i for i, p in enumerate(players)}; pf_ = pfc.set_index("player_id"); pfold_s = pf_.fold.astype(int).reindex(players).to_numpy(); plab_s = pf_["rank"].map(b.RANK).reindex(players).to_numpy().astype(int)
    pr = np.load(RV / "m96_pairs.npz"); ow, op, w = pr["owner"], pr["opp"], pr["weight"]; same = (plab_s[ow] == plab_s[op]).astype(float); C = len(players)
    S_den = np.bincount(ow, weights=w, minlength=C); S_num = np.bincount(ow, weights=w * same, minlength=C)
    pk = pd.DataFrame({"k": ow.astype(np.int64) * C + op, "w": w, "ws": w * same}).groupby("k").sum(); P_den, P_num = pk.w.to_dict(), pk.ws.to_dict()
    MU = {}  # level means of the same-level share, leaving out a fold's owners and opponents
    for v in FOLDS:
        s = (pfold_s[ow] != v) & (pfold_s[op] != v); MU[v] = np.bincount(plab_s[ow][s], weights=(w * same)[s], minlength=10) / np.maximum(np.bincount(plab_s[ow][s], weights=w[s], minlength=10), 1e-9)
    print(f"identified pairs: {len(ow)}; owners with weight >= {NMIN:g}: {int((S_den >= NMIN).sum())}; level means of the same-level share (fold 1 left out): " + " ".join(f"{LEVELS[c]} {MU[1][c]:.2f}" for c in range(10)))
    cand_i = np.array([pidx.get(p, -1) for p in best]); tgt_i = np.array([pidx.get(p, -1) for p in zc["pid"].astype(str)]); assert (cand_i >= 0).all() and (tgt_i >= 0).all() and (plab_s[cand_i] == zc["opp_lab"]).all()
    key2 = cand_i.astype(np.int64) * C + tgt_i; den = S_den[cand_i] - np.array([P_den.get(k, 0.0) for k in key2]); num = S_num[cand_i] - np.array([P_num.get(k, 0.0) for k in key2])
    mu_r = np.stack([MU[v] for v in FOLDS])[np.clip(zc["fold"].astype(int), 1, 4) - 1, plab_s[cand_i]]; rel = (num + KAP * mu_r) / (den + KAP) - mu_r; known = den >= NMIN
    ones = np.ones(len(zc["y"]), bool); _, _, band_all, g_all = s2.fit(zc, ones); conf = (band_all >= 5) & (g_all == 0); cuts = np.quantile(rel[conf & known], [1 / 3, 2 / 3]); ter = np.where(known, np.searchsorted(cuts, rel, side="right"), -1)
    d = (zc["opp_lab"] - zc["y"]).astype(int)
    print(f"near-certain even readings: {int(conf.sum())}; candidate's stability known for {float(known[conf].mean()):.3f} of them; cuts of the level-adjusted stability {cuts[0]:+.3f} / {cuts[1]:+.3f}")
    print("same-level rate | within one | n by tercile (near-certain even readings): " + "  ".join(f"{nm}: {float((d[conf & (ter == k)] == 0).mean()):.3f} | {float((np.abs(d[conf & (ter == k)]) <= 1).mean()):.3f} | {int((conf & (ter == k)).sum())}" for nm, k in (("unknown", -1), ("least stable", 0), ("middle", 1), ("most stable", 2))))
    LF = {v: m3.load_fold(v) for v in FOLDS}; z5 = V["5e"]; row = {k: i for i, k in enumerate(z5["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(z5["key"], zc["key"])
    players_q = q["players"].astype(str); plab = q["plab"]; pfold = pf_.fold.astype(int).reindex(players_q).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}; diag = []
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    Lg = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = Lg * Tn if temp else Lg
                Ls[A][name] = cache[spec]
        # the stability input: tables from the other folds, applied to this fold's games
        tr = zc["fold"] != v; edges, cal, _ = m87.corrected(zc, tr); _, _, band_tr, g_tr = s2.fit(zc, tr); sel = tr & (band_tr >= 5) & (g_tr == 0)
        cnt = np.zeros((4, 7)); np.add.at(cnt, (np.where(ter[sel] >= 0, ter[sel], 3), np.clip((zc["opp_lab"][sel] - zc["y"][sel]).astype(int), -3, 3) + 3), 1.0)
        pooled = (cnt.sum(0) + 1.0) / (cnt.sum() + 7.0); Ts = (cnt[:3] + 1.0) / (cnt[:3].sum(1, keepdims=True) + 7.0); dl = np.clip(np.log(Ts) - np.log(pooled)[None], -3.0, 3.0)  # [tercile, 7 cells of candidate label - target level]; 2026-11-15 06:10: the first version (10 x 10 cells per tercile, shrunk towards the pooled table) had the sparse-cell flaw found in M156
        te = np.flatnonzero(zc["fold"] == v); tb = zc["target_black"][te]; bnd = np.where(tb, np.searchsorted(edges[False], zc["opp_m"][te]), np.searchsorted(edges[True], zc["opp_m"][te])); gg = s2.group_of(zc["hd"][te], tb)
        use = (gg == 0) & (bnd >= 4) & (ter[te] >= 0); a = sc_id * cal[0]["a"][np.minimum(bnd, len(cal[0]["a"]) - 1)]; ev_g = np.zeros((len(te), 10)); ev_g[use] = a[use, None] * dl[ter[te][use][:, None], np.clip(zc["opp_lab"][te][use].astype(int)[:, None] - np.arange(10)[None, :], -3, 3) + 3]
        evd = dict(zip(zc["key"][te], ev_g)); Es = np.stack([sum((evd.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]) * Tn
        hc = (gg == 0) & (bnd >= 5) & (ter[te] >= 0); yy = zc["y"][te].astype(int); ll = zc["opp_lab"][te].astype(int); gain = dl[ter[te][hc], np.clip(ll[hc] - yy[hc], -3, 3) + 3]; diag.append((int(hc.sum()), float(gain.mean()) if hc.any() else 0.0, float(use.sum() / max(((gg == 0) & (bnd >= 4)).sum(), 1))))
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Es=Es, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)
        print(f"fold {v}: inputs ready; near-certain even readings with a known tercile {diag[-1][0]}, held-out log-likelihood gain per such reading {diag[-1][1]:+.4f} nats; share of the readings (band >= 4) that get the correction {diag[-1][2]:.3f}; questions with a non-zero input {float((np.abs(Es).sum(1) > 0).mean()):.3f}", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {"diag": diag}; ok = []
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731

    def pool_decisions(X, extra=()):
        out = {}
        for v in FOLDS:
            trf = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in trf]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in trf]), np.concatenate([Fd[u]["n"] for u in trf]), np.concatenate([Fd[u]["cw"] for u in trf]), extra)
            out[v] = (np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], extra)), int), coef)
        return out
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], "Es": f["Es"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f"\nseeds {seeds}: ct106 rule {bal(S_old):.4f}")
        dd = pool_decisions(X, ("Es",)); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
        pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]; good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); ok.append(bool(good))
        print(f"  + stability input: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; coefficient of the input " + " ".join(f"n={s}: {np.mean([dd[v][1][s][8] for v in FOLDS]):+.2f}" for s in SIZES), flush=True)
        rep[f"seeds_{seeds[0]}"] = {**d_, "per_fold": pf, "meets_m37": bool(good)}
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets, the pool with the stability input against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (RV / "m146_stability_pool.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m146_stability_pool.json")


if __name__ == "__main__":
    main()
