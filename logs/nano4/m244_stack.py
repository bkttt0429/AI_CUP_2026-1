#!/usr/bin/env python
"""Section 4M M244 (2026-11-19), report only: is anything left in the pool's own inputs that the linear pool does not use?
Boosted trees (xgboost, multi-class, base margin = log of the linear pool's posterior) on every pool input (base log posterior,
look-alike reading, the five voters' evidence, the new source's evidence, the pooled log posterior) and ten question-level
scalars; nested cross-fitting (the training rows carry pooled posteriors whose coefficients saw neither their own fold nor the
target fold); depth and rounds chosen inside the training folds.   python -I logs/nano4/m244_stack.py --sets SQ4 --json m244_stack.json
(The file is a copy of m243_gate.py; its original header follows.)
Section 4M M243 (2026-11-19), report only: should the look-alike reading and the base posterior get a different weight
in questions WITHOUT clear anchor evidence? Extra pool inputs = the look-alike reading / the base log posterior multiplied by the
indicator "no clear anchor" (or "at most two"), one coefficient per question size each; paired difference against the pool of the
same identifier.   python -I logs/nano4/m243_gate.py --sets SQ4 --json m243_pool.json
(The file is a copy of m232_pool.py; its original header follows.)
Section 4M M232 part B (2026-11-18), report only: does the SPREAD of a question's anchor readings tilt the answer?
A copy of m231_dedup.py. If the label were the peak rank of the period (not a centre), a question whose anchors are spread
widely should be answered above the centre of its anchors; the pool multiplies the anchors' evidence and so estimates the
centre. New pool inputs (one coefficient per question size, as every extra input):
  tilt  = (s - mean of s at this question size) * (level - 4.5); s = sd of the readings of the anchors with clear evidence
          (max |evidence| > 0.5; reading = expected level of the evidence after removing its flat floor), needs >= 3, else 0
  tiltn = (number of clear anchors - its mean at this question size) * (level - 4.5)   (control)
Compared with the pool of the same identifier without the new inputs by the paired difference.

  python -I logs/nano4/m232_pool.py --sets SQ4 --json m232_pool.json
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
import m105_voters as m5  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m141_opp_lookalike as m41  # noqa: E402  (fit / apply of the L0 pool with extra inputs)
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--json", default="m165_pool.json"); ap.add_argument("--dump", default=None, help="npz for the per-question decisions of seeds 0-2 (truth, size, player, the ct106 rule, every set)"); a_ = ap.parse_args(); sets = [tuple(x.split("+")) for x in a_.sets]
    out_json = ROOT / "results/r_variants" / a_.json
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    xs = list(dict.fromkeys(t for s_ in sets for t in s_))
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], *[f"{x}~{STAT}" for x in xs]])
    miss = [x for x in xs if f"{x}~{STAT}" not in used]
    if miss:
        sys.exit(f"identifier files missing: {miss}")
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    dr = lambda x: round(m5.density_ratio(V[f"{x}~{STAT}"], T[f"{x}~{STAT}"], key_pid, dec, ct9s, nominal), 3)  # noqa: E731
    chk = {x: dr(x) for x in ids}; assert all(abs(chk[x] - scale[x]) < 2e-3 for x in ids), (chk, scale)  # the stored ratios of the five voters are reproduced
    xscale = {x: dr(x) for x in xs}; print("test-side density ratio (m15): " + ", ".join(f"{x} {v:.3f}" for x, v in {**chk, **xscale}.items()), flush=True)
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    xspec = {x: (f"{x}~{STAT}", 1, 1, 1, 1, 1, float(xscale[x])) for x in xs}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz")
    cur = [(v_["qvec"], v_["protos"], m) for m in range(v_["qvec"].shape[0])]
    MEM = {"R": cur}; print(f"look-alike members: {len(cur)} ({la.VAR}), unchanged", flush=True)

    def lookalike(mem, idx, owner, nq, net, cand):
        """Centred log reading [nq, 10]: mean target-side vector of each question (queries idx, question index owner) against the prototypes of network `net`."""
        tq = np.zeros((len(mem), nq, 128))
        for j, (qv, _, m) in enumerate(mem):
            np.add.at(tq[j], owner, qv[m, idx].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        lr_, _, _ = la.reading(tq, [(pr[m, net] if cand is None else pr[m, net][cand]).astype(np.float64) for _, pr, m in mem], plab if cand is None else plab[cand], la.BE)
        return lr_ - lr_.mean(0, keepdims=True)
    Fd = {}; REPEAT = {}; BESTC = {}

    def best_of(idn):
        if idn not in BESTC:
            z_ = np.load(ROOT / "results/r_variants" / ("m82_val.npz" if idn == "5e" else f"m84_{idn}_val.npz")); BESTC[idn] = dict(zip(z_["key"].astype(str).tolist(), z_["f5e_opp_best" if idn == "5e" else "f_opp_best"].astype(str).tolist()))
        return BESTC[idn]
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}

        def ev_of(spec, dedup=False):
            if (spec, dedup) not in cache:
                idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                if not dedup:
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk])
                else:
                    bc = best_of(idn); rows_ = []
                    for g_ in gk:
                        grp = {}
                        for k in g_:
                            if k in ev:
                                grp.setdefault(bc.get(str(k), str(k)), []).append(ev[k])
                        rows_.append(sum((np.mean(vs, 0) for vs in grp.values()), m3.Z10) + b.U_ENC * sum((ee.get(k, m3.Z10) for k in g_), m3.Z10))
                    L = np.stack(rows_)
                cache[(spec, dedup)] = L * Tn if temp else L
            return cache[(spec, dedup)]
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}
        Lx = {x: ev_of(xspec[x]) for x in xs}
        sp0 = xspec[xs[0]]; ev0 = m3.evidence(V[sp0[0]], v, sp0[2], sp0[4], sp0[5], sp0[6]); sq = np.full(len(gk), np.nan); cq = np.zeros(len(gk)); mq = np.full(len(gk), np.nan)
        for j, g_ in enumerate(gk):
            E_ = np.array([ev0[k] for k in g_ if k in ev0 and float(np.abs(ev0[k]).max()) > 0.5]); cq[j] = len(E_)
            if len(E_) >= 3:
                w_ = np.exp(E_); w_ = w_ - w_.min(1, keepdims=True); mu_ = (w_ * np.arange(10)).sum(1) / np.maximum(w_.sum(1), 1e-12); sq[j] = float(mu_.std(ddof=1)); mq[j] = float(mu_.mean())
        hs = np.array([float(np.mean([z["hd"][row[k]] > 0 for k in g_ if k in row] or [0.0])) for g_ in gk]); bs = np.array([float(np.mean([bool(z["target_black"][row[k]]) for k in g_ if k in row] or [0.5])) for g_ in gk])
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)
        RD = {k: lookalike(mem, Q + gi, qi, len(y), v - 1, cand) for k, mem in MEM.items()}
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, sq=sq, cq=cq, mq=mq, hs=hs, bs=bs, cw=cw, wq=np.array([W_FIN[int(m)] for m in n]), **RD)
        print(f"fold {v}: inputs ready ({len(y)} questions); look-alike reading alone, level of its maximum equals the truth: " + ", ".join(f"{k} {float((RD[k].argmax(1) == y).mean()):.3f}" for k in MEM), flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); cw_all = np.concatenate([Fd[v]["cw"] for v in FOLDS]); rep = {"density_ratio": xscale}; ok = []; keep = {}
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    base_of = lambda v, seeds: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # noqa: E731

    def pool_decisions(X, extra=()):
        out = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra)
            out[v] = (np.asarray(b.decide(m41.apply(coef, X[v], Fd[v]["n"], extra)), int), coef)
        return out
    VARI = [(" + ".join(s_) + tag, ex) for s_ in sets for tag, ex in (("", s_), (" | look-alike gated on NO clear anchor", (*s_, "R0")), (" | base gated on NO clear anchor", (*s_, "lp0")), (" | both gated on NO clear anchor", (*s_, "R0", "lp0")),
                                                                        (" | look-alike gated on <= 2 clear anchors", (*s_, "R2")), (" | both gated on <= 2 clear anchors", (*s_, "R2", "lp2")))]
    cq_a = np.concatenate([Fd[v]["cq"] for v in FOLDS])
    print("validation questions without a clear anchor (" + xs[0] + ") / with at most two, by size: " + ", ".join(f"{s}: {float((cq_a[n == s] == 0).mean()):.3f} / {float((cq_a[n == s] <= 2).mean()):.3f}" for s in SIZES), flush=True)
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"], "R0": f["R"] * (f["cq"] == 0)[:, None], "R2": f["R"] * (f["cq"] <= 2)[:, None]}; X[v]["lp0"] = X[v]["logp"] * (f["cq"] == 0)[:, None]; X[v]["lp2"] = X[v]["logp"] * (f["cq"] <= 2)[:, None]
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]
        print(f"\nseeds {seeds}: ct104b rule {bal([SM[Fd[v]['dc'], Fd[v]['y']] for v in FOLDS]):.4f}, ct106 rule {bal(S_old):.4f}")
        if seeds == (0, 1, 2):
            keep = {"X": X, "ref": ref}
        import xgboost as xgb
        extra = (xs[0],); keys = ["logp", "R", *NAMES, xs[0]]; KKv = np.arange(10.0)
        SCAL = ["n_games", "clear_anchors", "sd_of_readings", "mean_reading", "base_entropy", "base_max", "handicap_share", "black_share", "pool_max", "pool_minus_base_level"]; FN = [f"{k}_{c}" for k in [*keys, "logpool"] for c in range(10)] + SCAL

        def feats(v, Pp):
            f = Fd[v]; lp = X[v]["logp"]; P = np.exp(lp); P = P / P.sum(1, keepdims=True)
            sc_ = np.column_stack([f["n"], f["cq"], np.nan_to_num(f["sq"], nan=-1.0), np.nan_to_num(f["mq"], nan=-1.0), -(P * np.log(np.maximum(P, 1e-12))).sum(1), P.max(1), f["hs"], f["bs"], Pp.max(1), Pp @ KKv - P @ KKv])
            return np.column_stack([*[X[v][k] for k in keys], np.log(np.maximum(Pp, 1e-12)), sc_])

        def pool_post(tr_folds, v_apply):
            Xt_ = {k: np.concatenate([X[u][k] for u in tr_folds]) for k in keys}; cf = m41.fit(Xt_, np.concatenate([Fd[u]["y"] for u in tr_folds]), np.concatenate([Fd[u]["n"] for u in tr_folds]), np.concatenate([Fd[u]["cw"] for u in tr_folds]), extra)
            return m41.apply(cf, X[v_apply], Fd[v_apply]["n"], extra)

        def dm(F, m_, yv_=None, w_=None):
            return xgb.DMatrix(F, label=yv_, weight=None if w_ is None else w_ / w_.mean(), base_margin=m_, feature_names=FN, nthread=12)
        CFG = [(d_, r_) for d_ in (2, 3) for r_ in (30, 100, 300)]; s_lin, s_stk = [], []; imp = {}; chosen = []
        for v in FOLDS:
            others = [u for u in FOLDS if u != v]; Pv = pool_post(others, v); Ptr = {u: pool_post([w for w in others if w != u], u) for u in others}
            Fu = {u: feats(u, Ptr[u]) for u in others}; Mu = {u: np.log(np.maximum(Ptr[u], 1e-12)) for u in others}; inner = {c: [] for c in CFG}
            for h in others:
                tr2 = [u for u in others if u != h]; Dt = dm(np.concatenate([Fu[u] for u in tr2]), np.concatenate([Mu[u] for u in tr2]), np.concatenate([Fd[u]["y"] for u in tr2]), np.concatenate([Fd[u]["cw"] for u in tr2])); Dh = dm(Fu[h], Mu[h])
                for d_ in (2, 3):
                    bst = xgb.train({"objective": "multi:softprob", "num_class": 10, "eta": 0.05, "max_depth": d_, "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 5.0, "lambda": 5.0, "nthread": 12, "seed": 244}, Dt, num_boost_round=300)
                    for r_ in (30, 100, 300):
                        inner[(d_, r_)].append(balanced_mean_over_n(SM[np.asarray(b.decide(bst.predict(Dh, iteration_range=(0, r_))), int), Fd[h]["y"]], Fd[h]["y"], Fd[h]["n"], sizes=SIZES))
            lin_inner = float(np.mean([balanced_mean_over_n(SM[np.asarray(b.decide(Ptr[h]), int), Fd[h]["y"]], Fd[h]["y"], Fd[h]["n"], sizes=SIZES) for h in others]))
            best = max(CFG, key=lambda c: (round(float(np.mean(inner[c])), 5), -c[1], -c[0])); chosen.append((best, float(np.mean(inner[best]) - lin_inner)))
            Dall = dm(np.concatenate([Fu[u] for u in others]), np.concatenate([Mu[u] for u in others]), np.concatenate([Fd[u]["y"] for u in others]), np.concatenate([Fd[u]["cw"] for u in others]))
            bst = xgb.train({"objective": "multi:softprob", "num_class": 10, "eta": 0.05, "max_depth": best[0], "subsample": 0.7, "colsample_bytree": 0.5, "min_child_weight": 5.0, "lambda": 5.0, "nthread": 12, "seed": 244}, Dall, num_boost_round=best[1])
            Pst = bst.predict(dm(feats(v, Pv), np.log(np.maximum(Pv, 1e-12)))); s_lin.append(SM[np.asarray(b.decide(Pv), int), Fd[v]["y"]]); s_stk.append(SM[np.asarray(b.decide(Pst), int), Fd[v]["y"]])
            for k_, g_ in bst.get_score(importance_type="total_gain").items():
                imp[k_] = imp.get(k_, 0.0) + g_
            print(f"   fold {v}: chosen depth {best[0]}, rounds {best[1]} (inner gain over the linear pool {chosen[-1][1]:+.4f}; all settings " + " ".join(f"d{c[0]}r{c[1]} {float(np.mean(inner[c]) - lin_inner):+.4f}" for c in CFG) + f"); answers changed {float((np.asarray(b.decide(Pst), int) != np.asarray(b.decide(Pv), int)).mean()):.3f}", flush=True)
        A_, B_ = np.concatenate(s_stk), np.concatenate(s_lin); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(s_stk, s_lin, ys, ns)]
        tot = sum(imp.values()); grp_ = {}
        for k_, g_ in imp.items():
            base_k = k_.rsplit("_", 1)[0] if k_ not in SCAL else k_; grp_[base_k] = grp_.get(base_k, 0.0) + g_ / max(tot, 1e-12)
        print(f"  M244 stacker (boosted trees on top of the linear pool, pool without guards {bal(s_lin):.4f}): {bal(s_stk):.4f}  minus the linear pool {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x:+.4f}" for m_, x in d_["by_n"].items()), flush=True)
        print("     share of the total gain by input: " + ", ".join(f"{k_} {g_:.3f}" for k_, g_ in sorted(grp_.items(), key=lambda kv: -kv[1])[:14]), flush=True)
        out = {"stacker": {**d_, "per_fold": pf, "chosen": chosen, "gain_share": grp_}}
        rep[f"seeds_{seeds[0]}"] = out
    out_json.write_text(json.dumps(rep, indent=1, default=float)); print(f"\nREPORT ONLY (no file is built); written {out_json.name}")
    if a_.dump:
        np.savez(ROOT / "results/r_variants" / a_.dump, **{k.replace(" + ", "+"): v for k, v in dump.items()}); print(f"per-question decisions of seeds 0-2 written to {a_.dump}")


if __name__ == "__main__":
    main()
