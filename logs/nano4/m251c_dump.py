#!/usr/bin/env python
"""M250 (2026-11-19), report only: the identified opponents' OWN base-model offsets (out-of-fold base expected level minus label, centred inside
the label) as a level tilt of the pool (design doc M250):
  tilto = (sum of the offsets of the clear anchors with a known offset / (their number + 1) - its mean at this question size) * (level - 4.5)
  tiltp = the same with the offsets permuted among the players of the same label (control)
This file is a copy of m249_pool.py (itself a copy of m232_pool.py); the original header follows.
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
import re
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


MOVE = re.compile(r";([BW])\[([a-z]*)\]"); HDRE = re.compile(r"HD\[(\d+)\]")


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
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)
        RD = {k: lookalike(mem, Q + gi, qi, len(y), v - 1, cand) for k, mem in MEM.items()}
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        bc0 = best_of(sp0[0]); anc = [[(bc0.get(str(k)), str(k)) for k in g_ if k in ev0 and float(np.abs(ev0[k]).max()) > 0.5] for g_ in gk]
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, sq=sq, cq=cq, mq=mq, anc=anc, cw=cw, wq=np.array([W_FIN[int(m)] for m in n]), **RD)
        print(f"fold {v}: inputs ready ({len(y)} questions); look-alike reading alone, level of its maximum equals the truth: " + ", ".join(f"{k} {float((RD[k].argmax(1) == y).mean()):.3f}" for k in MEM), flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); cw_all = np.concatenate([Fd[v]["cw"] for v in FOLDS]); rep = {"density_ratio": xscale}; ok = []; keep = {}
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    base_of = lambda v, seeds: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # noqa: E731

    def pool_decisions(X, extra=()):
        out = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coef = m41.fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra)
            post_ = m41.apply(coef, X[v], Fd[v]["n"], extra); out[v] = (np.asarray(b.decide(post_), int), coef, post_)
        return out
    VARI = [(" + ".join(s_) + tag, ex) for s_ in sets for tag, ex in (("", s_), (" | offset tilt", (*s_, "tilto")), (" | permuted offset tilt", (*s_, "tiltp")))]
    kc = (np.arange(10) - 4.5)[None, :]; sq_a, cq_a, mq_a = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("sq", "cq", "mq")); ms_ = {s: float(np.nanmean(sq_a[n == s])) for s in SIZES}; mc_ = {s: float(cq_a[n == s].mean()) for s in SIZES}
    print("questions with >= 3 clear anchors (" + xs[0] + "), by size: " + ", ".join(f"{s}: {float(np.isfinite(sq_a[n == s]).mean()):.3f} (mean sd of the readings {ms_[s]:.3f}, mean number of clear anchors {mc_[s]:.2f})" for s in SIZES), flush=True)
    for v in FOLDS:
        f = Fd[v]; t1 = np.zeros((len(f["y"]), 10)); t2 = np.zeros_like(t1)
        for s in SIZES:
            m = f["n"] == s; t1[m] = np.where(np.isnan(f["sq"][m]), 0.0, f["sq"][m] - ms_[s])[:, None] * kc; t2[m] = (f["cq"][m] - mc_[s])[:, None] * kc
        f["tilt"], f["tiltn"] = t1, t2
    ok_ = np.isfinite(sq_a); cen_ = lambda a: a - np.array([a[ok_ & (y == c)].mean() if (ok_ & (y == c)).any() else 0.0 for c in range(10)])[y]  # noqa: E731  centred inside the true level
    xs_, ys_ = cen_(np.where(ok_, sq_a, 0.0)), cen_(np.where(ok_, mq_a - y, 0.0))
    for lab_, msk in (("all levels", ok_), ("low  D-A", ok_ & (y <= 3)), ("high 1D-6D", ok_ & (y >= 4))):
        print(f"  validation questions, (mean reading - true level) on the sd of the readings, centred inside the true level, {lab_:12s}: slope {float((xs_[msk] * ys_[msk]).sum() / (xs_[msk] ** 2).sum()):+.3f}  r {float(np.corrcoef(xs_[msk], ys_[msk])[0, 1]):+.3f}  ({int(msk.sum())} questions)", flush=True)
    plab_of = dict(zip(players.tolist(), plab.astype(int).tolist())); fold_of = dict(zip(players.tolist(), pfold.tolist()))
    for seeds in ((0, 1, 2), (6, 7, 8)):
        num, den, lab_ = {}, {}, {}
        for v in FOLDS:
            yv, nv, pv, _ = LF[v]; Ev = base_of(v, seeds) @ np.arange(10.0)
            for p_, yy, nn, e_ in zip(np.asarray(pv).astype(str), np.asarray(yv), np.asarray(nv), Ev):
                num[p_] = num.get(p_, 0.0) + nn * (e_ - yy); den[p_] = den.get(p_, 0.0) + nn; lab_[p_] = int(yy)
        raw = {p_: num[p_] / den[p_] for p_ in num}; mu_l = {c: float(np.mean([raw[p_] for p_ in raw if lab_[p_] == c])) for c in range(10)}; off = {p_: raw[p_] - mu_l[lab_[p_]] for p_ in raw}
        rng = np.random.default_rng(20261119); offp = {}
        for c in range(10):
            ps_ = sorted(p_ for p_ in off if lab_[p_] == c); offp.update(zip(ps_, rng.permutation([off[p_] for p_ in ps_]).tolist()))
        ov = np.array(list(off.values()))
        print(f"\nseeds {seeds}: out-of-fold offsets of {len(off)} players (base expected level - label): mean by label " + " ".join(f"{mu_l[c]:+.2f}" for c in range(10)) + f"; centred sd {float(ov.std()):.3f}", flush=True)
        fo, fp, ko, pairs_o, pairs_d, n_anc, n_known, n_same = {}, {}, {}, [], [], 0, 0, 0
        for v in FOLDS:
            f = Fd[v]; a1 = np.zeros(len(f["y"])); a2 = np.zeros(len(f["y"])); kk_ = np.zeros(len(f["y"]))
            for j, lst in enumerate(f["anc"]):
                n_anc += len(lst); ks = [c for c, _ in lst if c in off]; n_known += len(ks); n_same += sum(fold_of.get(c) == v for c in ks)
                if ks:
                    a1[j] = sum(off[c] for c in ks) / (len(ks) + 1.0); a2[j] = sum(offp[c] for c in ks) / (len(ks) + 1.0); kk_[j] = len(ks)
                    pairs_o += [off[c] for c in ks]; pairs_d += [plab_of.get(c, lab_[c]) - int(f["y"][j]) for c in ks]
            fo[v], fp[v], ko[v] = a1, a2, kk_
        fo_a, fp_a, ko_a = (np.concatenate([d_[v] for v in FOLDS]) for d_ in (fo, fp, ko)); Pb_ = np.concatenate([base_of(v, seeds) for v in FOLDS]); res_ = y - Pb_ @ np.arange(10.0); mo_ = {s: float(fo_a[n == s].mean()) for s in SIZES}; mp_ = {s: float(fp_a[n == s].mean()) for s in SIZES}
        print(f"  clear anchors {n_anc}, with a known offset {n_known} ({n_known / max(n_anc, 1):.3f}; candidate in the question's own fold: {n_same}); questions with >= 1 known offset: " + ", ".join(f"n={s}: {float((ko_a[n == s] > 0).mean()):.3f}" for s in SIZES), flush=True)
        po_, pd_ = np.array(pairs_o), np.array(pairs_d, float); print(f"  per clear anchor: (candidate label - true level) on the candidate's offset: slope {float(np.polyfit(po_, pd_, 1)[0]):+.3f}, r {float(np.corrcoef(po_, pd_)[0, 1]):+.3f} ({len(po_)} anchors; mechanism (ii) predicts a negative slope)", flush=True)
        for lab2, arr in (("offset input", fo_a), ("permuted control", fp_a)):
            print(f"  base error (true level - base expected level) on the {lab2}, questions with >= 1 known offset: " + ", ".join(f"n={s}: sd of the input {float(arr[(n == s) & (ko_a > 0)].std()):.3f}, r {float(np.corrcoef(arr[(n == s) & (ko_a > 0)], res_[(n == s) & (ko_a > 0)])[0, 1]):+.3f}, slope {float(np.polyfit(arr[(n == s) & (ko_a > 0)], res_[(n == s) & (ko_a > 0)], 1)[0]):+.3f}" for s in SIZES), flush=True)
        m3_ = ko_a >= 3; print(f"  the same with >= 3 known offsets ({int(m3_.sum())} questions): r {float(np.corrcoef(fo_a[m3_], res_[m3_])[0, 1]):+.3f}, slope {float(np.polyfit(fo_a[m3_], res_[m3_], 1)[0]):+.3f}; control r {float(np.corrcoef(fp_a[m3_], res_[m3_])[0, 1]):+.3f}", flush=True)
        for v in FOLDS:
            f = Fd[v]; f["tilto"] = (fo[v] - np.array([mo_[int(m)] for m in f["n"]]))[:, None] * kc; f["tiltp"] = (fp[v] - np.array([mp_[int(m)] for m in f["n"]]))[:, None] * kc
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"], "tilt": f["tilt"], "tiltn": f["tiltn"], "tilto": f["tilto"], "tiltp": f["tiltp"]}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]
        print(f"\nseeds {seeds}: ct104b rule {bal([SM[Fd[v]['dc'], Fd[v]['y']] for v in FOLDS]):.4f}, ct106 rule {bal(S_old):.4f}")
        if seeds == (0, 1, 2):
            keep = {"X": X, "ref": ref}
        out = {}; kept = {}
        if a_.dump and seeds == (0, 1, 2):
            dump = {"y": y, "n": n, "pid": pid, "fold": np.concatenate([np.full(len(Fd[v]["y"]), v) for v in FOLDS]), "ct106_rule": np.concatenate([ref[v] for v in FOLDS]), "ct104b_rule": np.concatenate([Fd[v]["dc"] for v in FOLDS])}
        for name, extra in VARI:
            dd = pool_decisions(X, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]
            if name == VARI[0][0] and seeds == (0, 1, 2):
                np.savez(ROOT / "results/r_variants/m251_val_post.npz", y=y, n=n, pid=np.asarray(pid).astype(str), fold=np.concatenate([np.full(len(Fd[v]["y"]), v) for v in FOLDS]), post=np.concatenate([dd[v][2] for v in FOLDS]),
                         dec=np.concatenate([mf7.guard2(dd[v][0], ref[v]) for v in FOLDS]), plain=np.concatenate([dd[v][0] for v in FOLDS]))
                print(f"ct112 rule on the validation questions: {bal(new_):.4f} (without guards {bal(plain):.4f}); posterior and decisions written to m251_val_post.npz", flush=True); return
            A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]
            good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {name:36s}: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items())
                  + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}; coefficients (mean of the four fits) " + " | ".join(
                      f"n={s}: alpha {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}, v5 {np.mean([dd[v][1][s][3] for v in FOLDS]):.2f}, reading {np.mean([dd[v][1][s][7] for v in FOLDS]):.2f}" + (", new " + " ".join(f"{np.mean([dd[v][1][s][8 + j] for v in FOLDS]):+.2f}" for j in range(len(extra)))) for s in SIZES), flush=True)
            kept[name] = (A_, new_); out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good), "score": bal(new_), "score_plain": bal(plain)}
            if a_.dump and seeds == (0, 1, 2):
                dump[name] = np.concatenate([mf7.guard2(dd[v][0], ref[v]) for v in FOLDS])
        A0, n0 = kept[VARI[0][0]]
        for name, _ in VARI[1:]:
            A1, n1 = kept[name]; d2 = paired_diff(A1, A0, y, pid, n, sizes=SIZES, n=5000); pf2 = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(n1, n0, ys, ns)]
            print(f"    ({name}) minus ({VARI[0][0]}): {d2['diff']:+.4f} [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf2) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d2["by_n"].items()), flush=True); out[name]["minus_first"] = {**d2, "per_fold": pf2}
        rep[f"seeds_{seeds[0]}"] = out
    out_json.write_text(json.dumps(rep, indent=1, default=float)); print(f"\nREPORT ONLY (no file is built); written {out_json.name}")
    if a_.dump:
        np.savez(ROOT / "results/r_variants" / a_.dump, **{k.replace(" + ", "+"): v for k, v in dump.items()}); print(f"per-question decisions of seeds 0-2 written to {a_.dump}")


if __name__ == "__main__":
    main()
