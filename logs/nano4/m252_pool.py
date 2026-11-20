#!/usr/bin/env python
"""M252 (2026-11-19), report only: within-question coincidence of the identified opponent (design doc M252).
A game whose top-1 candidate is also the top-1 of another game of the same question is flagged; its true-match share a is re-estimated
by cell (question size x group x band class) from the label-difference histogram (cross-fitted), a' = max(a_b, a_cell):
  rep  = sum over the flagged games of (log evidence with a' - log evidence with a_b), cooled like the other evidence
  repc = the same with the flags permuted inside the cell (control)
This file is a copy of m250_pool.py with the offset inputs replaced; the older headers follow.
M250 (2026-11-19), report only: the identified opponents' OWN base-model offsets (out-of-fold base expected level minus label, centred inside
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
    ap.add_argument("--json", default="m252_pool.json"); ap.add_argument("--dump", default=None, help="npz for the per-question decisions of seeds 0-2 (truth, size, player, the ct106 rule, every set)"); a_ = ap.parse_args(); sets = [tuple(x.split("+")) for x in a_.sets]
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
    spz = xspec[xs[0]]; z0 = V[spz[0]]; bcz = best_of(spz[0]); BEST = np.array([bcz[str(k)] for k in z0["key"]]); m83_, m87_ = m3.m83, m3.m87; assert np.array_equal(z0["key"], z["key"])
    CLS = np.array([0, 0, 1, 2, 3, 4, 4, 4, 4]); INST = {}; d0_ = np.clip(z0["opp_lab"].astype(int) - z0["y"].astype(int), -9, 9) + 9; AGR = np.linspace(0.0, 0.98, 99); TAB = {}; LREL = {}; HEAD = {}

    def a_ml(idx_d, K, P0):
        h = np.bincount(idx_d, minlength=19).astype(float); ll = (h[None, :] * np.log(AGR[:, None] * K[None, :] + (1 - AGR[:, None]) * P0[None, :] + 1e-12)).sum(1); return float(AGR[int(np.argmax(ll))])
    for u in FOLDS:
        _, nu_, _, qdu = LF[u]; rr, qq, ss, ff = [], [], [], []
        for j, g_ in enumerate(qdu["game_keys"]):
            r_ = np.array([row[k] for k in g_ if k in row], int)
            if len(r_):
                _, inv, cnt = np.unique(BEST[r_], return_inverse=True, return_counts=True); rr.append(r_); qq.append(np.full(len(r_), j)); ss.append(np.full(len(r_), int(nu_[j]))); ff.append(cnt[inv] >= 2)
        INST[u] = dict(row=np.concatenate(rr), q=np.concatenate(qq), size=np.concatenate(ss), flag=np.concatenate(ff))
    print("M252 flagged share of the (question, game) instances, validation: " + ", ".join(f"n={s}: {float(np.concatenate([INST[u]['flag'][INST[u]['size'] == s] for u in FOLDS]).mean()):.4f}" for s in SIZES), flush=True)
    tz = np.load(ROOT / "results/r_variants" / f"m84_{spz[0]}_test.npz"); tqid = tz["qid"].astype(str); tbest = tz["f_opp_best"].astype(str); tfl = np.zeros(len(tqid), bool); tsz = np.zeros(len(tqid), int)
    for q_ in np.unique(tqid):
        m_ = np.flatnonzero(tqid == q_); _, inv, cnt = np.unique(tbest[m_], return_inverse=True, return_counts=True); tfl[m_] = cnt[inv] >= 2; tsz[m_] = len(m_)
    print("M252 flagged share of the games, test (no labels): " + ", ".join(f"n={s}: {float(tfl[tsz == s].mean()):.4f} ({int((tsz == s).sum())} games)" for s in sorted(set(tsz.tolist()))), flush=True)
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
        tr0 = z0["fold"] != v; _, _, band0, g0 = s2.fit(z0, tr0); _, cal0, _ = m87_.corrected(z0, tr0); R0 = m83_.tables(z0, tr0, cal0, band0, g0, m83_.KAPPA); sc0, bmin0 = float(spz[6]), int(spz[5])
        ci = {k_: np.concatenate([INST[u][k_] for u in FOLDS if u != v]) for k_ in ("row", "size", "flag")}; cg, ccl = g0[ci["row"]], CLS[band0[ci["row"]]]; A1 = {}
        for s in SIZES:
            for k in (0, 1, 2):
                for c_ in range(5):
                    bm = (cg == k) & (ccl == c_); m1 = bm & (ci["size"] == s) & ci["flag"]
                    if m1.sum() < 40:
                        m1 = bm & ci["flag"]
                    A1[(s, k, c_)] = a_ml(d0_[ci["row"][m1]], cal0[k]["K"], cal0[k]["P0"]) if m1.sum() >= 40 else np.nan
                    mall = bm & (ci["size"] == s); TAB.setdefault((s, k, c_), []).append((int((bm & (ci["size"] == s) & ci["flag"]).sum()), A1[(s, k, c_)], a_ml(d0_[ci["row"][mall]], cal0[k]["K"], cal0[k]["P0"]) if mall.sum() >= 40 else np.nan, float(cal0[k]["a"][band0[ci["row"][mall]]].mean()) if mall.any() else np.nan))
        iv = INST[v]; rw = iv["row"]; kg = g0[rw]; bd = band0[rw]; cl = CLS[bd]; okg = (kg >= 0) & (kg <= 2); ab = np.zeros(len(rw)); ratio = np.ones((len(rw), 10))
        for k in (0, 1, 2):
            s_ = kg == k; ab[s_] = cal0[k]["a"][bd[s_]]; ratio[s_] = R0[k][:, z0["opp_lab"][rw[s_]]].T
        a_old = np.where(okg, sc0 * ab * (bd >= bmin0), 0.0); old_log = np.log((1 - a_old)[:, None] + a_old[:, None] * ratio)
        chk_ = np.array([ev0[k_] for k_ in z0["key"][rw[:4000]]]); assert np.allclose(chk_, old_log[:4000], atol=1e-8), float(np.abs(chk_ - old_log[:4000]).max())  # the production evidence is reproduced
        ah = np.array([A1.get((int(s), int(k), int(c_)), np.nan) for s, k, c_ in zip(iv["size"], kg, cl)]); rngc = np.random.default_rng(2520 + v); flc = iv["flag"].copy()
        for cell in set(zip(iv["size"].tolist(), kg.tolist(), cl.tolist())):
            m_ = np.flatnonzero((iv["size"] == cell[0]) & (kg == cell[1]) & (cl == cell[2])); flc[m_] = rngc.permutation(iv["flag"][m_])
        Tq = np.array([b.TN[int(m)] for m in n])[:, None]; res_ = {}
        for nm_, fl_ in (("rep", iv["flag"]), ("repc", flc)):
            up = fl_ & okg & np.isfinite(ah); a_new = np.where(up, sc0 * np.maximum(ab, np.nan_to_num(ah)), a_old); dl = np.log((1 - a_new)[:, None] + a_new[:, None] * ratio) - old_log; L_ = np.zeros((len(y), 10)); np.add.at(L_, iv["q"], dl); res_[nm_] = L_ * Tq
            if nm_ == "rep":
                an_true = a_new
        Fd[v]["rep"], Fd[v]["repc"] = res_["rep"], res_["repc"]
        fl = iv["flag"] & okg; dlab = np.abs(z0["opp_lab"][rw].astype(int) - z0["y"][rw].astype(int))
        for c_ in range(5):
            for f_ in (0, 1):
                m_ = (kg == 0) & (cl == c_) & (fl == bool(f_)); e_ = LREL.setdefault((c_, f_), [0, 0, 0]); e_[0] += int(m_.sum()); e_[1] += int((dlab[m_] == 0).sum()); e_[2] += int((dlab[m_] <= 1).sum())
        for s in SIZES:
            ms_q = n == s; mi = iv["size"] == s; h_ = HEAD.setdefault(s, [0, 0, 0.0, 0.0, 0, 0, 0]); h_[0] += int(ms_q.sum()); h_[1] += int((mi & fl).sum()); h_[2] += float(a_old[mi & fl].sum()); h_[3] += float(an_true[mi & fl].sum()); h_[4] += int((mi & fl & (an_true - a_old >= 0.2)).sum())
            grp = {}
            for q_, c_id, ao, an, f_ in zip(iv["q"][mi], BEST[rw[mi]], a_old[mi], an_true[mi], fl[mi]):
                if f_:
                    e_ = grp.setdefault((int(q_), c_id), [0.0, 0.0]); e_[0] = max(e_[0], float(ao)); e_[1] = max(e_[1], float(an))
            h_[5] += len(grp); h_[6] += sum(1 for ao, an in grp.values() if ao < 0.5 * sc0 and an >= 0.5 * sc0)
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
    VARI = [(" + ".join(s_) + tag, ex) for s_ in sets for tag, ex in (("", s_), (" | coincidence", (*s_, "rep")), (" | shuffled coincidence", (*s_, "repc")))]
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
    print("\nM252 cells (size, group, band class): flagged games in the three calibration folds | a of the flagged (label fit) | a of all games of the cell (label fit) | mean production a_b; means over the four calibrations")
    for (s, k, c_), rows_ in sorted(TAB.items()):
        nf = np.mean([r_[0] for r_ in rows_]); print(f"   n={s:2d} group {k} class {c_}: {nf:7.0f} | " + " | ".join(("  nan" if not np.isfinite(np.nanmean([r_[j] for r_ in rows_]) if any(np.isfinite(r_[j]) for r_ in rows_) else np.nan) else f"{np.nanmean([r_[j] for r_ in rows_]):.3f}") for j in (1, 2, 3)), flush=True)
    print("M252 even games, top-1 candidate's label vs the truth (same level | within one), by band class: " + "; ".join(f"class {c_}: flagged {LREL[(c_, 1)][1] / max(LREL[(c_, 1)][0], 1):.3f} | {LREL[(c_, 1)][2] / max(LREL[(c_, 1)][0], 1):.3f} ({LREL[(c_, 1)][0]}), not flagged {LREL[(c_, 0)][1] / max(LREL[(c_, 0)][0], 1):.3f} | {LREL[(c_, 0)][2] / max(LREL[(c_, 0)][0], 1):.3f} ({LREL[(c_, 0)][0]})" for c_ in range(5)), flush=True)
    for s in SIZES:
        h_ = HEAD[s]; print(f"M252 headroom n={s}: {h_[0]} questions, flagged games per question {h_[1] / h_[0]:.3f}, their mean a before {h_[2] / max(h_[1], 1):.3f} -> after {h_[3] / max(h_[1], 1):.3f}; games raised by >= 0.2 per question {h_[4] / h_[0]:.3f}; flagged candidates per question {h_[5] / h_[0]:.3f}, of which NEW clear anchors (no game >= 0.5 of the scale before, >= 0.5 after) per question {h_[6] / h_[0]:.4f}", flush=True)
    plab_of = dict(zip(players.tolist(), plab.astype(int).tolist())); fold_of = dict(zip(players.tolist(), pfold.tolist()))
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"], "tilt": f["tilt"], "tiltn": f["tiltn"], "rep": f["rep"], "repc": f["repc"]}
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
