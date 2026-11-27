#!/usr/bin/env python
"""M258 (2026-11-20), report only: do the test question NUMBERS keep the order in which the questions were made? (design doc M258,
registered before any reading; copy of m246_scan.py with another covariate family plus a label-free order test; nothing is built)

Part 1 (label-free, null = a random order inside each half, 2,000 permutations), five primary statistics:
  (a) autocorrelation of the pooled expected level along the number, mean of lags 1-3
  (b) similarity of the own-side identity prototypes (six networks, same-colour pairs, standardised) at lags 1-3
  (c) share of question pairs at lags 1-3 whose confident anchors (SQ4 margin >= q90) name the same training player
  (d) order kept between the halves: sum of Z[i, j] * Z[i + 1, j + 1] over the Public x Private similarity matrix (clipped at +-3)
  (e) the same sum inside each half
Part 2: fair composite likelihood of the Public readings (as M246) for tilts along the number (Legendre 1-3) and towards the mean
pooled level of the neighbouring numbers (+-1, +-2, +-5); permutations stratified by question size.

  python -I logs/nano4/m258_qorder.py --perms 200
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m229_like as ml  # noqa: E402  (pair_dist, LATER)

LEV = led.LEVELS; K = np.arange(10.0); SIZES = (5, 10, 15, 20)
MODEL = {"ct40", "ct42", "ct44", "ct64", "ct71", "ct72", "ct73", "ct75n", "ct75s", "ct75u", "ct75s_out", "ct75s_out_up", "ct104", "ct104b", "ct105", "ct106", "ct109", "ct112"}
BAND = {"probe_4D5D_locked", "probe_A2B", "probe_2D3D", "probe_3D4D", "probe_C2B", "probe_5D4D", "probe_D2C", "probe_5D6D", "probe_1D2A", "probe_B2", "probe_noLA9"}
NEW = [("upload_probe/probe_D2C.csv", 0.8379152), ("upload_probe/probe_5D6D.csv", 0.8331743), ("upload_probe/probe_1D2A.csv", 0.8410758), ("upload_probe/probe_B2.csv", 0.8284334), ("upload_probe/probe_noLA9.csv", 0.8523970)]
GAM = (-0.7, -0.4, -0.2, -0.1, 0.1, 0.2, 0.4, 0.7); KINDS = ("shift",); MOVE = re.compile(r";([BW])\[([a-z]*)\]")
G = {}


def soft(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def tilt(LP, x, kind, g):
    return soft(LP + g * x[:, None] * K[None, :]) if kind == "shift" else soft(LP * np.clip(1 + g * x, 0.3, 3.0)[:, None])


def parts(Pp):
    """Log-likelihood of the Public readings under truths ~ Pp [400, 10]: model-driven steps, band probes, exact composition."""
    A, pairs, cnt = G["A"], G["pairs"], G["cnt"]; out = []
    for js in (G["jm"], G["jb"]):
        t = 0.0
        for j in js:
            k, p, idx, sols = pairs[j]; D, n = ml.pair_dist(Pp, A[p], A[k], idx); t += math.log(max(sum(D[n + x, n + y] for x, y in sols), 1e-300))
        out.append(t)
    mu = Pp.sum(0); var = (Pp * (1 - Pp)).sum(0); out.append(float((-0.5 * (cnt - mu) ** 2 / var - 0.5 * np.log(2 * math.pi * var)).sum()))
    return np.array(out)


def scan(X):
    """Gain against the current posterior [covariate, kind, gamma, part]."""
    LP, ref = G["LP"], G["ref"]; out = np.zeros((X.shape[1], len(KINDS), len(GAM), 3))
    for j in range(X.shape[1]):
        for a, kind in enumerate(KINDS):
            for c, g in enumerate(GAM):
                out[j, a, c] = parts(tilt(LP, X[:, j], kind, g)) - ref
    return out


def work(task):
    fam, r = task; X = G["X"][fam]; rng = np.random.default_rng(258000 + r); perm = np.arange(len(X))
    if fam == "A":
        for s in SIZES:
            m = np.flatnonzero(G["n"] == s); perm[m] = rng.permutation(m)
    else:
        perm = rng.permutation(len(X))
    return fam, r, scan(X[perm]).sum(-1)


def question_stats():
    """Per question from the released files: mean number of moves, share of games whose last move is the player's (games ending in a pass left out), share ending in a pass."""
    csv.field_size_limit(10 ** 9); out = {}
    for half in ("public", "private"):
        with open(ROOT / f"data/tests/rank_prediction_test_{half}.csv") as fh:
            for row in csv.DictReader(fh):
                n = int(row["num_games"]); ln, own, ps = [], [], []
                for i in range(1, n + 1):
                    mv = MOVE.findall(row[f"sgf_{i}"]); ln.append(len(mv)); last = mv[-1]; is_pass = last[1] in ("", "tt"); ps.append(float(is_pass))
                    if not is_pass:
                        own.append(float(last[0] == row[f"color_{i}"]))
                out[row["question_id"]] = (float(np.mean(ln)), float(np.mean(own)) if own else 0.5, float(np.mean(ps)), n)
    return out


def order_tests(qid, nq, black, hd1, el, eb, ansh, moves, gq, om, best, rep):
    """Part 1, label-free: does the question number carry structure? Null = a random order inside each half."""
    rng = np.random.default_rng(258); NP = 2000; num = np.array([int(x.split("_")[-1]) for x in qid]); pub = np.char.startswith(qid, "pub"); t0 = time.time()
    H = [np.flatnonzero(pub)[np.argsort(num[pub])], np.flatnonzero(~pub)[np.argsort(num[~pub])]]; assert all(np.array_equal(num[h], np.arange(1, 401)) for h in H)
    q = np.load(mb7.CACHE / "queries.npz"); ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(gq)
    S = np.zeros((800, 800)); nn = 0; chk = []
    for name in ("S7t", "S6at", "S4e"):
        V = np.load(mb7.CACHE / f"var_{name}.npz")["qvec"]
        for m in range(V.shape[0]):
            X = V[m][tgt].astype(np.float64); X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12); T = np.zeros((800, X.shape[1])); np.add.at(T, gq, X); T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
            S += T @ T.T; nn += 1; chk.append((float((X * T[gq]).sum(1).mean()), float((X * T[rng.permutation(800)][gq]).sum(1).mean())))
        del V
    S /= nn; print(f"== PART 1 (label-free order tests). identity prototypes from {nn} networks; check: a game's cosine to its own question's prototype {np.mean([c[0] for c in chk]):.3f}, to another question's {np.mean([c[1] for c in chk]):.3f}", flush=True)
    same = (black[:, None] == black[None, :]) & ~np.eye(800, dtype=bool); Sm = np.where(same, S, np.nan); Z = (S - np.nanmean(Sm, 1, keepdims=True)) / np.nanstd(Sm, 1, keepdims=True); Z = np.where(same, (Z + Z.T) / 2, 0.0); Zc = np.clip(Z, -3.0, 3.0)
    conf = om >= np.quantile(om, 0.9); _, codes = np.unique(best[conf], return_inverse=True); M = np.zeros((800, int(codes.max()) + 1)); M[gq[conf], codes] = 1.0; C = ((M @ M.T) > 0).astype(float); np.fill_diagonal(C, 0.0)
    print(f"   confident anchors (margin >= q90 of the test games): {int(conf.sum())} games, {M.shape[1]} different training players; share of all question pairs sharing one {float(C.sum() / (800 * 799)):.4f}; same-colour pairs with Z >= 3: {int((Z >= 3).sum() // 2)}")
    xs = np.zeros(800)
    for h in H:
        xs[h] = (el[h] - el[h].mean()) / el[h].std()

    def stats(orders, lags=(1, 2, 3)):
        """[half, 3]: (a) autocorrelation of the pooled level, (b) mean standardised prototype similarity of same-colour pairs, (c) share of pairs with a common anchor."""
        out = []
        for o in orders:
            a = float(np.mean([(xs[o[:-k]] * xs[o[k:]]).mean() for k in lags])); sb = nb_ = 0.0; sc = nc = 0.0
            for k in lags:
                i, j = o[:-k], o[k:]; m_ = same[i, j]; sb += float(Z[i, j][m_].sum()); nb_ += float(m_.sum()); sc += float(C[i, j].sum()); nc += len(i)
            out.append((a, sb / max(nb_, 1.0), sc / nc))
        return np.array(out)

    def coadj(o1, o2):
        A = Zc[np.ix_(o1, o2)]; return float((A[:-1, :-1] * A[1:, 1:]).sum())
    A0 = Zc[np.ix_(H[0], H[1])]; seen = float(A0[173, 267] * A0[174, 268]); obs = stats(H); Td = coadj(H[0], H[1]); Te = np.array([coadj(h, h) for h in H])
    null_s = np.zeros((NP, 2, 3)); null_d = np.zeros(NP); null_e = np.zeros((NP, 2)); LAGS = range(1, 11); curve = np.array([stats(H, (k,)) for k in LAGS]); ncurve = np.zeros((500, len(LAGS), 2, 3))
    for r in range(NP):
        P_ = [rng.permutation(h) for h in H]; null_s[r] = stats(P_); null_d[r] = coadj(H[0], P_[1]); null_e[r] = [coadj(p_, p_) for p_ in P_]
        if r < 500:
            ncurve[r] = [stats(P_, (k,)) for k in LAGS]
    p2 = lambda o, n_: (1 + int((np.abs(n_ - n_.mean()) >= abs(o - n_.mean())).sum())) / (len(n_) + 1)  # noqa: E731  two-sided
    names = ("(a) autocorrelation of the pooled expected level, lags 1-3", "(b) prototype similarity of same-colour neighbours, lags 1-3 (z)", "(c) share of neighbours with a common confident anchor, lags 1-3"); prim = {}
    print(f"   {NP} permutations of the order inside each half ({time.time() - t0:.0f} s). statistic: observed both halves (Public / Private) | permutation mean, sd | two-sided p")
    for j, nm in enumerate(names):
        o = float(obs[:, j].mean()); n_ = null_s[:, :, j].mean(1); p = p2(o, n_); same_sign = bool(np.sign(obs[0, j] - null_s[:, 0, j].mean()) == np.sign(obs[1, j] - null_s[:, 1, j].mean()))
        print(f"   {nm}: {o:+.4f} ({obs[0, j]:+.4f} / {obs[1, j]:+.4f}) | {n_.mean():+.4f}, {n_.std():.4f} | p {p:.4f} (Public alone {p2(obs[0, j], null_s[:, 0, j]):.3f}, Private alone {p2(obs[1, j], null_s[:, 1, j]):.3f}; same direction in both halves: {same_sign})")
        prim[nm[:3]] = {"obs": o, "public": float(obs[0, j]), "private": float(obs[1, j]), "null_mean": float(n_.mean()), "null_sd": float(n_.std()), "p": p, "same_sign": same_sign}
    pd_ = p2(Td - max(seen, 0.0), null_d); print(f"   (d) order kept between the halves, sum of Z[i,j] Z[i+1,j+1] (clipped): {Td:+.1f}; without the one term seen before registration ({seen:+.1f}; pub_q_0174/0175 x priv_q_0268/0269): {Td - max(seen, 0.0):+.1f} | {null_d.mean():+.1f}, {null_d.std():.1f} | p {pd_:.4f} (with the seen term {p2(Td, null_d):.4f})")
    pe_ = p2(float(Te.sum()), null_e.sum(1)); print(f"   (e) the same inside each half: {Te.sum():+.1f} ({Te[0]:+.1f} / {Te[1]:+.1f}) | {null_e.sum(1).mean():+.1f}, {null_e.sum(1).std():.1f} | p {pe_:.4f} (Public alone {p2(Te[0], null_e[:, 0]):.3f}, Private alone {p2(Te[1], null_e[:, 1]):.3f})")
    prim["(d)"] = {"obs": Td - max(seen, 0.0), "with_seen": Td, "null_mean": float(null_d.mean()), "null_sd": float(null_d.std()), "p": pd_}; prim["(e)"] = {"obs": float(Te.sum()), "null_mean": float(null_e.sum(1).mean()), "null_sd": float(null_e.sum(1).std()), "p": pe_}
    print("   report only - by lag (both halves; z against 500 permutations): lag | (a) | (b) | (c)")
    for a_i, k in enumerate(LAGS):
        o = curve[a_i].mean(0); n_ = ncurve[:, a_i].mean(1); print(f"      {k:2d} | " + " | ".join(f"{o[j]:+.4f} (z {(o[j] - n_[:, j].mean()) / max(n_[:, j].std(), 1e-12):+.2f})" for j in range(3)))
    # covariates along the number
    print("   report only - covariates along the number: correlation with the number (Public / Private) | spread of the means of 10 blocks of 40 against permutation (ratio, p) | agreement of neighbours (lag 1) against permutation (p)")
    cov = {"question size": nq.astype(float), "black": black.astype(float), "share of HD1 games": hd1, "pooled expected level": el, "base expected level": eb, "share of games with a confident anchor (q70)": ansh, "mean number of moves": moves}; rep["covariates"] = {}
    for nm, v in cov.items():
        cr = [float(np.corrcoef(np.arange(400), v[h])[0, 1]) for h in H]; blk = lambda o: float(np.mean([v[o_].reshape(10, 40).mean(1).var() for o_ in o]))  # noqa: E731
        ob = blk(H); nbk = np.array([blk([rng.permutation(h) for h in H]) for _ in range(1000)]); pb = (1 + int((nbk >= ob).sum())) / 1001
        if nm in ("question size", "black"):
            agr = lambda o: float(np.mean([(v[o_[:-1]] == v[o_[1:]]).mean() for o_ in o]))  # noqa: E731
        else:
            agr = lambda o: float(np.mean([np.corrcoef(v[o_[:-1]], v[o_[1:]])[0, 1] for o_ in o]))  # noqa: E731
        oa = agr(H); na = np.array([agr([rng.permutation(h) for h in H]) for _ in range(1000)]); pa = p2(oa, na)
        print(f"      {nm:44s}: {cr[0]:+.3f} / {cr[1]:+.3f} | {ob / nbk.mean():.2f}, p {pb:.3f} | {oa:+.3f} (permutation {na.mean():+.3f}), p {pa:.3f}"); rep["covariates"][nm] = {"corr_number": cr, "block_ratio": ob / float(nbk.mean()), "p_block": pb, "lag1": oa, "p_lag1": pa}
    # similarity between the halves by the difference of the numbers
    Zx = Z[np.ix_(H[0], H[1])]; Mx = same[np.ix_(H[0], H[1])].astype(float); ds = [d for d in range(-399, 400) if 400 - abs(d) >= 80]

    def prof(perm):
        A = Zx[:, perm]; B = Mx[:, perm]; return np.array([np.trace(A, d) / max(np.trace(B, d), 1.0) for d in ds])
    po = prof(np.arange(400)); pn = np.array([prof(rng.permutation(400)) for _ in range(300)]); mu, sd = pn.mean(0), pn.std(0); zo = (po - mu) / sd; zn = (pn - mu) / sd; mx = float(np.abs(zo).max()); pmx = (1 + int((np.abs(zn).max(1) >= mx).sum())) / 301
    band = [i for i, d in enumerate(ds) if abs(d) <= 5]; bo = float(po[band].mean()); bn_ = pn[:, band].mean(1)
    print(f"   report only - Public x Private similarity by the difference of the numbers d (|d| <= 320): largest |z| {mx:.2f} at d = {ds[int(np.abs(zo).argmax())]:+d} (family-wise p {pmx:.3f}); band |d| <= 5: {bo:+.4f} (permutation {bn_.mean():+.4f}, sd {bn_.std():.4f}, p {p2(bo, bn_):.3f}); the five largest z: " + ", ".join(f"d {ds[i]:+d}: {zo[i]:+.2f}" for i in np.argsort(-zo)[:5]))
    # the 25 mutual pairs of M251b (seen before registration)
    pr = json.load(open(ROOT / "results/r_variants/m251b_votes.json"))["pairs"]; hn = lambda s: (s.startswith("pub"), int(s.split("_")[-1]))  # noqa: E731
    cross = [(hn(p["a"])[1], hn(p["b"])[1]) if hn(p["a"])[0] else (hn(p["b"])[1], hn(p["a"])[1]) for p in pr if hn(p["a"])[0] != hn(p["b"])[0]]; within = [abs(hn(p["a"])[1] - hn(p["b"])[1]) for p in pr if hn(p["a"])[0] == hn(p["b"])[0]]

    def nco(cp):
        return sum(1 for i in range(len(cp)) for j in range(i + 1, len(cp)) if abs(cp[i][0] - cp[j][0]) == 1 and cp[i][0] - cp[j][0] == cp[i][1] - cp[j][1])
    oc = nco(cross); nc_ = np.array([nco(list(zip(rng.choice(400, len(cross), replace=False).tolist(), rng.choice(400, len(cross), replace=False).tolist()))) for _ in range(20000)])
    print(f"   seen before registration - M251b's {len(pr)} mutual pairs: {len(cross)} between the halves, consecutive on both sides in the same direction {oc} (random numbers: mean {nc_.mean():.4f}, P(>= {oc}) = {float((nc_ >= oc).mean()):.4f}); inside a half {len(within)} pairs, differences of the numbers {sorted(within)} (random: median {400 / 3:.0f})")
    ps = [prim[k]["p"] for k in ("(a)", "(b)", "(c)", "(d)", "(e)")]; hit = [k for k in ("(a)", "(b)", "(c)", "(d)", "(e)") if prim[k]["p"] <= 0.01 and prim[k].get("same_sign", True)]
    verdict = "STRUCTURE: the numbers keep an order (next step registered separately)" if hit else ("closed: the numbers are a random order" if min(ps) > 0.05 else "in between: recorded only (unless part 2 passes its family test)")
    print(f"   REGISTERED (five primary statistics, Bonferroni 0.01 each): p = " + ", ".join(f"{k} {prim[k]['p']:.4f}" for k in ("(a)", "(b)", "(c)", "(d)", "(e)")) + f"  ->  {verdict}\n", flush=True); rep.update(primary=prim, verdict=verdict, seen_pairs={"cross": cross, "coadjacent": oc, "p_random": float((nc_ >= oc).mean())})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--perms", type=int, default=200); a_ = ap.parse_args(); t0 = time.time()
    z = np.load(ROOT / "results/r_variants/m237_grid.npz", allow_pickle=True); names = [str(x) for x in z["names"]]; P0 = z["P"][names.index("current")].astype(np.float64); qid = z["qid"].astype(str); nq = z["nq"].astype(int)
    base = z["base"].astype(np.float64); R = z["R_cur"].astype(np.float64); pub = np.char.startswith(qid, "pub"); pi = np.flatnonzero(pub); pos = {q: i for i, q in enumerate(qid)}
    # ------------------------------------------------------------ covariates (all label-free)
    t82 = np.load(ROOT / "results/r_variants/m82_test.npz"); sq = np.load(ROOT / "results/r_variants/m84_SQ4~m15_test.npz"); assert np.array_equal(t82["qid"], sq["qid"])
    gq = np.array([pos[q] for q in t82["qid"].astype(str)]); hd = t82["hd"].astype(int); tb = t82["target_black"].astype(float); om = sq["f_opp_m"].astype(np.float64); ol = sq["f_opp_lab"].astype(np.float64)
    assert hd.max() <= 1 and np.array_equal(np.bincount(gq, minlength=800), nq), "games and questions are not aligned"
    per = lambda v: np.bincount(gq, weights=v, minlength=800)  # noqa: E731
    black = per(tb) / nq; assert set(np.unique(black)) <= {0.0, 1.0}, "a test question has both colours"
    hd1 = per((hd == 1).astype(float)) / nq; conf = om >= np.quantile(om, 0.7); an = per(conf.astype(float)); ce = conf & (hd == 0); ne = per(ce.astype(float)); s1 = per(ce * ol); s2_ = per(ce * ol * ol)
    eb = base @ K; el = P0 @ K; ela = soft(R) @ K; amean = np.where(ne > 0, s1 / np.maximum(ne, 1), np.nan); asd = np.where(ne >= 2, np.sqrt(np.maximum(s2_ / np.maximum(ne, 1) - (s1 / np.maximum(ne, 1)) ** 2, 0.0)), np.nan)
    ent = lambda P: -(P * np.log(np.maximum(P, 1e-300))).sum(1)  # noqa: E731
    qs = question_stats(); st = np.array([qs[q] for q in qid]); assert np.array_equal(st[:, 3].astype(int), nq)
    num = np.array([int(x.split("_")[-1]) for x in qid]); u = (num - 200.5) / 199.5; rep_order = {}
    order_tests(qid, nq, black, hd1, el, eb, an / nq, st[:, 0], gq, om, sq["f_opp_best"].astype(str), rep_order)

    def nb(k):
        """Mean pooled expected level of the questions numbered within +-k in the same half, minus the question's own."""
        out = np.zeros(800)
        for half in (pub, ~pub):
            h = np.flatnonzero(half)[np.argsort(num[half])]; x = el[h]; c = np.concatenate([[0.0], np.cumsum(x)]); lo = np.maximum(np.arange(400) - k, 0); hi = np.minimum(np.arange(400) + k, 399); out[h] = (c[hi + 1] - c[lo] - x) / (hi - lo) - x
        return out
    cov = {"question number (linear)": u, "question number (quadratic)": (3 * u * u - 1) / 2, "question number (cubic)": (5 * u ** 3 - 3 * u) / 2, "neighbours +-1: mean pooled level - own": nb(1),
           "neighbours +-2: mean pooled level - own": nb(2), "neighbours +-5: mean pooled level - own": nb(5)}
    cn = list(cov); XA = np.zeros((800, len(cn)))
    for j, k in enumerate(cn):
        v = cov[k]; ok = np.isfinite(v); XA[ok, j] = (v[ok] - v[ok].mean()) / max(v[ok].std(), 1e-9)  # questions without the quantity sit at the mean (0)
    bn = [f"question size {s}" for s in SIZES]; XB = np.stack([(nq == s).astype(float) for s in SIZES], 1); XB = (XB - XB.mean(0)) / XB.std(0)
    print(f"covariates: {len(cn)} + {len(bn)} size dummies; defined on (questions): " + ", ".join(f"{k.split(' ')[0]}.. {int(np.isfinite(cov[k]).sum())}" for k in cn))
    print("   Public mean / Private mean (raw units): " + " | ".join(f"{k}: {np.nanmean(cov[k][pub]):.3f} / {np.nanmean(cov[k][~pub]):.3f}" for k in cn), flush=True)
    # ------------------------------------------------------------ the Public readings
    ups = [*led.KNOWN_SCORES.items(), *json.load(open(led.LEDGER)).items(), *ml.LATER, *NEW]; ups = [(f, s) for f, s in ups if (ROOT / f).exists() or print(f"   (missing, left out: {f})")]
    A = np.stack([led.decisions(f).reindex(qid).to_numpy()[pi].astype(int) for f, _ in ups]); S = np.array([s for _, s in ups]); pairs = []
    for k in range(1, len(ups)):
        d = (A[:k] != A[k]).sum(1); p = int(np.flatnonzero(d == d.min())[-1]); idx = np.flatnonzero(A[p] != A[k])
        if len(idx) == 0:
            continue
        sols = led.decompose(400 * (S[k] - S[p]), len(idx))
        if sols:
            pairs.append((k, p, idx, sols))
        else:
            print(f"   (no decomposition, left out: {ups[k][0]} against {ups[p][0]}, {len(idx)} questions)")
    name = lambda k: ups[k][0].split("/")[-1].replace("submission_rank_", "").replace(".csv", "")  # noqa: E731
    jm = [j for j, x in enumerate(pairs) if name(x[0]) in MODEL]; jb = [j for j, x in enumerate(pairs) if name(x[0]) in BAND]
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; cnt = np.array([comp[lv] for lv in LEV], float)
    print(f"{len(ups)} uploads, {len(pairs)} pairs: model-driven {len(jm)} ({sum(len(pairs[j][2]) for j in jm)} question changes), band probes {len(jb)} ({sum(len(pairs[j][2]) for j in jb)}), not used (chosen with Public information) {len(pairs) - len(jm) - len(jb)}")
    print("   band probes found: " + ", ".join(f"{name(pairs[j][0])} vs {name(pairs[j][1])} ({len(pairs[j][2])})" for j in jb), flush=True)
    LP = np.log(np.maximum(P0[pi], 1e-300)); G.update(A=A, pairs=pairs, jm=jm, jb=jb, cnt=cnt, LP=LP, n=nq[pi], X={"A": XA[pi], "B": XB[pi]}); G["ref"] = parts(P0[pi])
    print(f"current posterior: log-likelihood model-driven {G['ref'][0]:.2f}, band probes {G['ref'][1]:.2f}, composition {G['ref'][2]:.2f}; one evaluation takes {time.time() - t0:.1f} s incl. loading", flush=True)
    one = np.ones((len(pi), 1)); g1 = scan(one)[0]
    print("\n== reference (closed in M229 / M235): the same tilts for every question (x = 1); fair gain by gamma " + " ".join(f"{g:+.1f}" for g in GAM))
    for a, kind in enumerate(KINDS):
        print(f"   {kind}: " + " ".join(f"{v:+.2f}" for v in g1[a].sum(-1)))
    # ------------------------------------------------------------ observed scan and permutations
    t1 = time.time(); obs = {"A": scan(G["X"]["A"]), "B": scan(G["X"]["B"])}; print(f"\nobserved scan done in {time.time() - t1:.0f} s; {a_.perms} permutations on {len(os.sched_getaffinity(0))} processes", flush=True)
    tasks = [(fam, r) for r in range(1, a_.perms + 1) for fam in ("A", "B")]; null = {"A": [], "B": []}
    with Pool(len(os.sched_getaffinity(0))) as pool:
        for i, (fam, r, gain) in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
            null[fam].append(gain.max(-1))  # [covariate, kind]: best gamma
            if (i + 1) % 80 == 0:
                print(f"   {i + 1} / {len(tasks)} permutation tasks, {time.time() - t1:.0f} s", flush=True)
    rep = {"gammas": GAM, "kinds": KINDS, "reference": g1.sum(-1).tolist(), "families": {}}
    lev = np.array(LEV); dec = lambda P: mf7.guard2(np.where(z["locked"], z["l28"], np.asarray(mb7.b.decide(P), int)), z["c106"])  # noqa: E731
    d0 = dec(P0); LPall = np.log(np.maximum(P0, 1e-300))
    for fam, nm, Xall in (("A", cn, XA), ("B", bn, XB)):
        o = obs[fam]; fair = o.sum(-1); best = fair.max(-1); N = np.array(null[fam]); T = float(best.max()); Tn = N.reshape(len(N), -1).max(1); pg = (1 + int((Tn >= T).sum())) / (len(Tn) + 1)
        i4p, i4m = GAM.index(0.4), GAM.index(-0.4)
        print(f"\n== family {fam} ({'stratified by question size' if fam == 'A' else 'unstratified'} permutations): largest fair gain {T:+.2f}; under permutation the largest gain has median {np.median(Tn):+.2f}, 90 % {np.quantile(Tn, 0.9):+.2f}, 95 % {np.quantile(Tn, 0.95):+.2f}, max {Tn.max():+.2f}  ->  FAMILY-WISE p = {pg:.3f}")
        print("   covariate | tilt: best gamma, fair gain (model-driven + band probes + composition), own permutation p (not corrected), mean loss of a +-0.4 tilt")
        rows = {}
        for j, k in enumerate(nm):
            for a, kind in enumerate(KINDS):
                c = int(fair[j, a].argmax()); pj = (1 + int((N[:, j, a] >= best[j, a]).sum())) / (len(N) + 1); loss = -(fair[j, a, i4p] + fair[j, a, i4m]) / 2
                print(f"   {k:48s} | {kind}: gamma {GAM[c]:+.1f}, {best[j, a]:+6.2f} ({o[j, a, c, 0]:+.2f} {o[j, a, c, 1]:+.2f} {o[j, a, c, 2]:+.2f}), p {pj:.3f}, loss at +-0.4 {loss:5.2f}")
                rows[f"{k} | {kind}"] = {"gamma": GAM[c], "gain": float(best[j, a]), "parts": o[j, a, c].tolist(), "p": pj, "loss_04": float(loss), "by_gamma": fair[j, a].tolist()}
        order = np.dstack(np.unravel_index(np.argsort(-best, axis=None), best.shape))[0][:3]; print("   the three largest settings and the answers they would change under the ct112 rule (Public / Private; kinds):")
        for j, a in order:
            c = int(fair[j, a].argmax()); dd = dec(tilt(LPall, Xall[:, j], KINDS[a], GAM[c])); ch = dd != d0; kinds = {}
            for u, w in zip(d0[ch], dd[ch]):
                kinds[f"{lev[u]}>{lev[w]}"] = kinds.get(f"{lev[u]}>{lev[w]}", 0) + 1
            print(f"      {nm[j]} | {KINDS[a]} {GAM[c]:+.1f}: {int((ch & pub).sum())} / {int((ch & ~pub).sum())}   " + ", ".join(f"{u} {w}" for u, w in sorted(kinds.items(), key=lambda kv: -kv[1])[:8]))
        rep["families"][fam] = {"largest_gain": T, "p_family": pg, "null_quantiles": {"median": float(np.median(Tn)), "q90": float(np.quantile(Tn, 0.9)), "q95": float(np.quantile(Tn, 0.95)), "max": float(Tn.max())}, "rows": rows}
        print(f"   REGISTERED: family-wise p {'<= 0.05 -> the largest setting is a candidate direction (next step registered separately)' if pg <= 0.05 else '> 0.05 -> no covariate direction is visible with the present readings'}", flush=True)
    rep["order"] = rep_order; (ROOT / "results/r_variants/m258_qorder.json").write_text(json.dumps(rep, indent=1)); print(f"\nwritten results/r_variants/m258_qorder.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
