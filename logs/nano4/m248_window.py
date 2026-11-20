#!/usr/bin/env python
"""M248 (2026-11-19), report only, label-free: are the games of a test question a narrower slice of an account's history than a
random sample (the validation questions are random samples)? (design doc M248, registered before any reading)

Three independent homogeneity readings, test questions against the 20 scored games of every validation player, same colour only:
  S1  within-unit variance of the labels of confidently identified opponents (even games, SQ4 margin above the validation q90 .. q99)
  S2  mean cosine between two own-side identity vectors of the same unit (the look-alike members)
  S3  share of pairs of even games of the same unit with an identical opening (first 4 / 6 plies, up to the 8 board symmetries)
Strata (S1: rounded mean anchor level; S2 / S3: level x colour) are weighted as on the test side. Bootstrap over units.

  python -I logs/nano4/m248_window.py
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m58_decode as led  # noqa: E402
import m137b_build as mb7  # noqa: E402

la, CACHE = mb7.la, mb7.CACHE; LEV = led.LEVELS; K = np.arange(10.0); MOVE = re.compile(r";[BW]\[([a-z]{2})\]"); HDRE = re.compile(r"HD\[(\d+)\]"); QS = (0.90, 0.95, 0.98, 0.99); RNG = np.random.default_rng(248)


def std_pair(T, V):
    """Stratum-standardised rates (sum num / sum den per stratum, strata weighted by the test side's den). T, V = (stratum, num, den)."""
    ns = int(max(T[0].max(), V[0].max())) + 1; tn, td = np.bincount(T[0], T[1], ns), np.bincount(T[0], T[2], ns); vn, vd = np.bincount(V[0], V[1], ns), np.bincount(V[0], V[2], ns)
    ok = (td > 0) & (vd > 0)
    if not ok.any():
        return float("nan"), float("nan")
    w = td[ok] / td[ok].sum()
    return float((w * tn[ok] / td[ok]).sum()), float((w * vn[ok] / vd[ok]).sum())


def boot(T, V, mode, n=500):
    a, b_ = std_pair(T, V); out = []
    for _ in range(n):
        i = RNG.integers(0, len(T[0]), len(T[0])); j = RNG.integers(0, len(V[0]), len(V[0])); x, y = std_pair(tuple(t[i] for t in T), tuple(v[j] for v in V))
        if np.isfinite(x) and np.isfinite(y) and (mode != "ratio" or y > 0):
            out.append(x / y if mode == "ratio" else x - y)
    lo, hi = np.percentile(out, [2.5, 97.5]) if len(out) >= 50 else (np.nan, np.nan); return a, b_, ((a / b_ if b_ > 0 else np.nan) if mode == "ratio" else a - b_), float(lo), float(hi)


def canon(coords):
    """Canonical form of a list of sgf coordinates under the 8 board symmetries."""
    xy = [(ord(c[0]) - 97, ord(c[1]) - 97) for c in coords]; best = None
    for f in (lambda x, y: (x, y), lambda x, y: (18 - x, y), lambda x, y: (x, 18 - y), lambda x, y: (18 - x, 18 - y), lambda x, y: (y, x), lambda x, y: (18 - y, x), lambda x, y: (y, 18 - x), lambda x, y: (18 - y, 18 - x)):
        t = tuple(f(x, y) for x, y in xy)
        if best is None or t < best:
            best = t
    return best


def pair_counts(unit, key):
    """Per unit: pairs of games with the same key, and all pairs."""
    df = pd.DataFrame({"u": unit, "k": key}); c = df.groupby(["u", "k"]).size(); same = (c * (c - 1) / 2).groupby(level=0).sum(); n = df.groupby("u").size(); return same.reindex(n.index).to_numpy(), (n * (n - 1) / 2).to_numpy(), n.index.to_numpy()


def main():
    t0 = time.time(); rep = {}
    dv = np.load(ROOT / "results/r_variants/m82_val.npz"); sv = np.load(ROOT / "results/r_variants/m84_SQ4~m15_val.npz"); assert np.array_equal(dv["key"], sv["key"])
    keyv = dv["key"].astype(str); pidv = dv["pid"].astype(str); yv = dv["y"].astype(int); hdv = dv["hd"].astype(int); tbv = dv["target_black"].astype(bool); foldv = dv["fold"].astype(int); omv = sv["f_opp_m"].astype(np.float64); olv = sv["f_opp_lab"].astype(int)
    dt = np.load(ROOT / "results/r_variants/m82_test.npz"); st = np.load(ROOT / "results/r_variants/m84_SQ4~m15_test.npz"); assert np.array_equal(dt["qid"], st["qid"])
    qt = dt["qid"].astype(str); hdt = dt["hd"].astype(int); tbt = dt["target_black"].astype(bool); omt = st["f_opp_m"].astype(np.float64); olt = st["f_opp_lab"].astype(int)
    up, up_names = pd.factorize(pidv); uv, uv_names = pd.factorize(pd.Series(pidv) + np.where(tbv, ":B", ":W")); ut, ut_names = pd.factorize(qt); ans = led.decisions("submission_rank_ct116.csv"); lev_t = ans.reindex(ut_names).to_numpy().astype(int)
    col_t = np.zeros(len(ut_names), int); col_t[ut] = tbt.astype(int); lev_v = np.zeros(len(uv_names), int); lev_v[uv] = yv; col_v = np.zeros(len(uv_names), int); col_v[uv] = tbv.astype(int); fold_v = np.zeros(len(uv_names), int); fold_v[uv] = foldv
    print(f"validation: {len(keyv)} games, {len(set(pidv))} players, {len(uv_names)} (player, colour) units; test: {len(qt)} games, {len(ut_names)} questions; games with two or more handicap stones: validation {float((hdv >= 2).mean()):.3f}, test {float((hdt >= 2).mean()):.3f}")
    # ------------------------------------------------------------ S1: within-unit variance of the anchors' labels
    print("\n== S1 labels of confidently identified opponents in even games: within-unit variance (pooled, strata = rounded mean anchor level, weighted as on the test side)")
    print("   threshold | share of even games above it: validation, test | units with >= 2 anchors: validation, test | variance: test, validation, RATIO [95 %] | share of anchor pairs with the same label: test, validation, ratio [95 %]")

    def units_of(unit, lab, nu):
        C = np.zeros((nu, 10)); np.add.at(C, (unit, lab), 1.0); k = C.sum(1); ok = k >= 2; C = C[ok]; k = k[ok]; mean = (C @ K) / k; s2 = ((C @ (K * K)) - k * mean * mean) / (k - 1); same = (C * (C - 1) / 2).sum(1); pairs = k * (k - 1) / 2
        return np.clip(np.rint(mean), 0, 9).astype(int), (k - 1) * s2, k - 1, same, pairs
    for q_ in QS:
        thr = {c: float(np.quantile(omv[(hdv == 0) & (tbv == c)], q_)) for c in (False, True)}; mv = (hdv == 0) & (omv >= np.where(tbv, thr[True], thr[False])); mt = (hdt == 0) & (omt >= np.where(tbt, thr[True], thr[False]))
        sv_, nv_, dv_, smv, prv = units_of(uv[mv], olv[mv], len(uv_names)); st_, nt_, dt_, smt, prt = units_of(ut[mt], olt[mt], len(ut_names))
        if len(st_) < 5:
            print(f"   q{int(q_ * 100)} | {float(mv[hdv == 0].mean()):.3f}, {float(mt[hdt == 0].mean()):.3f} | {len(sv_)}, {len(st_)} | too few test units, left out", flush=True); continue
        a, b_, r, lo, hi = boot((st_, nt_, dt_), (sv_, nv_, dv_), "ratio"); a2, b2, r2, lo2, hi2 = boot((st_, smt, prt), (sv_, smv, prv), "ratio")
        print(f"   q{int(q_ * 100)} | {float(mv[hdv == 0].mean()):.3f}, {float(mt[hdt == 0].mean()):.3f} | {len(sv_)}, {len(st_)} | {a:.3f}, {b_:.3f}, {r:.3f} [{lo:.3f}, {hi:.3f}] | {a2:.3f}, {b2:.3f}, {r2:.3f} [{lo2:.3f}, {hi2:.3f}]", flush=True)
        rep.setdefault("S1", {})[f"q{int(q_ * 100)}"] = {"share_val": float(mv[hdv == 0].mean()), "share_test": float(mt[hdt == 0].mean()), "var_test": a, "var_val": b_, "ratio": r, "ci": [lo, hi], "same_test": a2, "same_val": b2, "same_ratio": r2, "same_ci": [lo2, hi2]}
        sp_, np_, dp_, _, _ = units_of(up[mv], olv[mv], len(up_names)); a3, b3, r3, lo3, hi3 = boot((st_, nt_, dt_), (sp_, np_, dp_), "ratio")
        print(f"        report only, validation units = players with both colours ({len(sp_)} units): variance {b3:.3f}, ratio {r3:.3f} [{lo3:.3f}, {hi3:.3f}]", flush=True)
    # ------------------------------------------------------------ S2: coherence of the own-side identity vectors
    q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), keyv); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); qvec = v_["qvec"]; M = qvec.shape[0]
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(qt)

    def coherence(vec, unit, nu):
        U = vec.astype(np.float64); U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-9); S = np.zeros((nu, U.shape[1])); np.add.at(S, unit, U); k = np.bincount(unit, minlength=nu).astype(float)
        return np.where(k >= 2, ((S * S).sum(1) - k) / np.maximum(k * (k - 1), 1), np.nan), k
    okv = hdv <= 1; cv = np.zeros(len(uv_names)); ct_ = np.zeros(len(ut_names))
    for m in range(M):
        c1, kv_ = coherence(qvec[m, Q + np.flatnonzero(okv)], uv[okv], len(uv_names)); c2, kt_ = coherence(qvec[m, tgt], ut, len(ut_names)); cv += c1 / M; ct_ += c2 / M
    sT = lev_t * 2 + col_t; sV = lev_v * 2 + col_v; gT = np.isfinite(ct_); gV = np.isfinite(cv); T2 = (sT[gT], ct_[gT], np.ones(int(gT.sum()))); print(f"\n== S2 mean cosine between two own-side identity vectors of the same unit ({M} members of {la.VAR}; strata level x colour, weighted as on the test side)")
    a, b_, d, lo, hi = boot(T2, (sV[gV], cv[gV], np.ones(int(gV.sum()))), "diff"); print(f"   test {a:.4f}, validation {b_:.4f}, difference {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"); folds = {}
    for f in sorted(set(fold_v.tolist())):
        gf = gV & (fold_v == f); folds[int(f)] = std_pair(T2, (sV[gf], cv[gf], np.ones(int(gf.sum()))))[1]
    print("   validation by fold (each fold has its own network): " + ", ".join(f"fold {f} {x:.4f}" for f, x in folds.items()) + f"   -> range {min(folds.values()):.4f} .. {max(folds.values()):.4f}")
    for nm, mk in (("test 5 games", kt_ == 5), ("test 10 games", kt_ == 10), ("test 15 games", kt_ == 15), ("test 20 games", kt_ == 20)):
        g_ = gT & mk; x, y = std_pair((sT[g_], ct_[g_], np.ones(int(g_.sum()))), (sV[gV], cv[gV], np.ones(int(gV.sum())))); print(f"   {nm}: {x:.4f} (validation with the same strata weights {y:.4f})")
    rep["S2"] = {"test": a, "val": b_, "diff": d, "ci": [lo, hi], "val_by_fold": folds}; del qvec
    # ------------------------------------------------------------ S3: repeated openings
    need = set(keyv[hdv == 0]); op = {}
    for rk in LEV:
        df = pd.read_csv(ROOT / f"data/training/train_{rk}.csv", usecols=["game_id", "sgf_content"], dtype=str, keep_default_na=False)
        for gid, sgf in zip(df.game_id, df.sgf_content):
            kk = f"{rk}:{gid}"
            if kk in need:
                mv_ = MOVE.findall(sgf[:120]); op[kk] = (canon(mv_[:4]), canon(mv_[:6])) if len(mv_) >= 6 else None
        del df
    print(f"\n== S3 identical openings between two even games of the same unit (validation games found: {len(op)} of {len(need)}; {time.time() - t0:.0f} s)")
    iv = np.array([i for i in np.flatnonzero(hdv == 0) if op.get(keyv[i]) is not None]); tu, tk4, tk6 = [], [], []
    csv.field_size_limit(10 ** 9)
    for half in ("public", "private"):
        with open(ROOT / f"data/tests/rank_prediction_test_{half}.csv") as fh:
            for row in csv.DictReader(fh):
                for i in range(1, int(row["num_games"]) + 1):
                    s = row[f"sgf_{i}"]; mv_ = MOVE.findall(s[:120])
                    if int(HDRE.search(s).group(1)) == 0 and len(mv_) >= 6:
                        tu.append(row["question_id"]); tk4.append(str(canon(mv_[:4]))); tk6.append(str(canon(mv_[:6])))
    name_t = {n_: i for i, n_ in enumerate(ut_names)}
    for P, kv_keys, kt_keys in ((4, [str(op[keyv[i]][0]) for i in iv], tk4), (6, [str(op[keyv[i]][1]) for i in iv], tk6)):
        smv, prv, uu = pair_counts(uv[iv], kv_keys); smt, prt, tt = pair_counts(np.array([name_t[x] for x in tu]), kt_keys); gv_, gt_ = prv > 0, prt > 0
        a, b_, r, lo, hi = boot((sT[tt][gt_], smt[gt_], prt[gt_]), (sV[uu][gv_], smv[gv_], prv[gv_]), "ratio")
        print(f"   first {P} plies: share of pairs with the same opening, test {a:.4f}, validation {b_:.4f}, RATIO {r:.3f} [{lo:.3f}, {hi:.3f}]   (units: test {int(gt_.sum())}, validation {int(gv_.sum())})")
        rep.setdefault("S3", {})[f"plies{P}"] = {"test": a, "val": b_, "ratio": r, "ci": [lo, hi]}
        for nm, mk in (("black units", 1), ("white units", 0)):
            x, y = std_pair((sT[tt][gt_ & (col_t[tt] == mk)], smt[gt_ & (col_t[tt] == mk)], prt[gt_ & (col_t[tt] == mk)]), (sV[uu][gv_ & (col_v[uu] == mk)], smv[gv_ & (col_v[uu] == mk)], prv[gv_ & (col_v[uu] == mk)])); print(f"      {nm}: test {x:.4f}, validation {y:.4f}, ratio {x / y:.3f}")
    s1 = rep["S1"]["q98"]; s3 = rep["S3"]["plies4"]; s2 = rep["S2"]; lo_f, hi_f = min(folds.values()), max(folds.values())
    homo = [s1["ratio"] <= 0.85 and s1["ci"][1] < 1, s2["diff"] > 0 and s2["ci"][0] > 0 and s2["test"] > hi_f, s3["ratio"] >= 1.15 and s3["ci"][0] > 1]
    hete = [s1["ratio"] >= 1.15 and s1["ci"][0] > 1, s2["diff"] < 0 and s2["ci"][1] < 0 and s2["test"] < lo_f, s3["ratio"] <= 0.85 and s3["ci"][1] < 1]
    verdict = "MORE HOMOGENEOUS than a random sample" if sum(homo) >= 2 else "MORE SPREAD than a random sample" if sum(hete) >= 2 else "not distinguishable from a random sample"
    print(f"\nREGISTERED: more homogeneous (S1, S2, S3) = {homo}; more spread = {hete}  ->  {verdict}"); rep["verdict"] = verdict
    (ROOT / "results/r_variants/m248_window.json").write_text(json.dumps(rep, indent=1, default=float)); print(f"written results/r_variants/m248_window.json; total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
