#!/usr/bin/env python
"""Section 4M M229 (2026-11-18), report only: are the pool's global weights calibrated on the TEST side? A copy of m163_pool.py
(the ct112 rule: pool + the anchor evidence of SQ4). It (1) prints the cross-validated score when the anchor evidence, the base
log posterior or the look-alike reading is multiplied by a factor at apply time (coefficients as fitted), and (2) stores the
test-side pooled posterior for a grid of the three factors in results/r_variants/m229_grid.npz. No file is built.
The grid is scored against the Public readings by logs/nano4/m229_like.py.

  python -I logs/nano4/m229_scan.py --new SQ4 --control SPALm --json m229_cv.json
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


TA, AL, RR = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0), (0.8, 1.0, 1.25), (0.5, 1.0, 1.5)  # factors on the anchor evidence, the base log posterior, the look-alike reading


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True); ap.add_argument("--also", nargs="*", default=[]); ap.add_argument("--control", default=None)
    ap.add_argument("--name", default="ct111"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default=None)
    ap.add_argument("--build-anyway", action="store_true", help="dry run of the test-side code when the test did not pass: needs --out-dir (never the project root)"); ap.add_argument("--out-dir", default=None); a_ = ap.parse_args()
    assert not a_.build_anyway or a_.out_dir, "--build-anyway writes outside the project only: give --out-dir"
    out_json = ROOT / "results/r_variants" / (a_.json or f"m163_pool_{a_.new}.json")
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    xs = [a_.new, *a_.also, *([a_.control] if a_.control else [])]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], *[f"{x}~{STAT}" for x in xs]])
    if f"{a_.new}~{STAT}" not in used:
        sys.exit(f"the identifier files of {a_.new}~{STAT} are missing")
    xs = [x for x in xs if f"{x}~{STAT}" in used]
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
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}

        def ev_of(spec):
            if spec not in cache:
                idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
            return cache[spec]
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}
        Lx = {x: ev_of(xspec[x]) for x in xs}
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)
        RD = {k: lookalike(mem, Q + gi, qi, len(y), v - 1, cand) for k, mem in MEM.items()}
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, cw=cw, wq=np.array([W_FIN[int(m)] for m in n]), **RD)
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
    VARI = [("PRIMARY: + new source", "R", (a_.new,)), *[(f"+ {x} only", "R", (x,)) for x in xs if x != a_.new]]
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"]}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]
        print(f"\nseeds {seeds}: ct104b rule {bal([SM[Fd[v]['dc'], Fd[v]['y']] for v in FOLDS]):.4f}, ct106 rule {bal(S_old):.4f}")
        if seeds == (0, 1, 2):
            keep = {"X": X, "ref": ref}
        out = {}
        for name, rkey, extra in VARI:
            Xv = {v: {**X[v], "R": Fd[v][rkey]} for v in FOLDS}; dd = pool_decisions(Xv, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]
            A_, B_ = np.concatenate(new_), np.concatenate(S_old); d_ = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]
            good = (d_["ci95"][0] > 0) and (d_["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {name:36s}: {bal(new_):.4f}  {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items())
                  + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}; coefficients (mean of the four fits) " + " | ".join(
                      f"n={s}: alpha {np.mean([dd[v][1][s][0] for v in FOLDS]):.2f}, v5 {np.mean([dd[v][1][s][3] for v in FOLDS]):.2f}, reading {np.mean([dd[v][1][s][7] for v in FOLDS]):.2f}" + (f", new {np.mean([dd[v][1][s][8] for v in FOLDS]):+.2f}" if extra else "") for s in SIZES), flush=True)
            if name.startswith("PRIMARY"):
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)), flush=True)
            if name.startswith("PRIMARY"):
                for axis, vals in (("anchor evidence", TA), ("base posterior", AL), ("look-alike reading", RR)):
                    row = []
                    for m_ in vals:
                        fac = lambda k, m_=m_, axis=axis: (m_ if (axis == "anchor evidence" and k not in ("logp", "R")) or (axis == "base posterior" and k == "logp") or (axis == "look-alike reading" and k == "R") else 1.0)  # noqa: E731
                        S_ = [SM[mf7.guard2(np.asarray(b.decide(m41.apply(dd[v][1], {k: x * fac(k) for k, x in Xv[v].items()}, Fd[v]["n"], extra)), int), ref[v]), Fd[v]["y"]] for v in FOLDS]; row.append(f"{m_:g}: {bal(S_):.4f}")
                    print(f"        cross-validation, {axis} multiplied by -> score: " + "  ".join(row), flush=True)
            out[name] = {**d_, "per_fold": pf, "meets_m37": bool(good), "score": bal(new_), "score_plain": bal(plain)}
        rep[f"seeds_{seeds[0]}"] = out
    passed = all(ok); print(f"\nREGISTERED (M37 on both seed sets, the pool with the new source against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    out_json.write_text(json.dumps(rep, indent=1, default=float))
    if a_.no_build or not (passed or a_.build_anyway):
        return
    out_dir = Path(a_.out_dir) if a_.out_dir else ROOT; post_dir = Path(a_.out_dir) if a_.out_dir else ROOT / "results/r_variants"
    if not passed:
        print("DRY RUN OF THE BUILDER: the registered test did not pass, the file below is NOT a candidate")
    # ---------------------------------------------------------------- test side
    extra = (a_.new,); X012 = keep["X"]; Xcur = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}; Xnew = Xcur
    coef = m41.fit(Xnew, y, n, cw_all, extra); coef[15] = (coef[10] + coef[20]) / 2
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c104b, c106 = (sub(k) for k in ("ct9", "ct28", "ct104b", "ct106")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}
    for name, spec in [*[(k, rec2[k]) for k in NAMES], (a_.new, xspec[a_.new])]:
        idn, enc, cc, temp, pi0, bmin, sc = spec; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)
    ti = np.flatnonzero(q["qtest"] >= 0); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000
    R_t = lookalike(MEM["R"], tgt, gq, 800, 4, None)
    tq0 = np.zeros((800, 128)); np.add.at(tq0, gq, cur[0][0][cur[0][2], tgt].astype(np.float64)); tq0 /= np.maximum(np.linalg.norm(tq0, axis=1, keepdims=True), 1e-9)
    own = float(((cur[0][0][cur[0][2], tgt].astype(np.float64) @ tq0.T).argmax(1) == gq).mean())
    print(f"alignment check (member 0): a test game's target-side vector is closest to its own question's mean vector in {own:.3f} of the games"); assert own > 0.5, "test games and questions are not aligned"
    X0 = {"logp": np.log(np.maximum(P9, 1e-12)), "R": R_t, **{k: LQ[k] for k in NAMES}}; c0 = mb7.fit(Xcur, y, n, cw_all, "L0"); c0[15] = ((c0[10][0] + c0[20][0]) / 2, (c0[10][1] + c0[20][1]) / 2)  # exactly ct106's builder
    d0 = mf7.guard2(np.where(locked, l28, np.asarray(b.decide(mb7.apply(c0, X0, nq_t.astype(int))), int)), rk(c104b).astype(int)); assert (lev[d0] == c106).all(), int((lev[d0] != c106).sum()); print("checks: without the new inputs the builder reproduces ct106")
    Xt = {**X0, a_.new: LQ[a_.new]}; Pl = m41.apply(coef, Xt, nq_t.astype(int), extra); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin = lev[mf7.guard2(d_plain, rk(c106).astype(int))]; assert (fin[locked] == c28[locked]).all()
    grid, Ps = [], []
    for ta in TA:
        for al in AL:
            for rr in RR:
                Ps.append(m41.apply(coef, {k: x * (al if k == "logp" else rr if k == "R" else ta) for k, x in Xt.items()}, nq_t.astype(int), extra).astype(np.float64)); grid.append((ta, al, rr))
    ref_P = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True)["learned"].astype(np.float64); i0 = grid.index((1.0, 1.0, 1.0))
    print(f"grid of {len(grid)} settings; setting (1, 1, 1) against the stored posterior of ct112: max abs difference {float(np.abs(Ps[i0] - ref_P).max()):.2e}; answers of ct112 reproduced: {bool((fin == sub('ct112')).all())}")
    np.savez(ROOT / "results/r_variants/m229_grid.npz", grid=np.array(grid), P=np.array(Ps), qid=qids, base=P9, locked=locked, l28=l28, c106=rk(c106).astype(int), nq=nq_t.astype(int)); print("written results/r_variants/m229_grid.npz (no file is built)")
    return
    print("coefficients on all folds: " + " | ".join(f"n={s}: alpha {coef[s][0]:.2f}, v5 {coef[s][3]:.2f}, reading {coef[s][7]:.2f}, new source {coef[s][8]:+.2f}" for s in SIZES))
    print("test-side evidence of the new source: share of questions with a non-zero row " + f"{float((np.abs(LQ[a_.new]).sum(1) > 0).mean()):.3f} (v5: {float((np.abs(LQ['v5']).sum(1) > 0).mean()):.3f}); mean absolute log evidence {float(np.abs(LQ[a_.new]).mean()):.4f} (v5: {float(np.abs(LQ['v5']).mean()):.4f})")
    for nm, o in (("ct106", c106), ("ct104b", c104b)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c106[fin != c106], fin[fin != c106])]).value_counts(); print("kinds of changes against ct106: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    for i in np.flatnonzero(fin != c106):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct104b {c104b[i]}, ct106 {c106[i]} -> {fin[i]}")
    f = out_dir / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]
    print(f"{f.name} written (not copied to upload_next), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")
    if not passed:
        return
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct106": qids[fin != c106].tolist(), "public": int(((fin != c106) & pub).sum()), "private": int(((fin != c106) & ~pub).sum())}
    out_json.write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
