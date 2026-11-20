#!/usr/bin/env python
"""Section 4M M171 (2026-11-15), REPORT ONLY: candidate-specific readings in the anchor evidence (design doc M171).

A copy of m169_pool.py with one more kind of source and one switch:
  TAG@cand   the identifier TAG as an extra source whose evidence uses the candidate's own reading: in the even-game group the
             ratio of the unchanged evidence code is multiplied by M[c, t] (results/r_variants/m171_cand.npz, the fold's
             version: computed without the opponents of the validation fold) and capped at 30 again;
  V@cand     (a token inside a set) the six anchor sources already in the pool (v44 and the five voters) are replaced by
             their versions with the candidate's own reading.
Sets as in m165_pool.py; the paired difference of every set against the first one.

  python -I logs/nano4/m171_pool.py --sets SX4L SX4L@cand SX4L@cand+V@cand
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
    xs = list(dict.fromkeys(t for s_ in sets for t in s_)); xs = [x for x in xs if x != "V@cand"]; xb = list(dict.fromkeys(x.split("@")[0] for x in xs))  # xb = the identifiers behind the sources
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], *[f"{x}~{STAT}" for x in xb]])
    miss = [x for x in xb if f"{x}~{STAT}" not in used]
    if miss:
        sys.exit(f"identifier files missing: {miss}")
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    dr = lambda x: round(m5.density_ratio(V[f"{x}~{STAT}"], T[f"{x}~{STAT}"], key_pid, dec, ct9s, nominal), 3)  # noqa: E731
    chk = {x: dr(x) for x in ids}; assert all(abs(chk[x] - scale[x]) < 2e-3 for x in ids), (chk, scale)  # the stored ratios of the five voters are reproduced
    xscale = {x: dr(x.split("@")[0]) for x in xs}; FLAG = {x: np.load(ROOT / "results/r_variants" / f"m169_{x.split('@')[1]}_val.npz") for x in xs if "@" in x and not x.endswith("@cand")}
    for x, f_ in FLAG.items():
        assert np.array_equal(f_["key"], V["5e"]["key"]), f"{x}: the flag file is not aligned with the validation rows"
    print("test-side density ratio (m15): " + ", ".join(f"{x} {v:.3f}" for x, v in {**chk, **xscale}.items()), flush=True)
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    xspec = {x: (f"{x.split('@')[0]}~{STAT}", 1, 1, 1, 1, 1, float(xscale[x]), ("cand" if x.endswith("@cand") else x) if "@" in x else None) for x in xs}
    MC = np.load(ROOT / "results/r_variants/m171_cand.npz", allow_pickle=True); pix = {p: i for i, p in enumerate(MC["players"].astype(str))}; CAND = {}

    def cand_of(idn):
        """Index of the first candidate of every validation row of identifier idn."""
        if idn not in CAND:
            best = np.load(ROOT / "results/r_variants/m82_val.npz", allow_pickle=True)["f5e_opp_best"] if idn == "5e" else np.load(ROOT / "results/r_variants" / f"m84_{idn}_val.npz", allow_pickle=True)["f_opp_best"]
            CAND[idn] = np.array([pix[p] for p in best.astype(str)])
        return CAND[idn]

    def evidence_cand(z, v, cc, pi0, band_min, scale, ci):
        """m103_stability.evidence with the candidate's own reading in the even-game group."""
        s2_, m87_, m83_ = m3.s2, m3.m87, m3.m83; tr = z["fold"] != v; edges, cal, band_all, g_all = s2_.fit(z, tr)
        if pi0:
            _, cal, _ = m87_.corrected(z, tr)
        R = m83_.tables(z, tr, cal, band_all, g_all, m83_.KAPPA) if cc else None; te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]; M = MC["M_fold"][FOLDS.index(v)]
        band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2_.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
        for k in (0, 1, 2):
            s_ = g == k; a = (scale * cal[k]["a"][band[s_]] * (band[s_] >= band_min))[:, None]; ratio = R[k][:, lab[s_]].T if cc else cal[k]["ratio"][lab[s_][:, None] - np.arange(10)[None, :] + 9]
            if k == 0:
                ratio = np.minimum(ratio * M[ci[te][s_]], 30.0)
            out[s_] = np.log((1 - a) + a * ratio)
        return dict(zip(z["key"][te], out))
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
                idn, enc, cc, temp, pi0, bmin, sc = spec[:7]; split = spec[7] if len(spec) > 7 else None
                if split == "cand":
                    ev = evidence_cand(V[idn], v, cc, pi0, bmin, sc, cand_of(idn))
                elif split:  # one calibration per part of the flag; games without a flag keep the unsplit calibration
                    fl = FLAG[split]["flag"]; ev = {}; whole = m3.evidence(V[idn], v, cc, pi0, bmin, sc)
                    for o in sorted(set(int(x) for x in fl if x >= 0)):
                        m_ = fl == o; ev.update(m3.evidence({k: x[m_] for k, x in V[idn].items()}, v, cc, pi0, bmin, sc))
                    ev.update({k: whole[k] for k in V[idn]["key"][(fl < 0) & (V[idn]["fold"] == v)]})
                else:
                    ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc)
                ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
            return cache[spec]
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {name: ev_of(spec) for name, spec in rec.items()}
        Lx = {x: ev_of(xspec[x]) for x in xs}; LsC = {name: ev_of((*spec[:7], "cand")) for name, spec in rec2.items()} if any("V@cand" in s_ for s_ in sets) else {}
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v)
        RD = {k: lookalike(mem, Q + gi, qi, len(y), v - 1, cand) for k, mem in MEM.items()}
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, Lx=Lx, LsC=LsC, cw=cw, wq=np.array([W_FIN[int(m)] for m in n]), **RD)
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
    VARI = [(" + ".join(s_), "R", s_) for s_ in sets]
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = base_of(v, seeds); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}, **f["Lx"]}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        dd0 = pool_decisions(X); ref = {v: mf7.guard2(dd0[v][0], Fd[v]["dc"]) for v in FOLDS}; S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]
        print(f"\nseeds {seeds}: ct104b rule {bal([SM[Fd[v]['dc'], Fd[v]['y']] for v in FOLDS]):.4f}, ct106 rule {bal(S_old):.4f}")
        if seeds == (0, 1, 2):
            keep = {"X": X, "ref": ref}
        out = {}; kept = {}
        if a_.dump and seeds == (0, 1, 2):
            dump = {"y": y, "n": n, "pid": pid, "fold": np.concatenate([np.full(len(Fd[v]["y"]), v) for v in FOLDS]), "ct106_rule": np.concatenate([ref[v] for v in FOLDS]), "ct104b_rule": np.concatenate([Fd[v]["dc"] for v in FOLDS])}
        for name, rkey, extra in VARI:
            vc = "V@cand" in extra; extra = tuple(x for x in extra if x != "V@cand"); Xv = {v: {**X[v], "R": Fd[v][rkey], **(Fd[v]["LsC"] if vc else {})} for v in FOLDS}; dd = pool_decisions(Xv, extra); new_ = [SM[mf7.guard2(dd[v][0], ref[v]), Fd[v]["y"]] for v in FOLDS]; plain = [SM[dd[v][0], Fd[v]["y"]] for v in FOLDS]
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
        for name, _, _ in VARI[1:]:
            A1, n1 = kept[name]; d2 = paired_diff(A1, A0, y, pid, n, sizes=SIZES, n=5000); pf2 = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(n1, n0, ys, ns)]
            print(f"    ({name}) minus ({VARI[0][0]}): {d2['diff']:+.4f} [{d2['ci95'][0]:+.4f}, {d2['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf2) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d2["by_n"].items()), flush=True); out[name]["minus_first"] = {**d2, "per_fold": pf2}
        rep[f"seeds_{seeds[0]}"] = out
    out_json.write_text(json.dumps(rep, indent=1, default=float)); print(f"\nREPORT ONLY (no file is built); written {out_json.name}")
    if a_.dump:
        np.savez(ROOT / "results/r_variants" / a_.dump, **{k.replace(" + ", "+"): v for k, v in dump.items()}); print(f"per-question decisions of seeds 0-2 written to {a_.dump}")


if __name__ == "__main__":
    main()
