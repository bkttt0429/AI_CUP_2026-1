#!/usr/bin/env python
"""Section 4M M129 (2026-11-14): one bundle test of two small, independent gains (design doc M129).

  A  the banding statistic of the opponent anchors: 0 = first minus second (current); 1 = m15 (first minus the mean of the 2nd-5th
     fused score) for recipe v5 only (M126); 2 = m15 for the five recipes v3, v5, s2, s3c, s3 (identifier files m84_<id>~m15_*.npz
     of scripts/m_series/m127_conf_family.py; the test-side density ratio of every m15 identifier computed as in m105_voters.py).
  B  the look-alike reading of M123 (mean target-side identity vector of the question, var_S4e, against the prototypes of the seen
     players; balanced level weights, sharpness 20, centred per level), pooled into the base probabilities with a weight per
     question size from {0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}.
Protocol (M86's): on folds {1, 2} the best (A, three weights) by the class-balanced score after the ct75u composition (seeds 0-2)
is chosen and scored on folds {3, 4}, and the other way round; seeds 6-8 use the same choices. Registered: the cross-fitted
choice against "all off" (= the ct75u rule) - CI lower bound > 0, >= 3 of 4 folds positive, difference >= 0.0025, on both seed
sets. On a pass the choice made on all four folds is built as --name (ct28's 38 locked answers untouched; with everything off the
builder must reproduce ct44 and ct75u question by question; the file is never overwritten).

  python -I scripts/m_series/m129_bundle2.py [--name ct104]
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m103_stability as m3  # noqa: E402
import m105_voters as m5  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402  (reading, pooled, VAR, BE)
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); WS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0); AS = (0, 1, 2); STAT = "m15"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct104")
    ap.add_argument("--drop", nargs="*", default=[], help="M129b: kinds of changes that are not applied, OLD>NEW with level names, NEW may be * (e.g. 'A>*' '5D>6D'); the ct75u answer stays")
    ap.add_argument("--picks-from", default=None, help="M129b: reuse the cross-fitted choices of this m129 JSON instead of choosing again")
    ap.add_argument("--json", default="m129_bundle2.json"); a_ = ap.parse_args()
    drops = [(LEVELS.index(x.split(">")[0]), None if x.split(">")[1] == "*" else LEVELS.index(x.split(">")[1])) for x in a_.drop]

    def keep_old(d_new, d_old):
        """The decisions with the dropped kinds of changes undone."""
        d = d_new.copy()
        for o, w_ in drops:
            m = (d_old == o) & (d_new != d_old) if w_ is None else (d_old == o) & (d_new == w_)
            d[m] = d_old[m]
        return d
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); missing = [f"{x}~{STAT}" for x in ids if f"{x}~{STAT}" not in used]
    if missing:
        sys.exit(f"identifier files missing: {missing}")
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0); dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(P), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    scale = {x: round(m5.density_ratio(V[f"{x}~{STAT}"], T[f"{x}~{STAT}"], key_pid, dec, ct9s, nominal), 3) for x in ids}
    print("test-side density ratio of the m15 identifiers: " + ", ".join(f"{x} {scale[x]:.3f} (m12: {REC[k][6]})" for k, x in zip(FIVE, ids)), flush=True)
    rec = {0: dict(REC), 1: dict(REC), 2: dict(REC)}
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; new = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, scale[idn]); rec[2][k] = new
        if k == "v5":
            rec[1][k] = new
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy(); v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A in AS:
            Ls[A] = {}
            for name, spec in rec[A].items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); Fd[v] = (y, n, pid, Ls, lr_ - lr_.mean(0, keepdims=True))
    half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; picks = None; ok = []; rep = {"density_ratio": scale}; ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2))
    ng_ = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").n_games.astype(int).reindex(pid.astype(str)).to_numpy()
    for seeds in ((0, 1, 2), (6, 7, 8)):
        SU = {}
        for v in FOLDS:
            yv, nv, _, Ls, RD = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            _, du0 = mf.rule75u(P, Ls[0])
            for A in AS:
                for w in WS:
                    _, du = mf.rule75u(P if w == 0 else la.pooled(P, RD, w), Ls[A]); SU.setdefault((A, w), []).append(SM[keep_old(du, du0), yv])
        fsz = lambda k, m: np.array([balanced_mean_over_n(s_, yy, nn, sizes=(m,)) for s_, yy, nn in zip(SU[k], ys, ns)])  # noqa: E731
        fs = lambda k: np.array([balanced_mean_over_n(s_, yy, nn, sizes=SIZES) for s_, yy, nn in zip(SU[k], ys, ns)])  # noqa: E731

        def assemble(choice):
            """Per fold the scores of the choice {fold: (A, {size: weight})}."""
            out = []
            for v in FOLDS:
                A, wm = choice[v]; x = np.zeros(len(Fd[v][0]))
                for m in SIZES:
                    s_ = Fd[v][1] == m; x[s_] = SU[(A, wm[m])][fi[v]][s_]
                out.append(x)
            return out
        if picks is None:
            def best(fl):
                idx = [fi[v] for v in fl]; cands = []
                for A in AS:
                    wm = {m: max(WS, key=lambda c: fsz((A, c), m)[idx].mean()) for m in SIZES}; cands.append((float(np.mean([fsz((A, wm[m]), m)[idx].mean() for m in SIZES])), A, wm))
                return max(cands, key=lambda c: c[0])[1:]
            picks = {**{v: best(half[1]) for v in half[0]}, **{v: best(half[0]) for v in half[1]}}; final = best(list(FOLDS))
            if a_.picks_from:  # M129b: the choices of the registered run, not chosen again
                pj = json.load(open(ROOT / "results/r_variants" / a_.picks_from))["picks"]; conv = lambda x: (int(x[0]), {int(k): float(v) for k, v in x[1].items()})  # noqa: E731
                picks = {**{v: conv(pj["second_half"]) for v in half[0]}, **{v: conv(pj["first_half"]) for v in half[1]}}; final = conv(pj["all_folds"]); print(f"choices taken from {a_.picks_from}; kinds of changes not applied: {a_.drop}")
            print(f"chosen on folds {half[0]}: A={best(half[0])[0]}, weights {best(half[0])[1]} | on folds {half[1]}: A={best(half[1])[0]}, weights {best(half[1])[1]} | on all four: A={final[0]}, weights {final[1]}")
            print("  class-balanced score after the ct75u composition, difference to all off (seeds 0-2, mean of the four folds):")
            for A in AS:
                print(f"    A={A}: " + "  ".join(f"w={w}: {fs((A, w)).mean() - fs((0, 0.0)).mean():+.4f}" for w in WS))
            rep["picks"] = {"first_half": [best(half[0])[0], {str(k): v for k, v in best(half[0])[1].items()}], "second_half": [best(half[1])[0], {str(k): v for k, v in best(half[1])[1].items()}], "all_folds": [final[0], {str(k): v for k, v in final[1].items()}]}
        new, old = assemble(picks), SU[(0, 0.0)]; A_, B_ = np.concatenate(new), np.concatenate(old); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
        pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new, old, ys, ns)]; good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); ok.append(good)
        print(f"seeds {seeds}: cross-fitted bundle against the ct75u rule: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} -> {'PASS' if good else 'no'}")
        print("    single switches at the all-folds choice (not cross-fitted, report): " + "  ".join(f"{nm}: {float(np.mean([balanced_mean_over_n(a, yy, nn, sizes=SIZES) for a, yy, nn in zip(assemble({v: ch for v in FOLDS}), ys, ns)]) - fs((0, 0.0)).mean()):+.4f}" for nm, ch in (("A only", (final[0], {m: 0.0 for m in SIZES})), ("B only", (0, final[1])), ("both", final))))
        print("    by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)) + " | players by games in the data set: " + "  ".join(f"{nm_}: {float((A_ - B_)[s_].mean()):+.4f}" for nm_, s_ in (("1-40", ng_ <= 40), ("41-170", (ng_ > 40) & (ng_ <= 170)), ("171+", ng_ > 170))), flush=True)
        rep[f"seeds_{seeds[0]}"] = {**dd, "per_fold": pf, "pass": bool(good)}
    passed = bool(all(ok)); print(f"PRE-REGISTERED (M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep["passed"] = passed; rep["drop"] = a_.drop; (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))
    if not passed:
        print("no file: the registered test did not pass"); return
    # ---------------------------------------------------------------- test side (m123's builder, the voters' evidence per switch A)
    A_fin, w_fin = final
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c75u = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75u")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()

    def evidence_of(A):
        LQ = {}
        for name in ("v44",) + FIVE:
            idn, enc, cc, temp, pi0, bmin, sc = rec[A][name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)
        return LQ

    def compose(P, LQ):
        dR, dA = {}, {}
        for name in LQ:
            d0, da, dr, _, region = mb.decisions(P, LQ[name], 0.5, 3.0, 0.6); dR[name] = np.where(locked, l28, dr); dA[name] = np.where(locked, l28, da)
        base = np.where(locked, l28, d0); d44 = dR["v44"]; keep = d44 != base; fin = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE])
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != d44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != base); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]
        return d44, fin
    d44_0, fin_0 = compose(P9, evidence_of(0)); assert (lev[d44_0] == c44).all(), int((lev[d44_0] != c44).sum()); assert (lev[fin_0] == c75u).all(), int((lev[fin_0] != c75u).sum()); print("checks: with every switch off the builder reproduces the files ct44 and ct75u")
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); own = float(((QV[0, tgt].astype(np.float64) @ tq[0].T).argmax(1) == gq).mean()); assert own > 0.5, "test games and questions are not aligned"
    lr, _, _ = la.reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, la.BE); lr = lr - lr.mean(0, keepdims=True); wq = np.array([{5: w_fin[5], 10: w_fin[10], 15: (w_fin[10] + w_fin[20]) / 2, 20: w_fin[20]}[int(m)] for m in nq_t])
    P9n = la.pooled(P9, lr, wq) if any(w_fin.values()) else P9; _, fin_1 = compose(P9n, evidence_of(A_fin)); n_before = int((lev[fin_1] != c75u).sum()); fin_1 = keep_old(fin_1, fin_0); fin = lev[fin_1]; assert (fin[locked] == c28[locked]).all()
    if drops:
        print(f"kinds of changes not applied {a_.drop}: {n_before} -> {int((fin != c75u).sum())} questions differ from ct75u")
    Lm = evidence_of(A_fin)["v5"]; post = P9n * np.exp(Lm - Lm.max(1, keepdims=True)); post /= post.sum(1, keepdims=True); L0 = evidence_of(0)["v5"]; post0 = P9 * np.exp(L0 - L0.max(1, keepdims=True)); post0 /= post0.sum(1, keepdims=True)
    np.savez(ROOT / "results/r_variants/m129_test_posterior.npz", qid=qids, bundle=post, current=post0, base=P9)  # for the ledger back-test: the posterior behind the decisions (look-alike reading x v5's evidence under the chosen statistic), and the current one
    for nm, parts in (("A only", (P9, evidence_of(A_fin))), ("B only", (P9n, evidence_of(0)))):
        f_ = lev[compose(*parts)[1]]; print(f"  {nm}: {int((f_ != c75u).sum())} questions differ from ct75u (Public {int(((f_ != c75u) & pub).sum())}, Private {int(((f_ != c75u) & ~pub).sum())})")
    for nm, o in (("ct75u", c75u), ("ct44", c44), ("ct28", c28)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    print("predicted levels, ct75u -> " + a_.name + ": " + "  ".join(f"{lv} {int((c75u == lv).sum())}->{int((fin == lv).sum())}" for lv in LEVELS))
    for i in np.flatnonzero(fin != c75u):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct44 {c44[i]}, ct75u {c75u[i]} -> {fin[i]}")
    f = ROOT / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]; (ROOT / "upload_next" / f.name).write_bytes(raw)
    print(f"{f.name} written, md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}; choice A={A_fin}, weights {w_fin}")
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct75u": qids[fin != c75u].tolist(), "public": int(((fin != c75u) & pub).sum()), "private": int(((fin != c75u) & ~pub).sum())}
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
