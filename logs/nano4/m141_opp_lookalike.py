#!/usr/bin/env python
"""Section 4M M141 (2026-11-15): the opponent-side look-alike reading as a new input of the learned pool (design doc M141).

Reading: every even game's opponent-side identity vector (var_S4e) against the prototypes of the labelled training players
(m123's reading: balanced level weights, sharpness 20) -> a log distribution over the ten levels; a question takes the mean over
its even games, centred per level over the questions. The pool of M137 (L0: alpha, six betas, gamma) gets one more coefficient
per question size. Composition: the four guards relative to the current final's rule (ct106 = L0 + guards relative to the
ct104b rule). Registered: the pool with the new input against the ct106 rule, M37 on seeds 0-2 and 6-8. On a pass the test-side
file is built (ct28's locked answers untouched, guards relative to ct106's answers; never overwrites).

  python -I logs/nano4/m141_opp_lookalike.py [--name ct107] [--no-build]
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, mb, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.mb, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE, RIDGE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE, mb7.RIDGE
EXTRA = ("Ro",)  # the new inputs, one coefficient each per question size (start 0)


def fit(X, y, n, w, extra, steps=300):
    """{size: theta = alpha, six betas, gamma, one per extra input} (no level offset: the L0 pool)."""
    out = {}
    for s in SIZES:
        m = n == s
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0); E = [torch.tensor(X[k][m]) for k in extra]
        yt, wt = torch.tensor(y[m]), torch.tensor(w[m])
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]] + [0.0] * len(extra), dtype=torch.float64)
        dl = torch.zeros(len(th0), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([dl], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); th = th0 + dl
            z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R
            for i, e in enumerate(E):
                z = z + th[8 + i] * e
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (dl ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        out[s] = (th0 + dl).detach().numpy().copy()
    return out


def apply(coef, X, n, extra):
    P = np.zeros_like(X["logp"])
    for s in sorted(set(int(x) for x in n)):
        m = n == s; th = coef[s]
        z = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + sum(th[8 + i] * X[k][m] for i, k in enumerate(extra))
        z = z - z.max(1, keepdims=True); e = np.exp(z); P[m] = e / e.sum(1, keepdims=True)
    return P


def opp_reading(vecs, protos, labels, beta, owner, n_q, use):
    """[n_q, 10]: per question the mean over its used games of the per-game log reading of the opponent-side vectors (zeros when
    a question has no used game). vecs [members, G, 128] unit vectors, owner [G] question index, use [G] bool."""
    lr, _, _ = la.reading(vecs[:, use], protos, labels, beta)
    out = np.zeros((n_q, 10)); cnt = np.zeros(n_q); np.add.at(out, owner[use], lr); np.add.at(cnt, owner[use], 1.0)
    return out / np.maximum(cnt, 1.0)[:, None], cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct107"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default="m141_opp_lookalike.json"); a_ = ap.parse_args()
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); hd_val = np.asarray(z["hd"])
    assert np.array_equal(q["qkey"][:Q].astype(str), z["key"]) and (q["side"][:Q] == 1).all() and (q["side"][Q:2 * Q] == 0).all(), "queries: the first Q entries must be the opponent side, the next Q the target side"
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-9)  # noqa: E731
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        protos = [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)]
        lr_, _, _ = la.reading(tq, protos, plab[cand], la.BE)
        ov = unit(QV[:, gi].astype(np.float64))  # the opponent-side vectors of the questions' games (a game used by several questions appears once per question)
        Ro, cnt = opp_reading(ov, protos, plab[cand], la.BE, qi, len(y), hd_val[gi] == 0)
        Ro50, _ = opp_reading(ov, protos, plab[cand], 50.0, qi, len(y), hd_val[gi] == 0)
        Ro1, _ = opp_reading(ov, protos, plab[cand], la.BE, qi, len(y), hd_val[gi] <= 1)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        cen = lambda x: x - x.mean(0, keepdims=True)  # noqa: E731
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=cen(lr_), Ro=cen(Ro), Ro50=cen(Ro50), Ro1=cen(Ro1), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)
        print(f"fold {v}: inputs ready ({len(y)} questions; share of questions without an even game {float((cnt == 0).mean()):.3f}; opponent reading alone: level of its maximum equals the truth {float((Ro.argmax(1) == y)[cnt > 0].mean()):.3f}, within one {float((np.abs(Ro.argmax(1) - y) <= 1)[cnt > 0].mean()):.3f})", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; ok = []; X012 = None
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    VARI = {"L0 (ct106 rule)": (), "+ opponent reading": ("Ro",), "+ opponent reading, beta 50": ("Ro50",), "+ opponent reading, with HD1 games": ("Ro1",)}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X, S = {}, {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], "Ro": f["Ro"], "Ro50": f["Ro50"], "Ro1": f["Ro1"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
            S.setdefault("ct104b rule", []).append(SM[f["dc"], f["y"]])
        if seeds == (0, 1, 2):
            X012 = X
        coefs = {}
        for name, extra in VARI.items():
            for v in FOLDS:
                tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
                coef = fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra); coefs[(name, v)] = coef
                ds = np.asarray(b.decide(apply(coef, X[v], Fd[v]["n"], extra)), int)
                if not extra:
                    Fd[v]["d106"] = mf7.guard2(ds, Fd[v]["dc"]); S.setdefault(name, []).append(SM[Fd[v]["d106"], Fd[v]["y"]])
                else:
                    S.setdefault(name, []).append(SM[mf7.guard2(ds, Fd[v]["d106"]), Fd[v]["y"]]); S.setdefault(name + " (plain)", []).append(SM[ds, Fd[v]["y"]])
        print(f"seeds {seeds}: ct104b rule {bal(S['ct104b rule']):.4f}, ct106 rule {bal(S['L0 (ct106 rule)']):.4f}")
        out = {}
        for name, extra in VARI.items():
            if not extra:
                continue
            A_, B_ = np.concatenate(S[name]), np.concatenate(S["L0 (ct106 rule)"]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[name], S["L0 (ct106 rule)"], ys, ns)]
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {name:36s}: {bal(S[name]):.4f}; against the ct106 rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items())
                  + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(S[name + ' (plain)']):.4f}; coefficient of the new input " + " ".join(f"n={s}: {np.mean([coefs[(name, v)][s][8] for v in FOLDS]):+.2f}" for s in SIZES))
            if name == "+ opponent reading":
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)))
                print("        other coefficients (mean of the four fits): " + " | ".join(f"n={s}: alpha {np.mean([coefs[(name, v)][s][0] for v in FOLDS]):.2f}, v5 {np.mean([coefs[(name, v)][s][2] for v in FOLDS]):.2f}, own look-alike {np.mean([coefs[(name, v)][s][7] for v in FOLDS]):.2f}" for s in SIZES), flush=True)
            out[name] = {**dd, "per_fold": pf, "meets_m37": bool(good)}
        rep[f"seeds_{seeds[0]}"] = out
    passed = all(ok); print(f"REGISTERED (M37 on both seed sets, the pool with the opponent reading against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))
    if not passed or a_.no_build:
        return
    # ---------------------------------------------------------------- test side
    extra = ("Ro",); Xall = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}
    coef = fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), extra); coef[15] = (coef[10] + coef[20]) / 2
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c104b, c106 = (sub(k) for k in ("ct9", "ct28", "ct104b", "ct106")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}
    for name in NAMES:
        idn, enc, cc, temp, pi0, bmin, sc = rec2[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)
    ti = np.flatnonzero(q["qtest"] >= 0); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); protos = [PR[m, 4].astype(np.float64) for m in range(NM)]
    tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; opp = ti[q["side"][ti] == 1]; opp = opp[np.argsort(q["qtest"][opp])]; assert len(tgt) == len(opp) == len(tqid) == 10000
    tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
    lr, _, _ = la.reading(tq, protos, plab, la.BE); lr = lr - lr.mean(0, keepdims=True)
    hd_t = np.asarray(T["5e"]["hd"]); Ro_t, cnt_t = opp_reading(unit(QV[:, opp].astype(np.float64)), protos, plab, la.BE, gq, 800, hd_t == 0); Ro_t = Ro_t - Ro_t.mean(0, keepdims=True)
    Xt = {"logp": np.log(np.maximum(P9, 1e-12)), "R": lr, "Ro": Ro_t, **{k: LQ[k] for k in NAMES}}
    X0 = {k: v for k, v in Xt.items() if k != "Ro"}; c0 = mb7.fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), "L0"); c0[15] = ((c0[10][0] + c0[20][0]) / 2, (c0[10][1] + c0[20][1]) / 2)  # exactly ct106's builder
    d0 = mf7.guard2(np.where(locked, l28, np.asarray(b.decide(mb7.apply(c0, X0, nq_t.astype(int))), int)), rk(c104b).astype(int)); assert (lev[d0] == c106).all(), int((lev[d0] != c106).sum()); print("checks: without the new input the builder reproduces ct106")
    Pl = apply(coef, Xt, nq_t.astype(int), extra); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin = lev[mf7.guard2(d_plain, rk(c106).astype(int))]; assert (fin[locked] == c28[locked]).all()
    np.savez(ROOT / "results/r_variants/m141_test_posterior.npz", qid=qids, learned=Pl, base=P9)
    print(f"test questions without an even game: {int((cnt_t == 0).sum())}; coefficients of the opponent reading " + " ".join(f"n={s}: {coef[s][8]:+.2f}" for s in SIZES))
    for nm, o in (("ct106", c106), ("ct104b", c104b)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c106[fin != c106], fin[fin != c106])]).value_counts(); print("kinds of changes against ct106: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    for i in np.flatnonzero(fin != c106):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct104b {c104b[i]}, ct106 {c106[i]} -> {fin[i]}")
    f = ROOT / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]
    print(f"{f.name} written (not copied to upload_next), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct106": qids[fin != c106].tolist(), "public": int(((fin != c106) & pub).sum()), "private": int(((fin != c106) & ~pub).sum())}
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
