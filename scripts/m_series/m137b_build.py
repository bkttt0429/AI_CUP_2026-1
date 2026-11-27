#!/usr/bin/env python
"""Section 4M M137b (2026-11-14): the learned log-linear pool of M137 - ablations, the cross-validation under ct104b's two
composition guards, and (only when the registered build rule holds) the test-side file.

Variants: L = the registered model (alpha, six betas, gamma, a level offset; one set per question size); L0 = no level offset;
L1 = no offset and alpha fixed at 1; Lb = only the offset (the rest at the hand-set values). Guarded: a variant's decisions with
the changes "out of A" and "5D -> 6D" undone relative to the ct75u rule's answers, compared with the ct104b rule (the M129 rule
under the same guards). Build rule (design doc M137b): L0 under the guards must meet M37 on both seed sets; L replaces it only if
it is better by >= 0.002 on both. Test side: coefficients fitted on all four folds (seeds 0-2), 15-game questions use the mean
of the 10- and 20-game coefficients; ct28's 38 locked answers untouched; the guards relative to ct75u; never overwrites a file.

  python -I scripts/m_series/m137b_build.py [--name ct105] [--no-build]
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); NAMES = ("v44",) + FIVE; STAT = "m15"
W_FIN = {5: 0.5, 10: 0.5, 20: 0.1}; RIDGE = 1e-3
DROPS = [(LEVELS.index("A"), None), (LEVELS.index("5D"), LEVELS.index("6D"))]
VARIANTS = {"L": (True, True, True), "L0": (False, True, True), "L1": (False, False, True), "Lb": (True, False, False)}  # (offset, alpha, source weights) learned


def keep_old(d_new, d_old):
    d = d_new.copy()
    for o, w_ in DROPS:
        m = (d_old == o) & (d_new != d_old) if w_ is None else (d_old == o) & (d_new == w_)
        d[m] = d_old[m]
    return d


def fit(X, y, n, w, variant, steps=300):
    """{size: (theta [8] = alpha, six betas, gamma; offset [10])} of one variant."""
    use_b, use_a, use_s = VARIANTS[variant]; out = {}
    free = torch.tensor([1.0 if use_a else 0.0] + [1.0 if use_s else 0.0] * 7, dtype=torch.float64)
    for s in SIZES:
        m = n == s
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0)
        yt, wt = torch.tensor(y[m]), torch.tensor(w[m])
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]], dtype=torch.float64)
        dl = torch.zeros(8, dtype=torch.float64, requires_grad=True); bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([dl, bias], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            th = th0 + free * dl; bb = bias if use_b else bias * 0.0
            z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R + bb
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (((free * dl) ** 2).sum() + (bb ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        out[s] = ((th0 + free * dl).detach().numpy().copy(), (bias.detach().numpy().copy() if use_b else np.zeros(10)))
    return out


def apply(coef, X, n):
    P = np.zeros_like(X["logp"])
    for s in sorted(set(int(x) for x in n)):
        m = n == s; th, bias = coef[s]
        z = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + bias
        z = z - z.max(1, keepdims=True); e = np.exp(z); P[m] = e / e.sum(1, keepdims=True)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct105"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default="m137b_build.json"); a_ = ap.parse_args()
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
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
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)
        print(f"fold {v}: inputs ready ({len(y)} questions)", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; passes = {k: [] for k in VARIANTS}; gdiff = {k: [] for k in VARIANTS}; X012 = None
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X, S = {}, {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["du"] = du
            for nm, d_ in (("m129_rule", db), ("ct104b_rule", keep_old(db, du)), ("ct75u_rule", du)):
                S.setdefault(nm, []).append(SM[d_, f["y"]])
        if seeds == (0, 1, 2):
            X012 = X
        for var in VARIANTS:
            for v in FOLDS:
                tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
                coef = fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), var)
                ds = np.asarray(b.decide(apply(coef, X[v], Fd[v]["n"])), int)
                S.setdefault(var, []).append(SM[ds, Fd[v]["y"]]); S.setdefault(var + "+guards", []).append(SM[keep_old(ds, Fd[v]["du"]), Fd[v]["y"]])
        print(f"seeds {seeds}: ct75u rule {bal(S['ct75u_rule']):.4f}, M129 rule {bal(S['m129_rule']):.4f}, ct104b rule (guards) {bal(S['ct104b_rule']):.4f}")
        out = {}
        for var in VARIANTS:
            line = f"  {var:3s}"
            for new, old in ((var, "m129_rule"), (var + "+guards", "ct104b_rule")):
                A_, B_ = np.concatenate(S[new]), np.concatenate(S[old]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
                pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[new], S[old], ys, ns)]
                good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
                line += f" | {'guarded vs ct104b rule' if '+' in new else 'plain vs M129 rule'}: {bal(S[new]):.4f} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}"
                out[f"{new} - {old}"] = {**dd, "per_fold": pf, "meets_m37": bool(good), "level": bal(S[new])}
                if "+" in new:
                    passes[var].append(bool(good)); gdiff[var].append(float(dd["diff"]))
                    lv = "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10))
            print(line); print(f"        guarded, by true level: {lv}", flush=True)
        rep[f"seeds_{seeds[0]}"] = out
    choice = None
    if all(passes["L0"]):
        choice = "L0"
    if all(passes["L"]) and all(g - h >= 0.002 for g, h in zip(gdiff["L"], gdiff["L0"])):
        choice = "L"
    print(f"REGISTERED BUILD RULE: L0 under the guards {'meets' if all(passes['L0']) else 'does not meet'} M37 on both seed sets; L {'meets' if all(passes['L']) else 'does not meet'} it; L minus L0 under the guards "
          + " / ".join(f"{g - h:+.4f}" for g, h in zip(gdiff["L"], gdiff["L0"])) + f" -> {'build ' + choice if choice else 'NO FILE'}")
    rep["choice"] = choice; (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))
    if choice is None or a_.no_build:
        return
    # ---------------------------------------------------------------- test side (the inputs exactly as m129_bundle2.py builds them)
    Xall = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}
    coef = fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), choice); coef[15] = ((coef[10][0] + coef[20][0]) / 2, (coef[10][1] + coef[20][1]) / 2)
    print("coefficients fitted on all four folds (seeds 0-2): " + " | ".join(f"n={s}: alpha {coef[s][0][0]:.2f}, beta " + " ".join(f"{k} {coef[s][0][1 + i]:+.2f}" for i, k in enumerate(NAMES)) + f", gamma {coef[s][0][7]:.2f}" for s in SIZES))
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731
    c9, c28, c44, c75u, c104b = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75u", "ct104b")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()

    def evidence_of(rec):
        LQ = {}
        for name in NAMES:
            idn, enc, cc, temp, pi0, bmin, sc = rec[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)
        return LQ

    def compose(P, LQ):
        dR, dA = {}, {}
        for name in LQ:
            d0, da, dr, _, region = mb.decisions(P, LQ[name], 0.5, 3.0, 0.6); dR[name] = np.where(locked, l28, dr); dA[name] = np.where(locked, l28, da)
        base = np.where(locked, l28, d0); d44 = dR["v44"]; keep = d44 != base; fin = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE])
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != d44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != base); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]
        return d44, fin
    d44_0, fin_0 = compose(P9, evidence_of(rec0)); assert (lev[d44_0] == c44).all() and (lev[fin_0] == c75u).all(); print("checks: with the learned pool off the builder reproduces the files ct44 and ct75u")
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))
    for m in range(NM):
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
    lr, _, _ = la.reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, la.BE); lr = lr - lr.mean(0, keepdims=True)
    LQ2 = evidence_of(rec2); Xt = {"logp": np.log(np.maximum(P9, 1e-12)), "R": lr, **{k: LQ2[k] for k in NAMES}}
    Pl = apply(coef, Xt, nq_t.astype(int)); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin_i = keep_old(d_plain, fin_0); fin = lev[fin_i]; assert (fin[locked] == c28[locked]).all()
    np.savez(ROOT / "results/r_variants/m137_test_posterior.npz", qid=qids, learned=Pl, base=P9)
    print(f"guards: {int((lev[d_plain] != c75u).sum())} -> {int((fin != c75u).sum())} questions differ from ct75u")
    for nm, o in (("ct104b", c104b), ("ct75u", c75u), ("ct44", c44), ("ct28", c28)):
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")
    print("predicted levels, ct104b -> " + a_.name + ": " + "  ".join(f"{lv}{int((c104b == lv).sum())}->{int((fin == lv).sum())}" for lv in LEVELS))
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c104b[fin != c104b], fin[fin != c104b])]).value_counts(); print("kinds of changes against ct104b: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))
    for i in np.flatnonzero(fin != c104b):
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct75u {c75u[i]}, ct104b {c104b[i]} -> {fin[i]}")
    f = ROOT / f"submission_rank_{a_.name}.csv"
    if f.exists():
        sys.exit(f"{f.name} exists already: nothing overwritten")
    with open(f, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]
    print(f"{f.name} written (not copied to upload_next: the ledger back-test comes first), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}; variant {choice}")
    rep["file"] = {"name": f.name, "md5": md5, "variant": choice, "differs_from_ct104b": qids[fin != c104b].tolist(), "public": int(((fin != c104b) & pub).sum()), "private": int(((fin != c104b) & ~pub).sum())}
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
