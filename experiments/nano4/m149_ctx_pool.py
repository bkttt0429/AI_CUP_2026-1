#!/usr/bin/env python
"""Section 4M M149 (2026-11-15): the learned pool with weights that depend on where the question sits on the level axis
(design doc M149).

The L0 pool has one weight per input and question size: logit = alpha log P_base + sum_v beta_v L_v + gamma R. M132 found
structure by segment (look-alike readings help most where the base answers 1D, hurt where it answers B / 2D / 4D), but picking
30 segment weights on half of the folds over-fitted. Here the same idea as one smooth, jointly fitted model: every weight is a
piecewise-linear function of the base's expected level mu (three knots: 1.5 = kyu, 4.5 = low dan, 7.5 = high dan), fitted by
class-balanced cross-entropy with a ridge that pulls the three knot values of a weight together (strength --tie).
  V3 (PRIMARY): all eight weights depend on mu;   V2: only gamma (the look-alike reading);   V1: only alpha;
  VA: only the six anchor weights.
REGISTERED: V3 with tie = 0.01 against the ct106 rule (nested fit on the other three folds, the four guards relative to the
ct106 rule), M37 on seeds 0-2 and 6-8. The other variants and tie in {0.003, 0.03} are reported only.

  python -I logs/nano4/m149_ctx_pool.py
"""
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
import m141_opp_lookalike as m41  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

m3, mf, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE, RIDGE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE, mb7.RIDGE
KNOTS = np.array([1.5, 4.5, 7.5]); FREE = {"V3": list(range(8)), "V2": [7], "V1": [0], "VA": [1, 2, 3, 4, 5, 6]}


def hats(mu):
    """[N, 3] piecewise-linear weights of the three knots (flat outside)."""
    x = np.clip(mu, KNOTS[0], KNOTS[-1]); H = np.zeros((len(mu), 3)); lo = x < KNOTS[1]; t = np.where(lo, (x - KNOTS[0]) / (KNOTS[1] - KNOTS[0]), (x - KNOTS[1]) / (KNOTS[2] - KNOTS[1]))
    H[lo, 0] = 1 - t[lo]; H[lo, 1] = t[lo]; H[~lo, 1] = 1 - t[~lo]; H[~lo, 2] = t[~lo]
    return H


def feats(X):
    """[N, 8, 10]: the eight inputs in the pool's order (log P, six anchor sources, the look-alike reading)."""
    return np.stack([X["logp"], *[X[k] for k in NAMES], X["R"]], 1)


def fit(X, y, n, w, variant, tie, steps=300):
    out = {}; free = torch.zeros(8, dtype=torch.float64); free[FREE[variant]] = 1.0
    for s in SIZES:
        m = n == s; F = torch.tensor(feats(X)[m]); H = torch.tensor(hats(np.exp(X["logp"][m]) @ np.arange(10.0))); yt, wt = torch.tensor(y[m]), torch.tensor(w[m])
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]], dtype=torch.float64)
        dl = torch.zeros(8, dtype=torch.float64, requires_grad=True); dv = torch.zeros(3, 8, dtype=torch.float64, requires_grad=True)  # shared part, knot deviations
        opt = torch.optim.LBFGS([dl, dv], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); th = th0 + dl + (dv - dv.mean(0, keepdim=True)) * free  # [3, 8]
            zz = ((H @ th)[:, :, None] * F).sum(1)
            loss = (torch.nn.functional.cross_entropy(zz, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (dl ** 2).sum() + tie * ((dv - dv.mean(0, keepdim=True)) ** 2 * free).sum()
            loss.backward()
            return loss
        opt.step(closure)
        out[s] = (th0 + dl + (dv - dv.mean(0, keepdim=True)) * free).detach().numpy().copy()
    return out


def apply(coef, X, n):
    F = feats(X); H = hats(np.exp(X["logp"]) @ np.arange(10.0)); P = np.zeros_like(X["logp"])
    for s in sorted(set(int(x) for x in n)):
        m = n == s; zz = ((H[m] @ coef[s])[:, :, None] * F[m]).sum(1); zz = zz - zz.max(1, keepdims=True); e = np.exp(zz); P[m] = e / e.sum(1, keepdims=True)
    return P


def main():
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]]); rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}; z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]; Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    Lg = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = Lg * Tn if temp else Lg
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE); cw = np.zeros(len(y))
        for s in SIZES:
            for c in range(10):
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw); print(f"fold {v}: inputs ready", flush=True)
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; ok = []
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731
    cat = lambda X, tr, k: np.concatenate([X[u][k] for u in tr])  # noqa: E731
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X = {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0); X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)
        ref = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: cat(X, tr, k) for k in X[v]}; c0 = m41.fit(Xt, cat(Fd, tr, "y"), cat(Fd, tr, "n"), cat(Fd, tr, "cw"), ()); ref[v] = mf7.guard2(np.asarray(b.decide(m41.apply(c0, X[v], Fd[v]["n"], ())), int), Fd[v]["dc"])
        S_old = [SM[ref[v], Fd[v]["y"]] for v in FOLDS]; print(f"\nseeds {seeds}: ct106 rule {bal(S_old):.4f}"); out = {}
        for variant, tie in (("V3", 0.01), ("V3", 0.003), ("V3", 0.03), ("V2", 0.01), ("V1", 0.01), ("VA", 0.01)):
            new_, plain, cf = [], [], []
            for v in FOLDS:
                tr = [u for u in FOLDS if u != v]; Xt = {k: cat(X, tr, k) for k in X[v]}; coef = fit(Xt, cat(Fd, tr, "y"), cat(Fd, tr, "n"), cat(Fd, tr, "cw"), variant, tie); d_ = np.asarray(b.decide(apply(coef, X[v], Fd[v]["n"])), int)
                new_.append(SM[mf7.guard2(d_, ref[v]), Fd[v]["y"]]); plain.append(SM[d_, Fd[v]["y"]]); cf.append(coef)
            A_, B_ = np.concatenate(new_), np.concatenate(S_old); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000); pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new_, S_old, ys, ns)]
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3); prim = variant == "V3" and tie == 0.01
            print(f"  {variant} tie {tie:<5g}{' PRIMARY' if prim else '        '}: {bal(new_):.4f}  {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(plain):.4f}", flush=True)
            if prim:
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)))
                for s in SIZES:
                    th = np.mean([c_[s] for c_ in cf], 0); print(f"        n={s} weights at the knots (kyu / low dan / high dan): alpha " + "/".join(f"{x:.2f}" for x in th[:, 0]) + "; v5 " + "/".join(f"{x:.2f}" for x in th[:, 2]) + "; s2 " + "/".join(f"{x:.2f}" for x in th[:, 3]) + "; look-alike " + "/".join(f"{x:.2f}" for x in th[:, 7]))
            out[f"{variant}|{tie}"] = {**dd, "per_fold": pf, "meets_m37": bool(good)}
        rep[f"seeds_{seeds[0]}"] = out
    passed = all(ok); print(f"\nREGISTERED (V3, tie 0.01, M37 on both seed sets): {'PASS' if passed else 'NO'}"); rep["passed"] = passed
    (ROOT / "results/r_variants/m149_ctx_pool.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m149_ctx_pool.json")


if __name__ == "__main__":
    main()
