#!/usr/bin/env python
"""Section 4M M137 (2026-11-14): a learned log-linear pool of the inputs of the ct104b rule.

Inputs per question: log of the base probabilities, the anchor evidence of six recipes (v44 and the five voters, m15 statistic),
the look-alike reading. Model: logit = alpha_n log P + sum_v beta_{n,v} L_v + gamma_n R + b_n with one set of coefficients per
question size n (54 parameters), class-balanced cross-entropy, L2 shrinkage (1e-3) towards the hand-set values. Nested: the
coefficients of validation fold v are fitted on the other three folds. Registered comparison: the learned pool against the M129
rule without the composition guards (same inputs), M37 on seeds 0-2 and 6-8.

  python -I logs/nano4/m137_stacker.py [--json m137_stacker.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); NAMES = ("v44",) + FIVE; STAT = "m15"
W_FIN = {5: 0.5, 10: 0.5, 20: 0.1}; RIDGE = 1e-3


def fit(X, y, n, w, steps=300):
    """X: dict of [Q, 10] float arrays (logp, L_<name>, R); y [Q]; n [Q] sizes; w [Q] weights. Returns {size: (alpha, beta[6], gamma, bias[10])}."""
    out = {}
    for s in SIZES:
        m = n == s
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0)
        yt, wt = torch.tensor(y[m]), torch.tensor(w[m])
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]], dtype=torch.float64)
        th = th0.clone().requires_grad_(True); bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, bias], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R + bias
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (((th - th0) ** 2).sum() + (bias ** 2).sum())
            loss.backward()
            return loss
        opt.step(closure)
        out[s] = (th.detach().numpy().copy(), bias.detach().numpy().copy())
    return out


def apply(coef, X, n):
    P = np.zeros_like(X["logp"])
    for s in SIZES:
        m = n == s; th, bias = coef[s]
        z = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + bias
        z = z - z.max(1, keepdims=True); e = np.exp(z); P[m] = e / e.sum(1, keepdims=True)
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--json", default="m137_stacker.json"); a_ = ap.parse_args()
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
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        X, S = {}, {}
        for v in FOLDS:
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2])
            for nm, d_ in (("base", d0), ("ct75u_rule", du), ("m129_rule", db)):
                S.setdefault(nm, []).append(SM[d_, f["y"]])
            f["db"] = db
        coefs = {}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}
            coefs[v] = fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]))
            Ps = apply(coefs[v], X[v], Fd[v]["n"]); ds = np.asarray(b.decide(Ps), int); S.setdefault("learned", []).append(SM[ds, Fd[v]["y"]])
            keep = Fd[v]["db"] != np.asarray(b.decide(np.exp(X[v]["logp"])), int)  # the rule changed the base answer: keep the rule's answer there
            S.setdefault("learned_conservative", []).append(SM[np.where(keep, Fd[v]["db"], ds), Fd[v]["y"]])
        lvl = {k: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S[k], ys, ns)])) for k in S}
        print(f"seeds {seeds}: " + "  ".join(f"{k} {lvl[k]:.4f}" for k in ("base", "ct75u_rule", "m129_rule", "learned", "learned_conservative")))
        out = {"levels": lvl}
        for new, old in (("learned", "m129_rule"), ("learned_conservative", "m129_rule"), ("learned", "ct75u_rule"), ("learned", "base")):
            A_, B_ = np.concatenate(S[new]), np.concatenate(S[old]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[new], S[old], ys, ns)]
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"  {new} - {old}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items())
                  + f"  changed {float((A_ != B_).mean()):.3f}" + (f" -> {'PASS' if good else 'no'}" if old == "m129_rule" else ""))
            if (new, old) == ("learned", "m129_rule"):
                print("      by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)))
            out[f"{new} - {old}"] = {**dd, "per_fold": pf, "meets_m37": bool(good)}
        print("  coefficients (mean over the four fits): " + " | ".join(f"n={s}: alpha {np.mean([coefs[v][s][0][0] for v in FOLDS]):.2f}, beta " + " ".join(f"{k} {np.mean([coefs[v][s][0][1 + i] for v in FOLDS]):+.2f}" for i, k in enumerate(NAMES))
              + f", gamma {np.mean([coefs[v][s][0][7] for v in FOLDS]):.2f}, |bias| {np.mean([np.abs(coefs[v][s][1]).mean() for v in FOLDS]):.2f}" for s in SIZES), flush=True)
        out["coef"] = {str(v): {str(s): [coefs[v][s][0].tolist(), coefs[v][s][1].tolist()] for s in SIZES} for v in FOLDS}; rep[f"seeds_{seeds[0]}"] = out
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float)); print(f"written results/r_variants/{a_.json}")


if __name__ == "__main__":
    main()
