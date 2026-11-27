#!/usr/bin/env python
"""M257 (2026-11-20), report only: the ct112 rule on COLOUR-PURE validation questions (the test's construction), and whether
fitting the pool on pure questions helps (design doc M257, registered before any result).

Pure questions = M60's sets (results/trunk_seq/es/m60_pure{B,W}_val{v}.npz: chunks of 5 / 10 of a validation player's black /
white games in the stored order; base posterior of the same six members as the standard validation). Inputs of every pure
question as in production: anchor evidence of the six voters and SQ4 (only the question's games; encoder term, size
temperature), the look-alike reading from the question's games.
  R0  coefficients fitted on the MIXED standard questions of the other folds (the production practice)
  R1  fitted on the pure questions of the other folds (both colours together)
  R2  one set per colour
  R3  R0 + a level offset vector per colour
Scored on the pure questions (class-balanced, sizes 5 and 10, plain expected-score decisions).

  python -I logs/nano4/m257_pure.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m105_voters as m5  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m141_opp_lookalike as m41  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402

m3, la, b, s2, ES, FOLDS, REC, SM = mb7.m3, mb7.la, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM
NAMES, FIVE, STAT, W_FIN, CACHE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE; RV = ROOT / "results/r_variants"; XS = "SQ4"; EXTRA = (XS,); LV = np.arange(10.0); PS = (5, 10)


def fit_sizes(X, y, n, w, sizes, steps=300):
    """m141_opp_lookalike.fit for the given sizes only."""
    out = {}
    for s in sizes:
        m = n == s
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0); E = [torch.tensor(X[k][m]) for k in EXTRA]; yt, wt = torch.tensor(y[m]), torch.tensor(w[m])
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]] + [0.0] * len(EXTRA), dtype=torch.float64); dl = torch.zeros(len(th0), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([dl], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); th = th0 + dl; z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R
            for i, e in enumerate(E):
                z = z + th[8 + i] * e
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + m41.RIDGE * (dl ** 2).sum(); loss.backward(); return loss
        opt.step(closure); out[s] = (th0 + dl).detach().numpy().copy()
    return out


def logits(coef, X, n):
    Z = np.zeros_like(X["logp"])
    for s in sorted(set(int(x) for x in n)):
        m = n == s; th = coef[s]; Z[m] = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + sum(th[8 + i] * X[k][m] for i, k in enumerate(EXTRA))
    return Z


def soft(Z):
    Z = Z - Z.max(1, keepdims=True); P = np.exp(Z); return P / P.sum(1, keepdims=True)


def fit_offset(Z, y, w, steps=200):
    """A level offset vector (sum zero) on top of fixed logits, weighted likelihood."""
    zt, yt, wt = torch.tensor(Z), torch.tensor(y), torch.tensor(w); o = torch.zeros(10, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([o], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); loss = (torch.nn.functional.cross_entropy(zt + o[None], yt, reduction="none") * wt).sum() / wt.sum() + 1e-3 * (o ** 2).sum(); loss.backward(); return loss
    opt.step(closure); return o.detach().numpy().copy()


def cweights(y, n, sizes):
    cw = np.zeros(len(y))
    for s in sizes:
        for c in range(10):
            m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)
    return cw


def main():
    pj = json.load(open(RV / "m129_bundle2.json")); scale = pj["density_ratio"]; ids = [REC[k][0] for k in FIVE]
    V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids], f"{XS}~{STAT}"]); assert f"{XS}~{STAT}" in used
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}; base_of = lambda v: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in (0, 1, 2)], axis=0)  # noqa: E731
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(base_of(v)), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    xscale = round(m5.density_ratio(V[f"{XS}~{STAT}"], T[f"{XS}~{STAT}"], key_pid, dec, ct9s, nominal), 3); rec2 = dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    specs = {**{k: rec2[k] for k in NAMES}, XS: (f"{XS}~{STAT}", 1, 1, 1, 1, 1, float(xscale))}; print(f"test-side density ratio of {XS}: {xscale:.3f}", flush=True)
    z = V["5e"]; row = {str(k): i for i, k in enumerate(z["key"])}; tblack = dict(zip(z["key"].astype(str).tolist(), z["target_black"].astype(bool).tolist())); q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"].astype(str))
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); mem = [(v_["qvec"], v_["protos"], m) for m in range(v_["qvec"].shape[0])]

    def lookalike(idx, owner, nq, net, cand):
        tq = np.zeros((len(mem), nq, 128))
        for j, (qv, _, m) in enumerate(mem):
            np.add.at(tq[j], owner, qv[m, idx].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); lr_, _, _ = la.reading(tq, [pr[m, net][cand].astype(np.float64) for _, pr, m in mem], plab[cand], la.BE); return lr_ - lr_.mean(0, keepdims=True)
    D = {}  # fold -> set -> dict(X, y, n, pid, col)
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y).astype(int), np.asarray(n).astype(int), np.asarray(pid).astype(str); gk = [list(map(str, g_)) for g_ in qd["game_keys"]]; cand = np.flatnonzero(pfold != v); EV = {}
        for nm, (idn, enc, cc, temp, pi0, bmin, sc) in specs.items():
            EV[nm] = ({str(k): e for k, e in m3.evidence(V[idn], v, cc, pi0, bmin, sc).items()}, enc, temp)
        ee = {str(k): e for k, e in m3.evidence(V["enc"], v, 0, 1, 1, m3.R_ENC).items()}

        def inputs(gks, ns, P):
            Tn = np.array([b.TN[int(m)] for m in ns])[:, None]; X = {"logp": np.log(np.maximum(P, 1e-12))}
            for nm, (ev, enc, temp) in EV.items():
                L = np.stack([sum((ev.get(k, m3.Z10) + (b.U_ENC * ee.get(k, m3.Z10) if enc else 0.0) for k in g_), m3.Z10) for g_ in gks]); X[nm] = L * Tn if temp else L
            qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gks)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gks]).astype(int); X["R"] = lookalike(Q + gi, qi, len(gks), v - 1, cand); return X
        D[v] = {"mixed": dict(X=inputs(gk, n, base_of(v)), y=y, n=n, pid=pid, col=np.full(len(y), "M"))}
        keys20 = {p_: g_ for p_, n_, g_ in zip(pid, n, gk) if n_ == 20}; std = {tuple(g_) for g_ in gk}
        zm = np.load(ES / f"m60_mixed_val{v}.npz"); hit = tot = 0
        for p_, n_, qid_ in zip(zm["player_id"].astype(str), zm["n"].astype(int), zm["question_id"].astype(str)):
            if p_ in keys20:
                j = int(qid_.split("_")[-1]); tot += 1; hit += tuple(keys20[p_][j * n_:(j + 1) * n_]) in std
        parts = []
        for tag, isb in (("pureB", True), ("pureW", False)):
            zp = np.load(ES / f"m60_{tag}_val{v}.npz"); pp, nn, qq, yy, PP = zp["player_id"].astype(str), zp["n"].astype(int), zp["question_id"].astype(str), zp["y"].astype(int), zp["proba"].astype(np.float64); gks, keep = [], []
            for i, (p_, n_, qid_) in enumerate(zip(pp, nn, qq)):
                if p_ not in keys20:
                    continue
                ks = [k for k in keys20[p_] if k in tblack and tblack[k] == isb]; j = int(qid_.split("_")[-1]); g_ = ks[j * n_:(j + 1) * n_]
                if len(g_) == n_:
                    gks.append(g_); keep.append(i)
            keep = np.array(keep); parts.append(dict(X=inputs(gks, nn[keep], PP[keep]), y=yy[keep], n=nn[keep], pid=pp[keep], col=np.full(len(keep), "B" if isb else "W"), lost=int(len(pp) - len(keep))))
        D[v]["pure"] = dict(X={k: np.concatenate([p_["X"][k] for p_ in parts]) for k in parts[0]["X"]}, **{k: np.concatenate([p_[k] for p_ in parts]) for k in ("y", "n", "pid", "col")})
        print(f"fold {v}: mixed {len(y)} questions; M60's mixed chunks found among the standard questions {hit}/{tot}; pure questions black {len(parts[0]['y'])} (not rebuilt {parts[0]['lost']}), white {len(parts[1]['y'])} (not rebuilt {parts[1]['lost']}); sizes " + ", ".join(f"n={s}: {int((D[v]['pure']['n'] == s).sum())}" for s in PS), flush=True)
    cat = lambda st, k, vs: np.concatenate([D[u][st][k] for u in vs])  # noqa: E731
    catX = lambda st, vs: {k: np.concatenate([D[u][st]["X"][k] for u in vs]) for k in D[vs[0]][st]["X"]}  # noqa: E731
    res = {}; ZZ = {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; P = D[v]["pure"]; ym, nm_ = cat("mixed", "y", tr), cat("mixed", "n", tr); c0 = fit_sizes(catX("mixed", tr), ym, nm_, cweights(ym, nm_, (5, 10, 20)), (5, 10, 20))
        yp, np_, cp = cat("pure", "y", tr), cat("pure", "n", tr), cat("pure", "col", tr); Xp = catX("pure", tr); c1 = fit_sizes(Xp, yp, np_, cweights(yp, np_, PS), PS)
        Z0 = logits(c0, P["X"], P["n"]); Z1 = logits(c1, P["X"], P["n"]); Z2 = np.zeros_like(Z0); Z3 = Z0.copy(); Z0tr = logits(c0, Xp, np_)
        for c_ in ("B", "W"):
            mt = cp == c_; c2 = fit_sizes({k: x[mt] for k, x in Xp.items()}, yp[mt], np_[mt], cweights(yp[mt], np_[mt], PS), PS); mv_ = P["col"] == c_; Z2[mv_] = logits(c2, {k: x[mv_] for k, x in P["X"].items()}, P["n"][mv_])
            for s in PS:
                ms = mt & (np_ == s); Z3[mv_ & (P["n"] == s)] += fit_offset(Z0tr[ms], yp[ms], cweights(yp[ms], np_[ms], (s,)))[None]
        ZZ[v] = dict(R0=Z0, R1=Z1, R2=Z2, R3=Z3, base=P["X"]["logp"], mixed=logits(c0, D[v]["mixed"]["X"], D[v]["mixed"]["n"]), c0=c0, c1=c1)
    y, n, pid, col = (np.concatenate([D[v]["pure"][k] for v in FOLDS]) for k in ("y", "n", "pid", "col")); S = {k: np.concatenate([SM[np.asarray(b.decide(soft(ZZ[v][k])), int), D[v]["pure"]["y"]] for v in FOLDS]) for k in ("R0", "R1", "R2", "R3", "base")}
    bal = lambda s_, m_=None: float(np.mean([balanced_mean_over_n(s_[(fo == v) & (m_ if m_ is not None else True)], y[(fo == v) & (m_ if m_ is not None else True)], n[(fo == v) & (m_ if m_ is not None else True)], sizes=PS) for v in FOLDS]))  # noqa: E731
    fo = np.concatenate([np.full(len(D[v]["pure"]["y"]), v) for v in FOLDS]); ym = np.concatenate([D[v]["mixed"]["y"] for v in FOLDS]); nmx = np.concatenate([D[v]["mixed"]["n"] for v in FOLDS]); Sm = np.concatenate([SM[np.asarray(b.decide(soft(ZZ[v]["mixed"])), int), D[v]["mixed"]["y"]] for v in FOLDS]); fm = np.concatenate([np.full(len(D[v]["mixed"]["y"]), v) for v in FOLDS])
    print(f"\npure questions: {len(y)} (black {int((col == 'B').sum())}, white {int((col == 'W').sum())}); the rule on the MIXED standard questions, sizes 5 / 10 (plain decisions): {float(np.mean([balanced_mean_over_n(Sm[fm == v], ym[fm == v], nmx[fm == v], sizes=PS) for v in FOLDS])):.4f}")
    print(f"on the PURE questions: base only {bal(S['base']):.4f}; R0 (production coefficients) {bal(S['R0']):.4f} (black questions {bal(S['R0'], col == 'B'):.4f}, white {bal(S['R0'], col == 'W'):.4f})")
    for k, nm in (("R1", "R1 fitted on pure questions"), ("R2", "R2 one set per colour"), ("R3", "R3 production + offset per colour")):
        d_ = paired_diff(S[k], S["R0"], y, pid, n, sizes=PS); pf = [float(balanced_mean_over_n(S[k][fo == v], y[fo == v], n[fo == v], sizes=PS) - balanced_mean_over_n(S["R0"][fo == v], y[fo == v], n[fo == v], sizes=PS)) for v in FOLDS]
        chg = float(np.mean(np.concatenate([np.asarray(b.decide(soft(ZZ[v][k])), int) != np.asarray(b.decide(soft(ZZ[v]["R0"])), int) for v in FOLDS]))); res[k] = {**d_, "per_fold": pf, "changed": chg}
        print(f"   {nm:36s}: {bal(S[k]):.4f}  against R0 {d_['diff']:+.4f} [{d_['ci95'][0]:+.4f}, {d_['ci95'][1]:+.4f}]  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in d_["by_n"].items()) + "  folds " + " ".join(f"{x:+.4f}" for x in pf) + f"  answers changed {chg:.3f}")
    print("coefficients (mean of the four fits; alpha, v44, v3, v5, s2, s3c, s3, reading, SQ4): " + " | ".join(f"n={s}: mixed " + " ".join(f"{x:.2f}" for x in np.mean([ZZ[v]['c0'][s] for v in FOLDS], 0)) + " ; pure " + " ".join(f"{x:.2f}" for x in np.mean([ZZ[v]['c1'][s] for v in FOLDS], 0)) for s in PS))
    # the same player's black and white questions under R0 and under the base alone
    print("\nsame player, same size: expected level of the black question minus that of the white question (first question of each colour)")
    for k, nm in (("R0", "pooled posterior (R0)"), ("base", "base members only")):
        E = np.concatenate([soft(ZZ[v][k]) @ LV for v in FOLDS]); rows_ = []
        df = pd.DataFrame({"pid": pid, "n": n, "col": col, "E": E, "y": y}).groupby(["pid", "n", "col"], as_index=False).first().pivot_table(index=["pid", "n", "y"], columns="col", values="E").dropna().reset_index(); df["d"] = df["B"] - df["W"]
        print(f"   {nm:24s}: {len(df)} (player, size) cells; black - white mean {df.d.mean():+.3f} (se {df.d.std() / np.sqrt(len(df)):.3f}); by level: " + " ".join(f"{c}: {df.d[df.y == c].mean():+.2f}" for c in range(10)) + f"; black - truth {float((df.B - df.y).mean()):+.3f}, white - truth {float((df.W - df.y).mean()):+.3f}"); res[f"bw_{k}"] = {"n": int(len(df)), "mean": float(df.d.mean()), "by_level": {str(c): float(df.d[df.y == c].mean()) for c in range(10)}}
    (RV / "m257_pure.json").write_text(json.dumps(res, indent=1, default=float)); print("written results/r_variants/m257_pure.json")


if __name__ == "__main__":
    main()
