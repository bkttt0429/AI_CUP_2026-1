#!/usr/bin/env python
"""Section 4M M131 (2026-11-14, pilot, report only): do look-alikes share the persistent label OFFSET?

M109: two thirds of the lost points come from a per-player offset between the label and what the play shows. The look-alike
reading (M123 / M129) pools the LABELS of the training players nearest to the target in the identity space; most of a label is
the player's strength, which the base already estimates, so the reading enters with a small weight. Here the neighbours'
OFFSETS are read instead: for every training player of folds 1-4, offset = label - expected level under the out-of-fold base
ensemble (seeds 0-2), with the level-dependent mean (regression towards the middle) removed by question size; the target's
predicted offset = kernel mean (sharpness beta) of the offsets of the players of the other folds, similarity = the question's
mean target-side identity vector against their prototypes (the cross network of the target's fold, as M123).
Reported: correlation of the predicted offset with the target's own residual (question and player level, by size), against
the label reading's implied shift; and the score of the base probabilities tilted by exp(lambda * offset * level) (lambda per
size chosen on the other half of the folds) at the base level and after the ct75u composition, seeds 0-2 and 6-8.
  python -I logs/nano4/m131_offset_lookalike.py [--var S4e]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; LAMS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0); LV = np.arange(10.0)


def base_P(v, seeds):
    return np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)


def level_curve(mu, res, n, sizes=SIZES, bins=18):
    """Per question size the mean residual by expected level (binned, linear interpolation): {size: (centres, means)}."""
    out = {}
    for m in sizes:
        s = n == m; edges = np.quantile(mu[s], np.linspace(0, 1, bins + 1)); k = np.clip(np.searchsorted(edges, mu[s], side="right") - 1, 0, bins - 1)
        out[m] = (np.array([mu[s][k == j].mean() for j in range(bins)]), np.array([res[s][k == j].mean() for j in range(bins)]))
    return out


def curve_at(curve, mu, n):
    out = np.zeros(len(mu))
    for m, (c, v) in curve.items():
        s = n == m; out[s] = np.interp(mu[s], c, v)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--var", default="S4e"); ap.add_argument("--betas", nargs="+", type=float, default=[10.0, 20.0, 40.0]); a_ = ap.parse_args()
    V, T, key_pid, used = b.sources(["5e", "SP", "SP2e", "S2", "S3c", "S3"]); z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])
    players = q["players"].astype(str); plab = q["plab"]; pidx = {p: i for i, p in enumerate(players)}; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{a_.var}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    F = {v: m3.load_fold(v) for v in FOLDS}; P0 = {v: base_P(v, (0, 1, 2)) for v in FOLDS}
    y_all = np.concatenate([np.asarray(F[v][0]) for v in FOLDS]); n_all = np.concatenate([np.asarray(F[v][1]) for v in FOLDS]); f_all = np.concatenate([np.full(len(F[v][0]), v) for v in FOLDS]); mu_all = np.concatenate([P0[v] @ LV for v in FOLDS]); pid_all = np.concatenate([np.asarray(F[v][2]) for v in FOLDS])
    res_all = y_all - mu_all
    # per fold: the level curve from the other folds, the adjusted residuals, and the players' offsets (their own fold's adjusted residuals, all sizes)
    adj = np.zeros(len(y_all))
    for v in FOLDS:
        tr = f_all != v; adj[f_all == v] = res_all[f_all == v] - curve_at(level_curve(mu_all[tr], res_all[tr], n_all[tr]), mu_all[f_all == v], n_all[f_all == v])
    off = pd.Series(adj).groupby(pid_all).mean(); print(f"players with an out-of-fold offset: {len(off)}; sd of the residual {res_all.std():.3f}, after removing the level curve {adj.std():.3f}; sd of the players' offsets {off.std():.3f}")
    offv = np.full(len(players), np.nan); offv[[pidx[p] for p in off.index]] = off.to_numpy()
    pred = {be: np.zeros(len(y_all)) for be in a_.betas}; lab_shift = np.zeros(len(y_all)); Ls_all = {}; RD_all = {}
    for v in FOLDS:
        y, n, pid, qd = F[v]; gk = list(qd["game_keys"]); cand = np.flatnonzero((pfold != v) & (pfold > 0) & np.isfinite(offv)); qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9); S = sum(tq[m] @ PR[m, v - 1][cand].astype(np.float64).T for m in range(NM)) / NM
        for be in a_.betas:
            w = np.exp(be * (S - S.max(1, keepdims=True))); pred[be][f_all == v] = (w @ offv[cand]) / w.sum(1)
        candl = np.flatnonzero(pfold != v); lr_, _, _ = la.reading(tq, [PR[m, v - 1][candl].astype(np.float64) for m in range(NM)], plab[candl], la.BE); lab_shift[f_all == v] = np.exp(lr_) @ LV - P0[v] @ LV; RD_all[v] = lr_ - lr_.mean(0, keepdims=True)
        Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}
        for name, (idn, enc, cc, temp, pi0, bmin, scale) in REC.items():
            ev = m3.evidence(V[idn], v, cc, pi0, bmin, scale); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
            L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); Ls[name] = L * Tn if temp else L
        Ls_all[v] = Ls
    cq = lambda a, c, s: float(np.corrcoef(a[s], c[s])[0, 1])  # noqa: E731
    print("correlation of the predicted offset with the target's adjusted residual (question level; all / n=5 / n=10 / n=20), and at player level:")
    for be in a_.betas:
        pl = pd.DataFrame({"p": pid_all, "a": pred[be], "r": adj}).groupby("p").mean()
        print(f"  beta {be:g}: " + " / ".join(f"{cq(pred[be], adj, s):+.3f}" for s in (np.ones(len(adj), bool), n_all == 5, n_all == 10, n_all == 20)) + f"   players {float(np.corrcoef(pl.a, pl.r)[0, 1]):+.3f}   sd of the prediction {pred[be].std():.3f}")
    pl = pd.DataFrame({"p": pid_all, "a": lab_shift, "r": adj}).groupby("p").mean()
    print("  label reading's implied shift (mean level of the look-alikes minus the base's expected level): " + " / ".join(f"{cq(lab_shift, adj, s):+.3f}" for s in (np.ones(len(adj), bool), n_all == 5, n_all == 10, n_all == 20)) + f"   players {float(np.corrcoef(pl.a, pl.r)[0, 1]):+.3f}")
    be = 20.0 if 20.0 in a_.betas else a_.betas[0]; dh = pred[be]; half = (list(FOLDS[:2]), list(FOLDS[2:])); fi = {v: i for i, v in enumerate(FOLDS)}; rep = {}; picks = None
    ys = [np.asarray(F[v][0]) for v in FOLDS]; ns = [np.asarray(F[v][1]) for v in FOLDS]
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S0, SU = {}, {}
        for v in FOLDS:
            P = base_P(v, seeds); d = dh[f_all == v]; yv = ys[fi[v]]
            for lam in LAMS:
                Pt = P * np.exp(lam * d[:, None] * LV[None, :]); Pt /= Pt.sum(1, keepdims=True); d0, du = mf.rule75u(Pt, Ls_all[v]); S0.setdefault(lam, []).append(SM[d0, yv]); SU.setdefault(lam, []).append(SM[du, yv])
        fsz = lambda S, k, m: np.array([balanced_mean_over_n(s_, yy, nn, sizes=(m,)) for s_, yy, nn in zip(S[k], ys, ns)])  # noqa: E731
        if picks is None:
            best = lambda S, fl, m: max(LAMS, key=lambda c: fsz(S, c, m)[[fi[v] for v in fl]].mean())  # noqa: E731
            picks = {"base": {**{v: {m: best(S0, half[1], m) for m in SIZES} for v in half[0]}, **{v: {m: best(S0, half[0], m) for m in SIZES} for v in half[1]}}, "rule": {**{v: {m: best(SU, half[1], m) for m in SIZES} for v in half[0]}, **{v: {m: best(SU, half[0], m) for m in SIZES} for v in half[1]}}}
            for nm, S in (("base level", S0), ("after the ct75u composition", SU)):
                print(f"  {nm}, difference to no tilt by lambda (seeds 0-2, mean of the four folds):")
                for m in SIZES:
                    print(f"      n={m}: " + "  ".join(f"{lam:g}: {fsz(S, lam, m).mean() - fsz(S, 0.0, m).mean():+.4f}" for lam in LAMS[1:]))
            print(f"  lambda by size chosen on the other half: base level {picks['base']}; after the rule {picks['rule']}")
        y, n, pid = y_all, n_all, pid_all
        for nm, S, pk in (("base level", S0, picks["base"]), ("after the ct75u composition", SU, picks["rule"])):
            new = [np.select([nn == m for m in SIZES], [S[pk[v][m]][fi[v]] for m in SIZES]) for v, nn in zip(FOLDS, ns)]; old = S[0.0]; dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=3000)
            pf = [float(balanced_mean_over_n(a, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for a, c, yy, nn in zip(new, old, ys, ns)]
            print(f"seeds {seeds} [{nm}] cross-fitted tilt against no tilt: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()), flush=True)
            rep[f"{seeds[0]}|{nm}"] = {**dd, "per_fold": pf}
    (ROOT / "results/r_variants" / f"m131_offset_lookalike_{a_.var}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
