#!/usr/bin/env python
"""M255c (2026-11-20), report only: m255_val.py plus the distribution of |D| for same-player pairs and for twins, and how often the
median |D| of 20 pairs drawn with the test pairs' size mix (A = 5: 4 pairs, the rest from A = 10) reaches the value observed on the
20 test pairs (0.82). Nothing else is changed; writes results/r_variants/m255_dcheck.json.

Original header: M255 (2026-11-20), validation side of step 2 of the cross-competition linkage (design doc M255, registered before any result).

"Second question" posterior product: for a validation question A (5 or 10 games) and a 10-game question B of the SAME player
that shares no game with A,
    log P'(c) = log P_A(c) + a * [log P_base,B(c) - log pi(c)] + b * E_B(c)
P_A = the ct112 rule's cross-fitted pooled posterior (results/r_variants/m251_val_post.npz, seeds 0-2); P_base,B = the fold's base
members on B; E_B = the sum over B's games of the anchor evidence of the identifier SQ2 (statistic m15, calibration and the
test-side density ratio as in m163_pool.py; no encoder term); pi = the mean base posterior of the 10-game validation questions.
(a, b) per size of A, class-balanced likelihood on three folds, applied to the fourth. Decisions = expected score.
Report: A alone against the product (plain decisions; guards relative to A's guarded answer), the ablations b = 0 / a = 0 / a = b = 1,
the same players' 20-game questions as a reference, the label-free consistency statistic D = E[level | B alone] - E[level | A]
for the same player and for strangers, and (report only) B taken from a TWIN (mutual game-by-game favourites inside a fold).
Writes results/r_variants/m255_val.json (coefficients on all folds, pi, the density ratio, the quantiles of |D|).

  python -I logs/nano4/m255_dcheck.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m105_voters as m5  # noqa: E402
import m137b_build as mb7  # noqa: E402
import m137f_build2 as mf7  # noqa: E402
import m251b_votes as mv  # noqa: E402

m3, b, s2, ES, FOLDS, SM, CACHE = mb7.m3, mb7.b, mb7.s2, mb7.ES, mb7.FOLDS, mb7.SM, mb7.CACHE
STAT, X = "m15", "SQ2"; RNG = np.random.default_rng(2550); LV = np.arange(10.0)


def soft(L):
    L = L - L.max(1, keepdims=True); P = np.exp(L); return P / P.sum(1, keepdims=True)


def fit(lpa, lb, eb, y, w, mode):
    """(a, b) by the weighted multinomial likelihood; mode: both / base / anchor."""
    def nll(t):
        a_, b_ = (t[0], t[1]) if mode == "both" else (t[0], 0.0) if mode == "base" else (0.0, t[0])
        L = lpa + a_ * lb + b_ * eb; L = L - L.max(1, keepdims=True); return float(-(w * (L[np.arange(len(y)), y] - np.log(np.exp(L).sum(1)))).sum() / w.sum())
    r = minimize(nll, [0.5, 0.5] if mode == "both" else [0.5], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    return (float(r.x[0]), float(r.x[1])) if mode == "both" else (float(r.x[0]), 0.0) if mode == "base" else (0.0, float(r.x[0]))


def boot(diff, grp):
    up, inv = np.unique(grp, return_inverse=True); out = []
    for _ in range(2000):
        w = np.bincount(RNG.integers(0, len(up), len(up)), minlength=len(up))[inv]; out.append(float((diff * w).sum() / max(w.sum(), 1)))
    lo, hi = np.percentile(out, [2.5, 97.5]); return float(diff.mean()), float(lo), float(hi)


def main():
    V, T, key_pid, used = b.sources([f"{X}~{STAT}"]); assert f"{X}~{STAT}" in used
    LF = {v: m3.load_fold(v) for v in FOLDS}; dec = {}
    base_of = lambda v, seeds: np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # noqa: E731
    for v in FOLDS:
        y, n, pid, q_ = LF[v]; dec.update({p_: d_ for p_, n_, d_ in zip(pid, n, np.asarray(b.decide(base_of(v, (0, 1, 2))), int)) if n_ == 20})
    ct9s = pd.read_csv(ROOT / "submission_rank_ct9.csv", dtype=str).set_index("question_id").pred_rank.map(b.RANK); nominal = np.diff(np.concatenate([[0.0], s2.QS, [1.0]]))
    xscale = round(m5.density_ratio(V[f"{X}~{STAT}"], T[f"{X}~{STAT}"], key_pid, dec, ct9s, nominal), 3); print(f"test-side density ratio of {X} ({STAT}): {xscale:.3f}", flush=True)
    z = np.load(ROOT / "results/r_variants/m251_val_post.npz"); zf = z["fold"].astype(int); rep = {"identifier": X, "density_ratio": xscale}
    F = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y).astype(int), np.asarray(n).astype(int), np.asarray(pid).astype(str); gk = [list(map(str, g_)) for g_ in qd["game_keys"]]; rows = np.flatnonzero(zf == v)
        assert len(rows) == len(y) and (z["y"][rows].astype(int) == y).all() and (z["n"][rows].astype(int) == n).all() and (z["pid"][rows].astype(str) == pid).all(), f"fold {v}: stored posterior not aligned"
        ev = {str(k): e for k, e in m3.evidence(V[f"{X}~{STAT}"], v, 1, 1, 1, xscale).items()}; E = np.stack([sum((ev.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk])
        F[v] = dict(y=y, n=n, pid=pid, gk=gk, post=z["post"][rows].astype(np.float64), dec=z["dec"][rows].astype(int), E=E, covered=np.array([np.mean([k in ev for k in g_]) for g_ in gk]))
    for seeds in ((0, 1, 2),):
        tag = f"seeds_{seeds[0]}"; PB = {v: base_of(v, seeds) for v in FOLDS}; pi = np.concatenate([PB[v][F[v]["n"] == 10] for v in FOLDS]).mean(0); lpi = np.log(pi)
        print(f"\n== B side from base seeds {seeds}; pi = " + " ".join(f"{x:.3f}" for x in pi), flush=True)
        # ------------------------------------------------------------ pairs (A, B): same player, B a 10-game question, no shared game
        PR = {}
        for v in FOLDS:
            f = F[v]; by = {}
            for j, p_ in enumerate(f["pid"]):
                by.setdefault(p_, []).append(j)
            ia, ib, i20 = [], [], []
            for p_, js in by.items():
                tens = [j for j in js if f["n"][j] == 10]; tw = [j for j in js if f["n"][j] == 20]
                for j in js:
                    if f["n"][j] in (5, 10):
                        cand = [i for i in tens if i != j and not (set(f["gk"][i]) & set(f["gk"][j]))]
                        if cand:
                            ia.append(j); ib.append(cand[0]); i20.append(tw[0] if tw else -1)
            ia, ib, i20 = np.array(ia), np.array(ib), np.array(i20); PR[v] = dict(ia=ia, ib=ib, i20=i20, y=f["y"][ia], n=f["n"][ia], pid=f["pid"][ia], lpa=np.log(np.maximum(f["post"][ia], 1e-12)), lb=np.log(np.maximum(PB[v][ib], 1e-12)) - lpi, eb=f["E"][ib], decA=f["dec"][ia])
            cw = np.zeros(len(ia))
            for s in (5, 10):
                for c in range(10):
                    m = (PR[v]["n"] == s) & (PR[v]["y"] == c); cw[m] = 1.0 / max(int(m.sum()), 1)
            PR[v]["cw"] = cw
        print("pairs by fold and size of A: " + "; ".join(f"fold {v}: " + ", ".join(f"n={s} {int((PR[v]['n'] == s).sum())}" for s in (5, 10)) for v in FOLDS) + f"; share of B's games with an identifier row {np.mean([F[v]['covered'][PR[v]['ib']].mean() for v in FOLDS]):.3f}", flush=True)
        cat = lambda k, vs=FOLDS: np.concatenate([PR[v][k] for v in vs])  # noqa: E731
        out = {}; DK = {}
        for s in (5, 10):
            res = {}
            for mode in ("both", "base", "anchor", "naive"):
                S_new, S_old, S_g, grp, chg, coefs, post_all = [], [], [], [], [], [], {}
                for v in FOLDS:
                    tr = [u for u in FOLDS if u != v]; mt = cat("n", tr) == s; ab = (1.0, 1.0) if mode == "naive" else fit(cat("lpa", tr)[mt], cat("lb", tr)[mt], cat("eb", tr)[mt], cat("y", tr)[mt], cat("cw", tr)[mt], mode); coefs.append(ab)
                    p = PR[v]; m = p["n"] == s; Pn = soft(p["lpa"][m] + ab[0] * p["lb"][m] + ab[1] * p["eb"][m]); dn = np.asarray(b.decide(Pn), int); da = np.asarray(b.decide(np.exp(p["lpa"][m])), int)
                    S_new.append(SM[dn, p["y"][m]]); S_old.append(SM[da, p["y"][m]]); S_g.append(SM[mf7.guard2(dn, p["decA"][m]), p["y"][m]] - SM[p["decA"][m], p["y"][m]]); grp.append(p["pid"][m]); chg.append(dn != da); post_all[v] = Pn
                S_new, S_old, S_g, grp, chg = map(np.concatenate, (S_new, S_old, S_g, grp, chg)); d, lo, hi = boot(S_new - S_old, grp); dg, glo, ghi = boot(S_g, grp); am, bm = np.mean(coefs, 0)
                res[mode] = {"a": float(am), "b": float(bm), "score_A": float(S_old.mean()), "score_new": float(S_new.mean()), "diff": d, "ci": [lo, hi], "guarded_diff": dg, "guarded_ci": [glo, ghi], "changed": float(chg.mean()), "n": int(len(d_ := S_new))}
                print(f"  A = {s:2d} games + B = 10 games, {mode:6s}: a {am:.3f}, b {bm:.3f} | A alone {S_old.mean():.4f} -> product {S_new.mean():.4f}  {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | with the guards relative to A's answer {dg:+.4f} [{glo:+.4f}, {ghi:+.4f}] | answers changed {chg.mean():.3f} ({len(S_new)} pairs)", flush=True)
            # reference: the same players' 20-game question under the ct112 rule (guarded answer) against A's guarded answer
            ref = []
            for v in FOLDS:
                p = PR[v]; m = (p["n"] == s) & (p["i20"] >= 0); ref.append(SM[F[v]["dec"][p["i20"][m]], p["y"][m]] - SM[p["decA"][m], p["y"][m]])
            ref = np.concatenate(ref); print(f"  reference, the same players' 20-game question against A (both guarded): {ref.mean():+.4f}", flush=True); res["reference_20_games"] = float(ref.mean())
            # coefficients on all four folds
            mt = cat("n") == s; ab = fit(cat("lpa")[mt], cat("lb")[mt], cat("eb")[mt], cat("y")[mt], cat("cw")[mt], "both"); res["all_folds"] = {"a": ab[0], "b": ab[1]}
            # consistency statistic D: same player and strangers (B of another player of the same fold)
            Ds, Dx = [], []
            for v in FOLDS:
                p = PR[v]; m = np.flatnonzero(p["n"] == s); mu_a = np.exp(p["lpa"][m]) @ LV; mu_b = soft(lpi[None] + ab[0] * p["lb"][m] + ab[1] * p["eb"][m]) @ LV; Ds.append(mu_b - mu_a)
                perm = RNG.permutation(len(m)); ok = p["pid"][m][perm] != p["pid"][m]; Dx.append((mu_b[perm] - mu_a)[ok])
            Ds, Dx = np.concatenate(Ds), np.concatenate(Dx); DK.setdefault("same", {})[s] = np.abs(Ds); DK.setdefault("strangers", {})[s] = np.abs(Dx); qs = {str(q_): float(np.quantile(np.abs(Ds), q_)) for q_ in (0.5, 0.9, 0.95, 0.99)}
            res["D_same"] = {"mean": float(Ds.mean()), "sd": float(Ds.std()), "abs_quantiles": qs}; res["D_strangers"] = {"mean": float(Dx.mean()), "sd": float(Dx.std()), "abs_q50": float(np.quantile(np.abs(Dx), 0.5)), "share_above_same_q95": float((np.abs(Dx) > qs["0.95"]).mean()), "share_above_same_q99": float((np.abs(Dx) > qs["0.99"]).mean())}
            print(f"  D = E[level | B alone] - E[level | A], coefficients of all folds (a {ab[0]:.3f}, b {ab[1]:.3f}): same player mean {Ds.mean():+.3f}, sd {Ds.std():.3f}, |D| 50 / 90 / 95 / 99 % = " + " / ".join(f"{qs[k]:.2f}" for k in qs) + f"; strangers sd {Dx.std():.3f}, |D| median {np.quantile(np.abs(Dx), 0.5):.2f}, above the same-player 95 % / 99 % lines {float((np.abs(Dx) > qs['0.95']).mean()):.3f} / {float((np.abs(Dx) > qs['0.99']).mean()):.3f}", flush=True)
            out[str(s)] = res
        rep[tag] = {"pi": pi.tolist(), "sizes": out}
        if seeds != (0, 1, 2):
            continue
        KEEP = DK
        # ------------------------------------------------------------ report only: B taken from a twin (m251c_value.py's pairs)
        q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); vecs = []
        for name in mv.NETS:
            Vv = np.load(CACHE / f"var_{name}.npz")["qvec"]; vecs += [Vv[m] for m in range(Vv.shape[0])]
        dv = np.load(ROOT / "results/r_variants/m82_val.npz"); pidv = dv["pid"].astype(str); foldv = dv["fold"].astype(int); yv = dv["y"].astype(int); assert np.array_equal(q["qkey"][:Q].astype(str), dv["key"].astype(str)); tw = {}
        for v in FOLDS:
            idx = np.flatnonzero(foldv == v); u, names = pd.factorize(pidv[idx]); k = np.bincount(u); keep = k[u] >= 10; idx = idx[keep]; u, names = pd.factorize(pidv[idx]); nu = len(names); lab = np.zeros(nu, int); lab[u] = yv[idx]
            Fs = mv.scores(vecs, Q + idx, Q + idx, u, nu, u); W, fav, share, mut, _, _, kk = mv.votes(Fs, u, nu)
            tw[v] = [(str(names[a]), str(names[fav[a]]), float(min(share[a], share[fav[a]])), int(lab[a]), int(lab[fav[a]])) for a in range(nu) if mut[a]]
        del vecs
        for lo_ in (0.5, 0.7):
            for s in (5, 10):
                ab = (out[str(s)]["all_folds"]["a"], out[str(s)]["all_folds"]["b"]); rows_ = []
                for v in FOLDS:
                    f = F[v]; by = {}
                    for j, p_ in enumerate(f["pid"]):
                        by.setdefault(p_, []).append(j)
                    for a_, b_, sh, la_, lb_ in tw[v]:
                        if sh < lo_ or a_ not in by or b_ not in by:
                            continue
                        tens = [j for j in by[b_] if f["n"][j] == 10]
                        if not tens:
                            continue
                        jb = tens[0]; lb = np.log(np.maximum(PB[v][jb], 1e-12)) - lpi
                        for j in by[a_]:
                            if f["n"][j] == s:
                                lpa = np.log(np.maximum(f["post"][j], 1e-12)); Pn = soft((lpa + ab[0] * lb + ab[1] * f["E"][jb])[None]); dn = int(np.asarray(b.decide(Pn), int)[0]); da = int(np.asarray(b.decide(f["post"][j][None]), int)[0])
                                Dv = float(soft((lpi + ab[0] * lb + ab[1] * f["E"][jb])[None])[0] @ LV - f["post"][j] @ LV); rows_.append((a_, SM[dn, f["y"][j]] - SM[da, f["y"][j]], float(dn != da), float(la_ == lb_), float(abs(la_ - lb_) == 1), Dv))
                if rows_:
                    R = np.array([r[1:] for r in rows_], float); DK.setdefault(f"twin>={lo_}", {})[s] = np.abs(R[:, 4]); grp = np.array([r[0] for r in rows_]); d, lo, hi = boot(R[:, 0], grp); q99 = out[str(s)]["D_same"]["abs_quantiles"]["0.99"]; keep_ = np.abs(R[:, 4]) <= q99
                    print(f"  TWIN as B (report only), vote share >= {lo_}, A = {s:2d} games: {len(R)} questions of {len(set(grp))} players; twins' labels equal {R[:, 2].mean():.3f}, one apart {R[:, 3].mean():.3f}; product against A alone {d:+.4f} [{lo:+.4f}, {hi:+.4f}], answers changed {R[:, 1].mean():.3f}; D mean {R[:, 4].mean():+.3f}, sd {R[:, 4].std():.3f}; dropped by the |D| > 99 % rule {float((~keep_).mean()):.3f}, gain after the rule {float((R[:, 0] * keep_).mean()):+.4f}", flush=True)
                    rep.setdefault("twin", {})[f"{lo_}_{s}"] = {"n": int(len(R)), "players": int(len(set(grp))), "labels_equal": float(R[:, 2].mean()), "diff": d, "ci": [lo, hi], "D_mean": float(R[:, 4].mean()), "D_sd": float(R[:, 4].std()), "dropped": float((~keep_).mean()), "diff_after_rule": float((R[:, 0] * keep_).mean())}
    OBS = 0.82; out2 = {}
    print("\n== |D| by the kind of pair (50 / 75 / 90 %), and the chance that the median of 20 pairs (4 with A = 5, 16 with A = 10) is >= the 0.82 observed on the test pairs")
    for kind, d in KEEP.items():
        pr = float(np.mean([np.median(np.concatenate([RNG.choice(d[5], 4), RNG.choice(d[10], 16)])) >= OBS for _ in range(20000)])); out2[kind] = {"q": {str(s): [float(np.quantile(d[s], x)) for x in (0.5, 0.75, 0.9)] for s in (5, 10)}, "p_median_ge_obs": pr}
        print(f"   {kind:12s}: A = 5: " + " / ".join(f"{float(np.quantile(d[5], x)):.2f}" for x in (0.5, 0.75, 0.9)) + "; A = 10: " + " / ".join(f"{float(np.quantile(d[10], x)):.2f}" for x in (0.5, 0.75, 0.9)) + f"; P(median of 20 >= {OBS}) = {pr:.4f}")
    (ROOT / "results/r_variants/m255_dcheck.json").write_text(json.dumps(out2, indent=1)); print("written results/r_variants/m255_dcheck.json")


if __name__ == "__main__":
    main()
