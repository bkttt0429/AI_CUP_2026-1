"""Section 4M M80 (2026-11-10): is the gain of M79's "average" a tempering of over-counted anchor evidence, or the encoder?
Posterior ~ P * exp(T_n * L_fused + U_n * L_encoder); T, U depend only on the question size n and are CROSS-FITTED: for the held-out fold v
they maximise the class-balanced score at size n over the questions of the other three folds (grid; ties keep the value nearest to today's
T = 1, U = 0). Primary A: cross-fitted T (U = 0) vs T = 1. Primary B: cross-fitted (T, U) vs cross-fitted T. M37 rule, seeds 0-2 and 6-8.
Report only: fixed T = 0.5; fixed 0.5 / 0.5; the picks; the NLL-optimal T per n (calibration slope).
  python -I logs/nano4/m80_temper.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

GT = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25); GU = (0.0, 0.25, 0.5, 0.75, 1.0)  # preferred value first; a strict ">" keeps it on ties
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)
d = dict(np.load(ROOT / "results/r_variants/m73_val.npz"))
F = {**{k: d[k] for k in ("key", "fold", "y", "hd", "target_black")}, "opp_lab": d["f_opp_lab"], "opp_m": d["f_opp_m"], "imp_m": d["f_imp_m"]}
parts = [dict(np.load(ROOT / f"results/m77/val{v}.npz")) for v in FOLDS]; E = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_m")}


def per_query(z, v):
    """{game_key: log-evidence [10]} of fold v, calibrated on the other folds (M71 step 2)."""
    edges, cal, _, _ = s2.fit(z, z["fold"] != v); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]
    logE = np.zeros((len(te), 10))
    for k in (0, 1, 2):
        s = g == k; a = cal[k]["a"]; dd = lab[s][:, None] - np.arange(10)[None, :]
        logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])
    return dict(zip(z["key"][te], logE))


def post(P, L):
    Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); return Pf / Pf.sum(axis=1, keepdims=True)


def sc(P, L, y):
    """Official score of the expected-score decision, vectorised."""
    return SM[(post(P, L) @ SM).argmax(axis=1), y]


EV = {}
for v in FOLDS:  # evidence does not depend on the base model's seeds
    y, n, pid, q = load_fold(v); qf, qe = per_query(F, v), per_query(E, v); Lf = np.zeros((len(y), 10)); Le = np.zeros((len(y), 10)); kf = np.zeros(len(y)); ke = np.zeros(len(y))
    for i, gk in enumerate(q["game_keys"]):
        for k in gk:
            if k in qf:
                Lf[i] += qf[k]; kf[i] += np.abs(qf[k]).max() > 0.05
            if k in qe:
                Le[i] += qe[k]; ke[i] += np.abs(qe[k]).max() > 0.05
    EV[v] = dict(y=np.asarray(y), n=np.asarray(n), pid=np.asarray(pid), q=q, Lf=Lf, Le=Le, kf=kf, ke=ke)
n_all = np.concatenate([EV[v]["n"] for v in FOLDS]); kf_all = np.concatenate([EV[v]["kf"] for v in FOLDS]); ke_all = np.concatenate([EV[v]["ke"] for v in FOLDS])
print("games per question with non-trivial evidence (|log e| > 0.05), mean by n: " + "; ".join(f"n={m}: fused {kf_all[n_all == m].mean():.2f}, encoder {ke_all[n_all == m].mean():.2f}" for m in SIZES))


def pick(D, folds, with_enc):
    P = np.concatenate([D[u]["P"] for u in folds]); Lf = np.concatenate([EV[u]["Lf"] for u in folds]); Le = np.concatenate([EV[u]["Le"] for u in folds])
    y = np.concatenate([EV[u]["y"] for u in folds]); n = np.concatenate([EV[u]["n"] for u in folds]); out = {}
    for m in SIZES:
        s = n == m; best = (-1.0, None)
        for T in GT:
            for U in (GU if with_enc else (0.0,)):
                val = balanced(sc(P[s], T * Lf[s] + U * Le[s], y[s]), y[s])
                if val > best[0] + 1e-12:
                    best = (val, (T, U))
        out[int(m)] = best[1]
    return out


def run(seeds):
    D = {v: {"P": np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)} for v in FOLDS}
    v0 = FOLDS[0]; ref = evaluate(post(D[v0]["P"], EV[v0]["Lf"]), EV[v0]["y"], EV[v0]["q"], sizes=SIZES)[1]
    print(f"seeds {seeds}: vectorised scorer vs aicup.qeval.evaluate on fold {v0}: max |diff| {np.abs(ref - sc(D[v0]['P'], EV[v0]['Lf'], EV[v0]['y'])).max():.3g}")
    S = {k: [] for k in ("base", "fused", "t05", "t05e", "xT", "xTU")}; picks = {}
    for v in FOLDS:
        P, e = D[v]["P"], EV[v]; y, n = e["y"], e["n"]; oth = [u for u in FOLDS if u != v]; pt, ptu = pick(D, oth, False), pick(D, oth, True); picks[int(v)] = {"T": pt, "TU": ptu}
        Tq = np.array([pt.get(int(m), (1.0, 0.0))[0] for m in n])[:, None]; T2 = np.array([ptu.get(int(m), (1.0, 0.0))[0] for m in n])[:, None]; U2 = np.array([ptu.get(int(m), (1.0, 0.0))[1] for m in n])[:, None]
        S["base"].append(sc(P, np.zeros_like(P), y)); S["fused"].append(sc(P, e["Lf"], y)); S["t05"].append(sc(P, 0.5 * e["Lf"], y)); S["t05e"].append(sc(P, 0.5 * e["Lf"] + 0.5 * e["Le"], y))
        S["xT"].append(sc(P, Tq * e["Lf"], y)); S["xTU"].append(sc(P, T2 * e["Lf"] + U2 * e["Le"], y))
        print(f"  fold {v}: picked T " + " ".join(f"n={m}: {pt[m][0]}" for m in pt) + " | (T, U) " + " ".join(f"n={m}: {ptu[m]}" for m in ptu))
    ys = [EV[v]["y"] for v in FOLDS]; ns = [EV[v]["n"] for v in FOLDS]; y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate([EV[v]["pid"] for v in FOLDS]); out = {"picks": picks}

    def cmp(a, b, label):
        dd = paired_diff(np.concatenate(a), np.concatenate(b), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w_, yy, nn, sizes=SIZES) for x, w_, yy, nn in zip(a, b, ys, ns)]
        ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
        print(f"  {label:46s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")
        return {**dd, "per_fold": pf, "pass": bool(ok)}
    out["fused_vs_base"] = cmp(S["fused"], S["base"], "fused anchors (T = 1) vs base"); out["xT_vs_base"] = cmp(S["xT"], S["base"], "cross-fitted T vs base"); out["xTU_vs_base"] = cmp(S["xTU"], S["base"], "cross-fitted (T, U) vs base")
    out["A"] = cmp(S["xT"], S["fused"], "A: CROSS-FITTED T vs T = 1 (primary)"); out["B"] = cmp(S["xTU"], S["xT"], "B: CROSS-FITTED (T, U) vs CROSS-FITTED T (primary)")
    out["t05_vs_fused"] = cmp(S["t05"], S["fused"], "fixed T = 0.5 vs T = 1 (report)"); out["t05e_vs_t05"] = cmp(S["t05e"], S["t05"], "fixed 0.5 / 0.5 with encoder vs fixed T = 0.5 (report)")
    P = np.concatenate([D[v]["P"] for v in FOLDS]); Lf = np.concatenate([EV[v]["Lf"] for v in FOLDS]); Le = np.concatenate([EV[v]["Le"] for v in FOLDS]); grid = np.round(np.arange(0.1, 1.51, 0.1), 2); slope = {}
    for m in SIZES:
        s = n == m; nll = [balanced(-np.log(np.maximum(post(P[s], T * Lf[s])[np.arange(s.sum()), y[s]], 1e-12)), y[s]) for T in grid]; tb = float(grid[int(np.argmin(nll))])
        nl2 = [balanced(-np.log(np.maximum(post(P[s], tb * Lf[s] + U * Le[s])[np.arange(s.sum()), y[s]], 1e-12)), y[s]) for U in np.r_[0.0, grid]]; ub = float(np.r_[0.0, grid][int(np.argmin(nl2))])
        slope[int(m)] = {"T_nll": tb, "nll_at_T": float(min(nll)), "nll_at_1": float(nll[int(np.flatnonzero(grid == 1.0)[0])]), "U_nll_given_T": ub, "nll_at_TU": float(min(nl2))}
    print("  NLL-optimal temperature by n (report): " + "; ".join(f"n={m}: T {r['T_nll']} (NLL {r['nll_at_1']:.4f} at T=1 -> {r['nll_at_T']:.4f}), then U {r['U_nll_given_T']} ({r['nll_at_TU']:.4f})" for m, r in slope.items()))
    out["nll_slope"] = slope; out["final_T"] = pick(D, list(FOLDS), False); out["final_TU"] = pick(D, list(FOLDS), True)
    print(f"  fitted on all four folds: T {out['final_T']} | (T, U) {out['final_TU']}")
    return out


rep = {f"seeds_{s[0]}": run(s) for s in ((0, 1, 2), (6, 7, 8))}
A = all(r["A"]["pass"] for r in rep.values()); B = all(r["B"]["pass"] for r in rep.values()); rep["A_pass"] = bool(A); rep["B_pass"] = bool(B)
print(f"CONFIRMATORY (pre-registered): A tempering adopted: {'YES' if A else 'NO'}; B encoder on top of tempering adopted: {'YES' if B else 'NO'}")
Path(ROOT / "results/r_variants/m80_temper.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
