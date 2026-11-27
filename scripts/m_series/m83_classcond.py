"""Section 4M M83 (2026-11-11): class-conditional anchor evidence. Today e(c) = (1 - a_b) + a_b min(K(l - c) / P0(l - c), 30) with K, P0
functions of the difference only. Here one row per target class c: P0_c(l) from the low bands (<= q80) of the queries whose truth is c,
K_c(l) from the top bands (>= q98) minus (1 - a_top) P0_c, both shrunk with KAPPA pseudo-counts towards the pooled tables restricted to
the 10 legal labels and renormalised. Bands, a_b, groups and the cross-fit (fold v calibrated on the other folds) are unchanged.
Identifier: five components + matcher (m82_val.npz, f5e_). Primary: class-conditional vs pooled evidence, M37, seeds 0-2 and 6-8.
Report only: truncation-only (kappa = infinity); gain by true class.
  python -I scripts/m_series/m83_classcond.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

KAPPA = 50.0
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)


def tables(z, tr, cal, band, g, kappa):
    """{group: R [10 target classes, 10 labels]} = min(K_c(l) / P0_c(l), 30); kappa None = the pooled tables, truncated and renormalised."""
    y, lab, out = z["y"], z["opp_lab"], {}
    for k in (0, 1, 2):
        s = tr & (g == k); low, top = s & (band <= 1), s & (band >= 5); a_top, K, P0 = cal[k]["a_top"], cal[k]["K"], cal[k]["P0"]; R = np.zeros((10, 10))
        for c in range(10):
            pk, p0 = K[np.arange(10) - c + 9], P0[np.arange(10) - c + 9]; pk, p0 = pk / pk.sum(), p0 / p0.sum()
            if kappa is None:
                Kc, P0c = pk, p0
            else:
                n0 = np.bincount(lab[low & (y == c)], minlength=10).astype(float); P0c = (n0 + kappa * p0) / (n0.sum() + kappa)
                nt = np.bincount(lab[top & (y == c)], minlength=10).astype(float); Ht = (nt + kappa * ((1 - a_top) * p0 + a_top * pk)) / (nt.sum() + kappa)
                Kc = np.clip(Ht - (1 - a_top) * P0c, 0, None); Kc = Kc / Kc.sum()
            R[c] = np.minimum(Kc / P0c, 30.0)
        out[k] = R
    return out


def evidence(z, v, kappa="pooled"):
    """{game_key: log e [10]} of fold v, calibrated on the other folds. kappa "pooled" = today's evidence."""
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))
    R = None if kappa == "pooled" else tables(z, tr, cal, band_all, g_all, kappa)
    for k in (0, 1, 2):
        s = g == k; a = cal[k]["a"][band[s]][:, None]
        ratio = cal[k]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9] if R is None else R[k][:, lab[s]].T
        out[s] = np.log((1 - a) + a * ratio)
    return dict(zip(z["key"][te], out)), (R, cal)


def main():
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}
    VAR = {"pooled": "pooled", "truncated": None, "class": KAPPA}; EV = {name: {v: evidence(z, v, kap) for v in FOLDS} for name, kap in VAR.items()}
    R, cal = EV["class"][FOLDS[0]][1]; Rp = EV["truncated"][FOLDS[0]][1][0]
    print("HD0 evidence ratio K_c(l) / P0_c(l), rows = target class, columns = matched label (fold 1's calibration); class-conditional | pooled, truncated:")
    for c in range(10):
        print(f"  {LEVELS[c]:>2s} " + " ".join(f"{x:5.2f}" for x in R[0][c]) + "   | " + " ".join(f"{x:5.2f}" for x in Rp[0][c]))
    rep = {}
    for seeds in ((0, 1, 2), (6, 7, 8)):
        S = {name: [] for name in (*VAR, "base")}; ys, ns, pids = [], [], []
        for v in FOLDS:
            y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
            S["base"].append(SM[(P @ SM).argmax(1), y]); ys.append(np.asarray(y)); ns.append(np.asarray(n)); pids.append(np.asarray(pid))
            for name in VAR:
                ev = EV[name][v][0]; L = np.zeros_like(P)
                for i, gk in enumerate(q["game_keys"]):
                    for k in gk:
                        if k in ev:
                            L[i] += ev[k]
                Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); S[name].append(SM[(Pf @ SM).argmax(1), y])
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); res = {}

        def cmp(a, b, label):
            dd = paired_diff(np.concatenate(S[a]), np.concatenate(S[b]), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w, yy, nn, sizes=SIZES) for x, w, yy, nn in zip(S[a], S[b], ys, ns)]
            ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
            print(f"  {label:44s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")
            return {**dd, "per_fold": pf, "pass": bool(ok)}
        print(f"seeds {seeds}:")
        for name in VAR:
            res[f"{name}_vs_base"] = cmp(name, "base", f"{name} evidence vs base")
        res["primary"] = cmp("class", "pooled", "CLASS-CONDITIONAL vs POOLED (primary)"); res["truncated_vs_pooled"] = cmp("truncated", "pooled", "truncated vs pooled (report)"); res["class_vs_truncated"] = cmp("class", "truncated", "class-conditional vs truncated (report)")
        for name in VAR:
            gain = np.concatenate(S[name]) - np.concatenate(S["base"]); print(f"  gain by true class, {name:9s}: " + " ".join(f"{LEVELS[c]} {np.mean([gain[(y == c) & (n == m)].mean() for m in SIZES]):+.4f}" for c in range(10)))
        rep[f"seeds_{seeds[0]}"] = res
    ok = all(r["primary"]["pass"] for r in rep.values()); rep["adopt"] = bool(ok); print(f"PRE-REGISTERED: class-conditional evidence replaces the pooled one: {'YES' if ok else 'NO'}")
    Path(ROOT / "results/r_variants/m83_classcond.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
