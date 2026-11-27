"""Section 4M M71 (2026-11-10): the CV gain at the TEST's anchor density. In the test-regime validation table (m71_val_regime.npz) a random
share 1 - rho of the opponent queries is replaced by an impostor draw (the target-side label and margin of a random other query scored
on the same model colour, same fold), so real anchors are thinned to rho of their validation density; the evidence uses
a_b = scale x the cross-fitted validation a_b. Report only (no selection): scale = rho is the deployed setting.
  python -I scripts/m_series/m71_cv_density.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402


def thin(z, rho, seed):
    rng = np.random.default_rng(seed); out = {k: v.copy() for k, v in z.items()}
    drop = rng.random(len(z["y"])) > rho
    for f in FOLDS:
        for model_black in (True, False):
            src = np.flatnonzero((z["fold"] == f) & (z["target_black"] == model_black))  # impostors scored on this model colour
            dst = np.flatnonzero(drop & (z["fold"] == f) & (z["target_black"] != model_black))  # opponent queries scored on this model colour
            pick = rng.choice(src, len(dst))
            out["opp_lab"][dst], out["opp_m"][dst] = z["imp_lab"][pick], z["imp_m"][pick]
    return out


def gain(z_cal, z_eval, scale, seeds):
    S_new, S_old, pf, ys, ns, pids = [], [], [], [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        edges, cal, _, _ = s2.fit(z_cal, z_cal["fold"] != v)
        te = np.flatnonzero(z_eval["fold"] == v); tb = z_eval["target_black"][te]
        band = np.where(tb, np.searchsorted(edges[False], z_eval["opp_m"][te]), np.searchsorted(edges[True], z_eval["opp_m"][te]))
        g = s2.group_of(z_eval["hd"][te], tb); lab = z_eval["opp_lab"][te]; logE = np.zeros((len(te), 10))
        for k in (0, 1, 2):
            s = g == k; a = scale * cal[k]["a"]; dd = lab[s][:, None] - np.arange(10)[None, :]
            logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])
        key_row = {k: i for i, k in enumerate(z_eval["key"][te])}; L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)
        so, sn = evaluate(P, y, q, sizes=SIZES)[1], evaluate(Pf, y, q, sizes=SIZES)[1]
        S_new.append(sn); S_old.append(so); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(sn, y, n, sizes=SIZES) - balanced_mean_over_n(so, y, n, sizes=SIZES))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
    d = paired_diff(np.concatenate(S_new), np.concatenate(S_old), y, pid, n, sizes=SIZES, n=3000)
    return d, pf


def main():
    z = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); rep = {}
    for rho, scale, seeds, name in ((1.0, 1.0, (0, 1, 2), "full density, a x1.00 (CV as measured)"), (1.0, 0.74, (0, 1, 2), "full density, a x0.74 (under-weighted)"),
                                    (0.74, 0.74, (0, 1, 2), "TEST-LIKE density 0.74, a x0.74, seeds 0-2"), (0.74, 0.74, (6, 7, 8), "TEST-LIKE density 0.74, a x0.74, seeds 6-8"),
                                    (0.74, 1.0, (0, 1, 2), "density 0.74, a x1.00 (over-weighted)"), (0.5, 0.74, (0, 1, 2), "density 0.50, a x0.74 (if the density were lower still)")):
        ds = []
        for seed in (1, 2, 3) if rho < 1 else (1,):
            d, pf = gain(z, thin(z, rho, seed) if rho < 1 else z, scale, seeds); ds.append((d, pf))
        diff = float(np.mean([d["diff"] for d, _ in ds])); lo = float(np.mean([d["ci95"][0] for d, _ in ds])); hi = float(np.mean([d["ci95"][1] for d, _ in ds])); pf = np.mean([p for _, p in ds], axis=0)
        by_n = {m: float(np.mean([d["by_n"][m] for d, _ in ds])) for m in ds[0][0]["by_n"]}
        rep[name] = {"diff": diff, "ci95": [lo, hi], "per_fold": pf.tolist(), "by_n": by_n}
        print(f"{name:52s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in by_n.items()))
    Path(ROOT / "results/r_variants/m71_cv_density.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
