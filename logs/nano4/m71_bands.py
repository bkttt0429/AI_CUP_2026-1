"""Section 4M M71 correction (2026-11-10): which margin bands carry the CV gain? Test-like density (0.74, weights x0.74), anchors used only
in bands >= k. Pre-registered choice: the largest k whose gain keeps >= 80% of the all-bands gain; it must pass the M37 rule on both seed sets.
  python -I logs/nano4/m71_bands.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import m71_cv_density as cd  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402


def gain(z_cal, z_eval, scale, seeds, kmin):
    S_new, S_old, pf, ys, ns, pids, changed = [], [], [], [], [], [], 0
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        edges, cal, _, _ = s2.fit(z_cal, z_cal["fold"] != v)
        te = np.flatnonzero(z_eval["fold"] == v); tb = z_eval["target_black"][te]
        band = np.where(tb, np.searchsorted(edges[False], z_eval["opp_m"][te]), np.searchsorted(edges[True], z_eval["opp_m"][te]))
        g = s2.group_of(z_eval["hd"][te], tb); lab = z_eval["opp_lab"][te]; logE = np.zeros((len(te), 10))
        for k in (0, 1, 2):
            s = g == k; a = scale * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= kmin); dd = lab[s][:, None] - np.arange(10)[None, :]
            logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])
        key_row = {k: i for i, k in enumerate(z_eval["key"][te])}; L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)
        so, sn = evaluate(P, y, q, sizes=SIZES)[1], evaluate(Pf, y, q, sizes=SIZES)[1]; changed += int((np.abs(sn - so) > 1e-12).sum())
        S_new.append(sn); S_old.append(so); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(sn, y, n, sizes=SIZES) - balanced_mean_over_n(so, y, n, sizes=SIZES))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)
    d = paired_diff(np.concatenate(S_new), np.concatenate(S_old), y, pid, n, sizes=SIZES, n=3000)
    return d, pf, changed, len(y)


def main():
    z = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); rep = {}; names = ["<=q50", "q50-80", "q80-90", "q90-95", "q95-98", "q98-99", "q99-99.5", "q99.5-99.9", ">q99.9"]
    res = {}
    for kmin in (2, 3, 4, 5, 6, 7):
        out = []
        for seeds in ((0, 1, 2), (6, 7, 8)):
            ds = [gain(z, cd.thin(z, 0.74, seed), 0.74, seeds, kmin) for seed in (1, 2, 3)]
            diff = float(np.mean([d["diff"] for d, _, _, _ in ds])); lo = float(np.mean([d["ci95"][0] for d, _, _, _ in ds])); pf = np.mean([p for _, p, _, _ in ds], axis=0)
            ch = float(np.mean([c for _, _, c, _ in ds])); tot = ds[0][3]
            cond = (lo > 0) and (sum(e > 0 for e in pf) >= 3) and (diff >= 0.0025); out.append({"diff": diff, "ci_lo": lo, "per_fold": pf.tolist(), "changed_share": ch / tot, "pass": bool(cond)})
        res[kmin] = out
        print(f"bands >= {kmin} ({names[kmin]:>10s} and up): seeds 0-2 {out[0]['diff']:+.4f} (CI lower {out[0]['ci_lo']:+.4f}, folds " + " ".join(f"{e:+.4f}" for e in out[0]["per_fold"]) + f", {'PASS' if out[0]['pass'] else 'no'})"
              f" | seeds 6-8 {out[1]['diff']:+.4f} (CI lower {out[1]['ci_lo']:+.4f}, {'PASS' if out[1]['pass'] else 'no'}) | questions with a changed score {out[0]['changed_share']:.3f}", flush=True)
    full = np.mean([res[2][0]["diff"], res[2][1]["diff"]])
    ok = [k for k in res if np.mean([res[k][0]["diff"], res[k][1]["diff"]]) >= 0.8 * full and res[k][0]["pass"] and res[k][1]["pass"]]
    choice = max(ok) if ok else None
    print(f"all-bands gain (mean of the two seed sets) {full:+.4f}; pre-registered choice: the largest k keeping >= 80% of it and passing M37 on both -> k = {choice}")
    rep = {"by_kmin": res, "full": float(full), "choice": choice}
    Path(ROOT / "results/r_variants/m71_bands.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
