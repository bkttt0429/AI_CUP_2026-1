"""Section 4M M73 evaluation (2026-11-10): the anchor evidence built from the FUSED single-game identifier against the one built from the
opening component alone (both from m73_val.npz: same pools, same queries), through M71 step 2's cross-fitted mixture model.
Pre-registered: fused vs opening, paired; M37 rule on seeds 0-2 and 6-8. Report only: each against the base ensemble; band shares and agreement.
  python -I logs/nano4/m73_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

d = dict(np.load(ROOT / "results/r_variants/m73_val.npz"))
base = {k: d[k] for k in ("key", "fold", "y", "hd", "target_black")}
Z = {"fused": {**base, "opp_lab": d["f_opp_lab"], "opp_m": d["f_opp_m"], "imp_m": d["f_imp_m"], "imp_lab": d["f_imp_lab"]},
     "opening": {**base, "opp_lab": d["o_opp_lab"], "opp_m": d["o_opp_m"], "imp_m": d["o_imp_m"], "imp_lab": d["o_imp_lab"]}}
rep = {}
for name, z in Z.items():
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"]))
    hi, ihi = s & (band >= 6), s & (ib >= 6)
    rep[name] = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact_above_q99": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1_above_q99": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),
                 "impostor_exact_above_q99": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "a_by_band_hd0": cal[0]["a"].tolist(), "K_hd0": cal[0]["K"][7:12].tolist()}
    r = rep[name]
    print(f"{name:8s} HD0: opponent queries above the impostor q99 {r['hd0_share_above_q99']:.4f} (impostor 0.0100); exact / within-one agreement there {r['hd0_exact_above_q99']:.3f} / {r['hd0_within1_above_q99']:.3f} "
          f"(impostor exact {r['impostor_exact_above_q99']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]) + " | K(-2..2) " + " ".join(f"{x:.2f}" for x in cal[0]["K"][7:12]))


def scores_for(z, seeds):
    out = []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        te, logE, _ = s2.evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}; L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], evaluate(P, y, q, sizes=SIZES)[1], y, n, pid))
    return out


def compare(a, b, label):
    y, n, pid = (np.concatenate([x[k] for x in a]) for k in (2, 3, 4))
    dd = paired_diff(np.concatenate([x[0] for x in a]), np.concatenate([x[0] if b is not None else x[1] for x in (b if b is not None else a)]), y, pid, n, sizes=SIZES, n=5000)
    pf = [balanced_mean_over_n(x[0], x[2], x[3], sizes=SIZES) - balanced_mean_over_n((w[0] if b is not None else x[1]), x[2], x[3], sizes=SIZES) for x, w in zip(a, b if b is not None else a)]
    cond = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
    print(f"  {label:26s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if cond else 'no'}")
    return {**dd, "per_fold": pf, "pass": bool(cond)}


ok = []
for seeds in ((0, 1, 2), (6, 7, 8)):
    print(f"seeds {seeds}:"); F, O = scores_for(Z["fused"], seeds), scores_for(Z["opening"], seeds)
    r = {"fused_vs_base": compare(F, None, "fused anchors vs base"), "opening_vs_base": compare(O, None, "opening anchors vs base"), "fused_vs_opening": compare(F, O, "FUSED vs OPENING (primary)")}
    rep[f"seeds_{seeds[0]}"] = r; ok.append(r["fused_vs_opening"]["pass"])
rep["adopt_fused"] = bool(all(ok)); print(f"PRE-REGISTERED: replace the opening identifier by the fused one: {'YES' if all(ok) else 'NO'}")
Path(ROOT / "results/r_variants/m73_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
