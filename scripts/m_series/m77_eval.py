"""Section 4M M77 evaluation (2026-11-10): anchors from the whole-game identity encoder (results/m77/val1-4.npz, cross-fitted: the model
for fold v never saw fold v's players' games) against the opening-model anchors (m71_queries.npz; same regime, same queries), through
M71 step 2's cross-fitted mixture model. Stage 1: held-out identification of the all-rows model (>= 0.20 top-1). Stage 2 (adoption):
encoder vs opening anchors, paired, M37 rule on seeds 0-2 and 6-8.
  python -I logs/nano4/m77_eval.py
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

M = ROOT / "results/m77"; rep = {}
for v in (1, 2, 3, 4, -1):
    f = M / f"heldout_{v}.json"
    if f.exists():
        h = json.load(open(f)); best = max(h["history"], key=lambda e: e["top1"]); rep[f"heldout_{v}"] = best
        print(f"identification, model excluding fold {v}: {h['classes']} classes, held-out own games top-1 {best['top1']:.3f} top-5 {best['top5']:.3f} top-20 {best['top20']:.3f} (epoch {best['epoch']})")
if "heldout_-1" in rep:
    t1 = rep["heldout_-1"]["top1"]; print(f"STAGE 1 (all-rows model, 8,007 candidates; opening 0.091, five-component fusion 0.125): top-1 {t1:.3f} -> {'PASS (>= 0.20)' if t1 >= 0.20 else ('fuse with the five components' if t1 >= 0.125 else 'below the fusion: close')}")
parts = [dict(np.load(M / f"val{v}.npz")) for v in FOLDS]
enc = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}
o = dict(np.load(ROOT / "results/r_variants/m71_queries.npz")); opening = {k: o[k] for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_lab", "imp_m")}
Z = {"encoder": enc, "opening": opening}
for name, z in Z.items():
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"])); hi, ihi = s & (band >= 6), s & (ib >= 6)
    rep[name] = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),
                 "impostor_exact": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "share_above_q90": float((s & (band >= 3)).sum() / s.sum())}
    r = rep[name]; print(f"{name:8s} HD0 opponent queries above the impostor q99: {r['hd0_share_above_q99']:.4f} (above q90: {r['share_above_q90']:.3f}; impostor 0.0100 / 0.100); label agreement there exact {r['hd0_exact']:.3f} within one {r['hd0_within1']:.3f} "
                     f"(impostor exact {r['impostor_exact']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]) + " | K(-1,0,+1) " + " ".join(f"{x:.2f}" for x in cal[0]["K"][8:11]))


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
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], evaluate(P, y, q, sizes=SIZES)[1], y, n, pid, L))
    return out


def compare(a, b, label):
    y, n, pid = (np.concatenate([x[k] for x in a]) for k in (2, 3, 4)); new = np.concatenate([x[0] for x in a]); old = np.concatenate([(w[0] if b is not None else x[1]) for x, w in zip(a, b if b is not None else a)])
    dd = paired_diff(new, old, y, pid, n, sizes=SIZES, n=5000)
    pf = [balanced_mean_over_n(x[0], x[2], x[3], sizes=SIZES) - balanced_mean_over_n((w[0] if b is not None else x[1]), x[2], x[3], sizes=SIZES) for x, w in zip(a, b if b is not None else a)]
    ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
    print(f"  {label:34s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")
    return {**dd, "per_fold": pf, "pass": bool(ok)}


def both(E, O, seeds):
    """Evidence of both identifiers multiplied (report only: are they complementary?)."""
    out = []
    for v, e, o_ in zip(FOLDS, E, O):
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        L = 0.5 * (e[5] + o_[5]); Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)
        out.append((evaluate(Pf, y, q, sizes=SIZES)[1], e[1], y, n, pid, L))
    return out


ok = []
for seeds in ((0, 1, 2), (6, 7, 8)):
    print(f"seeds {seeds}:"); E, O = scores_for(Z["encoder"], seeds), scores_for(Z["opening"], seeds)
    r = {"encoder_vs_base": compare(E, None, "encoder anchors vs base"), "opening_vs_base": compare(O, None, "opening anchors vs base"), "encoder_vs_opening": compare(E, O, "ENCODER vs OPENING (adoption)"),
         "mean_of_both_vs_opening": compare(both(E, O, seeds), O, "average of both evidences vs opening")}
    rep[f"seeds_{seeds[0]}"] = r; ok.append(r["encoder_vs_opening"]["pass"])
rep["adopt_encoder"] = bool(all(ok)); print(f"STAGE 2 (pre-registered): the encoder replaces the opening model as the anchor source: {'YES' if all(ok) else 'NO'}")
Path(M / "m77_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
