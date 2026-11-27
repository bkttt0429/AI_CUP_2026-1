"""Section 4M M81 evaluation (2026-11-10): merge the shards of m81_tactic_id.py, check that the five-component outputs reproduce M73,
and compare the anchors built from the SIX-component identifier (five + tactical events) with the five-component ones through M71
step 2's cross-fitted mixture model. Pre-registered: six vs five, paired; M37 rule on seeds 0-2 and 6-8.
  python -I logs/nano4/m81_eval.py
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

OUT = ROOT / "results/m81"
meta = np.load(OUT / "meta.npz"); players, plab = meta["players"], meta["plab"]; Q, H, T = int(meta["Q"]), int(meta["H"]), int(meta["T"])
parts = [np.load(p) for p in sorted(OUT.glob("part*of*.npz"))]; idx = np.concatenate([p["idx"] for p in parts]); order = np.argsort(idx)
assert len(idx) == 2 * Q + 2 * T and np.array_equal(idx[order], np.concatenate([np.arange(2 * Q), np.arange(2 * Q + H, 2 * Q + H + 2 * T)])), "shards do not cover the queries"
col = {k: np.concatenate([p[k] for p in parts])[order] for k in ("b6", "m6", "b5", "m5")}
o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)
d73 = dict(np.load(ROOT / "results/r_variants/m73_val.npz")); t73 = dict(np.load(ROOT / "results/r_variants/m73_test.npz")); base = {k: d73[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
assert len(base["key"]) == Q and len(t73["qid"]) == T
val = dict(f_opp_lab=plab[col["b6"][o]], f_opp_m=col["m6"][o], f_opp_best=players[col["b6"][o]], f_imp_lab=plab[col["b6"][t_]], f_imp_m=col["m6"][t_],
           g_opp_lab=plab[col["b5"][o]], g_opp_m=col["m5"][o], g_opp_best=players[col["b5"][o]], g_imp_lab=plab[col["b5"][t_]], g_imp_m=col["m5"][t_])
np.savez_compressed(ROOT / "results/r_variants/m81_val.npz", **base, **val)
np.savez_compressed(ROOT / "results/r_variants/m81_test.npz", qid=t73["qid"], hd=t73["hd"], target_black=t73["target_black"],
                    f_opp_lab=plab[col["b6"][to]], f_opp_m=col["m6"][to], f_opp_best=players[col["b6"][to]], f_tgt_lab=plab[col["b6"][tt]], f_tgt_m=col["m6"][tt], f_tgt_best=players[col["b6"][tt]],
                    g_opp_lab=plab[col["b5"][to]], g_opp_m=col["m5"][to], g_opp_best=players[col["b5"][to]])
rep = {"report": json.loads((OUT / "report.json").read_text())}
same = float((val["g_opp_best"] == d73["f_opp_best"]).mean()); dm = float(np.abs(val["g_opp_m"] - d73["f_opp_m"]).max()); same_t = float((plab[col["b5"][to]] == t73["f_opp_lab"]).mean())
repro = same >= 0.999 and same_t >= 0.999; rep["reproduction"] = {"val_same_candidate": same, "val_max_margin_diff": dm, "test_same_label": same_t, "ok": bool(repro)}
print(f"five-component reproduction of M73: same top candidate on validation {same:.5f} (max margin difference {dm:.2g}), same label on test {same_t:.5f} -> {'OK' if repro else 'FAILED: results below are not to be trusted'}")
r = rep["report"]; print("weights (six): " + " ".join(f"{k} {v:.2f}" for k, v in r["weights6"].items()) + " | (five): " + " ".join(f"{k} {v:.2f}" for k, v in r["weights5"].items()) + f" | Dirichlet precision {r['dirichlet_precision']:.1f}")
print("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k, v in r["heldout_accuracy"].items()))
print(f"top candidate changed by the sixth component: validation opponent side {float((col['b6'][o] != col['b5'][o]).mean()):.3f}, test opponent side {float((col['b6'][to] != col['b5'][to]).mean()):.3f}")
Z = {"six": {**base, "opp_lab": val["f_opp_lab"], "opp_m": val["f_opp_m"], "imp_m": val["f_imp_m"], "imp_lab": val["f_imp_lab"]},
     "five": {**base, "opp_lab": val["g_opp_lab"], "opp_m": val["g_opp_m"], "imp_m": val["g_imp_m"], "imp_lab": val["g_imp_lab"]}}
for name, z in Z.items():
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"]))
    hi, ihi = s & (band >= 6), s & (ib >= 6)
    rep[name] = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact_above_q99": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1_above_q99": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),
                 "impostor_exact_above_q99": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "a_by_band_hd0": cal[0]["a"].tolist(), "K_hd0": cal[0]["K"][7:12].tolist()}
    q = rep[name]
    print(f"{name:5s} HD0: opponent queries above the impostor q99 {q['hd0_share_above_q99']:.4f} (impostor 0.0100); exact / within-one agreement there {q['hd0_exact_above_q99']:.3f} / {q['hd0_within1_above_q99']:.3f} "
          f"(impostor exact {q['impostor_exact_above_q99']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]) + " | K(-2..2) " + " ".join(f"{x:.2f}" for x in cal[0]["K"][7:12]))


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
    print(f"  {label:26s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val_:+.4f}" for m, val_ in dd["by_n"].items()) + f"  -> {'PASS' if cond else 'no'}")
    return {**dd, "per_fold": pf, "pass": bool(cond)}


ok = []
for seeds in ((0, 1, 2), (6, 7, 8)):
    print(f"seeds {seeds}:"); A, B = scores_for(Z["six"], seeds), scores_for(Z["five"], seeds)
    res = {"six_vs_base": compare(A, None, "six anchors vs base"), "five_vs_base": compare(B, None, "five anchors vs base"), "six_vs_five": compare(A, B, "SIX vs FIVE (primary)")}
    rep[f"seeds_{seeds[0]}"] = res; ok.append(res["six_vs_five"]["pass"])
rep["adopt_six"] = bool(all(ok) and repro)
print(f"PRE-REGISTERED: replace the five-component identifier by the six-component one: {'YES' if rep['adopt_six'] else 'NO'}" + ("" if repro else " (reproduction check failed)"))
Path(ROOT / "results/r_variants/m81_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
