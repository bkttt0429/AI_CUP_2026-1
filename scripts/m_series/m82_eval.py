"""Section 4M M82 evaluation (2026-11-10): merge the shards of m82_cond_id.py and compare the anchors of the four fusions (five, six =
+ tactic, 5e = five + matcher, seven) through M71 step 2's cross-fitted mixture model. Pre-registered primary: seven vs six when M81
adopted the six-component identifier (results/r_variants/m81_eval.json), else five + matcher vs five; M37 rule on seeds 0-2 and 6-8.
Sanity (pre-registered): the five-component anchors against the base must land in +0.010 .. +0.016.
  python -I logs/nano4/m82_eval.py
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

OUT = ROOT / "results/m82"; FUS = ("5", "6", "5e", "7")
meta = np.load(OUT / "meta.npz"); players, plab = meta["players"], meta["plab"]; Q, H, T = int(meta["Q"]), int(meta["H"]), int(meta["T"])
parts = [np.load(p) for p in sorted(OUT.glob("part*of*.npz"))]; idx = np.concatenate([p["idx"] for p in parts]); order = np.argsort(idx)
assert len(idx) == 2 * Q + 2 * T and np.array_equal(idx[order], np.concatenate([np.arange(2 * Q), np.arange(2 * Q + H, 2 * Q + H + 2 * T)])), "shards do not cover the queries"
col = {k: np.concatenate([p[k] for p in parts])[order] for k in parts[0].files if k != "idx"}
o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T)
d73 = dict(np.load(ROOT / "results/r_variants/m73_val.npz")); t73 = dict(np.load(ROOT / "results/r_variants/m73_test.npz")); base = {k: d73[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
assert len(base["key"]) == Q and len(t73["qid"]) == T
val, tst = dict(base), dict(qid=t73["qid"], hd=t73["hd"], target_black=t73["target_black"])
for f in FUS:
    b, m = col[f"b{f}"], col[f"m{f}"]
    val.update({f"f{f}_opp_lab": plab[b[o]], f"f{f}_opp_m": m[o], f"f{f}_opp_best": players[b[o]], f"f{f}_imp_lab": plab[b[t_]], f"f{f}_imp_m": m[t_]})
    tst.update({f"f{f}_opp_lab": plab[b[to]], f"f{f}_opp_m": m[to], f"f{f}_opp_best": players[b[to]], f"f{f}_tgt_lab": plab[b[tt]], f"f{f}_tgt_m": m[tt], f"f{f}_tgt_best": players[b[tt]]})
np.savez_compressed(ROOT / "results/r_variants/m82_val.npz", **val); np.savez_compressed(ROOT / "results/r_variants/m82_test.npz", **tst)
rep = {"report": json.loads((OUT / "report.json").read_text()), "embed": json.loads((OUT / "embed_report.json").read_text())}; r = rep["report"]; e = rep["embed"]
print("matcher alone, held-out own games (all-players network): top-1 / top-5 / top-20 " + " / ".join(f"{e['heldout_all_net'][k]:.3f}" for k in ("top1", "top5", "top20"))
      + " | players the network never saw vs the same players seen, top-1: " + "; ".join(f"fold {v}: {e[f'unseen_fold{v}']['unseen']['top1']:.3f} vs {e[f'unseen_fold{v}']['seen']['top1']:.3f}" for v in (1, 2, 3, 4)))
for name, w in r["weights"].items():
    print(f"weights ({name}): " + " ".join(f"{k} {v:.2f}" for k, v in w.items()))
print("held-out own games, top-1 / top-5 / top-20: " + "; ".join(f"{k} {v['top1']:.3f}/{v['top5']:.3f}/{v['top20']:.3f}" for k, v in r["heldout_accuracy"].items()))
print(f"same top candidate as M73's five-component run (different held-out games): validation opponent side {float((val['f5_opp_best'] == d73['f_opp_best']).mean()):.3f}")
Z = {f: {**base, "opp_lab": val[f"f{f}_opp_lab"], "opp_m": val[f"f{f}_opp_m"], "imp_m": val[f"f{f}_imp_m"], "imp_lab": val[f"f{f}_imp_lab"]} for f in FUS}
for name, z in Z.items():
    edges, cal, band, g = s2.fit(z, np.ones(len(z["y"]), bool)); s = g == 0
    ib = np.where(z["target_black"], np.searchsorted(edges[True], z["imp_m"]), np.searchsorted(edges[False], z["imp_m"]))
    hi, ihi = s & (band >= 6), s & (ib >= 6)
    rep[f"bands_{name}"] = q = {"hd0_share_above_q99": float(hi.sum() / s.sum()), "hd0_exact_above_q99": float((z["opp_lab"][hi] == z["y"][hi]).mean()), "hd0_within1_above_q99": float((np.abs(z["opp_lab"][hi] - z["y"][hi]) <= 1).mean()),
                                "impostor_exact_above_q99": float((z["imp_lab"][ihi] == z["y"][ihi]).mean()), "a_by_band_hd0": cal[0]["a"].tolist()}
    print(f"{name:3s} HD0: opponent queries above the impostor q99 {q['hd0_share_above_q99']:.4f}; exact / within-one agreement there {q['hd0_exact_above_q99']:.3f} / {q['hd0_within1_above_q99']:.3f} "
          f"(impostor exact {q['impostor_exact_above_q99']:.3f}); a by band " + " ".join(f"{x:.2f}" for x in cal[0]["a"]))


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
    cond = (dd["ci95"][0] > 0) and (sum(e_ > 0 for e_ in pf) >= 3) and (dd["diff"] >= 0.0025)
    print(f"  {label:34s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e_:+.4f}" for e_ in pf) + "  by n " + " ".join(f"{m}: {v_:+.4f}" for m, v_ in dd["by_n"].items()) + f"  -> {'PASS' if cond else 'no'}")
    return {**dd, "per_fold": pf, "pass": bool(cond)}


m81 = ROOT / "results/r_variants/m81_eval.json"; six = bool(json.loads(m81.read_text())["adopt_six"]) if m81.exists() else False
prim = ("7", "6") if six else ("5e", "5"); ok, sane = [], []
print(f"M81 adopted the six-component identifier: {'yes' if six else 'no'} -> primary = {prim[0]} vs {prim[1]}")
for seeds in ((0, 1, 2), (6, 7, 8)):
    print(f"seeds {seeds}:"); S = {f: scores_for(Z[f], seeds) for f in FUS}
    res = {f"{f}_vs_base": compare(S[f], None, f"{f} anchors vs base") for f in FUS}
    res["5e_vs_5"] = compare(S["5e"], S["5"], "five + matcher vs five" + (" (PRIMARY)" if not six else "")); res["7_vs_6"] = compare(S["7"], S["6"], "seven vs six" + (" (PRIMARY)" if six else "")); res["6_vs_5"] = compare(S["6"], S["5"], "six vs five (M81's contrast)")
    rep[f"seeds_{seeds[0]}"] = res; ok.append(res[f"{prim[0]}_vs_{prim[1]}"]["pass"]); sane.append(0.010 <= res["5_vs_base"]["diff"] <= 0.016)
rep["sanity_five"] = bool(all(sane)); rep["primary"] = f"{prim[0]}_vs_{prim[1]}"; rep["adopt_matcher"] = bool(all(ok) and all(sane))
print(f"sanity (five-component anchors vs base inside +0.010 .. +0.016 on both seed sets): {'OK' if all(sane) else 'FAILED'}")
print(f"PRE-REGISTERED: add the matcher to the identifier: {'YES' if rep['adopt_matcher'] else 'NO'}")
Path(ROOT / "results/r_variants/m82_eval.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
