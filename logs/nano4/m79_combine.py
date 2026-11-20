"""Section 4M M79 (2026-11-10): do the whole-game encoder's anchors add to the five-component identifier's anchors?
Per game each identifier gives (matched label, margin band) -> cross-fitted mixture weight a and log-evidence (M71 step 2). Combination,
fixed in advance: per game take the evidence of the identifier with the LARGER weight a ("max"). Report only: average, sum.
Confirmatory test: combined vs the five-component anchors alone, M37 rule on seeds 0-2 and 6-8.
  python -I logs/nano4/m79_combine.py [--enc-dir results/m77]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
import m71_step2 as s2  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--enc-dir", default="results/m77"); ap.add_argument("--out", default="results/r_variants/m79_combine.json"); a_ = ap.parse_args()
d = dict(np.load(ROOT / "results/r_variants/m73_val.npz"))
F = {**{k: d[k] for k in ("key", "fold", "y", "hd", "target_black")}, "opp_lab": d["f_opp_lab"], "opp_m": d["f_opp_m"], "imp_m": d["f_imp_m"]}
parts = [dict(np.load(ROOT / a_.enc_dir / f"val{v}.npz")) for v in FOLDS]; E = {k: np.concatenate([p[k] for p in parts]) for k in ("key", "fold", "y", "hd", "target_black", "opp_lab", "opp_m", "imp_m")}


def per_query(z, v):
    """{game_key: (log-evidence [10], weight a)} of fold v, calibrated on the other folds."""
    edges, cal, _, _ = s2.fit(z, z["fold"] != v); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]
    logE = np.zeros((len(te), 10)); w = np.zeros(len(te))
    for k in (0, 1, 2):
        s = g == k; a = cal[k]["a"]; dd = lab[s][:, None] - np.arange(10)[None, :]
        logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9]); w[s] = a[band[s]]
    return dict(zip(z["key"][te], zip(logE, w)))


def run(seeds):
    S = {k: [] for k in ("fused", "encoder", "max", "mean", "sum")}; base, ys, ns, pids = [], [], [], []; agree = []
    for v in FOLDS:
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        qf, qe = per_query(F, v), per_query(E, v); L = {k: np.zeros_like(P) for k in S}
        for i, gk in enumerate(q["game_keys"]):
            for k in gk:
                lf, wf = qf.get(k, (0.0, 0.0)); le, we = qe.get(k, (0.0, 0.0))
                L["fused"][i] += lf; L["encoder"][i] += le; L["max"][i] += le if we > wf else lf; L["mean"][i] += 0.5 * (lf + le); L["sum"][i] += lf + le
        for k in S:
            Pf = P * np.exp(L[k] - L[k].max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True); S[k].append(evaluate(Pf, y, q, sizes=SIZES)[1])
        base.append(evaluate(P, y, q, sizes=SIZES)[1]); ys.append(y); ns.append(n); pids.append(pid)
        both = [(qf[k][1], qe[k][1]) for k in qf if k in qe]; agree.append(np.mean([(wf >= 0.2) and (we >= 0.2) for wf, we in both]) / max(np.mean([(wf >= 0.2) or (we >= 0.2) for wf, we in both]), 1e-9))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); out = {}

    def cmp(a, b, label):
        dd = paired_diff(np.concatenate(a), np.concatenate(b), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w_, yy, nn, sizes=SIZES) for x, w_, yy, nn in zip(a, b, ys, ns)]
        ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
        print(f"  {label:40s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")
        return {**dd, "per_fold": pf, "pass": bool(ok)}
    print(f"seeds {seeds}: share of anchored games (weight >= 0.2 in either) where BOTH identifiers fire: {np.mean(agree):.3f}")
    for k in ("fused", "encoder", "max", "mean", "sum"):
        out[f"{k}_vs_base"] = cmp(S[k], base, f"{k} anchors vs base")
    out["primary"] = cmp(S["max"], S["fused"], "MAX-WEIGHT COMBINATION vs FUSED (primary)"); out["mean_vs_fused"] = cmp(S["mean"], S["fused"], "average vs fused (report)"); out["sum_vs_fused"] = cmp(S["sum"], S["fused"], "sum vs fused (report)")
    return out


rep = {f"seeds_{s[0]}": run(s) for s in ((0, 1, 2), (6, 7, 8))}
ok = all(r["primary"]["pass"] for r in rep.values()); near = all(r["primary"]["diff"] >= 0.0015 and r["primary"]["ci95"][0] > 0 for r in rep.values())
rep["adopt"] = bool(ok); rep["extend_training"] = bool(ok or near)
print(f"CONFIRMATORY (pre-registered): combined anchors replace the five-component anchors: {'YES' if ok else 'NO'}; extend the encoder's training (M77b): {'YES' if (ok or near) else 'NO'}")
Path(ROOT / a_.out).write_text(json.dumps(rep, indent=1, default=float)); print("written")
