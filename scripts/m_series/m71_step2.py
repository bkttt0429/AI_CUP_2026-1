"""Section 4M M71 step 2 (2026-11-10): fuse the opponent-anchor evidence into the ensemble posterior, cross-fitted over validation folds.
Reads results/r_variants/m71_queries.npz (step 1). For validation fold v the calibration uses the other three folds only:
band edges = impostor-margin quantiles (per model colour); groups = HD0, HD1 x target colour; P0(d) = label-difference distribution
(candidate label - target label) of the bands <= q80; a_b = clip(1 - impostor share / opponent share, 0, 0.98);
K(d) = the pooled bands > q98 minus (1 - a_top) P0, clipped at 0, normalised; per-game evidence e(c) = (1 - a_b) + a_b min(K(d)/P0(d), 30)
with d = candidate label - c. Question posterior = ensemble posterior x prod_g e_g(c); expected-score decision.
Base = standard 6 members (seeds 0-2); replication = standard seeds 6-8. Rule (M37): CI lower bound > 0, >= 3 of 4 folds > 0, diff >= 0.0025, in both.
  python -I logs/nano4/m71_step2.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

STD = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh")
QS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
D = np.arange(-9, 10)


def group_of(hd, tb):
    return np.where(hd == 0, 0, np.where(hd == 1, np.where(tb, 1, 2), -1))  # HD0; HD1 target Black; HD1 target White; -1 = no evidence


def fit(z, tr):
    """Calibration from the queries in `tr`: band edges per model colour, and per group P0, K, a_b."""
    edges = {True: np.quantile(z["imp_m"][tr & z["target_black"]], QS), False: np.quantile(z["imp_m"][tr & ~z["target_black"]], QS)}
    nominal = np.diff(np.concatenate([[0.0], QS, [1.0]]))
    band = np.where(z["target_black"], np.searchsorted(edges[False], z["opp_m"]), np.searchsorted(edges[True], z["opp_m"]))  # opponent plays the other colour
    g = group_of(z["hd"], z["target_black"]); d = z["opp_lab"] - z["y"]; cal = {}
    for k in (0, 1, 2):
        s = tr & (g == k); nb = len(QS) + 1
        share = np.array([(s & (band == b)).sum() for b in range(nb)]) / max(s.sum(), 1)
        a = np.clip(1 - nominal / np.maximum(share, 1e-9), 0, 0.98)
        hist = lambda m: (np.array([(d[m] == x).sum() for x in D]) + 1.0) / (m.sum() + len(D))  # noqa: E731
        P0 = hist(s & (band <= 1)); top = s & (band >= 5); a_top = float(np.clip(1 - nominal[5:].sum() / max(top.sum() / max(s.sum(), 1), 1e-9), 0, 0.98))
        K = np.clip(hist(top) - (1 - a_top) * P0, 0, None); K = K / K.sum()
        cal[k] = {"a": a, "ratio": np.minimum(K / P0, 30.0), "share": share, "a_top": a_top, "K": K, "P0": P0}
    return edges, cal, band, g


def evidence_per_query(z, v):
    """log e_g(c) [Q, 10] for the queries of fold v, calibrated on the other folds."""
    tr = z["fold"] != v; edges, cal, _, _ = fit(z, tr)
    te = np.flatnonzero(z["fold"] == v); out = np.zeros((len(te), 10))
    band = np.where(z["target_black"][te], np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te]))
    g = group_of(z["hd"][te], z["target_black"][te]); lab = z["opp_lab"][te]
    for k in (0, 1, 2):
        s = g == k
        dd = lab[s][:, None] - np.arange(10)[None, :]  # candidate label - c
        out[s] = np.log((1 - cal[k]["a"][band[s]])[:, None] + cal[k]["a"][band[s]][:, None] * cal[k]["ratio"][dd + 9])
    return te, out, cal


def run(name, seeds, z, rep):
    S_new, S_old, pf, ys, ns, pids, changed, total = [], [], [], [], [], [], 0, 0
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in seeds], axis=0)
        te, logE, cal = evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}
        L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf = Pf / Pf.sum(axis=1, keepdims=True)
        so, sn = evaluate(P, y, q, sizes=SIZES)[1], evaluate(Pf, y, q, sizes=SIZES)[1]
        changed += int((np.abs(sn - so) > 1e-12).sum()); total += len(y)
        S_new.append(sn); S_old.append(so); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(sn, y, n, sizes=SIZES) - balanced_mean_over_n(so, y, n, sizes=SIZES))
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); sn, so = np.concatenate(S_new), np.concatenate(S_old)
    d = paired_diff(sn, so, y, pid, n, sizes=SIZES, n=5000)
    cond = {"1_ci_lower_gt_0": bool(d["ci95"][0] > 0), "2_folds_3of4": bool(sum(e > 0 for e in pf) >= 3), "3_diff_ge_0.0025": bool(d["diff"] >= 0.0025)}
    rep[name] = {**d, "per_fold": pf, "base": float(balanced_mean_over_n(so, y, n, sizes=SIZES)), "questions_with_changed_score": changed, "questions": total, "conditions": cond, "pass": all(cond.values())}
    print(f"{name:22s} base {rep[name]['base']:.4f}  diff {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
          + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  score changed on {changed}/{total} questions -> {'PASS' if all(cond.values()) else 'no'}")
    return all(cond.values())


def main():
    z = dict(np.load(ROOT / "results/r_variants/m71_queries.npz", allow_pickle=False)); rep = {}
    _, cal, _, _ = fit(z, np.ones(len(z["y"]), bool))
    for k, nm in ((0, "HD0"), (1, "HD1 target Black"), (2, "HD1 target White")):
        c = cal[k]; print(f"{nm}: a by band " + " ".join(f"{x:.2f}" for x in c["a"]) + f" | a_top {c['a_top']:.2f} | K(d=-2..2) " + " ".join(f"{c['K'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2))
                          + " | P0(d=-2..2) " + " ".join(f"{c['P0'][9 + x]:.2f}" for x in (-2, -1, 0, 1, 2)) + " | K/P0 " + " ".join(f"{c['ratio'][9 + x]:.1f}" for x in (-2, -1, 0, 1, 2)))
        rep[f"cal_{nm}"] = {"a": c["a"].tolist(), "a_top": c["a_top"], "K": c["K"].tolist(), "P0": c["P0"].tolist()}
    p1 = run("base seeds 0-2", (0, 1, 2), z, rep); p2 = run("replication seeds 6-8", (6, 7, 8), z, rep)
    rep["adopt"] = bool(p1 and p2); print(f"ADOPT: {'YES' if p1 and p2 else 'NO'}")
    Path(ROOT / "results/r_variants/m71_step2.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
