#!/usr/bin/env python
"""Section 4M M135 (2026-11-14): does a gain of the base ensemble survive the layers above it?

The ct104b rule - the M129 bundle at its registered choice (the m15 banding statistic for the five voters, the look-alike reading
pooled with weights {5: 0.5, 10: 0.5, 20: 0.1}) after the ct75u composition, with the two composition guards (no change out of A,
no 5D -> 6D) - is evaluated in cross-validation on two base ensembles: the current members (m71_step2.STD) and the --new arms.
Nothing is chosen again: the evidence, the density ratios, the bundle's choice and the guards are those of ct104b; only the base
probabilities change. Reported for each seed set, paired by question (player bootstrap): the base decision, the ct75u rule and the
ct104b rule, new base minus current base; per fold, per question size, per true level.

  python -I logs/nano4/m135_final_cv.py --new C-pc-f0-x40-oh-ex T-pc-f0-x40-oh-ex --seeds 0 1 2 [--seeds2 6 7 8] [--json m135_final_cv.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m103_stability as m3  # noqa: E402
import m105_voters as m5  # noqa: E402
import m110_fusion as mf  # noqa: E402
import m123_lookalike_sym as la  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402

b, s2, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.ES, m3.FOLDS, m3.REC, m3.SM
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); STAT = "m15"
W_FIN = {5: 0.5, 10: 0.5, 20: 0.1}  # the registered choice of M129 (results/r_variants/m129_bundle2.json, all folds), A = 2
DROPS = [(LEVELS.index("A"), None), (LEVELS.index("5D"), LEVELS.index("6D"))]  # ct104b's guards


def keep_old(d_new, d_old):
    d = d_new.copy()
    for o, w_ in DROPS:
        m = (d_old == o) & (d_new != d_old) if w_ is None else (d_old == o) & (d_new == w_)
        d[m] = d_old[m]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", default=list(s2.STD))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds2", type=int, nargs="*", default=[])
    ap.add_argument("--json", default="m135_final_cv.json")
    a_ = ap.parse_args()
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); A_reg, w_reg = int(pj["picks"]["all_folds"][0]), {int(k): float(v) for k, v in pj["picks"]["all_folds"][1].items()}
    assert A_reg == 2 and w_reg == W_FIN, (A_reg, w_reg)
    scale = pj["density_ratio"]  # the m15 identifiers' test-side density ratios of the registered run (computed with the current base; kept fixed)
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])
    rec0, rec2 = dict(REC), dict(REC)
    for k in FIVE:
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))
    LF = {v: m3.load_fold(v) for v in FOLDS}
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); assert np.array_equal(q["qkey"][:Q].astype(str), z["key"])
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]
    Fd = {}
    for v in FOLDS:
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}
        for A, rec in ((0, rec0), (2, rec2)):
            Ls[A] = {}
            for name, spec in rec.items():
                if spec not in cache:
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L
                Ls[A][name] = cache[spec]
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))
        for m in range(NM):
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)
        Fd[v] = (y, n, pid, Ls, lr_ - lr_.mean(0, keepdims=True), np.array([W_FIN.get(int(m), (W_FIN[10] + W_FIN[20]) / 2) for m in n]))
        print(f"fold {v}: evidence and look-alike reading ready ({len(y)} questions)", flush=True)
    ys = [Fd[v][0] for v in FOLDS]; ns = [Fd[v][1] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in (0, 1, 2)); rep = {"old": a_.old, "new": a_.new}
    for seeds in [tuple(a_.seeds)] + ([tuple(a_.seeds2)] if a_.seeds2 else []):
        S = {}
        for tag, arms in (("old", a_.old), ("new", a_.new)):
            for v in FOLDS:
                yv, nv, _, Ls, RD, wq = Fd[v]
                P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds], axis=0)
                assert len(P) == len(yv), f"fold {v}: {len(P)} member rows for {len(yv)} validation questions"
                d_base = np.asarray(b.decide(P), int); _, du0 = mf.rule75u(P, Ls[0]); _, du = mf.rule75u(la.pooled(P, RD, wq), Ls[2]); d_fin = keep_old(du, du0)
                for lvl, d_ in (("base", d_base), ("ct75u_rule", du0), ("ct104b_rule", d_fin)):
                    S.setdefault((tag, lvl), []).append(SM[d_, yv])
        out = {}
        for lvl in ("base", "ct75u_rule", "ct104b_rule"):
            new, old = S[("new", lvl)], S[("old", lvl)]; A_, B_ = np.concatenate(new), np.concatenate(old); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(new, old, ys, ns)]
            lv_old = float(np.mean([balanced_mean_over_n(c, yy, nn, sizes=SIZES) for c, yy, nn in zip(old, ys, ns)])); lv_new = float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(new, ys, ns)]))
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)
            print(f"seeds {seeds} {lvl:12s}: current {lv_old:.4f} -> new base {lv_new:.4f}; paired {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
                  + "  by n " + " ".join(f"{m}: {v:+.4f}" for m, v in dd["by_n"].items()) + f"  decisions changed {float((A_ != B_).mean()):.3f} -> {'meets M37' if good else 'no'}")
            print("      by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)), flush=True)
            out[lvl] = {**dd, "per_fold": pf, "current": lv_old, "new": lv_new, "meets_m37": bool(good)}
        rep[f"seeds_{seeds[0]}"] = out
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float)); print(f"written results/r_variants/{a_.json}")


if __name__ == "__main__":
    main()
