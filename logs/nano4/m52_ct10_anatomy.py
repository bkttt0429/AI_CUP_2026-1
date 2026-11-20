"""Section 4M M52 anatomy of ct10 (2026-11-07; user: 「ct10 如何繼續提升？換更好的二分類模型？」). Report only, validation folds 1-4, ct9 =
C/T-pc-f0-x40-oh seeds 0-2, test-like questions (no HD >= 2 game), cross-fitted over the folds.
  (a) discrimination at A|1D on true-A / true-1D questions: AUC of ct9's own log P(A) - log P(1D); of the M46 'probs' specialist (ct10);
      of the 'full' specialist (+ 64 R columns). Equal AUCs = the specialist adds no discrimination.
  (b) bias-only variant: ct9's own odds P(1D) / P(A) scaled by a constant prior ratio r (A : 1D), folded in like ct10 (m = P(A) + P(1D) kept,
      expected-score decision); r = 1 reproduces ct9. The curve over r: flips 1D->A and A->1D, the class-balanced and the natural-prior
      (plain within n, mean over n) gain vs ct9; ct10's own flips and gains for comparison, and the same for the test file (flips on Public /
      Private per r from submission_rank_ct9_proba.csv).
  (c) oracle ceiling: a perfect A|1D oracle on the routed questions (decision A or 1D) -- the most any binary model could add -- under the
      uniform and the natural prior.
  python -I logs/nano4/m52_ct10_anatomy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, auc, decide, qscore  # noqa: E402
from m44_calib import logit_fit_auc  # noqa: E402
from m46_a1d import A, D1, fold_in, xprobs  # noqa: E402
from m47_prior import nplain  # noqa: E402

RATIOS = (1.0, 1.5, 2.0, 2.6, 3.5, 5.0, 8.0)


def scaled(P, r):
    """Bias-only specialist: s = P(1D | A or 1D) after multiplying the A odds by r."""
    pa, pd_ = P[:, A], P[:, D1]
    return pd_ / (pd_ + r * pa + 1e-12)


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float64)
        keep = ~(np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)
        D[v] = {"y": y[keep], "n": n[keep], "pid": pid[keep], "P": P[keep], "probs": xprobs(P, n)[keep], "full": np.hstack([xprobs(P, n), R])[keep]}
    # cross-fitted specialists
    S = {"probs": {}, "full": {}}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        sel = lambda u: (D[u]["y"] == A) | (D[u]["y"] == D1)  # noqa: E731
        lab = np.concatenate([(D[u]["y"][sel(u)] == D1).astype(float) for u in tr])
        for kind in S:
            X = np.concatenate([D[u][kind][sel(u)] for u in tr]); S[kind][v] = logit_fit_auc(X, lab, D[v][kind], None)
    y = np.concatenate([D[v]["y"] for v in FOLDS]); n = np.concatenate([D[v]["n"] for v in FOLDS]); P = np.concatenate([D[v]["P"] for v in FOLDS])
    s_probs = np.concatenate([S["probs"][v] for v in FOLDS]); s_full = np.concatenate([S["full"][v] for v in FOLDS])
    rep = {"questions": int(len(y))}
    # (a) discrimination
    a1 = (y == A) | (y == D1); lab = (y[a1] == D1).astype(int)
    own = np.log(np.clip(P[a1, D1], 1e-9, 1)) - np.log(np.clip(P[a1, A], 1e-9, 1))
    rep["auc"] = {"ct9_own_logodds": auc(own, lab), "specialist_probs": auc(s_probs[a1], lab), "specialist_full": auc(s_full[a1], lab)}
    print(f"(a) A|1D discrimination on {int(a1.sum())} true-A/1D questions: AUC ct9's own log odds {rep['auc']['ct9_own_logodds']:.4f}, "
          f"ct10's 'probs' specialist {rep['auc']['specialist_probs']:.4f}, 'full' specialist (+R) {rep['auc']['specialist_full']:.4f}", flush=True)
    # (b) bias-only curve vs ct10
    d9 = decide(P); s9 = qscore(d9, y); bal9 = balanced_mean_over_n(s9, y, n); nat9 = nplain(s9, n)
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); Pt = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); pub = pr["question_id"].str.startswith("pub").to_numpy()
    dt = decide(Pt)

    def describe(name, sv, st):
        Q, _ = fold_in(P, sv); d = decide(Q); s = qscore(d, y)
        f_up, f_down = int(((d9 == D1) & (d == A)).sum()), int(((d9 == A) & (d == D1)).sum())
        row = {"flips_1D_to_A": f_up, "flips_A_to_1D": f_down, "balanced_gain": float(balanced_mean_over_n(s, y, n) - bal9), "natural_gain": float(nplain(s, n) - nat9),
               "a_A": float(np.mean([(s - s9)[(y == A) & (n == m)].mean() for m in (5, 10, 20)])), "a_1D": float(np.mean([(s - s9)[(y == D1) & (n == m)].mean() for m in (5, 10, 20)]))}
        if st is not None:
            Qt, _ = fold_in(Pt, st); dd = decide(Qt)
            row["test_flips"] = {"public_1D_to_A": int(((dt == D1) & (dd == A))[pub].sum()), "public_A_to_1D": int(((dt == A) & (dd == D1))[pub].sum()),
                                 "private_1D_to_A": int(((dt == D1) & (dd == A))[~pub].sum()), "private_A_to_1D": int(((dt == A) & (dd == D1))[~pub].sum())}
        rep.setdefault("variants", {})[name] = row
        tf = row.get("test_flips", {})
        print(f"{name:14s} validation flips 1D->A {f_up:5d} A->1D {f_down:5d}; gain balanced {row['balanced_gain']:+.4f} natural {row['natural_gain']:+.4f}; a_A {row['a_A']:+.3f} a_1D {row['a_1D']:+.3f}"
              + (f"; test flips public 1D->A {tf['public_1D_to_A']} A->1D {tf['public_A_to_1D']}, private 1D->A {tf['private_1D_to_A']} A->1D {tf['private_A_to_1D']}" if tf else ""), flush=True)

    print(f"(b) ct9 reference: balanced {bal9:.4f}, natural {nat9:.4f}; bias-only curve (A odds x r) and ct10:", flush=True)
    for r in RATIOS:
        describe(f"bias r={r:g}", scaled(P, r), scaled(Pt, r))
    # ct10's deployed specialist for the test: trained on all four folds' A/1D questions (as m46 --write)
    sel_all = (y == A) | (y == D1)
    X_all = np.concatenate([D[v]["probs"] for v in FOLDS])
    st10 = logit_fit_auc(X_all[sel_all], (y[sel_all] == D1).astype(float), xprobs(Pt, pr["question_id"].map(lambda q: int(q.split("_")[-1]) * 0 + 0).to_numpy()) if False else None, None) if False else None
    describe("ct10 (cross-fit)", s_probs, None)
    describe("ct10 full (+R)", s_full, None)
    # (c) oracle ceiling on routed questions
    route = (d9 == A) | (d9 == D1)
    d_or = d9.copy(); fix = route & ((y == A) | (y == D1)); d_or[fix] = y[fix]
    s_or = qscore(d_or, y)
    rep["oracle"] = {"routed_questions": int(route.sum()), "balanced_gain": float(balanced_mean_over_n(s_or, y, n) - bal9), "natural_gain": float(nplain(s_or, n) - nat9),
                     "confusion": {"A->1D": float(np.mean([((d9 == D1) & (y == A) & (n == m)).sum() / max(((y == A) & (n == m)).sum(), 1) for m in (5, 10, 20)])),
                                   "1D->A": float(np.mean([((d9 == A) & (y == D1) & (n == m)).sum() / max(((y == D1) & (n == m)).sum(), 1) for m in (5, 10, 20)]))}}
    print(f"(c) perfect A|1D oracle on the {int(route.sum())} routed questions: gain balanced {rep['oracle']['balanced_gain']:+.4f}, natural {rep['oracle']['natural_gain']:+.4f}; "
          f"ct9 confusion A->1D {rep['oracle']['confusion']['A->1D']:.3f}, 1D->A {rep['oracle']['confusion']['1D->A']:.3f}", flush=True)
    Path(ROOT / "results/trunk_seq/es/m52_ct10_anatomy.json").write_text(json.dumps(rep, indent=1, default=float)); print("written", flush=True)


if __name__ == "__main__":
    main()
