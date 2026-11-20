"""Section 4M M63 (2026-11-09, written before the 11/10 readout): is there a composition / directional basis for the 4D->5D and B->C
bets of the kind A|1D had ("the test has more A than 1D near the boundary")? Report only.
(1) Directional statistic per boundary i|j from the decoded Public probes (ledger, results/trunk_seq/es/public_ledger_decoded.json and the
    design doc): among the flipped questions whose truth is i or j, the share whose truth is the UPPER class j, pooling the push i->j
    (truth j = "right") and the push j->i (truth j = "wrong"). Null = symmetric noise: every push has precision p0 (truth = target with
    probability p0, = source otherwise), so U = Bin(n_ij, p0) + Bin(n_ji, 1 - p0); exact one-sided tail P(U >= observed) for p0 = 0.38
    (pooled over the 12 pushes after ct13) and p0 = 0.5. A|1D is the reference: its two pushes agreed (truth A 14 of 16).
(2) Label-shift estimates of the test's class prior from ct9's test posteriors (submission_rank_ct9_proba.csv), per half and pooled:
    BBSE (Lipton et al. 2018): confusion matrix C[pred | true] of ct9's expected-score decisions on the CV folds (six members per fold),
    class-balanced and n-weighted to the test's n mix (5 / 10 / 15 / 20 at 25% each; CV has 5 / 10 / 20, so 20 stands for 15 and 20),
    solved by NNLS; EM (Saerens et al. 2002) on the posteriors with the training prior = share of validation players per class.
    Question bootstrap (1000) for 90% intervals of pi(5D)/pi(4D), pi(C)/pi(B), pi(A)/pi(1D), pi(2D)/pi(1D).
  python -I logs/nano4/m63_label_shift.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import binom

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402
from m43_gate import decide  # noqa: E402

R = {lv: i for i, lv in enumerate(LEVELS)}
MEMBERS = [(f"{t}-pc-f0-x40-oh", s) for t in ("C", "T") for s in (0, 1, 2)]
# decoded Public single pushes (source -> target): right = truth is the target, wrong = truth is the source, other = truth elsewhere.
# B->A is the ledger's MAP split of ct14 (X +7 over 13 determined-or-not questions, 5.8 / 7.2 by posterior mean); all others are exact.
PUSHES = {"1D->A": (8, 0, 0), "A->1D": (2, 6, 1), "2D->1D": (5, 6, 1), "1D->2D": (6, 5, 4), "3D->2D": (1, 6, 1), "3D->4D": (5, 7, 0),
          "4D->5D": (7, 2, 0), "5D->4D": (1, 6, 1), "B->C": (6, 3, 0), "C->D": (4, 4, 1), "5D->6D": (1, 6, 0), "B->A": (5.8, 7.2, 0),
          "C->B": (3, 4, 1), "A->B": (3, 3, 0), "4D->3D": (1, 4, 1)}  # 11/10 readout; 3D->2D re-decoded 11/10
P0_POOLED = 0.38
TEST_N_SHARE = {5: 0.25, 10: 0.25, 20: 0.5}


def tail_sum(n_up, n_down, u_obs, p0):
    """P(U >= u_obs) with U = Bin(n_up, p0) + Bin(n_down, 1 - p0); non-integer counts are rounded."""
    n_up, n_down, u_obs = int(round(n_up)), int(round(n_down)), int(round(u_obs))
    pa = binom.pmf(np.arange(n_up + 1), n_up, p0); pb = binom.pmf(np.arange(n_down + 1), n_down, 1 - p0)
    conv = np.convolve(pa, pb)
    return float(conv[u_obs:].sum())


def directional():
    out = {}
    for i in range(9):
        lo, hi = LEVELS[i], LEVELS[i + 1]
        up, down = PUSHES.get(f"{lo}->{hi}"), PUSHES.get(f"{hi}->{lo}")
        if up is None and down is None:
            continue
        U = (up[0] if up else 0) + (down[1] if down else 0); L = (up[1] if up else 0) + (down[0] if down else 0)
        n_up = (up[0] + up[1]) if up else 0; n_down = (down[0] + down[1]) if down else 0
        rec = {"pushes": [k for k in (f"{lo}->{hi}", f"{hi}->{lo}") if k in PUSHES], "truth_upper": U, "truth_lower": L, "share_upper": float(U / (U + L)) if U + L else None,
               "null_expected_upper_p038": float(P0_POOLED * n_up + (1 - P0_POOLED) * n_down),
               "p_upper_ge_obs_p038": tail_sum(n_up, n_down, U, P0_POOLED), "p_upper_ge_obs_p05": tail_sum(n_up, n_down, U, 0.5),
               "p_lower_ge_obs_p038": tail_sum(n_down, n_up, L, P0_POOLED)}
        out[f"{lo}|{hi}"] = rec
    return out


def cv_tables():
    Ps, ys, ns = [], [], []
    for v in FOLDS:
        y, n, _, _ = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t, s in MEMBERS], axis=0)
        Ps.append(P); ys.append(y); ns.append(n)
    P, y, n = np.concatenate(Ps), np.concatenate(ys), np.concatenate(ns)
    d = np.asarray(decide(P), int)
    w = np.zeros(len(y))
    for c in range(10):
        for m, share in TEST_N_SHARE.items():
            s = (y == c) & (n == m); w[s] = share / max(s.sum(), 1)
    C = np.zeros((10, 10))  # C[pred, true] = P(pred | true), class-balanced, n-weighted
    for c in range(10):
        s = y == c; C[:, c] = np.bincount(d[s], weights=w[s], minlength=10) / w[s].sum()
    prior_players = np.bincount(y[n == 20], minlength=10) / (n == 20).sum()  # one n=20 question per validation player
    return C, prior_players


def bbse(C, mu):
    pi, _ = nnls(C, mu); return pi / pi.sum() if pi.sum() > 0 else pi


def em(P, prior_train, iters=200):
    pi = np.full(10, 0.1)
    for _ in range(iters):
        W = P * (pi / prior_train)[None, :]; W /= W.sum(1, keepdims=True)
        new = W.mean(0)
        if np.abs(new - pi).max() < 1e-8:
            pi = new; break
        pi = new
    return pi


def ratios(pi):
    return {"5D/4D": float(pi[R["5D"]] / pi[R["4D"]]), "C/B": float(pi[R["C"]] / pi[R["B"]]), "A/1D": float(pi[R["A"]] / pi[R["1D"]]), "2D/1D": float(pi[R["2D"]] / pi[R["1D"]])}


def label_shift():
    C, prior_train = cv_tables()
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}).set_index("question_id")
    P_all = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); ids = pr.index.to_numpy()
    rng = np.random.default_rng(20261109)
    out = {"confusion_pred_given_true": C.round(3).tolist(), "prior_train_players": prior_train.round(4).tolist(), "halves": {}}
    for half, mask in (("public", np.char.startswith(ids.astype(str), "pub_")), ("private", np.char.startswith(ids.astype(str), "priv_")), ("all", np.ones(len(ids), bool))):
        P = P_all[mask]; d = np.asarray(decide(P), int); mu = np.bincount(d, minlength=10) / len(d)
        pi_b, pi_e = bbse(C, mu), em(P, prior_train)
        boots = {"bbse": [], "em": []}
        for _ in range(1000):
            k = rng.integers(0, len(P), len(P)); Pk = P[k]; dk = d[k]; muk = np.bincount(dk, minlength=10) / len(dk)
            boots["bbse"].append(ratios(bbse(C, muk))); boots["em"].append(ratios(em(Pk, prior_train, iters=60)))
        rec = {"questions": int(mask.sum()), "predicted_share": {lv: float(x) for lv, x in zip(LEVELS, mu)},
               "pi_bbse": {lv: float(x) for lv, x in zip(LEVELS, pi_b)}, "pi_em": {lv: float(x) for lv, x in zip(LEVELS, pi_e)}}
        for meth, pi in (("bbse", pi_b), ("em", pi_e)):
            rr = ratios(pi); ci = {k: (float(np.nanpercentile([b[k] for b in boots[meth]], 5)), float(np.nanpercentile([b[k] for b in boots[meth]], 95))) for k in rr}
            rec[f"ratios_{meth}"] = {k: {"point": rr[k], "ci90": ci[k]} for k in rr}
        out["halves"][half] = rec
    return out


def main():
    d = directional()
    print("directional statistic per boundary (share of truths on the UPPER side among the flipped questions with truth i or j; null p0 = 0.38 symmetric):")
    for k, r in d.items():
        print(f"  {k:6s} {','.join(r['pushes']):16s} upper {r['truth_upper']:5.1f} lower {r['truth_lower']:5.1f} share {r['share_upper']:.2f} | null E[upper] {r['null_expected_upper_p038']:.1f}; "
              f"P(upper >= obs) p0=.38 {r['p_upper_ge_obs_p038']:.4f}, p0=.5 {r['p_upper_ge_obs_p05']:.4f}; P(lower >= obs) {r['p_lower_ge_obs_p038']:.4f}")
    ls = label_shift()
    print("training prior (validation players per class):", " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, ls["prior_train_players"])))
    for half, r in ls["halves"].items():
        print(f"{half}: {r['questions']} q; ct9 predicted share " + " ".join(f"{lv} {x:.3f}" for lv, x in r["predicted_share"].items()))
        print(f"  pi BBSE " + " ".join(f"{lv} {x:.3f}" for lv, x in r["pi_bbse"].items()))
        print(f"  pi EM   " + " ".join(f"{lv} {x:.3f}" for lv, x in r["pi_em"].items()))
        for meth in ("bbse", "em"):
            print(f"  ratios {meth}: " + "; ".join(f"{k} {v['point']:.2f} [{v['ci90'][0]:.2f}, {v['ci90'][1]:.2f}]" for k, v in r[f"ratios_{meth}"].items()))
    Path(ES / "m63_label_shift.json").write_text(json.dumps({"directional": d, "label_shift": ls}, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
