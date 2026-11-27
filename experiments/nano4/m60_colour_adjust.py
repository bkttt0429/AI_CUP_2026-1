"""Section 4M M60 (b): a colour-conditional prior correction learned on colour-pure CV questions (the test's construction), cross-fitted over
folds 1-4. Report only. Inputs: results/trunk_seq/es/m60_{pureB,pureW,mixed}_val{v}.npz (ct9's six members per fold, m60_colour_pure_cv.py).
Arms (fit on three folds' questions, applied to the fourth; class-balanced weighted log-likelihood of the truth under softmax(log P + b) / T):
  shared  one bias vector b (10) for all pure questions;   colour  b_B and b_W separately;   colourT  b_colour plus a temperature per colour.
Decision = expected-score argmax. Score: class-balanced official score over n = 5 / 10 on pure questions (B and W together, as the test is),
paired player bootstrap vs raw ct9, per fold; also the same arms fitted on mixed questions applied to pure ones (control).
  python -I logs/nano4/m60_colour_adjust.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402

ES = ROOT / "results/trunk_seq/es"
SIZES = (5, 10)


def load(name, v):
    z = np.load(ES / f"m60_{name}_val{v}.npz", allow_pickle=False)
    return {k: z[k] for k in ("proba", "y", "n", "player_id")}


def weights(y, n):
    w = np.zeros(len(y))
    for c in range(10):
        for m in SIZES:
            s = (y == c) & (n == m); w[s] = 1.0 / max(s.sum(), 1)
    return w / w.sum()


def fit(P, y, n, temp=False):
    L = np.log(np.clip(P, 1e-9, 1)); w = weights(y, n)

    def nll(th):
        b = th[:10]; T = np.exp(th[10]) if temp else 1.0
        Z = L / T + b; Z -= Z.max(1, keepdims=True); lp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -(w * lp[np.arange(len(y)), y]).sum() + 1e-3 * (b ** 2).sum()
    th0 = np.zeros(11 if temp else 10)
    return minimize(nll, th0, method="L-BFGS-B").x


def apply(P, th, temp=False):
    L = np.log(np.clip(P, 1e-9, 1)); T = np.exp(th[10]) if temp else 1.0
    Z = L / T + th[:10]; Z -= Z.max(1, keepdims=True); Q = np.exp(Z); return Q / Q.sum(1, keepdims=True)


def main():
    D = {v: {k: load(k, v) for k in ("pureB", "pureW", "mixed")} for v in FOLDS}
    S = {arm: {} for arm in ("raw", "shared", "colour", "colourT", "mixed_fit")}
    fitted = {}
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]
        cat = lambda key, sets: np.concatenate([D[u][s][key] for u in tr for s in sets])  # noqa: E731
        th_sh = fit(cat("proba", ("pureB", "pureW")), cat("y", ("pureB", "pureW")), cat("n", ("pureB", "pureW")))
        th_c = {c: fit(cat("proba", (f"pure{c}",)), cat("y", (f"pure{c}",)), cat("n", (f"pure{c}",))) for c in "BW"}
        th_cT = {c: fit(cat("proba", (f"pure{c}",)), cat("y", (f"pure{c}",)), cat("n", (f"pure{c}",)), temp=True) for c in "BW"}
        th_m = fit(cat("proba", ("mixed",)), cat("y", ("mixed",)), cat("n", ("mixed",)))
        fitted[v] = {"shared": th_sh.tolist(), "B": th_c["B"].tolist(), "W": th_c["W"].tolist()}
        ys, ns, ps, sc = [], [], [], {a: [] for a in S}
        for c in "BW":
            d = D[v][f"pure{c}"]; P, y = d["proba"].astype(np.float64), d["y"]
            for arm, Q in (("raw", P), ("shared", apply(P, th_sh)), ("colour", apply(P, th_c[c])), ("colourT", apply(P, th_cT[c], temp=True)), ("mixed_fit", apply(P, th_m))):
                sc[arm].append(qscore(decide(Q), y))
            ys.append(y); ns.append(d["n"]); ps.append(d["player_id"])
        for arm in S:
            S[arm][v] = (np.concatenate(sc[arm]), np.concatenate(ys), np.concatenate(ns), np.concatenate(ps))
    rep = {"fitted_bias": fitted, "arms": {}}
    raw_all = np.concatenate([S["raw"][v][0] for v in FOLDS]); y = np.concatenate([S["raw"][v][1] for v in FOLDS])
    n = np.concatenate([S["raw"][v][2] for v in FOLDS]); pid = np.concatenate([S["raw"][v][3] for v in FOLDS])
    print(f"pure questions (B + W, folds 1-4): {len(y)}; raw ct9 class-balanced score {balanced_mean_over_n(raw_all, y, n, sizes=SIZES):.4f}")
    for arm in ("shared", "colour", "colourT", "mixed_fit"):
        new = np.concatenate([S[arm][v][0] for v in FOLDS])
        dd = paired_diff(new, raw_all, y, pid, n, sizes=SIZES, n=3000)
        pf = [balanced_mean_over_n(S[arm][v][0], S[arm][v][1], S[arm][v][2], sizes=SIZES) - balanced_mean_over_n(S["raw"][v][0], S["raw"][v][1], S["raw"][v][2], sizes=SIZES) for v in FOLDS]
        rep["arms"][arm] = {**dd, "per_fold": pf}
        print(f"{arm:9s}: {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf), flush=True)
    b = np.mean([fitted[v]["B"] for v in FOLDS], axis=0); w = np.mean([fitted[v]["W"] for v in FOLDS], axis=0)
    print("mean fitted bias, black-pure: " + " ".join(f"{lv} {x:+.2f}" for lv, x in zip(LEVELS, b - b.mean())))
    print("mean fitted bias, white-pure: " + " ".join(f"{lv} {x:+.2f}" for lv, x in zip(LEVELS, w - w.mean())))
    Path(ES / "m60_colour_adjust.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
