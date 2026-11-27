"""Section 4M M55 (C): per-boundary shift scan from unlabelled test predictions. Report only.
For boundary (i, j = i + 1): s = log P(j) - log P(i) under ct9; routed questions = ct9 decision in {i, j}.
Validation (folds 1-4, ct9 out-of-fold, HD-free, n = 5/10/20 weighted equally): for each true class c in {i-1, i, j, j+1} the weighted KDE
f_c(s) of the routed questions. Test: s of ct9's routed test questions (submission_rank_ct9_proba.csv).
  M0: g(s) = sum_c w_c f_c(s), w fitted by EM (composition free).
  M_i(delta): f_i replaced by f_i(s - delta) (true-i questions look delta more like j); M_j(delta): f_j(s - delta) (true-j look more like i
  when delta < 0). Profile likelihood over a delta grid; LR = 2 (max - M0).
Null calibration: 300 validation resamples of the test's routed size with M0's fitted composition; p = share of null LRs >= observed.
The known case A|1D (Public: A truths hidden in 1D decisions, i.e. M_i with delta > 0 at i = A) is the method check.
  python -I logs/nano4/m55_shift_scan.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide  # noqa: E402

DELTAS = np.round(np.arange(-1.5, 1.51, 0.1), 2)


def em(F, iters=300):
    w = np.full(F.shape[1], 1.0 / F.shape[1])
    for _ in range(iters):
        R = F * w; tot = R.sum(1, keepdims=True) + 1e-300; w_new = (R / tot).mean(0)
        if np.abs(w_new - w).max() < 1e-9:
            w = w_new; break
        w = w_new
    return w, float(np.log((F * w).sum(1) + 1e-300).sum())


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    ys, ns, Ps = [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        keep = ~(np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)
        ys.append(y[keep]); ns.append(n[keep]); Ps.append(P[keep])
    y, n, P = np.concatenate(ys), np.concatenate(ns), np.concatenate(Ps)
    wq = np.zeros(len(y))
    for c in range(10):
        for m in (5, 10, 20):
            sel = (y == c) & (n == m); wq[sel] = 1.0 / max(sel.sum(), 1)
    d = decide(P)
    Pt = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv")[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); dt = decide(Pt)
    rng = np.random.default_rng(0)
    rep = {}
    for i in range(9):
        j = i + 1
        s_val = np.log(np.clip(P[:, j], 1e-9, 1)) - np.log(np.clip(P[:, i], 1e-9, 1))
        s_t = np.log(np.clip(Pt[:, j], 1e-9, 1)) - np.log(np.clip(Pt[:, i], 1e-9, 1))
        rv, rt = (d == i) | (d == j), (dt == i) | (dt == j)
        classes = [c for c in (i - 1, i, j, j + 1) if 0 <= c <= 9]
        kdes = {c: gaussian_kde(s_val[rv & (y == c)], weights=wq[rv & (y == c)]) for c in classes}
        x = s_t[rt]

        def fit(xs):
            F0 = np.column_stack([kdes[c](xs) for c in classes])
            w0, ll0 = em(F0)
            best = {"i": (0.0, ll0), "j": (0.0, ll0)}
            curve = {"i": [], "j": []}
            for side, cc in (("i", i), ("j", j)):
                k = classes.index(cc)
                for dl in DELTAS:
                    F = F0.copy(); F[:, k] = kdes[cc](xs - dl)
                    _, ll = em(F); curve[side].append(ll)
                    if ll > best[side][1]:
                        best[side] = (float(dl), ll)
            return w0, ll0, best, curve

        w0, ll0, best, curve = fit(x)
        lr = {side: 2 * (best[side][1] - ll0) for side in ("i", "j")}
        # null: resample validation routed questions with the fitted composition
        null = {"i": [], "j": []}
        pools = {c: np.flatnonzero(rv & (y == c)) for c in classes}
        probs_c = {c: wq[pools[c]] / wq[pools[c]].sum() for c in classes}
        for _ in range(300):
            counts = rng.multinomial(len(x), w0)
            xs = np.concatenate([s_val[rng.choice(pools[c], cnt, p=probs_c[c])] for c, cnt in zip(classes, counts) if cnt > 0])
            _, l0, b, _ = fit(xs)
            for side in ("i", "j"):
                null[side].append(2 * (b[side][1] - l0))
        pv = {side: float((np.array(null[side]) >= lr[side]).mean()) for side in ("i", "j")}
        name = f"{LEVELS[i]}|{LEVELS[j]}"
        rep[name] = {"routed_test": int(len(x)), "composition_w": dict(zip([LEVELS[c] for c in classes], w0.tolist())),
                     "shift_i": {"delta": best["i"][0], "lr": lr["i"], "p": pv["i"]}, "shift_j": {"delta": best["j"][0], "lr": lr["j"], "p": pv["j"]}}
        print(f"{name:6s} routed test {len(x):3d}; composition " + " ".join(f"{LEVELS[c]} {w:.2f}" for c, w in zip(classes, w0))
              + f"; true-{LEVELS[i]} shift delta {best['i'][0]:+.1f} LR {lr['i']:.2f} p {pv['i']:.3f}; true-{LEVELS[j]} shift delta {best['j'][0]:+.1f} LR {lr['j']:.2f} p {pv['j']:.3f}", flush=True)
    Path(ROOT / "results/trunk_seq/es/m55_shift_scan.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
