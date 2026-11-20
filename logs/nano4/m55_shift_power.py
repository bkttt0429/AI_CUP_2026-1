"""Section 4M M55 (C) power check: would m55_shift_scan.py have seen an A|1D-sized shift at the other boundaries? Report only.
At each boundary, 200 simulated test samples of the test's routed size are drawn from the validation routed questions with the composition
fitted on the test (M0), and the true-i class (or the true-j class) shifted by delta in s = log P(j) - log P(i) (+0.5 and -0.5 on the j side:
the A|1D estimate was +0.5 on the i side). Power = share of simulated LRs above the 95% quantile of 200 unshifted simulations.
  python -I logs/nano4/m55_shift_power.py
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
from m55_shift_scan import DELTAS, em  # noqa: E402

SIMS = 200


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
    scan = json.load(open(ROOT / "results/trunk_seq/es/m55_shift_scan.json"))
    rng = np.random.default_rng(1)
    rep = {}
    for i in range(9):
        j = i + 1; name = f"{LEVELS[i]}|{LEVELS[j]}"
        s_val = np.log(np.clip(P[:, j], 1e-9, 1)) - np.log(np.clip(P[:, i], 1e-9, 1))
        rv = (d == i) | (d == j); rt = (decide(Pt) == i) | (decide(Pt) == j)
        classes = [c for c in (i - 1, i, j, j + 1) if 0 <= c <= 9]
        kdes = {c: gaussian_kde(s_val[rv & (y == c)], weights=wq[rv & (y == c)]) for c in classes}
        w0 = np.array([scan[name]["composition_w"][LEVELS[c]] for c in classes]); w0 = w0 / w0.sum()
        N = int(rt.sum())
        pools = {c: np.flatnonzero(rv & (y == c)) for c in classes}
        pc = {c: wq[pools[c]] / wq[pools[c]].sum() for c in classes}

        def lr_of(xs, side):
            F0 = np.column_stack([kdes[c](xs) for c in classes]); _, l0 = em(F0)
            k = classes.index(i if side == "i" else j); best = l0
            for dl in DELTAS:
                F = F0.copy(); F[:, k] = kdes[classes[k]](xs - dl); _, ll = em(F); best = max(best, ll)
            return 2 * (best - l0)

        def draw(shift_class, delta):
            counts = rng.multinomial(N, w0); parts = []
            for c, cnt in zip(classes, counts):
                if cnt == 0:
                    continue
                xs = s_val[rng.choice(pools[c], cnt, p=pc[c])]
                parts.append(xs + (delta if c == shift_class else 0.0))
            return np.concatenate(parts)

        out = {}
        for side, cls, delta in (("i", i, 0.5), ("j", j, -0.5)):
            null = [lr_of(draw(None, 0.0), side) for _ in range(SIMS)]; q95 = float(np.quantile(null, 0.95))
            alt = [lr_of(draw(cls, delta), side) for _ in range(SIMS)]
            out[side] = {"delta": delta, "q95_null": q95, "power": float((np.array(alt) > q95).mean()), "median_lr_alt": float(np.median(alt))}
        rep[name] = out
        print(f"{name:6s} N {N:3d}: true-{LEVELS[i]} shift +0.5 power {out['i']['power']:.2f} (null 95% {out['i']['q95_null']:.2f}, median alt LR {out['i']['median_lr_alt']:.2f}); "
              f"true-{LEVELS[j]} shift -0.5 power {out['j']['power']:.2f} (null 95% {out['j']['q95_null']:.2f}, median alt LR {out['j']['median_lr_alt']:.2f})", flush=True)
    Path(ROOT / "results/trunk_seq/es/m55_shift_power.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
