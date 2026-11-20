"""Section 4M M53 (2026-11-07; after ct13's Public 0.7748456 = all 8 of its 1D->A flips were true A): the same bias-only push at every
other adjacent boundary, as candidates for Public probes. Report only. Validation = folds 1-4, ct9 (C/T-pc seeds 0-2), test-like (no HD >= 2).
For each directed push i -> j (adjacent classes, j the target) and ratio r: on questions whose ct9 decision is i or j, m = P(i) + P(j) is kept
and P'(j) = m r P(j) / (r P(j) + P(i)); only questions whose decision moves i -> j change (as ct13). Reported: validation flips, precision
(share of flipped questions whose truth is j), the natural-prior gain (plain within n, mean over n), the class-balanced gain, and the number
of test questions flipped on Public / Private (from submission_rank_ct13_proba.csv's base, i.e. ct9's posteriors with ct13's A|1D change).
  python -I logs/nano4/m53_boundary_curves.py
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
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m47_prior import nplain  # noqa: E402

RATIOS = (1.5, 2.0, 2.6, 3.5)


def push(P, i, j, r):
    """Decisions after pushing i -> j with odds ratio r; only i -> j moves are applied."""
    d0 = decide(P); route = (d0 == i) | (d0 == j)
    Q = P.copy(); m = P[:, i] + P[:, j]; s = r * P[:, j] / (r * P[:, j] + P[:, i] + 1e-12)
    Q[route, j] = m[route] * s[route]; Q[route, i] = m[route] * (1 - s[route])
    d1 = decide(Q); d = d0.copy(); mv = (d0 == i) & (d1 == j); d[mv] = j
    return d, mv


def main():
    hd = pd.read_csv(ROOT / "results/data/hd_table.csv", dtype={"game_key": str}).drop_duplicates("game_key").set_index("game_key")["hd"].to_dict()
    ys, ns, Ps = [], [], []
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in (0, 1, 2)], axis=0)
        keep = ~(np.array([max(hd.get(g, 0.0) for g in ks) for ks in q["game_keys"]]) >= 2)
        ys.append(y[keep]); ns.append(n[keep]); Ps.append(P[keep])
    y, n, P = np.concatenate(ys), np.concatenate(ns), np.concatenate(Ps)
    d9 = decide(P); s9 = qscore(d9, y); bal9, nat9 = balanced_mean_over_n(s9, y, n), nplain(s9, n)
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); Pt = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); pub = pr["question_id"].str.startswith("pub").to_numpy()
    print(f"validation {len(y)} questions (no HD >= 2); ct9 balanced {bal9:.4f} natural {nat9:.4f}")
    print(f"{'push':10s} {'r':>4s} {'val flips':>9s} {'precision':>9s} {'gain nat':>9s} {'gain bal':>9s} {'pub':>4s} {'priv':>4s}")
    rep = {}
    for k in range(9):
        for i, j in ((k, k + 1), (k + 1, k)):
            name = f"{LEVELS[i]}->{LEVELS[j]}"
            for r in RATIOS:
                d, mv = push(P, i, j, r); s = qscore(d, y)
                prec = float((y[mv] == j).mean()) if mv.any() else float("nan")
                _, mvt = push(Pt, i, j, r)
                row = {"val_flips": int(mv.sum()), "precision": prec, "gain_natural": float(nplain(s, n) - nat9), "gain_balanced": float(balanced_mean_over_n(s, y, n) - bal9),
                       "public_flips": int(mvt[pub].sum()), "private_flips": int(mvt[~pub].sum())}
                rep[f"{name} r={r:g}"] = row
                print(f"{name:10s} {r:4.1f} {row['val_flips']:9d} {prec:9.3f} {row['gain_natural']:+9.4f} {row['gain_balanced']:+9.4f} {row['public_flips']:4d} {row['private_flips']:4d}", flush=True)
    Path(ROOT / "results/trunk_seq/es/m53_boundary_curves.json").write_text(json.dumps(rep, indent=1)); print("written")


if __name__ == "__main__":
    main()
