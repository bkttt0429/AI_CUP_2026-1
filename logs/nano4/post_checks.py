"""Cheap post-processing checks on the CV oh ensembles (folds 1-4, C/T-x40-oh seeds 0-2), each fitted on the other three folds:
(i) per-n temperature gamma_n (grid), (ii) geometric vs arithmetic member averaging, (iii) C/T arm weight. Paired player bootstrap vs the
arithmetic 6-member ensemble. Gemini's proposals 4.2 / 4.3 (2026-11-05); expected ~0.
  python -I logs/nano4/post_checks.py --out results/r_variants/post_checks.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}; FOLDS = (1, 2, 3, 4)


def load(v):
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
    M = {(a, s): np.load(ROOT / f"results/trunk_seq/es/val{v}/{a}_s{s}.npz")["proba"].astype(float) for a in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)}
    return meta["rank"].map(RANK).to_numpy(), meta.n_games.to_numpy(), meta.player_id.to_numpy(dtype=str), M


def sc(P, y):
    return np.array([score(int(expected_score_decision(p)), int(t)) for p, t in zip(P, y)])


def temper(P, g):
    Q = P ** g; return Q / Q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    D = {v: load(v) for v in FOLDS}
    arith = {v: np.mean(list(D[v][3].values()), axis=0) for v in FOLDS}
    geo = {}
    for v in FOLDS:
        L = np.mean([np.log(np.clip(P, 1e-9, 1)) for P in D[v][3].values()], axis=0); G = np.exp(L - L.max(1, keepdims=True)); geo[v] = G / G.sum(1, keepdims=True)
    gammas = np.linspace(0.5, 2.0, 31); wgrid = np.linspace(0, 1, 21)
    out = {"base": [], "temp_n": [], "geo": [], "ctw": []}; ys, ns, pids = [], [], []; fitted = {}
    for v in FOLDS:
        y, n, pid, M = D[v]; ys.append(y); ns.append(n); pids.append(pid)
        others = [u for u in FOLDS if u != v]
        # (i) per-n temperature fitted on the other folds
        g_n = {}
        for m in (5, 10, 20):
            best = max(gammas, key=lambda g: np.mean([balanced_mean_over_n(sc(temper(arith[u], g), D[u][0]), D[u][0], D[u][1], sizes=(m,)) for u in others]))
            g_n[m] = float(best)
        Pt = arith[v].copy()
        for m in (5, 10, 20):
            Pt[n == m] = temper(arith[v][n == m], g_n[m])
        # (iii) C/T weight fitted on the other folds
        def ctmix(u, w):
            C = np.mean([P for (arm, s), P in D[u][3].items() if arm.startswith("C")], axis=0); T = np.mean([P for (arm, s), P in D[u][3].items() if arm.startswith("T")], axis=0)
            return w * C + (1 - w) * T
        w_best = float(max(wgrid, key=lambda w: np.mean([balanced_mean_over_n(sc(ctmix(u, w), D[u][0]), D[u][0], D[u][1]) for u in others])))
        fitted[str(v)] = {"gamma_n": g_n, "w_C": w_best}
        out["base"].append(sc(arith[v], y)); out["temp_n"].append(sc(Pt, y)); out["geo"].append(sc(geo[v], y)); out["ctw"].append(sc(ctmix(v, w_best), y))
        print(f"fold {v}: gamma_n {g_n}  w_C {w_best:.2f}  base {balanced_mean_over_n(out['base'][-1], y, n):.4f} temp_n {balanced_mean_over_n(out['temp_n'][-1], y, n):.4f} geo {balanced_mean_over_n(out['geo'][-1], y, n):.4f} ctw {balanced_mean_over_n(out['ctw'][-1], y, n):.4f}", flush=True)
    y, n, pid = map(np.concatenate, (ys, ns, pids)); S = {k: np.concatenate(v) for k, v in out.items()}
    rep = {"fitted": fitted, "scores": {k: balanced_mean_over_n(s, y, n) for k, s in S.items()}, "pairs": {}}
    for k in ("temp_n", "geo", "ctw"):
        d = paired_diff(S[k], S["base"], y, pid, n); rep["pairs"][f"{k} vs base"] = d
        print(f"{k:7s} vs arithmetic base: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  by n {d['by_n']}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
