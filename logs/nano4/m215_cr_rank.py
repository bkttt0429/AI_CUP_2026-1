#!/usr/bin/env python
"""M215 (2026-11-17), probe: does the choice residual carry RANK information beyond the base ensemble?

Per game, from results/m197_choice64 (67 columns per stored ply: r64 + 3 policy scalars, int8 x20), the question player's own
plies (t < 256): the sums of the 67 columns over all plies and over three phases (t < 60, 60-150, >= 150), the sums of
squares, the ply counts. Per question: the pooled means (all plies; three phases) and the pooled standard deviation.
  small = the 67 means over all plies; full = means + 3 phase means + sd = 335 columns.
Evaluation as m74b_quant.py: (1) players' 20-game questions: partial correlation of the cross-fitted ridge composite with the
label given the ensemble's expected level E; (2) question level: multinomial logistic stack on [ensemble log-probabilities,
log n, features] against the same stack without the features, cross-fitted over the validation folds, the M37 rule
(CI lower bound > 0, >= 3 of 4 folds, difference >= 0.0025) on seeds 0-2 and 6-8.

  python -I logs/nano4/m215_cr_rank.py
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402
from m74b_quant import stack  # noqa: E402

CH = ROOT / "results/m197_choice64"; STD = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"); N_CR = 67; NS = 5 * N_CR + 4  # sums: all, 3 phases, squares; counts: all, 3 phases
NEED = set()


def read_part(path):
    """{game key: [NS] sufficient statistics of the subject's stored plies} for the needed games of one part."""
    with np.load(path, allow_pickle=False) as z:
        key = z["key"].astype(str); hit = np.flatnonzero(np.isin(key, list(NEED))) if len(NEED) else np.arange(0)
        if not len(hit):
            return {}
        OFF, T, MOV, SUBJ, XQ = z["OFF"], z["T"], z["MOV"], z["SUBJ"], z["XQ"]
    out = {}
    for g in hit:
        a, e = int(OFF[g]), int(OFF[g + 1]); own = MOV[a:e] == SUBJ[g]; x = XQ[a:e][own].astype(np.float32) / 20.0; t = T[a:e][own].astype(np.int64); s = np.zeros(NS)
        if len(x):
            ph = np.where(t < 60, 0, np.where(t < 150, 1, 2)); s[:N_CR] = x.sum(0); s[4 * N_CR:5 * N_CR] = (x * x).sum(0); s[5 * N_CR] = len(x)
            for k in range(3):
                m = ph == k; s[(1 + k) * N_CR:(2 + k) * N_CR] = x[m].sum(0); s[5 * N_CR + 1 + k] = m.sum()
        out[str(key[g])] = s
    return out


def features(S):
    """[Q, NS] summed statistics -> (small [Q, 67], full [Q, 335])."""
    n = np.maximum(S[:, 5 * N_CR:5 * N_CR + 1], 1); mean = S[:, :N_CR] / n; sd = np.sqrt(np.maximum(S[:, 4 * N_CR:5 * N_CR] / n - mean ** 2, 1e-9))
    ph = [S[:, (1 + k) * N_CR:(2 + k) * N_CR] / np.maximum(S[:, 5 * N_CR + 1 + k:5 * N_CR + 2 + k], 1) for k in range(3)]
    return mean, np.hstack([mean] + ph + [sd])


def main():
    data = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v); data[v] = [y, n, pid, q]; NEED.update(k for gk in q["game_keys"] for k in gk)
    files = sorted(CH.glob("part_s*of*_p*.npz")); print(f"{len(NEED)} games of the validation questions; {len(files)} residual parts", flush=True); per = {}
    with mp.get_context("fork").Pool(12) as pool:
        for d in pool.imap_unordered(read_part, files, chunksize=32):
            per.update(d)
    miss = len(NEED) - len(per); print(f"games found in the residual store: {len(per)} of {len(NEED)} (missing {miss}); mean stored own plies per game {np.mean([s[5 * N_CR] for s in per.values()]):.1f}", flush=True)
    if miss > 0.01 * len(NEED):
        sys.exit("more than 1% of the validation games have no residual rows - keys do not line up")
    zero = np.zeros(NS)
    for v in FOLDS:
        q = data[v][3]; S = np.stack([np.sum([per.get(k, zero) for k in gk], axis=0) for gk in q["game_keys"]]); data[v] += list(features(S))
    rep = {"games": len(per), "missing": miss}
    for name, col, lam in (("small (67 means)", 4, 10.0), ("full (335)", 5, 100.0)):
        comp, ys, Es = [], [], []
        for v in FOLDS:
            y, n, F = data[v][0], data[v][1], data[v][col]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in (0, 1, 2)], axis=0); E = P @ np.arange(10)
            tr = [data[u] for u in FOLDS if u != v]; Xtr = np.vstack([t[col][t[1] == 20] for t in tr]); ytr = np.concatenate([t[0][t[1] == 20] for t in tr])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z = (Xtr - mu) / sd; w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (ytr - ytr.mean()))
            m = n == 20; comp.append(((F[m] - mu) / sd) @ w); ys.append(y[m]); Es.append(E[m])
        c, yy, ee = np.concatenate(comp), np.concatenate(ys).astype(float), np.concatenate(Es)
        res = lambda a: a - np.polyval(np.polyfit(ee, a, 1), ee)  # noqa: E731
        rng = np.random.default_rng(0); rc, ry = res(c), res(yy); r = float(np.corrcoef(rc, ry)[0, 1]); bs = [np.corrcoef(rc[i], ry[i])[0, 1] for i in (rng.integers(0, len(c), len(c)) for _ in range(1000))]
        raw = float(np.corrcoef(np.argsort(np.argsort(c)), np.argsort(np.argsort(yy)))[0, 1]); rep[f"partial {name}"] = {"r": r, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "spearman_with_label": raw}
        print(f"(1) {name:17s}: partial r of the composite with the label given E {r:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] ({len(c)} players; composite against the label, Spearman {raw:.3f})", flush=True)
    ok = {}
    for name, col in (("small (67 means)", 4), ("full (335)", 5)):
        ok[name] = []
        for seeds in ((0, 1, 2), (6, 7, 8)):
            X0, Y = {}, {}
            for v in FOLDS:
                y, n = data[v][0], data[v][1]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in seeds], axis=0); X0[v] = np.hstack([np.log(np.clip(P, 1e-6, 1)), np.log(n)[:, None] - 2.3]); Y[v] = y
            sn, so, pf, ys, ns, pids = [], [], [], [], [], []
            for v in FOLDS:
                y, n, pid, q = data[v][:4]; tr = [u for u in FOLDS if u != v]; F_tr = np.vstack([data[u][col] for u in tr]); mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-9
                A0, B0 = np.vstack([X0[u] for u in tr]), X0[v]; A1, B1 = np.hstack([A0, (F_tr - mu) / sd]), np.hstack([B0, (data[v][col] - mu) / sd]); ytr = np.concatenate([Y[u] for u in tr])
                p0, p1 = stack(A0, ytr, B0, 10), stack(A1, ytr, B1, 10); s0, s1 = evaluate(p0, y, q, sizes=SIZES)[1], evaluate(p1, y, q, sizes=SIZES)[1]
                sn.append(s1); so.append(s0); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(s1, y, n, sizes=SIZES) - balanced_mean_over_n(s0, y, n, sizes=SIZES))
            y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); d = paired_diff(np.concatenate(sn), np.concatenate(so), y, pid, n, sizes=SIZES, n=5000)
            good = (d["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (d["diff"] >= 0.0025); ok[name].append(bool(good)); rep[f"stack {name} seeds {seeds[0]}"] = {**d, "per_fold": pf, "pass": bool(good)}
            print(f"(2) {name:17s} seeds {seeds}: stack with the features against the same stack without {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in d["by_n"].items()) + f"  -> {'PASS' if good else 'no'}", flush=True)
    rep["pass_small"] = bool(all(ok["small (67 means)"])); print(f"REGISTERED (primary = small): the choice residual carries rank information beyond the base ensemble: {'YES - go on to the member-level test' if rep['pass_small'] else 'NO - closed'}")
    (ROOT / "results/r_variants/m215_cr_rank.json").write_text(json.dumps(rep, indent=1, default=float)); print("written results/r_variants/m215_cr_rank.json")


if __name__ == "__main__":
    main()
