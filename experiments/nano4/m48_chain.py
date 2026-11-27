"""Section 4M M48 (2026-11-07; user: 類似 conv 的方式，每對相鄰等級都做二分類的專家): nine adjacent-pair logistic specialists chained into a
10-level posterior. Specialist k (levels k, k+1; trained on the validation questions of those two levels, cross-fitted over the folds) gives
logit_k = log P(k+1) / P(k); log Q(k) = sum_{j<k} logit_j, normalised. Weights 'bal' (the two levels equal) or 'nat' (natural shares, ct10's
way). Uses 'chain' (Q) and 'blend' (log P and log Q averaged). Expected-score decision; against ct9: class-balanced (primary) and the
test-like plain average (n-balanced); seed sets 0-2 and 3-5; per-boundary specialist AUC vs ct9's own log-odds.
  python -I logs/nano4/m48_chain.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m46_a1d import xprobs  # noqa: E402
from m47_prior import boot_players, nplain  # noqa: E402


def wlogit(X, y, w, Xte, iters=60):
    mu, sd = X.mean(0), X.std(0) + 1e-9; A = np.hstack([np.ones((len(X), 1)), (X - mu) / sd]); b = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ b)); H = A.T @ (A * (w * p * (1 - p))[:, None]) + 1e-4 * np.eye(A.shape[1]); step = np.linalg.solve(H, A.T @ (w * (p - y))); b -= step
        if np.abs(step).max() < 1e-8:
            break
    return np.hstack([np.ones((len(Xte), 1)), (Xte - mu) / sd]) @ b


def auc(s, lab):
    r = pd.Series(s).rank().to_numpy(); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(seeds):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in seeds], axis=0)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "X": xprobs(P, n)}
    out = {"auc": {}}
    for wk in ("bal", "nat"):
        L = {v: np.zeros((len(D[v]["y"]), 9)) for v in FOLDS}
        aucs = {k: ([], [], []) for k in range(9)}
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]
            for k in range(9):
                sel = lambda u: (D[u]["y"] == k) | (D[u]["y"] == k + 1)  # noqa: E731
                X = np.concatenate([D[u]["X"][sel(u)] for u in tr]); lab = np.concatenate([(D[u]["y"][sel(u)] == k + 1).astype(float) for u in tr])
                w = np.ones(len(lab)) if wk == "nat" else np.where(lab == 1, 0.5 / lab.mean(), 0.5 / (1 - lab.mean()))
                L[v][:, k] = wlogit(X, lab, w, D[v]["X"])
                m = sel(v); lp = np.log(np.clip(D[v]["P"][m], 1e-9, 1))
                aucs[k][0].append(L[v][m, k]); aucs[k][1].append(lp[:, k + 1] - lp[:, k]); aucs[k][2].append((D[v]["y"][m] == k + 1).astype(int))
        if wk == "bal":
            for k in range(9):
                s, b, lab = (np.concatenate(x) for x in aucs[k])
                out["auc"][f"{LEVELS[k]}|{LEVELS[k + 1]}"] = {"specialist": auc(s, lab), "ct9": auc(b, lab)}
        for use in ("chain", "blend"):
            Sn, So = {}, {}
            for v in FOLDS:
                lq = np.hstack([np.zeros((len(L[v]), 1)), np.cumsum(L[v], 1)]); lq -= lq.max(1, keepdims=True)
                lp = np.log(np.clip(D[v]["P"], 1e-9, 1))
                z = lq if use == "chain" else 0.5 * (lq + lp)
                Q = np.exp(z - z.max(1, keepdims=True)); Q /= Q.sum(1, keepdims=True)
                Sn[v], So[v] = qscore(decide(Q), D[v]["y"]), qscore(decide(D[v]["P"]), D[v]["y"])
            y, n, pid = (np.concatenate([D[v][kk] for v in FOLDS]) for kk in ("y", "n", "pid"))
            sn, so = np.concatenate([Sn[v] for v in FOLDS]), np.concatenate([So[v] for v in FOLDS])
            dd = paired_diff(sn, so, y, pid, n, n=3000)
            pf = [balanced_mean_over_n(Sn[v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(So[v], D[v]["y"], D[v]["n"]) for v in FOLDS]
            dpl = nplain(sn, n) - nplain(so, n)
            ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
            out[f"{wk}-{use}"] = {**dd, "per_fold": pf, "testlike_plain": dpl, "pass_new_rule": bool(ok)}
            print(f"seeds {seeds} {wk}-{use:5s}: class-balanced {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf)
                  + f"  test-like plain {dpl:+.4f}" + ("  -> PASS" if ok else "  -> no"), flush=True)
    return out


def main():
    rep = {"s012": run((0, 1, 2)), "s345": run((3, 4, 5))}
    print("per-boundary AUC (bal specialists, seeds 0-2): " + "  ".join(f"{k} {v['specialist']:.3f} vs {v['ct9']:.3f}" for k, v in rep["s012"]["auc"].items()))
    Path(ROOT / "results/trunk_seq/es/m48_chain.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
