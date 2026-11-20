"""Section 4M M75 (2026-11-10; user: 討論決策模型的方法有效嗎): how good is the expected-score (EV) decision on the current ensemble?
Out-of-fold probabilities of the standard 6 members (seeds 0-2; 6-8 as a second set), folds 1-4. Report only.
 1 decision rules on the same posterior: argmax, EV (in use), posterior median, rounded posterior mean
 2 where EV departs from argmax: how often, and what it earns there
 3 calibration of the decision model: the expected score it predicts for its own choice against the score it gets, by decile
 4 ceilings: the better of the posterior's two most likely levels; the truth among the top 1 / 2 / 3
 5 a LEARNED decision layer: multinomial logistic map from [log p, log n] trained on the realised score itself (cross-fitted),
   against the EV rule on the raw posterior -- does direct optimisation of the score beat the Bayes rule on this posterior?
  python -I logs/nano4/m75_decision.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

STD = ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh"); K = np.arange(10)
S = np.where(K[:, None] == K[None, :], 1.0, np.where(np.abs(K[:, None] - K[None, :]) == 1, np.exp(-1), 0.0))  # S[decision, truth]
sc = lambda d, y: S[d, y]  # noqa: E731


def learned(Xtr, ytr, Xte):
    Xtr, Xte = torch.tensor(Xtr), torch.tensor(Xte); y = torch.tensor(ytr); St = torch.tensor(S)
    W = torch.zeros(Xtr.shape[1], 10, dtype=torch.float64); W[:10] = torch.eye(10, dtype=torch.float64); W.requires_grad_(True); b = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    cw = (1.0 / torch.bincount(y, minlength=10).double())[y]; cw = cw / cw.sum()
    opt = torch.optim.LBFGS([W, b], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); p = torch.softmax(Xtr @ W + b, dim=1); loss = -(cw * (p * St[:, y].T).sum(1)).sum() + 1e-4 * ((W[:10] - torch.eye(10, dtype=torch.float64)) ** 2).sum(); loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return (Xte @ W + b).argmax(1).numpy()


rep = {}
for seeds in ((0, 1, 2), (6, 7, 8)):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in STD for s in seeds], axis=0); D[v] = (y, n, pid, P)
    y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in (0, 1, 2)); P = np.vstack([D[v][3] for v in FOLDS])
    ev_val = P @ S.T; d_ev = ev_val.argmax(1); d_arg = P.argmax(1); d_med = (P.cumsum(1) >= 0.5).argmax(1); d_mean = np.clip(np.rint(P @ K), 0, 9).astype(int)
    bal = lambda s, sizes=SIZES: float(balanced_mean_over_n(s, y, n, sizes=sizes))  # noqa: E731
    r = {"rules": {}}
    print(f"seeds {seeds}: {len(y)} validation questions")
    for name, d in (("argmax", d_arg), ("EV (in use)", d_ev), ("posterior median", d_med), ("rounded mean", d_mean)):
        s = sc(d, y); r["rules"][name] = {"score": bal(s), "exact": bal((d == y).astype(float)), "within1": bal((np.abs(d - y) <= 1).astype(float)), "by_n": {m: bal(s, (m,)) for m in SIZES}}
        print(f"  1 {name:17s} score {r['rules'][name]['score']:.4f}  exact {r['rules'][name]['exact']:.4f}  within one {r['rules'][name]['within1']:.4f}  by n " + " ".join(f"{m}: {val:.4f}" for m, val in r["rules"][name]["by_n"].items()))
    dd = paired_diff(sc(d_ev, y), sc(d_arg, y), y, pid, n, sizes=SIZES, n=3000); ch = d_ev != d_arg
    r["ev_vs_argmax"] = {"diff": dd["diff"], "ci95": dd["ci95"], "share_differ": float(ch.mean()), "score_ev_there": float(sc(d_ev, y)[ch].mean()), "score_argmax_there": float(sc(d_arg, y)[ch].mean()),
                         "ev_exact_there": float((d_ev == y)[ch].mean()), "argmax_exact_there": float((d_arg == y)[ch].mean())}
    e = r["ev_vs_argmax"]; print(f"  2 EV vs argmax {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]; they differ on {e['share_differ']:.3f} of questions; there EV scores {e['score_ev_there']:.3f} (exact {e['ev_exact_there']:.3f}) vs argmax {e['score_argmax_there']:.3f} (exact {e['argmax_exact_there']:.3f})")
    pred = ev_val.max(1); got = sc(d_ev, y); w = np.zeros(len(y))
    for c in range(10):
        for m in SIZES:
            k_ = (y == c) & (n == m); w[k_] = 1.0 / max(k_.sum(), 1)
    w /= w.sum(); order = np.argsort(pred); cuts = np.searchsorted(np.cumsum(w[order]), np.linspace(0, 1, 11)[1:-1]); bins = np.split(order, cuts)
    cal = [(float(np.average(pred[b], weights=w[b])), float(np.average(got[b], weights=w[b]))) for b in bins]
    r["calibration"] = {"predicted": float(np.average(pred, weights=w)), "realised": float(np.average(got, weights=w)), "deciles": cal}
    print(f"  3 the decision model predicts an expected score of {r['calibration']['predicted']:.4f} for its own choices and gets {r['calibration']['realised']:.4f}; by decile (predicted -> realised): " + " ".join(f"{a:.2f}->{b:.2f}" for a, b in cal))
    top = np.argsort(-P, axis=1)[:, :3]; best2 = np.maximum(sc(top[:, 0], y), sc(top[:, 1], y))
    r["ceilings"] = {"best_of_top2": bal(best2), "truth_in_top1": bal((top[:, 0] == y).astype(float)), "truth_in_top2": bal((top[:, :2] == y[:, None]).any(1).astype(float)), "truth_in_top3": bal((top == y[:, None]).any(1).astype(float))}
    c = r["ceilings"]; print(f"  4 ceilings: a perfect chooser between the two most likely levels would score {c['best_of_top2']:.4f}; the truth is the most likely level {c['truth_in_top1']:.3f}, among the top two {c['truth_in_top2']:.3f}, top three {c['truth_in_top3']:.3f}")
    s_l = []
    for v in FOLDS:
        tr = [u for u in FOLDS if u != v]; X = lambda u: np.hstack([np.log(np.clip(D[u][3], 1e-6, 1)), np.log(D[u][1])[:, None] - 2.3])  # noqa: E731
        s_l.append(sc(learned(np.vstack([X(u) for u in tr]), np.concatenate([D[u][0] for u in tr]), X(v)), D[v][0]))
    dl = paired_diff(np.concatenate(s_l), sc(d_ev, y), y, pid, n, sizes=SIZES, n=3000)
    pf = [float(balanced_mean_over_n(a, D[v][0], D[v][1], sizes=SIZES) - balanced_mean_over_n(sc(d_ev, y)[np.concatenate([np.full(len(D[u][0]), u) for u in FOLDS]) == v], D[v][0], D[v][1], sizes=SIZES)) for a, v in zip(s_l, FOLDS)]
    r["learned_vs_ev"] = {**dl, "per_fold": pf}
    print(f"  5 learned decision layer (trained on the score itself) vs the EV rule {dl['diff']:+.4f} [{dl['ci95'][0]:+.4f}, {dl['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in pf))
    rep[f"seeds_{seeds[0]}"] = r
Path(ES / "m75_decision.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")
