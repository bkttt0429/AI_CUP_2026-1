"""Post-hoc levers on the stored C+T 12-model CV probabilities (folds 1-4, results/trunk_seq/es/val{v}/{C,T}_s{0..5}.npz), no retraining:
  (1) temperature scaling fitted on the other three folds (by NLL and by official score);
  (2) prior-shift EM (Saerens et al. 2002) on pseudo test sets;
  (3) balanced assignment: exactly Q/10 questions per class (linear_sum_assignment on the expected-score matrix) on balanced
      800-question pseudo tests, against the independent expected-score decision; and its harm on NATURAL (training-prior) pseudo tests;
  (4) EM-estimated counts + assignment (adaptive), and oracle counts (upper bound);
  (5) averaging with the stored MLP0 probabilities (results/endgame/mlp_val{v}.npz, same validation questions).
Scores: plain mean (what a leaderboard computes) and the class-balanced mean over n (our CV statistic)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n, nll_brier, paired_diff  # noqa: E402

ES_DIR = ROOT / "results/trunk_seq/es"
RANK = {lv: i for i, lv in enumerate(LEVELS)}
S = np.array([[score(k, t) for t in range(10)] for k in range(10)])  # S[pred, true]
FOLDS = (1, 2, 3, 4)
N_DRAWS, Q = 50, 800


def load(v):
    m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]
    y, n, pid = np.array([RANK[r] for r in m["rank"]]), np.array(m["n_games"]), np.array(m["player_id"], dtype=str)
    P = np.mean([np.load(ES_DIR / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in "CT" for s in range(6)], axis=0)
    M = np.load(ROOT / f"results/endgame/mlp_val{v}.npz")["MLP0"].astype(np.float64)
    assert P.shape == M.shape == (len(y), 10)
    return y, n, pid, P, M


def es_decide(P):
    return np.argmax(P @ S.T, axis=1)


def scores_of(pred, y):
    return S[pred, y]


def temper(P, T):
    L = np.log(np.clip(P, 1e-12, None)) / T
    L -= L.max(axis=1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(axis=1, keepdims=True)


def em_prior(P, iters=100, prior0=None):
    """Saerens-style EM for the test prior, the model's own prior taken as uniform (class-balanced training)."""
    pi0 = np.full(10, 0.1) if prior0 is None else prior0
    pi = pi0.copy()
    for _ in range(iters):
        W = P * (pi / pi0)
        W /= W.sum(axis=1, keepdims=True)
        new = W.mean(axis=0)
        if np.abs(new - pi).max() < 1e-8:
            pi = new
            break
        pi = new
    W = P * (pi / pi0)
    return pi, W / W.sum(axis=1, keepdims=True)


def assign(P, counts):
    """Predictions maximising the total expected score with exactly counts[k] questions in class k (sum = len(P))."""
    cols = np.repeat(np.arange(10), counts)
    assert len(cols) == len(P)
    E = P @ S.T  # [Q, 10] expected score of predicting k
    rows, picked = linear_sum_assignment(-E[:, cols])
    pred = np.empty(len(P), dtype=int)
    pred[rows] = cols[picked]
    return pred


def rounded_counts(pi, total):
    raw = pi * total
    c = np.floor(raw).astype(int)
    rem = total - c.sum()
    c[np.argsort(-(raw - c))[:rem]] += 1
    return c


data = {v: load(v) for v in FOLDS}
rep = {"base": {}, "temperature": {}, "pseudo": {}, "mlp_mix": {}}
# ---- base reproduction
for v in FOLDS:
    y, n, pid, P, M = data[v]
    pred = es_decide(P)
    assert np.array_equal(pred[:50], [expected_score_decision(p) for p in P[:50]])
    rep["base"][v] = {"ct12": balanced_mean_over_n(scores_of(pred, y), y, n), "mlp0": balanced_mean_over_n(scores_of(es_decide(M), y), y, n),
                      "pmax_lt_0.4_share": float((P.max(axis=1) < 0.4).mean()), "nll_ct12": float(nll_brier(P, y)[0].mean())}
print("base per fold", {v: {k: round(x, 4) for k, x in d.items()} for v, d in rep["base"].items()})
print("CT12 mean over folds", round(np.mean([rep["base"][v]["ct12"] for v in FOLDS]), 4), "MLP0", round(np.mean([rep["base"][v]["mlp0"] for v in FOLDS]), 4))

# ---- (1) temperature, nested
grid = np.round(np.arange(0.5, 3.01, 0.05), 2)
for v in FOLDS:
    others = [f for f in FOLDS if f != v]
    nll_T, sc_T = {}, {}
    for T in grid:
        nll_T[T] = np.mean([nll_brier(temper(data[f][3], T), data[f][0])[0].mean() for f in others])
        sc_T[T] = np.mean([balanced_mean_over_n(scores_of(es_decide(temper(data[f][3], T)), data[f][0]), data[f][0], data[f][1]) for f in others])
    T_nll, T_sc = min(nll_T, key=nll_T.get), max(sc_T, key=sc_T.get)
    y, n, pid, P, M = data[v]
    base = balanced_mean_over_n(scores_of(es_decide(P), y), y, n)
    rep["temperature"][v] = {"T_by_nll": float(T_nll), "score_T_nll": balanced_mean_over_n(scores_of(es_decide(temper(P, T_nll)), y), y, n) - base,
                             "T_by_score": float(T_sc), "score_T_score": balanced_mean_over_n(scores_of(es_decide(temper(P, T_sc)), y), y, n) - base}
print("temperature (nested, delta vs base per fold)", {v: {k: (round(x, 4) if isinstance(x, float) else x) for k, x in d.items()} for v, d in rep["temperature"].items()})

# ---- (2)-(4) pseudo tests
rng = np.random.RandomState(0)
methods = ("indep", "em", "assign_uniform", "assign_em", "assign_oracle")
for design in ("balanced", "natural"):
    acc = {m: {"plain": [], "balanced": []} for m in methods}
    prior_err = []
    for v in FOLDS:
        y, n, pid, P, M = data[v]
        for _ in range(N_DRAWS):
            if design == "balanced":
                idx = np.concatenate([rng.choice(np.flatnonzero(y == k), Q // 10, replace=False) for k in range(10)])
            else:
                idx = rng.choice(len(y), Q, replace=False)
            Pq, yq, nq = P[idx], y[idx], n[idx]
            true_counts = np.bincount(yq, minlength=10)
            pi, W = em_prior(Pq)
            prior_err.append(float(np.abs(pi - true_counts / Q).sum()))
            preds = {"indep": es_decide(Pq), "em": es_decide(W), "assign_uniform": assign(Pq, np.full(10, Q // 10)),
                     "assign_em": assign(Pq, rounded_counts(pi, Q)), "assign_oracle": assign(Pq, true_counts)}
            for m, pr in preds.items():
                s = scores_of(pr, yq)
                acc[m]["plain"].append(float(s.mean()))
                acc[m]["balanced"].append(balanced_mean_over_n(s, yq, nq))
    out = {}
    for m in methods:
        out[m] = {k: {"mean": float(np.mean(acc[m][k])), "sd": float(np.std(acc[m][k]))} for k in ("plain", "balanced")}
        if m != "indep":
            for k in ("plain", "balanced"):
                d = np.array(acc[m][k]) - np.array(acc["indep"][k])
                out[m][f"delta_{k}"] = {"mean": float(d.mean()), "sd": float(d.std()), "share_positive": float((d > 0).mean()), "min": float(d.min()), "max": float(d.max())}
    out["em_prior_L1_error_mean"] = float(np.mean(prior_err))
    rep["pseudo"][design] = out
    print(f"== pseudo tests, {design} design ({len(FOLDS) * N_DRAWS} draws of {Q} questions)")
    for m in methods:
        line = f"  {m:15s} plain {out[m]['plain']['mean']:.4f}  balanced {out[m]['balanced']['mean']:.4f}"
        if m != "indep":
            dp, db = out[m]["delta_plain"], out[m]["delta_balanced"]
            line += f"  | delta plain {dp['mean']:+.4f} (sd {dp['sd']:.4f}, >0 {dp['share_positive']:.2f}, min {dp['min']:+.4f})  balanced {db['mean']:+.4f} (sd {db['sd']:.4f})"
        print(line)
    print(f"  EM prior L1 error {out['em_prior_L1_error_mean']:.3f}")

# ---- (5) MLP0 mixing
ws = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
per_w = {w: [] for w in ws}
S_ct, S_mix, ys, pids, ns = [], [], [], [], []
for v in FOLDS:
    y, n, pid, P, M = data[v]
    for w in ws:
        per_w[w].append(balanced_mean_over_n(scores_of(es_decide(w * P + (1 - w) * M), y), y, n))
    S_ct.append(scores_of(es_decide(P), y))
    S_mix.append(scores_of(es_decide(0.8 * P + 0.2 * M), y))
    ys.append(y); pids.append(pid); ns.append(n)
rep["mlp_mix"] = {"curve": {str(w): float(np.mean(per_w[w])) for w in ws},
                  "w0.8_vs_ct12": paired_diff(np.concatenate(S_mix), np.concatenate(S_ct), np.concatenate(ys), np.concatenate(pids), np.concatenate(ns), n=2000)}
print("MLP0 mix curve (w on CT12)", {k: round(x, 4) for k, x in rep["mlp_mix"]["curve"].items()}, "w=0.8 vs CT12", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in rep["mlp_mix"]["w0.8_vs_ct12"].items() if k != "by_n"})
(ES_DIR / "posthoc_levers.json").write_text(json.dumps(rep, indent=1, default=float))
print("written", ES_DIR / "posthoc_levers.json")
