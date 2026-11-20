"""ct8 = a model's rank probabilities fused with the question-level identification of the player among the training players
(M24; the rule was pre-registered in reports/architecture_v2_2026-11-02.md section 4M before the gate numbers were seen).
Calibration on ident_pool's gate outputs, per n group (5-7 / 8-12 / 13-17 / 18-20 <-> test n 5 / 10 / 15 / 20), feature
x = log1p(fused margin): logit P(top-1 correct | x) on A (in-pool); logit P(in pool | x) on A (1) vs B (0) at equal priors,
shifted by log(p / (1 - p)) for the prevalence p of the n group. a_q = P(in) P(correct), capped at 0.98;
P_mix = (1 - a_q) P_model + a_q onehot(label of the matched player); decision = expected-score argmax of P_mix.
Gate: the 4,000 inner held-out questions with the ct3 ensemble (12 production models) as P_model: Delta A_n (in-pool) and
Delta B_n (out-of-pool) of the rule against the plain decision; expected test gain = 1/4 sum_n [p_n Delta A_n + (1 - p_n) Delta B_n]
with p_n the mixture estimates of ident_pool.json; paired player bootstrap. Test: ct7 probabilities -> submission_rank_ct8.csv.
  python -I logs/nano4/ident_ct8.py --ident results/r_variants/ident_pool.json --out results/r_variants/ident_ct8.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(next((sys.argv[i + 1] for i, x in enumerate(sys.argv[:-1]) if x == "--root"), Path.cwd())).resolve()
for _p in (HERE, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402


def find_input(root, rel, name):
    for q in (Path(root) / rel / name, Path(root) / name, HERE / "inputs" / name, HERE / name):
        if q.exists():
            return q
    raise FileNotFoundError(f"{name}: not under {root}/{rel}, {root}, or {HERE}/inputs")

RANK = {lv: i for i, lv in enumerate(LEVELS)}; K = len(LEVELS)
NGROUP = {5: range(5, 8), 10: range(8, 13), 15: range(13, 18), 20: range(18, 21)}
P_COLS = [f"p_{lv}" for lv in LEVELS]
CAP = 0.98


def group_of(n):
    g = np.zeros(len(n), int)
    for k, r in NGROUP.items():
        g[np.isin(n, list(r))] = k
    return g


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit_fit(x, y, l2=1e-3, iters=200):
    """y ~ sigmoid(b0 + b1 x) by Newton's method with a tiny ridge; returns [b0, b1]."""
    X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(X @ b); W = p * (1 - p)
        g = X.T @ (y - p) - l2 * b; H = -(X.T * W) @ X - l2 * np.eye(2)
        step = np.linalg.solve(H, g); b = b - step
        if np.abs(step).max() < 1e-9:
            break
    return b


def calibrate(gate):
    g = group_of(gate["n"]); xA, xB = np.log1p(gate["A_margin"]), np.log1p(gate["B_margin"]); cal = {}
    for k in NGROUP:
        sel = g == k
        cal[k] = {"correct": logit_fit(xA[sel], gate["A_correct"][sel].astype(float)).tolist(),
                  "inpool": logit_fit(np.concatenate([xA[sel], xB[sel]]), np.concatenate([np.ones(sel.sum()), np.zeros(sel.sum())])).tolist()}
    return cal


def rule(P_model, margin, label_top1, n, prior, cal):
    g = group_of(n); a = np.zeros(len(n)); x = np.log1p(margin)
    for k in NGROUP:
        sel = g == k
        if not sel.any():
            continue
        p = float(np.clip(prior[k], 0.01, 0.99)); c = cal[k]
        pc = sigmoid(c["correct"][0] + c["correct"][1] * x[sel]); pin = sigmoid(c["inpool"][0] + c["inpool"][1] * x[sel] + np.log(p / (1 - p)))
        a[sel] = np.minimum(pc * pin, CAP)
    Pm = (1 - a)[:, None] * P_model + a[:, None] * np.eye(K)[label_top1]
    return np.array([expected_score_decision(p) for p in Pm]), a, Pm


def scores(dec, y):
    return np.array([score(int(d), int(t)) for d, t in zip(dec, y)])


def group_mean(v, g):
    return {k: float(v[g == k].mean()) for k in NGROUP if (g == k).any()}


def boot(diff, g, pid, reps=2000, seed=0):
    """Player bootstrap of the mean over n groups of the group means of diff."""
    rng = np.random.default_rng(seed); players, inv = np.unique(pid, return_inverse=True); out = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(players), len(players)), minlength=len(players))[inv].astype(float)
        vals = [np.sum(w[g == k] * diff[g == k]) / max(np.sum(w[g == k]), 1e-9) for k in NGROUP if (g == k).any()]
        out.append(np.mean(vals))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ident", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT)); ap.add_argument("--gate-probs", default=None); ap.add_argument("--ct7", default=None); ap.add_argument("--ct7-sub", default=None); ap.add_argument("--submission", default=None)
    a = ap.parse_args(); rep = {}; root = Path(a.root)
    a.gate_probs = a.gate_probs or str(find_input(root, "results/r_variants", "prod_heldout.P.npz")); a.ct7 = a.ct7 or str(find_input(root, ".", "submission_rank_ct7_proba.csv"))
    a.ct7_sub = a.ct7_sub or str(find_input(root, ".", "submission_rank_ct7.csv")); a.submission = a.submission or str(root / "submission_rank_ct8.csv")
    ident = json.loads(Path(a.ident).read_text()); gate = np.load(Path(a.ident).with_suffix(".gate.npz"), allow_pickle=True); test = np.load(Path(a.ident).with_suffix(".test.npz"), allow_pickle=True)
    prior = {int(k): (float(v) if v is not None else 0.1) for k, v in ident["test"]["mixture_in_pool_share_by_n"].items()}
    rep["prior_in_pool_by_n"] = prior
    cal = calibrate(gate); rep["calibration"] = {str(k): v for k, v in cal.items()}
    # ---- gate: ct3 ensemble as the model
    z = np.load(a.gate_probs, allow_pickle=True); P3 = np.mean([z[k] for k in z.files if k.startswith("ct3__")], axis=0)
    y, n, pid = gate["y"].astype(int), gate["n"].astype(int), gate["pid"]; g = group_of(n)
    assert np.array_equal(z["y"].astype(int), y) and np.array_equal(z["n"].astype(int), n), "gate question order differs"
    base_dec = np.array([expected_score_decision(p) for p in P3]); s_base = scores(base_dec, y)
    decA, aA, _ = rule(P3, gate["A_margin"], gate["A_top1_label"].astype(int), n, prior, cal); sA = scores(decA, y)
    decB, aB, _ = rule(P3, gate["B_margin"], gate["B_top1_label"].astype(int), n, prior, cal); sB = scores(decB, y)
    dA, dB = sA - s_base, sB - s_base
    pri = np.array([prior[k] for k in g]); comb = pri * dA + (1 - pri) * dB
    rep["gate"] = {"base_score_by_n": group_mean(s_base, g), "deltaA_by_n": group_mean(dA, g), "deltaB_by_n": group_mean(dB, g),
                   "expected_test_gain": float(np.mean(list(group_mean(comb, g).values()))), "expected_test_gain_ci95": boot(comb, g, pid),
                   "deltaA_mean": float(np.mean(list(group_mean(dA, g).values()))), "deltaA_ci95": boot(dA, g, pid),
                   "deltaB_mean": float(np.mean(list(group_mean(dB, g).values()))), "deltaB_ci95": boot(dB, g, pid),
                   "A_changed_share": float((decA != base_dec).mean()), "B_changed_share": float((decB != base_dec).mean()),
                   "A_a_mean": float(aA.mean()), "B_a_mean": float(aB.mean()), "A_share_a_gt_0.5": float((aA > 0.5).mean()), "B_share_a_gt_0.5": float((aB > 0.5).mean()),
                   "A_precision_when_a_gt_0.5": float(gate["A_correct"][aA > 0.5].mean()) if (aA > 0.5).any() else None}
    # oracle references: the rule when the status is known (prior 1 on A, prior 0.01 on B)
    decA1, _, _ = rule(P3, gate["A_margin"], gate["A_top1_label"].astype(int), n, {k: 0.99 for k in NGROUP}, cal)
    rep["gate"]["deltaA_if_known_in_pool"] = float(np.mean(list(group_mean(scores(decA1, y) - s_base, g).values())))
    # reliability of a on A: bins
    bins = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]); idx = np.digitize(aA, bins) - 1
    rep["gate"]["A_reliability"] = [{"a_range": [float(bins[i]), float(bins[i + 1])], "questions": int((idx == i).sum()), "mean_a": float(aA[idx == i].mean()) if (idx == i).any() else None,
                                     "correct_rate": float(gate["A_correct"][idx == i].mean()) if (idx == i).any() else None} for i in range(len(bins) - 1)]
    print("gate (ct3 as the model): base by n", {k: round(v, 4) for k, v in rep["gate"]["base_score_by_n"].items()})
    print("  Delta A (in-pool) by n", {k: round(v, 4) for k, v in rep["gate"]["deltaA_by_n"].items()}, "mean %.4f CI %s" % (rep["gate"]["deltaA_mean"], np.round(rep["gate"]["deltaA_ci95"], 4).tolist()))
    print("  Delta B (out-of-pool) by n", {k: round(v, 4) for k, v in rep["gate"]["deltaB_by_n"].items()}, "mean %.4f CI %s" % (rep["gate"]["deltaB_mean"], np.round(rep["gate"]["deltaB_ci95"], 4).tolist()))
    print("  expected test gain %.4f CI %s; A changed %.3f (a>0.5: %.3f, precision %s); B changed %.3f (a>0.5: %.3f); Delta A if known in pool %.4f" % (
        rep["gate"]["expected_test_gain"], np.round(rep["gate"]["expected_test_gain_ci95"], 4).tolist(), rep["gate"]["A_changed_share"], rep["gate"]["A_share_a_gt_0.5"], rep["gate"]["A_precision_when_a_gt_0.5"],
        rep["gate"]["B_changed_share"], rep["gate"]["B_share_a_gt_0.5"], rep["gate"]["deltaA_if_known_in_pool"]))
    print("  reliability on A:", [(r["a_range"], r["questions"], None if r["correct_rate"] is None else round(r["correct_rate"], 3)) for r in rep["gate"]["A_reliability"]])
    passed = rep["gate"]["expected_test_gain"] >= 0.005 and all(v > -0.002 for v in rep["gate"]["deltaB_by_n"].values()) and rep["gate"]["deltaA_ci95"][0] > 0
    rep["gate"]["passed_preregistered_gate"] = bool(passed)
    # ---- test: ct7 probabilities
    p7 = pd.read_csv(a.ct7).set_index("question_id"); qid = test["qid"]; P7 = p7.loc[qid, P_COLS].to_numpy(dtype=float)
    sub7 = pd.read_csv(a.ct7_sub, dtype=str); ct7 = sub7.set_index("question_id")["pred_rank"].reindex(qid).map(RANK).to_numpy()
    dec8, a8, _ = rule(P7, test["margin"], test["top1_label"].astype(int), test["n"].astype(int), prior, cal)
    dec7 = np.array([expected_score_decision(p) for p in P7])
    rep["test"] = {"questions": int(len(qid)), "ct7_decision_reproduced": float((dec7 == ct7).mean()), "changed_vs_ct7": int((dec8 != ct7).sum()),
                   "changed_by_n": {int(k): int(((dec8 != ct7) & (test["n"] == k)).sum()) for k in NGROUP}, "a_mean": float(a8.mean()), "share_a_gt_0.5": float((a8 > 0.5).mean()),
                   "share_a_gt_0.9": float((a8 > 0.9).mean()), "matched_label_equals_ct7_when_a_gt_0.5": float((test["top1_label"][a8 > 0.5] == ct7[a8 > 0.5]).mean()) if (a8 > 0.5).any() else None,
                   "changed": [{"question_id": str(q), "n": int(nn), "ct7": LEVELS[int(c7)], "ct8": LEVELS[int(c8)], "a": round(float(aa), 3), "matched_label": LEVELS[int(ml)], "matched_player": str(mp), "margin": round(float(m), 2)}
                               for q, nn, c7, c8, aa, ml, mp, m in zip(qid, test["n"], ct7, dec8, a8, test["top1_label"], test["top1_player"], test["margin"]) if c7 != c8]}
    out = sub7.copy(); m8 = dict(zip(qid, [LEVELS[int(d)] for d in dec8])); out["pred_rank"] = out["question_id"].map(m8)
    assert out.pred_rank.notna().all() and len(out) == len(sub7)
    out.to_csv(a.submission, index=False)
    print("test: ct7 decision reproduced %.3f; ct8 changes %d of %d (by n %s); a>0.5 %.3f, a>0.9 %.3f; matched label = ct7 when a>0.5: %s; gate %s; written %s" % (
        rep["test"]["ct7_decision_reproduced"], rep["test"]["changed_vs_ct7"], len(qid), rep["test"]["changed_by_n"], rep["test"]["share_a_gt_0.5"], rep["test"]["share_a_gt_0.9"],
        rep["test"]["matched_label_equals_ct7_when_a_gt_0.5"], "PASSED" if passed else "not passed", a.submission))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
