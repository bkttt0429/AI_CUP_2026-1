"""Section 4M M47 (2026-11-07): the test set's class prior estimated without labels (Saerens et al. 2002 EM; the members were trained with
class-balanced weights, so the training prior is uniform), and the prior-adjusted expected-score decision.
  1. EM on each validation fold (labels unused) vs its true class shares.
  2. EM on ct9's 800 test posteriors (and Public / Private); references by simulation from the validation questions: 800 questions drawn
     class-balanced (a uniform-prior test) and 800 drawn at random (a validation-prior test); the L1 distance of each EM estimate to
     uniform, 1,000 draws each.
  3. On the validation folds, each fold's own EM prior adjusts ct9; the test-like plain average (plain mean within n = 5, 10, 20, then the
     mean over n) and the class-balanced score against ct9's decision, player bootstrap; seed sets 0-2 and 3-5.
  4. With --write: submission_rank_ct11.csv = ct9's test posteriors adjusted by the test EM prior.
  python -I logs/nano4/m47_prior.py --write
"""
import argparse
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


def em_prior(P, iters=500, tol=1e-10):
    pi = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(iters):
        Q = P * (pi * P.shape[1]); Q /= Q.sum(1, keepdims=True); new = Q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    Q = P * (pi * P.shape[1]); Q /= Q.sum(1, keepdims=True)
    return pi, Q


def nplain(s, n):
    return float(np.mean([s[n == m].mean() for m in np.unique(n)]))


def boot_players(s_new, s_old, n, pid, B=5000, seed=0):
    rng = np.random.default_rng(seed); players = np.unique(pid); idx = {p: np.flatnonzero(pid == p) for p in players}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(players, len(players))]); out.append(nplain(s_new[ii], n[ii]) - nplain(s_old[ii], n[ii]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def load(seeds):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in seeds], axis=0)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P}
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    rep = {}
    for seeds in ((0, 1, 2), (3, 4, 5)):
        D = load(seeds); tag = "".join(map(str, seeds)); r = {"folds": {}}
        S_new, S_old = {}, {}
        for v in FOLDS:
            pi, Q = em_prior(D[v]["P"]); true = np.bincount(D[v]["y"], minlength=10) / len(D[v]["y"])
            r["folds"][str(v)] = {"em": pi.tolist(), "true": true.tolist(), "l1": float(np.abs(pi - true).sum())}
            S_new[v], S_old[v] = qscore(decide(Q), D[v]["y"]), qscore(decide(D[v]["P"]), D[v]["y"])
        y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in ("y", "n", "pid"))
        sn, so = np.concatenate([S_new[v] for v in FOLDS]), np.concatenate([S_old[v] for v in FOLDS])
        d_plain = nplain(sn, n) - nplain(so, n); ci = boot_players(sn, so, n, pid)
        pf = [nplain(S_new[v], D[v]["n"]) - nplain(S_old[v], D[v]["n"]) for v in FOLDS]
        bal = float(np.mean([balanced_mean_over_n(S_new[v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(S_old[v], D[v]["y"], D[v]["n"]) for v in FOLDS]))
        ok = ci[0] > 0 and sum(e > 0 for e in pf) >= 3 and d_plain >= 0.0025
        r.update({"testlike_plain_diff": d_plain, "ci95": ci, "per_fold": pf, "class_balanced_diff": bal, "pass_new_rule_plain": bool(ok)})
        rep[tag] = r
        print(f"seeds {seeds}: EM L1 to the true fold prior " + " ".join(f"{r['folds'][str(v)]['l1']:.3f}" for v in FOLDS)
              + f"; prior-adjusted vs ct9: test-like plain {d_plain:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf)
              + f"; class-balanced {bal:+.4f}" + ("  -> PASS (new rule, plain)" if ok else "  -> no"), flush=True)
    D = load((0, 1, 2))
    Pv = np.concatenate([D[v]["P"] for v in FOLDS]); yv = np.concatenate([D[v]["y"] for v in FOLDS])
    val_prior = np.bincount(yv, minlength=10) / len(yv)
    print("validation true prior: " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, val_prior)))
    rng = np.random.default_rng(0); uni = np.full(10, 0.1)
    null_bal, null_nat = [], []
    by_class = [np.flatnonzero(yv == k) for k in range(10)]
    for _ in range(300):
        ib = np.concatenate([rng.choice(by_class[k], 80) for k in range(10)]); null_bal.append(np.abs(em_prior(Pv[ib])[0] - uni).sum())
        inat = rng.choice(len(yv), 800, replace=False); null_nat.append(np.abs(em_prior(Pv[inat])[0] - uni).sum())
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv"); cols = [f"p_{lv}" for lv in LEVELS]; Pt = pr[cols].to_numpy(float)
    pub = pr["question_id"].str.startswith("pub").to_numpy()
    out = {}
    for name, m in (("all", np.ones(len(Pt), bool)), ("public", pub), ("private", ~pub)):
        pi, _ = em_prior(Pt[m]); out[name] = pi
        print(f"test {name:7s} EM prior: " + " ".join(f"{lv} {x:.3f}" for lv, x in zip(LEVELS, pi)) + f"   L1 to uniform {np.abs(pi - uni).sum():.3f}, to the validation prior {np.abs(pi - val_prior).sum():.3f}")
    l1_test = float(np.abs(out["all"] - uni).sum())
    q_bal = float(np.percentile(null_bal, 95)); m_nat = float(np.median(null_nat))
    print(f"reference (800 simulated questions): L1 to uniform of the EM prior -- class-balanced draws median {np.median(null_bal):.3f}, 95% {q_bal:.3f}; "
          f"natural draws median {m_nat:.3f} [{np.percentile(null_nat, 5):.3f}, {np.percentile(null_nat, 95):.3f}]; test {l1_test:.3f}")
    rep["test"] = {k: v.tolist() for k, v in out.items()}; rep["test_l1_uniform"] = l1_test; rep["null_balanced_q95"] = q_bal; rep["null_natural_median"] = m_nat
    rep["test_prior_unbalanced"] = bool(l1_test > q_bal)
    Path(ROOT / "results/trunk_seq/es/m47_prior.json").write_text(json.dumps(rep, indent=1, default=float))
    if a.write:
        _, Q = em_prior(Pt)
        d9 = pd.read_csv(ROOT / "submission_rank_ct9.csv").set_index("question_id")["pred_rank"].reindex(pr["question_id"]).to_numpy()
        d10 = pd.read_csv(ROOT / "submission_rank_ct10.csv").set_index("question_id")["pred_rank"].reindex(pr["question_id"]).to_numpy()
        d11 = np.array([LEVELS[i] for i in decide(Q)])
        order = pd.read_csv(ROOT / "submission_rank_ct9.csv")["question_id"]
        pd.DataFrame({"question_id": pr["question_id"], "pred_rank": d11}).set_index("question_id").loc[order].reset_index().to_csv(ROOT / "submission_rank_ct11.csv", index=False)
        o = pd.DataFrame(Q, columns=cols); o["question_id"] = pr["question_id"]; o.to_csv(ROOT / "submission_rank_ct11_proba.csv", index=False)
        for nm, dd in (("ct9", d9), ("ct10", d10)):
            ch = d11 != dd; print(f"ct11 vs {nm}: {ch.sum()} of 800 differ (public {ch[pub].sum()}, private {ch[~pub].sum()})")
        print("ct11 predicted counts", pd.Series(d11).value_counts().reindex(LEVELS).to_dict())


if __name__ == "__main__":
    main()
