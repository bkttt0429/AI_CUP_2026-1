"""Section 4M M46 (2026-11-07; user: 「加進去試試看，給我上傳檔案」): the A|1D logistic specialist folded into ct9's posterior. Only questions
whose ct9 decision is A or 1D are touched: m = P(A) + P(1D) is kept, P'(1D) = m s, P'(A) = m (1 - s) with s the specialist's P(1D | A or 1D, x),
then the expected-score decision. Specialists: 'probs' (ct9's log P, entropy, top-2 margin, n_games: deployable on the test set) and 'full'
(+ the 64 R columns; control only). CV: cross-fitted on the four folds, seed sets 0-2 and 3-5, against ct9's decision. Then, with
--write, the 'probs' specialist trained on all four folds (seeds 0-2) applied to submission_rank_ct9_proba.csv -> submission_rank_ct10.csv.
  python -I logs/nano4/m46_a1d.py --write
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
from aicup.qeval import balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, load_fold  # noqa: E402
from m43_gate import FOLDS, decide, qscore  # noqa: E402
from m44_calib import logit_fit_auc  # noqa: E402

A, D1 = LEVELS.index("A"), LEVELS.index("1D")


def xprobs(P, n):
    lp = np.log(np.clip(P, 1e-9, 1)); ent = -(P * lp).sum(1, keepdims=True); s = np.sort(P, 1)
    return np.hstack([lp, ent, (s[:, -1] - s[:, -2])[:, None], n[:, None].astype(float)])


def fold_in(P, s):
    d = decide(P); route = (d == A) | (d == D1)
    Q = P.copy(); m = P[:, A] + P[:, D1]
    Q[route, D1] = m[route] * s[route]; Q[route, A] = m[route] * (1 - s[route])
    return Q, route


def load(seeds):
    D = {}
    for v in FOLDS:
        y, n, pid, q = load_fold(v)
        P = np.mean([np.load(ES / f"val{v}/{t}_s{s}.npz")["proba"].astype(np.float64) for t in ("C-pc-f0-x40-oh", "T-pc-f0-x40-oh") for s in seeds], axis=0)
        R = np.load(ROOT / f"results/trunk_seq/r_val{v}_f0_x40.npz")["X_va"].astype(np.float64)
        D[v] = {"y": y, "n": n, "pid": pid, "P": P, "probs": xprobs(P, n), "full": np.hstack([xprobs(P, n), R])}
    return D


def cv(seeds):
    D = load(seeds); out = {}
    for kind in ("probs", "full"):
        S_new, S_old, chg = {}, {}, 0
        for v in FOLDS:
            tr = [u for u in FOLDS if u != v]
            sel = lambda u: (D[u]["y"] == A) | (D[u]["y"] == D1)  # noqa: E731
            X = np.concatenate([D[u][kind][sel(u)] for u in tr]); lab = np.concatenate([(D[u]["y"][sel(u)] == D1).astype(float) for u in tr])
            s = logit_fit_auc(X, lab, D[v][kind], None)
            Q, route = fold_in(D[v]["P"], s)
            dn, do = decide(Q), decide(D[v]["P"]); chg += int((dn != do).sum())
            S_new[v], S_old[v] = qscore(dn, D[v]["y"]), qscore(do, D[v]["y"])
        y, n, pid = (np.concatenate([D[v][k] for v in FOLDS]) for k in ("y", "n", "pid"))
        sn, so = np.concatenate([S_new[v] for v in FOLDS]), np.concatenate([S_old[v] for v in FOLDS])
        dd = paired_diff(sn, so, y, pid, n, n=5000)
        pf = [balanced_mean_over_n(S_new[v], D[v]["y"], D[v]["n"]) - balanced_mean_over_n(S_old[v], D[v]["y"], D[v]["n"]) for v in FOLDS]
        ok = dd["ci95"][0] > 0 and sum(e > 0 for e in pf) >= 3 and dd["diff"] >= 0.0025
        out[kind] = {**dd, "per_fold_diff": pf, "plain_diff": float((sn - so).mean()), "changed": chg, "pass_new_rule": bool(ok)}
        print(f"seeds {seeds} {kind:5s}: class-balanced {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf)
              + f"  plain {out[kind]['plain_diff']:+.4f}  changed {chg} of {len(y)}" + ("  -> PASS (new rule)" if ok else "  -> no"), flush=True)
    return out


def write():
    D = load((0, 1, 2))
    sel = {v: (D[v]["y"] == A) | (D[v]["y"] == D1) for v in FOLDS}
    X = np.concatenate([D[v]["probs"][sel[v]] for v in FOLDS]); lab = np.concatenate([(D[v]["y"][sel[v]] == D1).astype(float) for v in FOLDS])
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv")
    cols = [f"p_{lv}" for lv in LEVELS]; P = pr[cols].to_numpy(float)
    ng = pd.concat([pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{s}.csv", usecols=["question_id", "num_games"]) for s in ("public", "private")]).set_index("question_id")["num_games"]
    n = ng.reindex(pr["question_id"]).to_numpy(float); assert np.isfinite(n).all()
    s = logit_fit_auc(X, lab, xprobs(P, n), None)
    Q, route = fold_in(P, s)
    d9 = pd.read_csv(ROOT / "submission_rank_ct9.csv").set_index("question_id")["pred_rank"].reindex(pr["question_id"]).to_numpy()
    assert (np.array([LEVELS[i] for i in decide(P)]) == d9).all(), "ct9's decisions are not reproduced from its probabilities"
    d10 = np.array([LEVELS[i] for i in decide(Q)])
    sub = pd.DataFrame({"question_id": pr["question_id"], "pred_rank": d10})
    order = pd.read_csv(ROOT / "submission_rank_ct9.csv")["question_id"]
    sub.set_index("question_id").loc[order].reset_index().to_csv(ROOT / "submission_rank_ct10.csv", index=False)
    out = pd.DataFrame(Q, columns=cols); out["question_id"] = pr["question_id"]; out.to_csv(ROOT / "submission_rank_ct10_proba.csv", index=False)
    pub = pr["question_id"].str.startswith("pub").to_numpy(); ch = d10 != d9
    print(f"ct10 written: routed {route.sum()} questions (ct9 decision A or 1D), {ch.sum()} of 800 decisions differ from ct9 (public {ch[pub].sum()}, private {ch[~pub].sum()})")
    print("changes: " + ", ".join(f"{a}->{b} {c}" for (a, b), c in pd.Series(list(zip(d9[ch], d10[ch]))).value_counts().items()))
    print("ct10 predicted counts", pd.Series(d10).value_counts().reindex(LEVELS).to_dict())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    rep = {"s012": cv((0, 1, 2)), "s345": cv((3, 4, 5))}
    Path(ROOT / "results/trunk_seq/es/m46_a1d.json").write_text(json.dumps(rep, indent=1, default=float))
    if a.write:
        write()


if __name__ == "__main__":
    main()
