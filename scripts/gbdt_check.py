#!/usr/bin/env python
"""XGBoost vs the MLP on the question features, and stacked on ens_top3 -- folds 1-4 CV (issue #6, 2026-10-26).

The questions, training questions and folds are scripts/l1b_cv.py's (the L3 ladder's): validation
pilot.pilot_questions per fold, training questions qfeatures.sample_questions --per-player per player with
RandomState(42 + v) over the training folds. Two feature sets: hsl (the 24 question features of the 18 Human SL
profiles) and fused (hsl + the 16 of the stage-1 L1b heads, --l1b). Two models per feature set and validation
fold: MLPModel(hidden=32) (the production model) and XGBoost with fixed hyperparameters (XGB_PARAMS, set before
any run; no tuning on the validation folds) and class-balanced sample weights (rankmodel.class_weights, the
MLP's rule). Stacking: a model's probabilities as a fourth member of ens_top3 (seed-averaged r3_pma /
r4_xattn_t1 / r5_ctm, uniform over members), the expected-score decision on the mean.

Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): xgb_hsl vs mlp_hsl,
and ens_top3 + xgb_hsl vs ens_top3. Reported only: the same two for fused, and ens_top3 + mlp_fused + xgb_fused.

  python -I scripts/gbdt_check.py --l1b results/l1b/scores results/l1b/scores_B --out results/gbdt/gbdt_check.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):  # `python -I` adds neither; scripts/ holds l1b_cv and l1b_stack_check
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel, class_weights  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_stack_check as stack  # noqa: E402

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=1.0, objective="multi:softprob", tree_method="hist", random_state=0)
PCOLS = [f"p{k}" for k in range(10)]
NEED = 3


def xgb_proba(X_tr, y_tr, X_va, n_jobs):
    """XGB_PARAMS through xgboost's native API (the sklearn wrapper would add scikit-learn as a dependency)."""
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    rounds = p.pop("n_estimators")
    params = {"eta": p.pop("learning_rate"), "seed": p.pop("random_state"), **p, "num_class": 10, "nthread": n_jobs}
    booster = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr, weight=class_weights(y_tr)), num_boost_round=rounds)
    return booster.predict(xgb.DMatrix(X_va)).reshape(len(X_va), 10)


def cv_probabilities(hsl_dir, l1b_dirs, folds, per_player, n_jobs, log=print):
    """DataFrame (fold, question_id, player_id, y, n_games) and {name: P [Q, 10]} for mlp/xgb x hsl/fused."""
    where = l1b_cv.locate_folds(l1b_dirs, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, where[f], f)
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "players": players, "val": l1b_cv.pilot_questions(players)}
    keys, probs = [], {f"{m}_{s}": [] for m in ("mlp", "xgb") for s in ("hsl", "fused")}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds) if f != v], ignore_index=True)
        X_tr, y_tr, _ = l1b_cv.feature_sets(pd.concat([data[f]["hsl"] for f in folds if f != v], ignore_index=True),
                                            pd.concat([data[f]["l1b"] for f in folds if f != v], ignore_index=True), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = l1b_cv.feature_sets(data[v]["hsl"], data[v]["l1b"], val)
        for s in ("hsl", "fused"):
            probs[f"mlp_{s}"].append(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]))
            probs[f"xgb_{s}"].append(xgb_proba(X_tr[s], y_tr, X_va[s], n_jobs))
        log(f"fold {v}: " + "  ".join(f"{k} {evaluate(p[-1], y_va, val)[0]['balanced_mean_over_n']:.4f}" for k, p in probs.items()), flush=True)
        keys.append(pd.DataFrame({"fold": v, "question_id": val["question_id"].to_numpy(), "player_id": val["player_id"].to_numpy(dtype=str),
                                  "y": y_va, "n_games": n_va}))
    return pd.concat(keys, ignore_index=True), {k: np.concatenate(p) for k, p in probs.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="the stage-1 L1b score directories (fused features)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--cv-dir", default=str(ROOT / "results" / "ladder" / "cv"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    keys, probs = cv_probabilities(a.hsl, a.l1b, folds, a.per_player, a.n_jobs)
    keys = keys.astype({"fold": int, "question_id": str})
    y, pid, n, fold = keys["y"].to_numpy(dtype=int), keys["player_id"].to_numpy(dtype=str), keys["n_games"].to_numpy(), keys["fold"].to_numpy()
    members = stack.member_probabilities(a.cv_dir)
    mem = {}
    for m in stack.MEMBERS:
        per_seed = []
        for s in stack.SEEDS:
            P, y_m = stack.align(members[m][s], keys)
            if not np.array_equal(y_m, y):
                raise ValueError(f"{m} seed {s}: labels differ")
            per_seed.append(P)
        mem[m] = np.mean(per_seed, axis=0)
    ens = [mem[m] for m in stack.MEMBERS]
    arms = {**probs, "ens_top3": np.mean(ens, axis=0),
            "ens_top3+xgb_hsl": np.mean(ens + [probs["xgb_hsl"]], axis=0), "ens_top3+mlp_hsl": np.mean(ens + [probs["mlp_hsl"]], axis=0),
            "ens_top3+xgb_fused": np.mean(ens + [probs["xgb_fused"]], axis=0), "ens_top3+mlp_fused": np.mean(ens + [probs["mlp_fused"]], axis=0),
            "ens_top3+mlp_fused+xgb_fused": np.mean(ens + [probs["mlp_fused"], probs["xgb_fused"]], axis=0)}
    scores = {k: evaluate(P, y, keys)[1] for k, P in arms.items()}
    nll = {k: evaluate(P, y, keys)[0]["nll_balanced"] for k, P in arms.items()}
    head = {k: {"balanced_mean_over_n": balanced_mean_over_n(s, y, n), "nll_balanced": nll[k],
                "balanced_by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},
                "per_fold": [balanced_mean_over_n(s[fold == v], y[fold == v], n[fold == v]) for v in folds]} for k, s in scores.items()}

    def compare(a_, b_):
        d = paired_diff(scores[a_], scores[b_], y, pid, n, n=a.n_boot)
        pf = [x - z for x, z in zip(head[a_]["per_fold"], head[b_]["per_fold"])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(x > 0 for x in pf) >= NEED), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    pre = [("xgb_hsl", "mlp_hsl"), ("ens_top3+xgb_hsl", "ens_top3")]
    rep = [("xgb_fused", "mlp_fused"), ("ens_top3+xgb_fused", "ens_top3"), ("ens_top3+mlp_fused+xgb_fused", "ens_top3"),
           ("ens_top3+xgb_fused", "ens_top3+mlp_fused")]
    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in pre}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in rep}}
    for k, h in head.items():
        print(f"{k:30s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for group, cs in comps.items():
        for k, c in cs.items():
            lo, hi = c["ci95"]
            print(f"[{group}] {k}: {c['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    import xgboost
    summary = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "xgb_params": XGB_PARAMS, "xgboost_version": xgboost.__version__,
               "l1b_dirs": a.l1b, "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(summary, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
