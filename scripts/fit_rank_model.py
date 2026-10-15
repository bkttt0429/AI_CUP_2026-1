#!/usr/bin/env python
"""Choose the L3/L4 model by leave-one-fold-out CV on the training folds, refit it on all of
them, then evaluate every candidate once on the evaluation fold.

Questions: training uses random subsets of each player's scored games (sizes 5-20,
`--per-player` per player); validation and evaluation use the nested 1x20 / 2x10 / 4x5
carvings. The primary metric is the class-balanced official score (mean of per-rank means)
averaged over question sizes 5, 10 and 20; the question-mean score is reported too.
Writes results/rank_model_eval.json and results/models/rank_model.json.
"""
import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.evidence import bootstrap_ci  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.qfeatures import FEATURE_NAMES, build_matrix, sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel, SoftmaxModel, ThresholdModel  # noqa: E402
from aicup.scoring import fold_players, player_games  # noqa: E402

SIZES = (5, 10, 20)
CANDIDATES = {
    "threshold_pmi": lambda: ThresholdModel(feature=FEATURE_NAMES.index("pmi_sum")),
    "threshold_peak": lambda: ThresholdModel(feature=FEATURE_NAMES.index("peak_interp")),
    "softmax_l2_1e-4": lambda: SoftmaxModel(l2=1e-4),
    "softmax_l2_1e-3": lambda: SoftmaxModel(l2=1e-3),
    "softmax_l2_1e-2": lambda: SoftmaxModel(l2=1e-2),
    "mlp_h32": lambda: MLPModel(hidden=32),
    "mlp_h64": lambda: MLPModel(hidden=64),
}


def load_fold(fold, all_games, folds_df, scores_dir):
    paths = sorted(glob.glob(str(Path(scores_dir) / f"fold{fold}" / "games_s*_p*.csv")))
    if not paths:
        raise FileNotFoundError(f"no scored parts for fold {fold}")
    games = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    players = fold_players(all_games, folds_df, fold)
    want = {k for ks in players["game_keys"] for k in ks}
    have = set(games["game_key"])
    if want != have:
        raise ValueError(f"fold {fold}: {len(have)} games scored but {len(want)} selected (missing {len(want - have)})")
    if games.isna().any().any():
        raise ValueError(f"fold {fold}: NaN in scored games")
    return games, players


def evaluate(proba, y, qs):
    pred = np.array([expected_score_decision(p) for p in proba])
    s = np.array([score(p, t) for p, t in zip(pred, y)])
    n = qs["n_games"].to_numpy()

    def balanced(mask):
        return float(np.mean([s[mask & (y == k)].mean() for k in range(10) if (mask & (y == k)).any()]))

    out = {"balanced_by_n": {int(m): balanced(n == m) for m in SIZES}}
    out["balanced_mean_over_n"] = float(np.mean(list(out["balanced_by_n"].values())))
    out["question_mean"] = float(s.mean())
    out["per_class"] = {LEVELS[k]: float(s[y == k].mean()) for k in range(10)}
    out["exact"] = float((pred == y).mean())
    out["within1"] = float((np.abs(pred - y) <= 1).mean())
    conf = np.zeros((10, 10), dtype=int)
    np.add.at(conf, (y, pred), 1)
    out["confusion_rows_truth_cols_pred"] = conf.tolist()
    return out, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--eval-fold", type=int, default=0)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--per-player", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scores", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--out", default=str(ROOT / "results" / "rank_model_eval.json"))
    ap.add_argument("--model-out", default=str(ROOT / "results" / "models" / "rank_model.json"))
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    all_games = player_games()
    folds_df = pd.read_csv(ROOT / "splits" / "player_folds.csv")
    data = {}
    for f in a.train_folds + ([] if a.no_eval else [a.eval_fold]):
        games, players = load_fold(f, all_games, folds_df, a.scores)
        qs = pilot_questions(players)
        X, y, _ = build_matrix(games, qs)
        data[f] = {"games": games, "players": players, "qs": qs, "X": X, "y": y}
        print(f"fold {f}: {len(players)} players, {len(games)} games, {len(qs)} eval questions", flush=True)

    def train_set(train_folds, seed):
        rng = np.random.RandomState(seed)
        Xs, ys = [], []
        for f in train_folds:
            sq = sample_questions(data[f]["players"], a.per_player, rng)
            X, y, _ = build_matrix(data[f]["games"], sq)
            Xs.append(X)
            ys.append(y)
        return np.vstack(Xs), np.concatenate(ys)

    cv = {name: [] for name in CANDIDATES}
    for v in a.train_folds:
        X_tr, y_tr = train_set([f for f in a.train_folds if f != v], a.seed + v)
        for name, make in CANDIDATES.items():
            res, _ = evaluate(make().fit(X_tr, y_tr).predict_proba(data[v]["X"]), data[v]["y"], data[v]["qs"])
            cv[name].append(res)
        print(f"CV held-out fold {v}: " + "  ".join(f"{n} {cv[n][-1]['balanced_mean_over_n']:.4f}" for n in CANDIDATES), flush=True)
    cv_summary = {name: {"balanced_mean_over_n": float(np.mean([r["balanced_mean_over_n"] for r in rs])),
                         "balanced_by_n": {m: float(np.mean([r["balanced_by_n"][m] for r in rs])) for m in SIZES},
                         "per_fold": [r["balanced_mean_over_n"] for r in rs]}
                  for name, rs in cv.items()}
    best = max(cv_summary, key=lambda n: cv_summary[n]["balanced_mean_over_n"])
    print(f"CV choice: {best} ({cv_summary[best]['balanced_mean_over_n']:.4f})", flush=True)

    X_all, y_all = train_set(a.train_folds, a.seed)
    final = {name: make().fit(X_all, y_all) for name, make in CANDIDATES.items()}
    out = {"created": datetime.now().isoformat(timespec="seconds"), "train_folds": a.train_folds, "eval_fold": None if a.no_eval else a.eval_fold,
           "per_player": a.per_player, "seed": a.seed, "features": FEATURE_NAMES, "profiles": PROFILES,
           "train_questions": int(len(y_all)), "cv": cv_summary, "choice": best}
    if not a.no_eval:
        e = data[a.eval_fold]
        out["eval"] = {}
        for name, model in final.items():
            res, s = evaluate(model.predict_proba(e["X"]), e["y"], e["qs"])
            if name == best:
                res["balanced_ci95"] = list(bootstrap_ci(s, e["qs"]["player_id"].to_numpy(), e["y"], balanced=True))
                res["question_mean_ci95"] = list(bootstrap_ci(s, e["qs"]["player_id"].to_numpy(), e["y"]))
            out["eval"][name] = res
        r = out["eval"][best]
        print(f"fold {a.eval_fold} ({best}): balanced {r['balanced_mean_over_n']:.4f} CI {r['balanced_ci95']}, "
              f"by n {r['balanced_by_n']}, question mean {r['question_mean']:.4f}", flush=True)
        for name in CANDIDATES:
            print(f"  {name:16s} CV {cv_summary[name]['balanced_mean_over_n']:.4f}  fold{a.eval_fold} {out['eval'][name]['balanced_mean_over_n']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.model_out).write_text(json.dumps({"name": best, "features": FEATURE_NAMES, "profiles": PROFILES,
                                             "train_folds": a.train_folds, "cv_balanced_mean_over_n": cv_summary[best]["balanced_mean_over_n"],
                                             "state": final[best].state_dict()}))
    print(f"wrote {a.out} and {a.model_out}")


if __name__ == "__main__":
    main()
