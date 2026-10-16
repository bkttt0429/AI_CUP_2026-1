"""The official AI CUP 2026 rank baseline (Tutorial rank-prediction.ipynb) under the project protocol.

The parser and the network are the Tutorial's own files, imported by path and never edited,
so features and model are the official ones. Only the data split is ours (METHOD_NOTES §2.1):
training and validation players come from folds 1-4, whole players at a time, and fold 0 is
used for evaluation only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from aicup.evidence import bootstrap_ci, bootstrap_ci_mean_over_sizes
from aicup.metric import LEVELS, expected_score_decision, score

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_DIR = ROOT / "repo" / "AICup-2026-Tutorial"
_TUTORIAL = {}
_PARSER = None


def load_tutorial(tutorial_dir=TUTORIAL_DIR):
    """(SGFParseRankPrediction, GoRankResNet) from the Tutorial checkout, loaded once per process."""
    key = str(Path(tutorial_dir).resolve())
    if key not in _TUTORIAL:
        mods = []
        for name in ("utils", "network"):
            spec = importlib.util.spec_from_file_location(f"aicup_tutorial_{name}", Path(tutorial_dir) / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            mods.append(mod)
        _TUTORIAL[key] = (mods[0].SGFParseRankPrediction, mods[1].GoRankResNet)
    return _TUTORIAL[key]


def game_features(sgf):
    """The notebook's sgf_to_features: uint8 [500, 19, 19] for one game, None if it does not parse."""
    global _PARSER
    if _PARSER is None:
        _PARSER = load_tutorial()[0]()
    feat, err = _PARSER.features_from_sgf_content(sgf)
    return None if err is not None else feat


def split_players(folds, train_folds=(1, 2, 3, 4), val_games_per_rank=10_000, seed=42):
    """(train_ids, val_ids), both sorted: whole players of `train_folds`, validation >= target games per rank.

    Ranks go in LEVELS order; a rank's players are sorted by id, permuted with one shared
    RandomState(seed) and taken in that order until their games reach `val_games_per_rank`.
    Players of other folds are in neither list.
    """
    pool = folds[folds["fold"].isin(list(train_folds))]
    rng = np.random.RandomState(seed)
    val = []
    for rank in [r for r in LEVELS if r in set(pool["rank"])]:
        cand = pool[pool["rank"] == rank].sort_values("player_id")
        ids, games = cand["player_id"].to_numpy(), cand["n_games"].to_numpy()
        taken = 0
        for i in rng.permutation(len(cand)):
            if taken >= val_games_per_rank:
                break
            val.append(ids[i])
            taken += int(games[i])
        if taken >= games.sum():
            raise ValueError(f"rank {rank}: validation would take all {int(games.sum())} training games")
    val_set = set(val)
    return sorted(p for p in pool["player_id"] if p not in val_set), sorted(val)


def eval_questions(games, folds, questions, fold=0):
    """The question sets of scripts/baseline_eval.py on one fold, carved 1x20 / 2x10 / 4x5 by pilot.pilot_questions.

    'fold{fold}': every player of the fold with >= 5 games, up to 20 of them (scoring.fold_players
    on `games`, the rows of scoring.player_games); 'pilot_fold{fold}': the 200-player pilot set
    of METHOD_NOTES §2.4.1 (pilot.select_players on `questions`, splits/questions_fold{fold}.csv).
    """
    from aicup.pilot import pilot_questions, select_players
    from aicup.scoring import fold_players

    return {f"fold{fold}": pilot_questions(fold_players(games, folds, fold)),
            f"pilot_fold{fold}": pilot_questions(select_players(questions, fold))}


def mean_probs(game_probs, questions):
    """[Q, 10]: per question, the mean softmax over its games that have one (the notebook's vote); NaN if none."""
    rows = []
    for keys in questions:
        ps = [game_probs[k] for k in keys if k in game_probs]
        rows.append(np.mean(ps, axis=0) if ps else np.full(len(LEVELS), np.nan))
    return np.array(rows, dtype=float).reshape(-1, len(LEVELS))


def decide(probs, rule):
    """Class per row: 'argmax' (the notebook) or 'expected_score' (metric.expected_score_decision).

    A NaN row (no game of the question parsed) gets class 0, the notebook's LEVELS[0] answer.
    """
    if rule not in ("argmax", "expected_score"):
        raise ValueError(f"unknown decision rule {rule!r}")
    out = []
    for p in np.asarray(probs, dtype=float):
        if np.isnan(p).any():
            out.append(0)
        else:
            out.append(int(np.argmax(p)) if rule == "argmax" else expected_score_decision(p))
    return np.array(out, dtype=int)


def report(pred, y, qs, sizes=(5, 10, 20)):
    """Official scores of question decisions, with the keys of results/rank_model_eval.json plus `by_n` details.

    The headline `balanced_mean_over_n` gets the CI of that same statistic (one player resample
    per replicate shared by every size), as scripts/fit_rank_model.py reports it; `by_n` and
    `balanced_by_n_ci95` hold each size on its own. CIs resample players within each rank.
    """
    pred, y = np.asarray(pred, dtype=int), np.asarray(y, dtype=int)
    s = np.array([score(p, t) for p, t in zip(pred, y)])
    players, n = qs["player_id"].to_numpy(), qs["n_games"].to_numpy()
    classes = range(len(LEVELS))

    def balanced(mask):
        return float(np.mean([s[mask & (y == k)].mean() for k in classes if (mask & (y == k)).any()]))

    def summary(mask):
        return {"exact": float((pred[mask] == y[mask]).mean()),
                "within1": float((np.abs(pred[mask] - y[mask]) <= 1).mean()),
                "per_class": {LEVELS[k]: float(s[mask & (y == k)].mean()) for k in classes if (mask & (y == k)).any()}}

    by_n = {}
    for m in sizes:
        mask = n == m
        by_n[m] = {"balanced": balanced(mask),
                   "balanced_ci95": list(bootstrap_ci(s[mask], players[mask], y[mask], balanced=True)),
                   "question_mean": float(s[mask].mean()),
                   "question_mean_ci95": list(bootstrap_ci(s[mask], players[mask], y[mask])),
                   "n_questions": int(mask.sum()), **summary(mask)}
    conf = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    np.add.at(conf, (y, pred), 1)
    keep = np.isin(n, sizes)
    return {"balanced_by_n": {m: by_n[m]["balanced"] for m in sizes},
            "balanced_by_n_ci95": {m: by_n[m]["balanced_ci95"] for m in sizes},
            "balanced_mean_over_n": float(np.mean([by_n[m]["balanced"] for m in sizes])),
            "balanced_mean_over_n_ci95": list(bootstrap_ci_mean_over_sizes(s[keep], players[keep], y[keep], n[keep])),
            "question_mean": float(s.mean()),
            "question_mean_ci95": list(bootstrap_ci(s, players, y)),
            **summary(np.ones(len(s), dtype=bool)),
            "confusion_rows_truth_cols_pred": conf.tolist(),
            "by_n": by_n}
