#!/usr/bin/env python
"""Seen-player check of the official baseline (METHOD_NOTES §2.1, §4) -> results/baseline_seen_players_eval.json.

Three training runs of scripts/baseline_train.py, each scored by scripts/baseline_eval.py on the
same fold-0 questions (aicup.baseline.eval_questions, set 'fold0'):
  base     unseen protocol, seed 42                         models/baseline_official
  seen     + every fold-0 game that no question asks         models/baseline_seen (baseline_extract.py --seen-fold 0)
  control  unseen protocol, seed 43                         models/baseline_official_s43
seen - base is the gain from knowing the evaluated players through their other games (plus 25%
more training data); control - base is what a retrain alone moves, the noise floor. Both are
paired on the same questions and restricted to the fold-0 players whose games entered training
('gained'); the evaluated players who gained none ('not_gained', exactly 5/10/15/20 games) see
only the extra data. Reported per decision rule: the class-balanced official score, mean over
5/10/20-game questions, with player-bootstrap CIs; per game, the log-probability and accuracy
of the true rank; and the change by how many of a player's games were added.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.baseline import decide, eval_questions, mean_probs  # noqa: E402
from aicup.evidence import bootstrap_ci, bootstrap_ci_mean_over_sizes  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX, score  # noqa: E402
from aicup.scoring import player_games  # noqa: E402

SIZES = (5, 10, 20)
RULES = {"argmax_mean_softmax": "argmax", "expected_score_mean_softmax": "expected_score"}
RUNS = ("base", "seen", "control")
PAIRS = {"seen_minus_base": ("seen", "base"), "control_minus_base": ("control", "base"),
         "seen_minus_control": ("seen", "control")}
BUCKETS = [(1, 4), (5, 49), (50, 199), (200, None)]   # games of the player added to training


def balanced_over_sizes(v, y, n, sizes=SIZES):
    """Mean over the question sizes present of the class-balanced mean of v (aicup.baseline.report's headline)."""
    return float(np.mean([np.mean([v[(n == m) & (y == k)].mean() for k in np.unique(y[n == m])])
                          for m in sizes if (n == m).any()]))


def paired(new, old, y, n, players, sizes=SIZES):
    """new - old on the same questions: the headline statistic of the difference with its player-bootstrap CI, and per size."""
    d = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    keep = np.isin(n, sizes)
    return {"delta": balanced_over_sizes(d, y, n, sizes),
            "ci95": list(bootstrap_ci_mean_over_sizes(d[keep], players[keep], y[keep], n[keep])),
            "by_n": {m: {"delta": balanced_over_sizes(d, y, n, (m,)),
                         "ci95": list(bootstrap_ci(d[n == m], players[n == m], y[n == m], balanced=True))}
                     for m in sizes if (n == m).any()}}


def true_logp(table, floor=1e-6):
    """log p(true rank) per game of a game_probs table (game_key 'RANK:game_id', p_<rank> columns rounded to 6 decimals)."""
    idx = table["game_key"].str.split(":").str[0].map(RANK_TO_IDX).to_numpy()
    p = table[[f"p_{lv}" for lv in LEVELS]].to_numpy()[np.arange(len(table)), idx]
    return pd.Series(np.log(np.maximum(p, floor)), index=table["game_key"].to_numpy())


def eval_summary(probs_dir):
    """Training and fold-0 headline of a run, from the baseline_eval.py JSON next to its probabilities."""
    path = Path(probs_dir).with_name(Path(probs_dir).name + "_eval.json")
    if not path.exists():
        return {"eval": None}
    r = json.loads(path.read_text())
    run = (r["training"] or {}).get("run") or {}
    return {"eval": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "checkpoint": r["checkpoint"], "seed": run.get("args", {}).get("seed"),
            "train_samples_per_epoch": run.get("train_samples_per_epoch"),
            "best_epoch": run.get("best_epoch"), "best_val_acc": run.get("best_val_acc"),
            "game_accuracy_fold0": r["game_accuracy_fold0"],
            "fold0_balanced_mean_over_n": {label: r["eval_sets"]["fold0"][label]["balanced_mean_over_n"] for label in RULES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "results" / "baseline_official"), help="dir of game_probs_fold0.csv")
    ap.add_argument("--seen", default=str(ROOT / "results" / "baseline_seen"))
    ap.add_argument("--control", default=str(ROOT / "results" / "baseline_official_s43"))
    ap.add_argument("--seen-features", default=str(ROOT / "data" / "baseline_seen"))
    ap.add_argument("--out", default=str(ROOT / "results" / "baseline_seen_players_eval.json"))
    a = ap.parse_args()

    folds = pd.read_csv(ROOT / "splits" / "player_folds.csv")
    qs = eval_questions(player_games(), folds, pd.read_csv(ROOT / "splits" / "questions_fold0.csv"))["fold0"]
    asked = {k for ks in qs["game_keys"] for k in ks}
    split = json.loads((Path(a.seen_features) / "split.json").read_text())
    train = pd.read_csv(Path(a.seen_features) / "train_games.csv")
    added = train[train["fold"] == 0]
    if split.get("seen_fold") != 0 or added.empty or added["game_key"].isin(asked).any():
        raise SystemExit(f"{a.seen_features} is not a --seen-fold 0 extraction without question games")
    gained = qs["player_id"].map(added.groupby("player_id").size()).fillna(0).astype(int).to_numpy()

    tables = {run: pd.read_csv(Path(getattr(a, run)) / "game_probs_fold0.csv") for run in RUNS}
    probs = {run: dict(zip(t["game_key"], t[[f"p_{lv}" for lv in LEVELS]].to_numpy())) for run, t in tables.items()}
    logp = {run: true_logp(t) for run, t in tables.items()}
    y_all, n_all, pl_all = qs["rank"].map(RANK_TO_IDX).to_numpy(), qs["n_games"].to_numpy(), qs["player_id"].to_numpy()
    scores = {rule: {run: np.array([score(p, t) for p, t in zip(decide(mean_probs(probs[run], qs["game_keys"]), rule), y_all)])
                     for run in RUNS} for rule in RULES.values()}

    def per_game(mask):
        owner = {k: p for p, ks in zip(pl_all[mask], qs["game_keys"][mask]) for k in ks}
        keys = sorted(owner)
        players, y = np.array([owner[k] for k in keys]), np.array([RANK_TO_IDX[k.split(":")[0]] for k in keys])
        lp = {run: logp[run].reindex(keys).to_numpy() for run in RUNS}
        acc = {run: np.array([float(np.argmax(probs[run][k]) == t) for k, t in zip(keys, y)]) for run in RUNS}
        out = {"games": len(keys), "mean_logp_true": {run: float(lp[run].mean()) for run in RUNS},
               "accuracy": {run: float(acc[run].mean()) for run in RUNS}}
        for name, (new, old) in PAIRS.items():
            out[name] = {"logp_delta": float((lp[new] - lp[old]).mean()),
                         "logp_delta_ci95": list(bootstrap_ci(lp[new] - lp[old], players, y)),
                         "accuracy_delta": float((acc[new] - acc[old]).mean()),
                         "accuracy_delta_ci95": list(bootstrap_ci(acc[new] - acc[old], players, y))}
        return out

    def block(mask, balanced=True):
        y, n, pl = y_all[mask], n_all[mask], pl_all[mask]
        out = {"players": int(len(set(pl))), "questions": int(mask.sum()),
               "questions_by_n": {m: int((n == m).sum()) for m in SIZES},
               "gained_games": {"median": float(np.median(gained[mask])), "min": int(gained[mask].min()),
                                "max": int(gained[mask].max())}}
        for label, rule in RULES.items():
            s = {run: scores[rule][run][mask] for run in RUNS}
            if balanced:
                out[label] = {"balanced_mean_over_n": {run: balanced_over_sizes(s[run], y, n) for run in RUNS},
                              **{name: paired(s[new], s[old], y, n, pl) for name, (new, old) in PAIRS.items()}}
            else:   # few players per class and size: the question mean, players resampled within rank
                out[label] = {"question_mean": {run: float(s[run].mean()) for run in RUNS},
                              **{name: {"delta": float((s[new] - s[old]).mean()),
                                        "ci95": list(bootstrap_ci(s[new] - s[old], pl, y))}
                                 for name, (new, old) in PAIRS.items()}}
        out["per_game"] = per_game(mask)
        return out

    result = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "what": "seen-player check of the official baseline: fold-0 questions scored by a model that also trained on "
                "the evaluated players' other games, against the unseen protocol and a seed-43 retrain of it",
        "runs": {run: {"probs": str(Path(getattr(a, run)) / "game_probs_fold0.csv"), **eval_summary(getattr(a, run))}
                 for run in RUNS},
        "seen_training": {"features": str(a.seen_features), **split["seen"],
                          "train_games_total": split["train"]["games"], "val_games": split["val"]["games"]},
        "question_set": "fold0 of scripts/baseline_eval.py: every fold-0 player with >= 5 games, up to 20 of them, "
                        "carved 1x20 / 2x10 / 4x5",
        "statistics": "official score 1 / e^-1 / 0; class-balanced mean over 5/10/20-game questions; deltas paired on the "
                      "same questions; 95% CIs resample players within rank (1,000 replicates, seed 42)",
        "gained": block(gained > 0),
        "not_gained": block(gained == 0),
        "by_gained_games": {f"{lo}-{hi}" if hi else f"{lo}+": block((gained >= lo) & (gained <= (hi or gained.max())), balanced=False)
                            for lo, hi in BUCKETS},
    }
    Path(a.out).write_text(json.dumps(result, indent=1))

    for name in ("gained", "not_gained"):
        b = result[name]
        print(f"{name}: {b['players']} players, {b['questions']} questions, median {b['gained_games']['median']:.0f} games added")
        for label in RULES:
            r = b[label]
            lv = "  ".join(f"{run} {r['balanced_mean_over_n'][run]:.4f}" for run in RUNS)
            ds = "  ".join(f"{p} {r[p]['delta']:+.4f} [{r[p]['ci95'][0]:+.3f}, {r[p]['ci95'][1]:+.3f}]" for p in PAIRS)
            print(f"  {label}: {lv} | {ds}")
        g = b["per_game"]
        print("  per game logp(true): " + "  ".join(f"{run} {g['mean_logp_true'][run]:.4f}" for run in RUNS)
              + " | " + "  ".join(f"{p} {g[p]['logp_delta']:+.4f}" for p in PAIRS))
    for bucket, b in result["by_gained_games"].items():
        r = b["argmax_mean_softmax"]
        print(f"added {bucket}: {b['players']} players, {b['questions']} questions | argmax question mean seen-base "
              f"{r['seen_minus_base']['delta']:+.4f} control-base {r['control_minus_base']['delta']:+.4f} | logp seen-base "
              f"{b['per_game']['seen_minus_base']['logp_delta']:+.4f} control-base {b['per_game']['control_minus_base']['logp_delta']:+.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
