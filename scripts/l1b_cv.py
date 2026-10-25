#!/usr/bin/env python
"""Folds 1-4 leave-one-fold-out CV of question features from Human SL, from the L1b heads, and fused.

Plan 2026-10-25-l1b-rank-heads, Task 7 (decision point 2). The protocol is the L3 ladder's: validation
questions pilot.pilot_questions (1 x 20, 2 x 10, 4 x 5 games per player), training questions
qfeatures.sample_questions with --per-player per player and RandomState(42 + v) over the training folds
in ascending order, one deterministic MLPModel(hidden=32) fit per feature set and validation fold,
scores pooled over the four validation folds, player bootstrap (n = --n-boot) for the paired
differences. Feature sets: hsl = the 24 question features of the 18 Human SL profiles (the production
model's), l1b = the 16 of the 10 L1b heads, fused = both (40). The two pre-specified comparisons are
l1b vs hsl and fused vs hsl; conditions as the ladder's: (1) CI lower bound > 0, (3) the point difference
> 0 in at least len(folds) - 1 folds, (4) point difference >= 0.005; (2) (every seed > 0) does not apply
to a deterministic fit and is reported as "n/a".

--l1b takes one or more l1b_score.py directories (model A's and model B's); every fold must be in exactly
one of them, and its games must be the same as --hsl's.

  python -I scripts/l1b_cv.py --l1b results/l1b/scores results/l1b/scores_B --hsl results/scores --folds 1 2 3 4 --out results/l1b/cv_summary.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.humansl import PROFILES  # noqa: E402
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import build_matrix, feature_names, sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import completed_parts, read_scored_games  # noqa: E402

FEATURE_SETS = ("hsl", "l1b", "fused")
COMPARISONS = (("l1b", "hsl"), ("fused", "hsl"))
MIN_DIFF = 0.005
TRAIN_SEED = 42


def locate_folds(l1b_dirs, folds):
    """{fold: directory} with exactly one l1b directory per fold; ValueError listing duplicates and missing folds."""
    where = {}
    dups = []
    for d in l1b_dirs:
        for f in folds:
            if completed_parts(Path(d) / f"fold{f}"):
                if f in where:
                    dups.append(f)
                where.setdefault(f, d)
    missing = [f for f in folds if f not in where]
    if dups or missing:
        raise ValueError(" ".join(filter(None, [f"fold {sorted(set(dups))} in more than one --l1b directory;" if dups else "",
                                                 f"no L1b scores for folds {missing} in {list(l1b_dirs)}" if missing else ""])))
    return where


def load_fold(hsl_dir, l1b_dir, fold):
    """(hsl games, l1b games) of one fold with the same game keys (ValueError otherwise)."""
    hsl = read_scored_games(Path(hsl_dir) / f"fold{fold}")
    l1b = read_scored_games(Path(l1b_dir) / f"fold{fold}")
    if set(hsl["game_key"]) != set(l1b["game_key"]):
        raise ValueError(f"fold {fold}: {hsl_dir} and {l1b_dir} score different games")
    return hsl, l1b


def feature_sets(hsl_games, l1b_games, questions):
    """{"hsl": X [Q, 24], "l1b": X [Q, 16], "fused": X [Q, 40]} of the questions, plus (y, n_games)."""
    X_h, y, n = build_matrix(hsl_games, questions, PROFILES)
    X_l, y2, _ = build_matrix(l1b_games, questions, LEVELS)
    assert np.array_equal(y, y2)
    return {"hsl": X_h, "l1b": X_l, "fused": np.hstack([X_h, X_l])}, y, n


def run_cv(hsl_dir, where, folds, folds_df, per_player, log=print):
    """Per validation fold the three fits; returns per-question tables and per-fold metrics."""
    data = {}
    for f in folds:
        hsl, l1b = load_fold(hsl_dir, where[f], f)
        players = players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "players": players, "val": pilot_questions(players)}
        log(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    rows, per_fold = [], {name: [] for name in FEATURE_SETS}
    nll = {name: [] for name in FEATURE_SETS}
    for v in folds:
        rng = np.random.RandomState(TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds) if f != v], ignore_index=True)
        train_hsl = pd.concat([data[f]["hsl"] for f in folds if f != v], ignore_index=True)
        train_l1b = pd.concat([data[f]["l1b"] for f in folds if f != v], ignore_index=True)
        X_tr, y_tr, _ = feature_sets(train_hsl, train_l1b, train_q)
        X_va, y_va, n_va = feature_sets(data[v]["hsl"], data[v]["l1b"], data[v]["val"])
        val = data[v]["val"]
        scores = {}
        for name in FEATURE_SETS:
            proba = MLPModel(hidden=32).fit(X_tr[name], y_tr).predict_proba(X_va[name])
            res, s = evaluate(proba, y_va, val)
            per_fold[name].append(res["balanced_mean_over_n"])
            nll[name].append(res["nll_balanced"])
            scores[name] = s
        log(f"CV held-out fold {v} ({len(train_q)} training questions): " + "  ".join(f"{n_} {per_fold[n_][-1]:.4f}" for n_ in FEATURE_SETS), flush=True)
        rows.append(pd.DataFrame({"fold": v, "player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n_games": n_va,
                                  **{f"s_{name}": scores[name] for name in FEATURE_SETS}}))
    return pd.concat(rows, ignore_index=True), per_fold, nll


def summarise(table, per_fold, nll, folds, n_boot, n_features):
    y, pid, n = table["y"].to_numpy(), table["player_id"].to_numpy(dtype=str), table["n_games"].to_numpy()
    out = {"feature_sets": {}, "comparisons": {}}
    for name in FEATURE_SETS:
        s = table[f"s_{name}"].to_numpy()
        out["feature_sets"][name] = {"balanced_mean_over_n": balanced_mean_over_n(s, y, n),
                                     "balanced_by_n": {str(m): balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},
                                     "per_fold": [float(x) for x in per_fold[name]], "nll_balanced": float(np.mean(nll[name])),
                                     "nll_balanced_per_fold": [float(x) for x in nll[name]], "n_features": n_features[name]}
    need = max(1, len(folds) - 1)
    for a, b in COMPARISONS:
        d = paired_diff(table[f"s_{a}"].to_numpy(), table[f"s_{b}"].to_numpy(), y, pid, n, n=n_boot)
        per_fold_diff = [float(x - z) for x, z in zip(per_fold[a], per_fold[b])]
        cond = {"1": bool(d["ci95"][0] > 0), "2": "n/a", "3": bool(sum(x > 0 for x in per_fold_diff) >= need), "4": bool(d["diff"] >= MIN_DIFF)}
        out["comparisons"][f"{a}_vs_{b}"] = {**d, "per_fold_diff": per_fold_diff, "conditions": cond,
                                             "pass": bool(cond["1"] and cond["3"] and cond["4"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py output directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated")
    try:
        where = locate_folds(a.l1b, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        table, per_fold, nll = run_cv(a.hsl, where, folds, folds_df, a.per_player)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    n_features = {"hsl": len(feature_names(PROFILES)), "l1b": len(feature_names(LEVELS)), "fused": len(feature_names(PROFILES)) + len(feature_names(LEVELS))}
    summary = {"folds": folds, "per_player": a.per_player, "n_boot": a.n_boot, "questions": int(len(table)), "hsl_dir": a.hsl,
               "l1b_dirs": {str(f): str(d) for f, d in where.items()}, **summarise(table, per_fold, nll, folds, a.n_boot, n_features),
               "min_diff": MIN_DIFF, **environment("cpu")}
    for name in FEATURE_SETS:
        fs = summary["feature_sets"][name]
        print(f"{name:6s} {fs['balanced_mean_over_n']:.4f}  by n " + " ".join(f"{m}: {fs['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + f"  NLL {fs['nll_balanced']:.4f}  folds " + " ".join(f"{x:.4f}" for x in fs["per_fold"]))
    for a_, b_ in COMPARISONS:
        c = summary["comparisons"][f"{a_}_vs_{b_}"]
        lo, hi = c["ci95"]
        print(f"{a_} vs {b_}: {c['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
              + f"  conditions 1 {'✓' if c['conditions']['1'] else '✗'} 2 n/a 3 {'✓' if c['conditions']['3'] else '✗'} 4 {'✓' if c['conditions']['4'] else '✗'}"
              + f"  -> {'PASS' if c['pass'] else 'no'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(summary, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
