#!/usr/bin/env python
"""Fit the submission's fused MLP (Human SL + L1b question features) exactly as scripts/l1b_fold0_check.py did.

Issue #9. Training questions: qfeatures.sample_questions, --per-player per player, one RandomState(42) over
--train-folds in ascending order (the L3 ladder's and the fold-0 check's rule). Features of a training fold
come from the L1b model that did not see it (the sidecar's train_folds): A (folds 1-2) feeds folds 3-4, B
(folds 3-4) feeds folds 1-2; a fold with no such model, or with several, is refused. The MLP is
rankmodel.MLPModel(hidden=32) on the 40 fused features (aicup.fused). The document written to --out
(rank_model_fused.json) records the features, the training folds and question hash, and the L1b models
(heads file, state and trunk sha256, training folds, feature folds) so predict_submission.py can check that the
shards were scored with exactly those models.

Fold 0 is re-scored as the fold-0 check did -- hsl alone, fused with each model's scores, and the mean of the
models' probabilities -- and the numbers go into the document (fold0_check) and to stdout; with --reference they
are compared with results/l1b/fold0_check.json. --fold0-proba writes the per-question fold-0 probabilities (the
offline side of scripts/check_fused_submission.py). Only the models that did not train on fold 0 take part (A and
B both; of the five cross-fit models of issue #16 -- f0..f4, each trained on the other four folds and feeding its
own fold -- only f0), so for a cross-fit document the fold-0 averages are over that one model and the submission's
average over all five has no held-out check (the document says so in fold0_check.models_seen_fold0).

With --opp NAME:OPP_SCORES_DIR --diff (issue #25) the document's MLP is the fused_opp_diff one (59 features: fused_opp + the three
opponent-minus-own pmi differences of aicup.fused.diff_features), the fold-0 table also carries fused_opp_diff_* and
fused_opp_diff_avg vs fused_opp_avg, and the default output is rank_model_fused_opp_diff.json.
With --opp ... --diff --hd-table HD_TABLE_CSV (issue #47; scripts/hd_table.py's table of dataset v1.1's HD[n]) the document's MLP is
the fused_opp_diff_hd one (64 features: + aicup.handicap's five handicap question features, looked up per game), the fold-0
table also carries fused_opp_diff_hd_* and fused_opp_diff_hd_avg vs fused_opp_diff_avg, the fused_opp_diff MLP's state is kept
(fused_opp_diff_state) and the default output is rank_model_fused_opp_diff_hd.json. predict_submission.py computes the same
features from the test files' HD[] and refuses the document when a game carries none.
With --opp NAME:OPP_SCORES_DIR (issue #12; one per --l1b model, l1b_score.py --opponent directories) the
document's MLP is the fused_opp one: the 40 fused features plus the 16 of the L1b sums over the opponent's
moves, built by scripts/l1b_opp_check.features exactly as scripts/l1b_opp_fold0_check.py did (the same
model feeds the target's and the opponent's features of a fold). The hsl and fused MLPs are still fitted
for the fold-0 table (hsl, fused_*, fused_opp_*; --reference then takes results/l1b/opp_fold0_check.json),
and the fused MLP's state is kept in the document (fused_state) as a reference.

  python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json \\
      B:results/l1b/scores3_B:results/l1b/heads3_B.json --out results/models/rank_model_fused.json \\
      --fold0-proba results/l1b/fold0_fused_proba.csv --reference results/l1b/fold0_check.json --reference-run disc
  python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json \\
      B:results/l1b/scores3_B:results/l1b/heads3_B.json --opp A:results/l1b/scores3_opp B:results/l1b/scores3_B_opp \\
      --out results/models/rank_model_fused_opp.json --fold0-proba results/l1b/fold0_fused_opp_proba.csv \\
      --reference results/l1b/opp_fold0_check.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup import fused  # noqa: E402
from aicup.handicap import hd_summary, question_hd_features, read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

DEFAULT_TRAIN_FOLDS = (1, 2, 3, 4)
DEFAULT_OUT = {"fused": ROOT / "results" / "models" / "rank_model_fused.json", "fused_opp": ROOT / "results" / "models" / "rank_model_fused_opp.json",
               "fused_opp_diff": ROOT / "results" / "models" / "rank_model_fused_opp_diff.json",
               "fused_opp_diff_hd": ROOT / "results" / "models" / "rank_model_fused_opp_diff_hd.json"}


def parse_l1b(spec):
    """"NAME:SCORES_DIR:HEADS_SIDECAR" -> (name, scores dir, sidecar path, sidecar dict)."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--l1b expects NAME:SCORES_DIR:HEADS_SIDECAR, got {spec!r}")
    name, scores, side = parts
    sidecar = json.loads(Path(side).read_text())
    for key in ("state_sha256", "train_folds"):
        if sidecar.get(key) is None:
            raise ValueError(f"{side}: sidecar has no {key}")
    return name, Path(scores), Path(side), sidecar


def parse_opp(spec, names):
    """"NAME:OPP_SCORES_DIR" -> (name, dir); the name must be one of the --l1b models'."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--opp expects NAME:OPP_SCORES_DIR, got {spec!r}")
    name, d = parts
    if name not in names:
        raise ValueError(f"--opp {spec!r}: no --l1b model is named {name!r} ({sorted(names)})")
    return name, Path(d)


def opponent_table(opp_dir, fold, hsl):
    """The l1b_score.py --opponent table of one fold; ValueError unless it covers exactly the Human SL table's games."""
    opp = read_scored_games(Path(opp_dir) / f"fold{fold}")
    if set(opp["game_key"]) != set(hsl["game_key"]):
        raise ValueError(f"fold {fold}: the opponent scores in {opp_dir} cover other games than the Human SL scores "
                         f"({len(set(opp['game_key']) ^ set(hsl['game_key']))} differ)")
    return opp


def feature_sources(models, folds):
    """{fold: model name} -- the one model whose train_folds exclude the fold; ValueError when none or several."""
    out = {}
    for f in folds:
        cands = [name for name, _, _, side in models if f not in set(side["train_folds"])]
        if len(cands) != 1:
            raise ValueError(f"fold {f}: {len(cands)} L1b models did not train on it ({cands}); exactly one must supply its features")
        out[f] = cands[0]
    return out


def questions_sha(train_q):
    h = hashlib.sha256()
    for qid, keys in zip(train_q["question_id"], train_q["game_keys"]):
        h.update(f"{qid}:{';'.join(keys)}\n".encode())
    return h.hexdigest()


def feature_sets(hsl, l1b, questions, opp=None, diff=False, hd=None):
    """l1b_cv.feature_sets (hsl, fused), plus l1b_opp_check.features' fused_opp when the opponent table is given and,
    with diff, fused_opp_diff (+ aicup.fused.question_diff_features; issue #25) and, with an HD table, fused_opp_diff_hd
    (+ aicup.handicap.question_hd_features; issue #47)."""
    if opp is None:
        return l1b_cv.feature_sets(hsl, l1b, questions)
    X, y, n = oc.features(hsl, l1b, opp, questions)
    if diff:
        X["fused_opp_diff"] = np.hstack([X["fused_opp"], fused.question_diff_features(l1b, opp, questions)])
    if hd is not None:
        X["fused_opp_diff_hd"] = np.hstack([X["fused_opp_diff"], question_hd_features(hd, questions)])
    return X, y, n


def fit(models, hsl_dir, folds, per_player, folds_df, log=print, opp_dirs=None, diff=False, hd=None):
    """({"fused": MLP, "hsl": MLP[, "fused_opp": MLP[, "fused_opp_diff": MLP[, "fused_opp_diff_hd": MLP]]]}, info) fitted on the
    training folds with the fold-0 checks' protocol; opp_dirs {model name: l1b_score.py --opponent directory} adds the fused_opp
    fit (the fold's opponent features come from the same model as its target features), diff the fused_opp_diff fit and an HD
    table (aicup.handicap.read_hd_table) the fused_opp_diff_hd fit."""
    by_name = {name: (scores, side) for name, scores, _, side in models}
    src = feature_sources(models, folds)
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, by_name[src[f]][0], f)
        data[f] = {"hsl": hsl, "l1b": l1b, "players": l1b_cv.players_from_scores(hsl, folds_df),
                   "opp": opponent_table(opp_dirs[src[f]], f, hsl) if opp_dirs else None}
        log(f"fold {f}: {len(data[f]['players'])} players, {len(hsl)} games, L1b features from {src[f]}"
            + (f" (opponent moves {int(data[f]['opp']['n_moves'].sum())} vs target {int(hsl['n_moves'].sum())})" if opp_dirs else ""), flush=True)
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds)], ignore_index=True)
    cat = lambda key: pd.concat([data[f][key] for f in sorted(folds)], ignore_index=True)  # noqa: E731
    X, y, _ = feature_sets(cat("hsl"), cat("l1b"), train_q, cat("opp") if opp_dirs else None, diff, hd)
    sets = ("fused", "hsl") + (("fused_opp",) if opp_dirs else ()) + (("fused_opp_diff",) if diff else ()) + (("fused_opp_diff_hd",) if hd is not None else ())
    log(f"{len(train_q)} training questions; fitting the {', '.join(sets)} MLPs", flush=True)
    mlps = {s: MLPModel(hidden=32).fit(X[s], y) for s in sets}
    return mlps, {"n_train": int(len(train_q)), "questions_sha256": questions_sha(train_q), "sources": src}


def fold0(models, hsl_dir, mlps, folds_df, n_boot, opp_dirs=None, diff=False, hd=None):
    """The fold-0 checks' numbers for these fits (hsl, fused_<model>, fused_avg and, with opp_dirs, fused_opp_<model>,
    fused_opp_avg; with diff also fused_opp_diff_<model>, fused_opp_diff_avg; with hd also fused_opp_diff_hd_<model>,
    fused_opp_diff_hd_avg) and the per-question probabilities table. Only the models out-of-fold for fold 0 take part (A and B
    both; of the five cross-fit models of issue #16 only the one that left fold 0 out), and the averages are over those;
    the result names them (fold0_models) and the models left aside (models_seen_fold0). ValueError when none is OOF."""
    oof = [m for m in models if 0 not in set(m[3]["train_folds"])]
    seen = [name for name, _, _, side in models if 0 in set(side["train_folds"])]
    if not oof:
        raise ValueError("no L1b model is out-of-fold for fold 0 (every sidecar's train_folds has 0); the fold-0 check cannot be made")
    hsl0 = l1b_cv.read_scored_games(Path(hsl_dir) / "fold0")
    val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))
    P, y, n = {}, None, None
    for name, scores, _, _ in oof:
        _, l1b0 = l1b_cv.load_fold(hsl_dir, scores, 0)
        X, y_, n_ = feature_sets(hsl0, l1b0, val, opponent_table(opp_dirs[name], 0, hsl0) if opp_dirs else None, diff, hd)
        if y is None:
            y, n = y_, n_
            P["hsl"] = mlps["hsl"].predict_proba(X["hsl"])
        assert np.array_equal(y, y_)
        P[f"fused_{name}"] = mlps["fused"].predict_proba(X["fused"])
        if opp_dirs:
            P[f"fused_opp_{name}"] = mlps["fused_opp"].predict_proba(X["fused_opp"])
        if diff:
            P[f"fused_opp_diff_{name}"] = mlps["fused_opp_diff"].predict_proba(X["fused_opp_diff"])
        if hd is not None:
            P[f"fused_opp_diff_hd_{name}"] = mlps["fused_opp_diff_hd"].predict_proba(X["fused_opp_diff_hd"])
    P["fused_avg"] = np.mean([P[f"fused_{name}"] for name, *_ in oof], axis=0)
    if opp_dirs:
        P["fused_opp_avg"] = np.mean([P[f"fused_opp_{name}"] for name, *_ in oof], axis=0)
    if diff:
        P["fused_opp_diff_avg"] = np.mean([P[f"fused_opp_diff_{name}"] for name, *_ in oof], axis=0)
    if hd is not None:
        P["fused_opp_diff_hd_avg"] = np.mean([P[f"fused_opp_diff_hd_{name}"] for name, *_ in oof], axis=0)
    pid = val["player_id"].to_numpy(dtype=str)
    res = {k: evaluate(p, y, val) for k, p in P.items()}
    s = {k: r[1] for k, r in res.items()}
    out = {"n_questions": int(len(y)), "n_players": int(len(set(pid))), "fold0_models": [name for name, *_ in oof], "models_seen_fold0": seen,
           "scores": {}, "vs_hsl": {}}
    for k in P:
        out["scores"][k] = {"balanced_mean_over_n": balanced_mean_over_n(s[k], y, n), "nll_balanced": res[k][0]["nll_balanced"],
                            "balanced_by_n": {str(m): balanced_mean_over_n(s[k], y, n, sizes=(m,)) for m in SIZES},
                            "per_class": {LEVELS[c]: float(np.mean(s[k][y == c])) for c in range(len(LEVELS)) if np.any(y == c)}}
        if k != "hsl":
            out["vs_hsl"][k] = paired_diff(s[k], s["hsl"], y, pid, n, n=n_boot)
    if opp_dirs:
        out["fused_opp_avg_vs_fused_avg"] = paired_diff(s["fused_opp_avg"], s["fused_avg"], y, pid, n, n=n_boot)
    if diff:
        out["fused_opp_diff_avg_vs_fused_opp_avg"] = paired_diff(s["fused_opp_diff_avg"], s["fused_opp_avg"], y, pid, n, n=n_boot)
    if hd is not None:
        out["fused_opp_diff_hd_avg_vs_fused_opp_diff_avg"] = paired_diff(s["fused_opp_diff_hd_avg"], s["fused_opp_diff_avg"], y, pid, n, n=n_boot)
    table = val[["question_id", "rank", "n_games", "player_id"]].copy()
    for k, p in P.items():
        for j, lv in enumerate(LEVELS):
            table[f"{k}_p_{lv}"] = p[:, j]
    return out, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, metavar="NAME:SCORES_DIR:HEADS_SIDECAR")
    ap.add_argument("--opp", nargs="*", default=[], metavar="NAME:OPP_SCORES_DIR",
                    help="l1b_score.py --opponent directories, one per --l1b model: fit the fused_opp MLP instead (issue #12)")
    ap.add_argument("--diff", action="store_true", help="with --opp: fit the fused_opp_diff MLP instead (+3 opponent-minus-own pmi features; issue #25)")
    ap.add_argument("--hd-table", default=None, help="with --opp --diff: fit the fused_opp_diff_hd MLP instead (+5 handicap question features from "
                    "scripts/hd_table.py's CSV of dataset v1.1's HD[n]; issue #47)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--train-folds", type=int, nargs="+", default=list(DEFAULT_TRAIN_FOLDS))
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", default=None, help=f"default {DEFAULT_OUT['fused'].name}, {DEFAULT_OUT['fused_opp'].name} with --opp, "
                    f"{DEFAULT_OUT['fused_opp_diff'].name} with --opp --diff")
    ap.add_argument("--fold0-proba", default=None, help="write fold 0's per-question probabilities (CSV) here")
    ap.add_argument("--reference", default=None,
                    help="results/l1b/fold0_check.json (its runs[--reference-run]) or results/l1b/opp_fold0_check.json to compare the fold-0 numbers with")
    ap.add_argument("--reference-run", default="disc")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    opponent = bool(a.opp)
    if a.diff and not opponent:
        ap.error("--diff needs --opp (the differences are between the opponent's and the target's L1b sums)")
    if a.hd_table and not a.diff:
        ap.error("--hd-table needs --opp --diff (the handicap features are appended to fused_opp_diff)")
    kind = "fused_opp_diff_hd" if a.hd_table else "fused_opp_diff" if a.diff else "fused_opp" if opponent else "fused"
    out = Path(a.out) if a.out else DEFAULT_OUT[kind]
    if out.exists() and not a.overwrite:
        print(f"{out} exists; pass --overwrite to refit", file=sys.stderr)
        return 2
    folds = sorted(set(a.train_folds))
    if 0 in folds:
        ap.error("fold 0 is the check, not a training fold (propose adding it separately)")
    try:
        models = [parse_l1b(spec) for spec in a.l1b]
        names = [name for name, *_ in models]
        opp_dirs = dict(parse_opp(spec, set(names)) for spec in a.opp) if opponent else None
        if opponent and sorted(opp_dirs) != sorted(names):
            raise ValueError(f"--opp must name every --l1b model exactly once: got {sorted(opp_dirs)}, models {sorted(names)}")
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        hd = read_hd_table(a.hd_table) if a.hd_table else None
        if hd is not None:
            cov = hd_summary(hd)
            print(f"HD table {a.hd_table}: {cov['games']} games, HD>=1 {cov['share_hd1']:.3f}, HD>=2 {cov['share_hd2']:.3f}, absent {cov['hd'].get('absent', 0)}", flush=True)
        mlps, info = fit(models, a.hsl, folds, a.per_player, folds_df, opp_dirs=opp_dirs, diff=a.diff, hd=hd)
        check, table = fold0(models, a.hsl, mlps, folds_df, a.n_boot, opp_dirs=opp_dirs, diff=a.diff, hd=hd)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    if check["models_seen_fold0"]:
        print(f"fold 0: models {check['models_seen_fold0']} trained on it and are left out of the check; the averages are over "
              f"{check['fold0_models']} only -- the submission's average over all {len(models)} models has no held-out check", flush=True)
    for k, v in check["scores"].items():
        d = check["vs_hsl"].get(k)
        extra = f"  vs hsl {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""
        print(f"fold 0 {k:14s} {v['balanced_mean_over_n']:.4f}  NLL {v['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {v['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + extra, flush=True)
    if opponent:
        d = check["fused_opp_avg_vs_fused_avg"]
        print(f"fold 0 fused_opp_avg vs fused_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    if a.diff:
        d = check["fused_opp_diff_avg_vs_fused_opp_avg"]
        print(f"fold 0 fused_opp_diff_avg vs fused_opp_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    if a.hd_table:
        d = check["fused_opp_diff_hd_avg_vs_fused_opp_diff_avg"]
        print(f"fold 0 fused_opp_diff_hd_avg vs fused_opp_diff_avg: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]", flush=True)
    if a.reference:
        ref = json.loads(Path(a.reference).read_text())
        ref = ref["runs"][a.reference_run]["scores"] if "runs" in ref else ref["scores"]  # fold0_check.json vs opp_fold0_check.json
        for k, v in check["scores"].items():
            if k in ref:
                delta = v["balanced_mean_over_n"] - ref[k]["balanced_mean_over_n"]
                print(f"  vs {a.reference} {k}: {delta:+.2e}" + ("  WARNING: differs" if abs(delta) > 1e-6 else ""))
    records = [fused.l1b_model_record(name, side_path.with_suffix(".pt").name, side, [f for f, s in info["sources"].items() if s == name])
               for name, _, side_path, side in models]
    extra = {"feature_source_by_fold": {str(f): s for f, s in info["sources"].items()}, "hsl_dir": str(a.hsl),
             "l1b_scores_dirs": {name: str(scores) for name, scores, _, _ in models}, "fold0_check": check, "hsl_state": mlps["hsl"].state_dict()}
    if opponent:
        extra.update({"opp_scores_dirs": {name: str(d) for name, d in opp_dirs.items()}, "fused_state": mlps["fused"].state_dict()})
    if a.diff:
        extra["fused_opp_state"] = mlps["fused_opp"].state_dict()
    if a.hd_table:
        extra.update({"fused_opp_diff_state": mlps["fused_opp_diff"].state_dict(), "hd_table": str(a.hd_table), "hd_coverage": hd_summary(hd)})
    doc = fused.fused_document(mlps[kind], records, folds, a.per_player, info["n_train"], info["questions_sha256"], extra=extra, feature_set=kind)
    fused.save_fused_model(doc, out)
    print(f"wrote {out}: {doc['name']}, {info['n_train']} training questions, L1b models {[r['name'] for r in records]}")
    if a.fold0_proba:
        Path(a.fold0_proba).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.fold0_proba, index=False)
        print(f"wrote {a.fold0_proba} ({len(table)} fold-0 questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
