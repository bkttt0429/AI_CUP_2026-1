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

Fold 0, which no model saw, is re-scored as the fold-0 check did -- hsl alone, fused with each model's
scores, and the mean of the models' probabilities -- and the numbers go into the document (fold0_check) and
to stdout; with --reference they are compared with results/l1b/fold0_check.json. --fold0-proba writes the
per-question fold-0 probabilities (the offline side of scripts/check_fused_submission.py).

  python -I scripts/fit_fused_model.py --l1b A:results/l1b/scores3:results/l1b/heads3_A.json \\
      B:results/l1b/scores3_B:results/l1b/heads3_B.json --out results/models/rank_model_fused.json \\
      --fold0-proba results/l1b/fold0_fused_proba.csv --reference results/l1b/fold0_check.json --reference-run disc
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
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
import l1b_cv  # noqa: E402

DEFAULT_TRAIN_FOLDS = (1, 2, 3, 4)


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


def fit(models, hsl_dir, folds, per_player, folds_df, log=print):
    """(fused MLP, hsl MLP, info) fitted on the training folds with the fold-0 check's protocol."""
    by_name = {name: (scores, side) for name, scores, _, side in models}
    src = feature_sources(models, folds)
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, by_name[src[f]][0], f)
        data[f] = {"hsl": hsl, "l1b": l1b, "players": l1b_cv.players_from_scores(hsl, folds_df)}
        log(f"fold {f}: {len(data[f]['players'])} players, {len(hsl)} games, L1b features from {src[f]}", flush=True)
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in sorted(folds)], ignore_index=True)
    X, y, _ = l1b_cv.feature_sets(pd.concat([data[f]["hsl"] for f in sorted(folds)], ignore_index=True),
                                  pd.concat([data[f]["l1b"] for f in sorted(folds)], ignore_index=True), train_q)
    log(f"{len(train_q)} training questions; fitting the fused and the hsl MLP", flush=True)
    mlp_f, mlp_h = MLPModel(hidden=32).fit(X["fused"], y), MLPModel(hidden=32).fit(X["hsl"], y)
    return mlp_f, mlp_h, {"n_train": int(len(train_q)), "questions_sha256": questions_sha(train_q), "sources": src}


def fold0(models, hsl_dir, mlp_f, mlp_h, folds_df, n_boot):
    """The fold-0 check's numbers for these fits and the per-question probabilities table."""
    names = [name for name, _, _, side in models if 0 not in set(side["train_folds"])]
    if len(names) != len(models):
        raise ValueError("every L1b model must be out-of-fold for fold 0")
    hsl0 = l1b_cv.read_scored_games(Path(hsl_dir) / "fold0")
    val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))
    P, y, n = {}, None, None
    for name, scores, _, _ in models:
        _, l1b0 = l1b_cv.load_fold(hsl_dir, scores, 0)
        X, y_, n_ = l1b_cv.feature_sets(hsl0, l1b0, val)
        if y is None:
            y, n = y_, n_
            P["hsl"] = mlp_h.predict_proba(X["hsl"])
        assert np.array_equal(y, y_)
        P[f"fused_{name}"] = mlp_f.predict_proba(X["fused"])
    P["fused_avg"] = np.mean([P[f"fused_{name}"] for name, *_ in models], axis=0)
    pid = val["player_id"].to_numpy(dtype=str)
    res = {k: evaluate(p, y, val) for k, p in P.items()}
    s = {k: r[1] for k, r in res.items()}
    out = {"n_questions": int(len(y)), "n_players": int(len(set(pid))), "scores": {}, "vs_hsl": {}}
    for k in P:
        out["scores"][k] = {"balanced_mean_over_n": balanced_mean_over_n(s[k], y, n), "nll_balanced": res[k][0]["nll_balanced"],
                            "balanced_by_n": {str(m): balanced_mean_over_n(s[k], y, n, sizes=(m,)) for m in SIZES},
                            "per_class": {LEVELS[c]: float(np.mean(s[k][y == c])) for c in range(len(LEVELS)) if np.any(y == c)}}
        if k != "hsl":
            out["vs_hsl"][k] = paired_diff(s[k], s["hsl"], y, pid, n, n=n_boot)
    table = val[["question_id", "rank", "n_games", "player_id"]].copy()
    for k, p in P.items():
        for j, lv in enumerate(LEVELS):
            table[f"{k}_p_{lv}"] = p[:, j]
    return out, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, metavar="NAME:SCORES_DIR:HEADS_SIDECAR")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--train-folds", type=int, nargs="+", default=list(DEFAULT_TRAIN_FOLDS))
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", default=str(ROOT / "results" / "models" / "rank_model_fused.json"))
    ap.add_argument("--fold0-proba", default=None, help="write fold 0's per-question probabilities (CSV) here")
    ap.add_argument("--reference", default=None, help="results/l1b/fold0_check.json to compare the fold-0 numbers with")
    ap.add_argument("--reference-run", default="disc")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.overwrite:
        print(f"{out} exists; pass --overwrite to refit", file=sys.stderr)
        return 2
    folds = sorted(set(a.train_folds))
    if 0 in folds:
        ap.error("fold 0 is the check, not a training fold (propose adding it separately)")
    try:
        models = [parse_l1b(spec) for spec in a.l1b]
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        mlp_f, mlp_h, info = fit(models, a.hsl, folds, a.per_player, folds_df)
        check, table = fold0(models, a.hsl, mlp_f, mlp_h, folds_df, a.n_boot)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    for k, v in check["scores"].items():
        d = check["vs_hsl"].get(k)
        extra = f"  vs hsl {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""
        print(f"fold 0 {k:10s} {v['balanced_mean_over_n']:.4f}  NLL {v['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {v['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + extra, flush=True)
    if a.reference:
        ref = json.loads(Path(a.reference).read_text())["runs"][a.reference_run]["scores"]
        for k, v in check["scores"].items():
            if k in ref:
                delta = v["balanced_mean_over_n"] - ref[k]["balanced_mean_over_n"]
                print(f"  vs {a.reference} [{a.reference_run}] {k}: {delta:+.2e}" + ("  WARNING: differs" if abs(delta) > 1e-6 else ""))
    records = [fused.l1b_model_record(name, side_path.with_suffix(".pt").name, side, [f for f, s in info["sources"].items() if s == name])
               for name, _, side_path, side in models]
    doc = fused.fused_document(mlp_f, records, folds, a.per_player, info["n_train"], info["questions_sha256"],
                               extra={"feature_source_by_fold": {str(f): s for f, s in info["sources"].items()}, "hsl_dir": str(a.hsl),
                                      "l1b_scores_dirs": {name: str(scores) for name, scores, _, _ in models}, "fold0_check": check,
                                      "hsl_state": mlp_h.state_dict()})
    fused.save_fused_model(doc, out)
    print(f"wrote {out}: {info['n_train']} training questions, L1b models {[r['name'] for r in records]}")
    if a.fold0_proba:
        Path(a.fold0_proba).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(a.fold0_proba, index=False)
        print(f"wrote {a.fold0_proba} ({len(table)} fold-0 questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
