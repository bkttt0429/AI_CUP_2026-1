#!/usr/bin/env python
"""Train a set rung on every training question of the given folds and save it for the submission pipeline.

Item 2 of the to-do list (issue #1, 2026-10-25). Unlike the CV (scripts/ladder_cv.py) nothing is held
out between folds: fold 0 is training data here, by the user's decision (its one evaluation is spent).
The training questions are qfeatures.sample_questions, --per-player per player, over the folds in
ascending order with one RandomState(aicup.ladder.TRAIN_SEED) -- fit_rank_model's train_set over all its
folds. The base logits of every training question come from r2_mlp24 cross-fitted by fold
(settrain.crossfit_base_logits: each fold's questions from a model fitted on the other folds), as the
CV's split_base does; the paired base for inference is r2_mlp24 fitted on every training question,
written to --base-out in the format of results/models/rank_model.json (name, features, profiles,
train_folds, per_player, state, provenance). Every --seeds then trains fit_set_model with the rung's
TrainCfg (inner early stopping on 10% of each rank's players, as the CV) and is saved by
aicup.setmodel_io as <out>/{rung}_s{seed}.pt + .json; the sidecar carries the training folds,
per_player, the training questions' sha256, and the paired base's file name and sha256. Existing model
files are refused unless --overwrite (exit 2); an existing --base-out is kept when it is this base (the
same training questions and parameters: the rungs of one ensemble share it) and refused otherwise.

  python -I scripts/train_set_model.py --rung r3_pma                   # folds 0-4, seeds 0 1 2, per player 32
  python -I scripts/train_set_model.py --rung r4_xattn_t1 --seeds 0 1 2
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.humansl import PROFILES  # noqa: E402
from aicup.ladder import (BASE_RUNG, FOLDS_CSV, SET_RUNGS, TRAIN_SEED, _check_players, _features, _labels,  # noqa: E402
                          _weighted_ce, environment, load_vectors, to_json, train_cfg)
from aicup.ladder_data import STATIC_RUNGS, pack_questions  # noqa: E402
from aicup.qfeatures import FEATURE_NAMES, sample_questions  # noqa: E402
from aicup.scoring import TRAIN_DIR, fold_players, player_games  # noqa: E402
from aicup.setmodel_io import file_sha256, save_set_model  # noqa: E402
from aicup.settrain import crossfit_base_logits, fit_set_model  # noqa: E402

DEFAULT_FOLDS = (0, 1, 2, 3, 4)
DEFAULT_PER_PLAYER = 32


def load_players_all(folds, train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):
    """{fold: scoring.fold_players selection} for every fold asked, fold 0 included (training data here)."""
    games, folds_df = player_games(train_dir), pd.read_csv(folds_csv)
    return {f: fold_players(games, folds_df, f) for f in folds}


def training_set(players, folds, per_player):
    """The training questions: sample_questions per fold in ascending order, one RandomState(TRAIN_SEED), a
    `fold` column -- fit_rank_model.train_set over all the folds."""
    rng = np.random.RandomState(TRAIN_SEED)
    return pd.concat([sample_questions(players[f], per_player, rng).assign(fold=f) for f in sorted(folds)], ignore_index=True)


def questions_sha(train_q):
    h = hashlib.sha256()
    for qid, keys in zip(train_q["question_id"], train_q["game_keys"]):
        h.update(f"{qid}:{';'.join(keys)}\n".encode())
    return h.hexdigest()


def fit_base(train_q, vec):
    """(cross-fitted base logits [N, 10] aligned with train_q, the paired base model fitted on every question, info).

    The static rung BASE_RUNG (r2_mlp24 on q24 features): crossfit_base_logits gives each fold's questions the
    logits of a model fitted on the other folds; the paired base is the same model fitted on all of them."""
    kind, rep, make = STATIC_RUNGS[BASE_RUNG]
    X, y = _features(train_q, vec, kind, rep, True), _labels(train_q)
    fold = train_q["fold"].to_numpy()
    if np.any(np.diff(fold) < 0):
        raise ValueError("training questions must be in ascending fold order")
    folds = sorted(set(fold.tolist()))
    by_fold, _ = crossfit_base_logits(make, {f: X[fold == f] for f in folds}, {f: y[fold == f] for f in folds}, X[:1])
    base = np.concatenate([by_fold[f] for f in folds])
    paired = make().fit(X, y)
    info = {"ce_train_crossfit": _weighted_ce(base, y), "n_train": int(len(y)), "folds": folds}
    return base, paired, info


def base_document(paired, folds, per_player, n_train, questions_sha256):
    """The paired base in results/models/rank_model.json's format (predict_submission.load_rank_model reads it)."""
    return {"name": "mlp_h32", "features": list(FEATURE_NAMES), "profiles": list(PROFILES), "train_folds": [int(f) for f in folds],
            "per_player": int(per_player), "train_questions": int(n_train), "questions_sha256": questions_sha256,
            "role": f"paired base of the set models trained on the same questions ({BASE_RUNG} fitted on all of them)",
            "created": datetime.now().isoformat(timespec="seconds"), **environment("cpu"), "state": paired.state_dict()}


def check_same_base(base_out, doc, atol=1e-8):
    """An existing paired base is kept only when it is this base: the same training questions (questions_sha256,
    train_folds, per_player, name) and the same parameters (within atol). Otherwise FileExistsError: --overwrite
    replaces it, and every set model paired with the old file is then to be retrained."""
    old = json.loads(Path(base_out).read_text())
    keys = ("questions_sha256", "train_folds", "per_player", "name")
    differ = [k for k in keys if old.get(k) != doc[k]]
    if differ:
        raise FileExistsError(f"{base_out} exists and was fitted on other training questions ("
                              + ", ".join(f"{k} {old.get(k)!r} vs {doc[k]!r}" for k in differ)
                              + "); pass --overwrite to replace it and retrain every set model paired with it")
    a, b = old.get("state", {}).get("params", {}), doc["state"]["params"]
    if set(a) != set(b) or any(np.shape(a[k]) != np.shape(b[k]) or not np.allclose(a[k], b[k], rtol=0, atol=atol) for k in b):
        raise FileExistsError(f"{base_out} exists with other parameters than this fit gives on the same questions (other "
                              "code or torch?); pass --overwrite to replace it and retrain every set model paired with it")


def train(rung, vec, players, folds, seeds, per_player, out_dir, base_out, device="cuda", overwrite=False, log=print):
    """Train and save `rung` for every seed; returns [(pt_path, sidecar)]. Writes the paired base to base_out first,
    or keeps an existing one that check_same_base accepts."""
    if rung not in SET_RUNGS:
        raise KeyError(f"unknown set rung {rung!r}; known: {list(SET_RUNGS)}")
    folds = sorted(int(f) for f in folds)
    if len(set(folds)) != len(folds) or len(folds) < 2:
        raise ValueError(f"need at least 2 distinct folds (the base is cross-fitted by fold), got {folds}")
    out_dir, base_out = Path(out_dir), Path(base_out)
    targets = [out_dir / f"{rung}_s{int(s)}.pt" for s in seeds]
    existing = [p for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} model files exist (e.g. {existing[0]}); pass --overwrite to replace them")
    rep, make, _ = SET_RUNGS[rung]
    t0 = time.time()
    train_q = training_set(players, folds, per_player)
    sha = questions_sha(train_q)
    base, paired, info = fit_base(train_q, vec)
    log(f"{len(train_q)} training questions over folds {folds} ({per_player} per player); cross-fitted base CE "
        f"{info['ce_train_crossfit']:.4f} ({time.time() - t0:.0f} s)")
    doc = base_document(paired, folds, per_player, len(train_q), sha)
    if base_out.exists() and not overwrite:
        check_same_base(base_out, doc)
        log(f"kept the existing paired base {base_out} (the same training questions and parameters)")
    else:
        base_out.parent.mkdir(parents=True, exist_ok=True)
        base_out.write_text(to_json(doc, indent=1) + "\n")
        log(f"wrote the paired base {base_out}")
    base_sha = file_sha256(base_out)
    data = pack_questions(train_q, vec, rep, True)
    saved = []
    for seed in seeds:
        t1 = time.time()
        fitted = fit_set_model(make, data, base, train_cfg(rung, int(seed), device))
        if not np.isfinite(fitted.losses).all():
            raise FloatingPointError(f"{rung} seed {seed}: non-finite training loss {fitted.losses}")
        pt = out_dir / f"{rung}_s{int(seed)}.pt"
        meta = save_set_model(fitted, rung, int(seed), pt, extra={
            "train_folds": folds, "per_player": int(per_player), "n_train_questions": int(len(train_q)), "questions_sha256": sha,
            "base_model": base_out.name, "base_model_sha256": base_sha, "base_ce_train_crossfit": info["ce_train_crossfit"],
            "wall_seconds": time.time() - t1})
        saved.append((pt, meta))
        log(f"wrote {pt}: best_epoch {fitted.best_epoch}, inner CE {fitted.inner_ce[0]:.4f} -> {fitted.inner_ce[fitted.best_eval]:.4f}, "
            f"{'stopped early' if fitted.stopped_early else 'ran to the end'} ({time.time() - t1:.0f} s)")
    return saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rung", required=True, choices=list(SET_RUNGS))
    ap.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-player", type=int, default=DEFAULT_PER_PLAYER)
    ap.add_argument("--vectors", default=str(ROOT / "results" / "ladder" / "vectors"))
    ap.add_argument("--out", default=str(ROOT / "results" / "models" / "set"), help="directory of the .pt + .json files")
    ap.add_argument("--base-out", default=str(ROOT / "results" / "models" / "rank_model_f01234.json"),
                    help="the paired base (r2_mlp24 on every training question), rank_model.json format")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    for f in folds:
        if not (Path(a.vectors) / f"fold{f}.npz").exists():
            raise FileNotFoundError(f"{Path(a.vectors) / f'fold{f}.npz'} not built (scripts/ladder_build_vectors.py)")
    vec, fold_of = load_vectors(a.vectors, folds)
    players = load_players_all(folds)
    _check_players(players, vec, fold_of, folds)
    try:
        train(a.rung, vec, players, folds, a.seeds, a.per_player, a.out, a.base_out, a.device, a.overwrite)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
