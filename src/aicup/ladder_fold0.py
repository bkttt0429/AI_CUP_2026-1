"""One-shot fold-0 evaluation of the aggregator-ladder rung chosen from the folds 1-4 CV.

Fold 0 is the held-out evaluation fold: its result is spent once, on the configuration chosen from
the CV summary, and never feeds back into a choice. run() therefore raises Refused (the CLI's exit
2) before reading any data when the output or the canonical results/ladder/fold0_eval.json
(CANONICAL_OUT) exists already, when the CV summary the rung was chosen from (chosen_from) is
missing, when that summary has no CV results for the requested configuration
(ladder._label(rung, per_player, counts)), or, for a set rung, when the seeds differ from that CV
label's seeds. The output is written once, atomically, and never over an existing file (write_once).

Protocol (plan 2026-10-19-l3-aggregator-ladder-ctm, Task 8): the rung is retrained on folds 1-4 with
the training questions of split v = 0 (ladder.training_questions: one RandomState(42), folds 1-4
ascending -- fit_rank_model's final model) and scored on fold 0's pilot_questions. A static rung is
fitted once (deterministic; seeds null). A set rung trains once per seed on the 4-fold cross-fitted
base (ladder.split_base with v = 0, computed once and shared by the seeds), and the per-question
score is the mean over seeds -- the statistic of the CV summary. Only the rung's own configuration (its
TrainCfg.pick tick) is scored: the report-only EXTRAS of the CV (more ticks, another tick) are not computed on fold 0, so
that nothing can be picked on it after the fact. Reported: the headline
balanced_mean_over_n with its player-bootstrap CI (evidence.bootstrap_ci_mean_over_sizes, n = 5000)
and two paired differences (qeval.paired_diff), both through paired(), which refuses frames whose
question ids differ in content or order:
- vs_base: the same retrain's base -- for a set rung softmax of split_base's validation base
  (Δ ≡ 0); for a static rung r2_mlp24 refitted on the same training questions (same per_player and
  counts), so the difference is exactly 0 for r2_mlp24 itself;
- vs_production: results/models/rank_model.json (rankmodel.from_state on qfeatures.build_matrix
  features), trained with 8 questions per player (PRODUCTION_NOTE).
"""
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.evidence import bootstrap_ci_mean_over_sizes
from aicup.ladder import (BASE_RUNG, CV_FOLDS, DEFAULT_PER_PLAYER, FOLDS_CSV, KEY_COLS, N_BOOT, ROOT, RUNGS, SET_RUNGS,
                          _check_players, _frame, _label, _plain, _same_keys, _set_split, _static_split, environment,
                          load_vectors, split_base, to_json, training_questions, validation_questions)
from aicup.qeval import SIZES, balanced, balanced_mean_over_n, nll_brier, paired_diff
from aicup.qfeatures import FEATURE_NAMES, SUM_COLS, build_matrix
from aicup.rankmodel import N_CLASSES, _softmax, from_state
from aicup.scoring import TRAIN_DIR, fold_players, player_games

EVAL_FOLD = 0
ALL_FOLDS = (EVAL_FOLD, *CV_FOLDS)
# the canonical result: check_request refuses whenever it exists, whatever `out` is (tests move it)
CANONICAL_OUT = ROOT / "results" / "ladder" / "fold0_eval.json"
OUT = CANONICAL_OUT  # the default output
CHOSEN_FROM = ROOT / "results" / "ladder" / "cv_summary.json"
VECTORS = ROOT / "results" / "ladder" / "vectors"
RANK_MODEL = ROOT / "results" / "models" / "rank_model.json"
PCOLS = [f"p{k}" for k in range(N_CLASSES)]
PRODUCTION_NOTE = (f"The production model (rank_model.json) was trained with 8 questions per player and this retrain with "
                   f"per_player of them, so vs_production mixes the rung with the training size; the training size "
                   f"alone is the CV comparison '{BASE_RUNG}@pp8 vs {BASE_RUNG}' of chosen_from "
                   f"(cv_pp8_vs_base, when present).")


class Refused(Exception):
    """Fold 0 must not be evaluated for this request (the CLI exits with 2); raised before any data is read."""


def repo_path(path):
    """The path relative to the repository root when inside it, else absolute."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def check_request(rung, out, chosen_from, per_player=DEFAULT_PER_PLAYER, counts=True, seeds=(0, 1, 2)):
    """Refused unless fold 0 may be evaluated: neither `out` nor CANONICAL_OUT may exist (checked
    first), `chosen_from` must be a CV summary with results for ladder._label(rung, per_player,
    counts) and, for a set rung, `seeds` must be that CV label's seeds (in any order; a static rung
    is fitted once and ignores them). Returns the provenance of the choice: chosen_from
    (repo-relative), its sha256, the CV label, the CV entry of that label and the summary's
    r2_mlp24@pp8 vs r2_mlp24 comparison (None when absent)."""
    out, chosen_from = Path(out), Path(chosen_from)
    if out.exists():
        raise Refused(f"{out} exists: fold 0 has been evaluated already, and it is evaluated only once")
    if CANONICAL_OUT.exists():
        raise Refused(f"{CANONICAL_OUT} exists: fold 0 has been evaluated already, and it is evaluated only once "
                      f"(this canonical result is checked whatever --out is)")
    if rung not in RUNGS:
        raise Refused(f"unknown rung {rung!r}; known: {list(RUNGS)}")
    if not chosen_from.is_file():
        raise Refused(f"{chosen_from} not found: fold 0 is evaluated only for a rung chosen from a CV summary "
                      f"(scripts/ladder_cv.py)")
    raw = chosen_from.read_bytes()
    try:
        summary = json.loads(raw)
    except ValueError as e:
        raise Refused(f"{chosen_from} is not a CV summary ({e})") from e
    rungs = summary.get("rungs") if isinstance(summary, dict) else None
    label = _label(rung, per_player, counts)
    if not isinstance(rungs, dict) or label not in rungs:
        have = sorted(rungs) if isinstance(rungs, dict) else []
        raise Refused(f"{chosen_from} has no CV results for {label!r} (it has {have}): fold 0 is evaluated only for a "
                      f"configuration chosen from the CV; give --per-player / --no-counts as in that CV run")
    cv = rungs[label] if isinstance(rungs[label], dict) else {}
    if rung in SET_RUNGS:  # the headline is the seed mean: the same seeds as the CV statistic it was chosen by
        cv_seeds = cv.get("seeds")
        if not isinstance(cv_seeds, list) or sorted(int(s) for s in seeds) != sorted(int(s) for s in cv_seeds):
            raise Refused(f"--seeds {sorted(int(s) for s in seeds)} differ from the seeds {cv_seeds} of {label!r} in "
                          f"{chosen_from}: a set rung is evaluated on fold 0 with the seeds of its CV results")
    pp8 = (summary.get("comparisons") or {}).get(f"{BASE_RUNG}@pp8 vs {BASE_RUNG}")
    return {"chosen_from": repo_path(chosen_from), "cv_summary_sha256": _sha256(raw), "cv_label": label,
            "cv": {k: cv.get(k) for k in ("balanced_mean_over_n", "per_fold", "seeds")},
            "cv_pp8_vs_base": None if pp8 is None else {k: pp8.get(k) for k in ("diff", "ci95")}}


# ---------------------------------------------------------------- inputs

def load_eval_players(folds=ALL_FOLDS, train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):
    """{fold: scoring.fold_players selection}, fold 0 included (ladder.load_players refuses fold 0 so
    that the CV never reads it); reads the training CSVs, ~30 s."""
    games, folds_df = player_games(train_dir), pd.read_csv(folds_csv)
    return {f: fold_players(games, folds_df, f) for f in folds}


def load_inputs(vectors_dir, folds=ALL_FOLDS, players=None):
    """(vec, fold_of, players) of the folds: the vectors first (a missing file fails before the ~30 s of
    players), then the players (load_eval_players when None), checked against each other."""
    vec, fold_of = load_vectors(vectors_dir, folds)
    players = load_eval_players(folds) if players is None else players
    _check_players(players, vec, fold_of, folds)
    return vec, fold_of, players


def check_production(doc, where="the production model"):
    """The production model must use qfeatures.FEATURE_NAMES and must not have seen fold 0."""
    if doc.get("features") != FEATURE_NAMES:
        raise ValueError(f"{where}: its features are not qfeatures.FEATURE_NAMES")
    if sorted(doc.get("train_folds") or []) != list(CV_FOLDS):
        raise ValueError(f"{where}: trained on folds {doc.get('train_folds')}, not {list(CV_FOLDS)}")


def _games(vec):
    """qfeatures.build_matrix's games frame from the vectors (their sums and n_moves are the score CSVs' values)."""
    return pd.DataFrame({"game_key": vec["game_key"], **{c: vec["sums"][:, i] for i, c in enumerate(SUM_COLS)},
                         "n_moves": vec["n_moves"]})


def production_frame(doc, vec, val_q):
    """Frame (ladder._frame) of the production model `doc` (rank_model.json's content) on the questions:
    rankmodel.from_state(doc["state"]) on qfeatures.build_matrix features, as fit_rank_model scored it."""
    check_production(doc)
    X, _, _ = build_matrix(_games(vec), val_q)
    return _frame(val_q, EVAL_FOLD, {"": from_state(doc["state"]).predict_proba(X)})


# ---------------------------------------------------------------- evaluation

def paired(a, b, what, n_boot=N_BOOT):
    """qeval.paired_diff of a's scores minus b's (frames with KEY_COLS and score) with its n_boot. The
    frames must hold identical questions in the same order -- question_id equal in content and order,
    and so every key column -- else ValueError (ladder._same_keys)."""
    _same_keys(a, b, what)
    y, n = a["y"].to_numpy(dtype=np.int64), a["n_games"].to_numpy(dtype=np.int64)
    d = paired_diff(a["score"].to_numpy(dtype=float), b["score"].to_numpy(dtype=float), y,
                    a["player_id"].to_numpy(dtype=str), n, n=n_boot)
    return {**d, "n_boot": n_boot}


def _block(frames):
    """(metrics, frame) of one model's frames on the same questions, one per seed (one for a static
    model): per question the mean score over seeds, then balanced_mean_over_n and balanced_by_n, and
    the class-balanced NLL / Brier of the seed-mean per-question values (as ladder._rung_summary)."""
    for i, f in enumerate(frames[1:], 1):
        _same_keys(frames[0], f, f"frame {i}")
    keys = frames[0][KEY_COLS]
    y, n = keys["y"].to_numpy(dtype=np.int64), keys["n_games"].to_numpy(dtype=np.int64)
    s = np.mean([f["score"].to_numpy(dtype=float) for f in frames], axis=0)
    nll, brier = (np.mean(v, axis=0) for v in zip(*(nll_brier(f[PCOLS].to_numpy(dtype=float), y) for f in frames)))
    metrics = {"balanced_mean_over_n": balanced_mean_over_n(s, y, n),
               "balanced_by_n": {m: balanced_mean_over_n(s, y, n, sizes=(m,)) for m in SIZES},
               "nll_balanced": balanced(nll, y), "brier": balanced(brier, y)}
    return metrics, keys.assign(score=s)


def evaluate_fold0(rung, vec, players, production, seeds=(0, 1, 2), per_player=DEFAULT_PER_PLAYER, counts=True,
                   device="cuda", n_boot=N_BOOT):
    """Retrain `rung` on folds 1-4 and evaluate it on fold 0 (see the module docstring). vec / players:
    the vectors and fold_players of folds 0-4; production: rank_model.json's content. Returns the
    result in its JSON form (string keys): settings, headline (with ci95, n_boot and, for a set rung,
    per_seed), base, production, vs_base, vs_production, the production note and diagnostics."""
    if rung not in RUNGS:
        raise KeyError(f"unknown rung {rung!r}; known: {list(RUNGS)}")
    is_set = rung in SET_RUNGS
    check_production(production)
    train_q, val_q = training_questions(players, EVAL_FOLD, CV_FOLDS, per_player), validation_questions(players, EVAL_FOLD)
    if is_set:
        seeds = [int(s) for s in seeds]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"a set rung needs distinct seeds, got {seeds}")
        runs, diag = {}, {"train_loss_last": {}}
        with tempfile.TemporaryDirectory(prefix="ladder_fold0_") as cache:  # the base is fitted once, for every seed
            _, base_val, diag["base"] = split_base(EVAL_FOLD, vec, train_q, val_q, counts, per_player, cache)
            for seed in seeds:
                t0 = time.time()
                # fold 0 reports only the chosen configuration (the rung's TrainCfg.pick tick); the EXTRAS live
                # in the CV summary, so that nothing is picked on fold 0 after the fact
                probas, info = _set_split(rung, EVAL_FOLD, vec, train_q, val_q, counts, per_player, seed, device, cache,
                                          extras=False)
                runs[seed] = _frame(val_q, EVAL_FOLD, probas)
                diag["train_loss_last"][seed] = info["train_loss"][-1] if info["train_loss"] else None
                print(f"{rung} seed {seed} trained ({time.time() - t0:.0f} s)", flush=True)
        frames, base = list(runs.values()), _frame(val_q, EVAL_FOLD, {"": _softmax(base_val)})
        base_def = f"softmax of the 4-fold cross-fitted {BASE_RUNG} validation base of the same training questions (Δ ≡ 0)"
    else:
        seeds = None
        probas, diag = _static_split(rung, vec, train_q, val_q, counts)
        frames = [_frame(val_q, EVAL_FOLD, probas)]
        base = frames[0] if rung == BASE_RUNG else _frame(val_q, EVAL_FOLD,
                                                          _static_split(BASE_RUNG, vec, train_q, val_q, counts)[0])
        base_def = f"{BASE_RUNG} refitted on the same training questions (folds 1-4, same per_player and counts)"

    head, rung_frame = _block(frames)
    base_m, base_frame = _block([base])
    prod_m, prod_frame = _block([production_frame(production, vec, val_q)])
    y, n = rung_frame["y"].to_numpy(dtype=np.int64), rung_frame["n_games"].to_numpy(dtype=np.int64)
    pid = rung_frame["player_id"].to_numpy(dtype=str)
    head["ci95"] = list(bootstrap_ci_mean_over_sizes(rung_frame["score"].to_numpy(dtype=float), pid, y, n, n=n_boot))
    head["n_boot"] = n_boot
    vs_base, vs_prod = (paired(rung_frame, base_frame, f"{rung} vs base", n_boot),
                        paired(rung_frame, prod_frame, f"{rung} vs production", n_boot))
    if is_set:
        head["per_seed"] = {s: balanced_mean_over_n(f["score"], y, n) for s, f in runs.items()}
        head["seed_sd"] = float(np.std(list(head["per_seed"].values()), ddof=1)) if len(seeds) > 1 else None
        for c, ref in ((vs_base, base_m), (vs_prod, prod_m)):
            c["per_seed"] = {s: v - ref["balanced_mean_over_n"] for s, v in head["per_seed"].items()}
    return _plain({"rung": rung, "kind": "set" if is_set else "static", "seeds": seeds, "per_player": int(per_player),
                   "counts": bool(counts), "eval_fold": EVAL_FOLD, "train_folds": list(CV_FOLDS),
                   "n_train_questions": len(train_q), "n_questions": len(val_q),
                   "n_players": int(val_q["player_id"].nunique()),
                   "headline": head, "base": {**base_m, "definition": base_def}, "production": prod_m,
                   "vs_base": vs_base, "vs_production": vs_prod, "note_production": PRODUCTION_NOTE, "diagnostics": diag})


# ---------------------------------------------------------------- output

def write_once(path, obj):
    """Write strict JSON (ladder.to_json) to `path` atomically and never over an existing file: the text
    goes to a temporary file beside it, hard-linked to `path` (FileExistsError when `path` exists,
    e.g. written by another run meanwhile) and then removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "x") as f:
            f.write(to_json(obj, indent=1) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def run(rung, out=OUT, chosen_from=CHOSEN_FROM, vectors_dir=VECTORS, rank_model=RANK_MODEL, seeds=(0, 1, 2),
        per_player=DEFAULT_PER_PLAYER, counts=True, device="cuda", n_boot=N_BOOT, players=None):
    """The one fold-0 evaluation: check_request (Refused before anything is read), then the inputs (a
    missing file fails before the players are loaded), evaluate_fold0, provenance (created; the code
    that runs -- ladder.environment: git_head, git_dirty, code_sha256, code_files, torch_version,
    device_name; chosen_from with its sha256; the production model's path and sha256) and write_once
    to `out`. players: {fold: fold_players frame} of folds 0-4, loaded when None. Returns the written result."""
    chosen = check_request(rung, out, chosen_from, per_player, counts, seeds)
    missing = [str(p) for p in [*(Path(vectors_dir) / f"fold{f}.npz" for f in ALL_FOLDS), Path(rank_model)] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"inputs not found: {', '.join(missing)} (vectors: scripts/ladder_build_vectors.py)")
    env = environment(device if rung in SET_RUNGS else "cpu")  # the code that runs, read before the hours of training
    raw = Path(rank_model).read_bytes()
    production = json.loads(raw)
    check_production(production, str(rank_model))
    vec, _, players = load_inputs(vectors_dir, ALL_FOLDS, players)
    res = evaluate_fold0(rung, vec, players, production, seeds, per_player, counts, device, n_boot)
    result = {"created": datetime.now().isoformat(timespec="seconds"), **env, **res, **chosen,
              "production_model": {"path": repo_path(rank_model), "sha256": _sha256(raw)}}
    write_once(out, result)
    return result
