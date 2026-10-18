"""Cross-validation of the L3 aggregator ladder on folds 1-4, its pooled paired summary and the gates.

Protocol (plan 2026-10-19-l3-aggregator-ladder-ctm, Global Constraints):
- Leave one fold out over the CV folds (1-4). Fold 0 is refused everywhere here: it is evaluated
  once, for the chosen rung only, by scripts/ladder_eval_fold0.py.
- Validation questions of split v: pilot.pilot_questions of fold v's scoring.fold_players selection
  (nested 1x20 / 2x10 / 4x5 carvings).
- Training questions of split v: rng = RandomState(42 + v) made once, then qfeatures.sample_questions
  per training fold in ascending order -- exactly fit_rank_model.train_set, so r2_mlp24 at
  per_player=8 reproduces results/rank_model_eval.json. Every rung and seed of v gets the same ones.
- Static rungs (ladder_data.STATIC_RUNGS: kind, rep, make_model) fit make_model() on the stacked
  training features. Set rungs (SET_RUNGS: rep, make(d_in), TrainCfg overrides) train a correction Δ
  to cross-fitted base logits (settrain.crossfit_base_logits with the r2_mlp24 model on q24
  features); the validation base is the r2_mlp24 static rung of the split, which split_base checks.
- Scores are always taken at the final tick; EXTRAS (more ticks at inference, each question's most
  certain tick) are extra columns of the same trained model and never enter the gates.

Files: run_rung_cv returns one frame per (rung, seed) -- question_id, fold, player_id, y, n_games,
p0..p9, score, plus p0@X..p9@X, score@X for each extra X -- with its settings and per-split
diagnostics in DataFrame.attrs. save_cv writes it as <cv_dir>/cv_filename(...) (fixed-width unicode
strings, attrs as a JSON string, loadable with allow_pickle=False); split_base caches the base
logits as <cv_dir>/base_v{v}[_pp{N}][_nocounts].npz.

summarize: per label (rung[@ppN][@nocounts][@extra]) the per-question score is the mean over seeds
first, then every metric is the class-balanced mean per question size averaged over sizes 5/10/20,
per validation fold and on the folds pooled. Comparisons are paired on the pooled questions:
qeval.paired_diff (player bootstrap, n=5000), per-fold and per-seed point differences and the
Nadeau-Bengio corrected interval (variance x 2.33 = 1 + J n_test / n_train for J = 4 folds) as a
conservative reference. gates: every pre-specified comparison of a gate must meet all of CONDITIONS,
on a summary bootstrapped with at least N_BOOT replicates.
"""
import hashlib
import json
import math
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.aggregators import CTM, DeepSets, PooledDelta, QglobDelta, SetTransformer, XAttnGRU
from aicup.ladder_data import STATIC_RUNGS, fit_clip_k, pack_questions, static_features
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.pilot import pilot_questions
from aicup.qeval import SIZES, balanced, balanced_mean_over_n, evaluate, nll_brier, paired_diff
from aicup.qfeatures import sample_questions
from aicup.rankmodel import N_CLASSES, _softmax, class_weights
from aicup.scoring import TRAIN_DIR, fold_players, player_games
from aicup.settrain import BASE_CLIP, TrainCfg, crossfit_base_logits, fit_set_model, predict_set_model

ROOT = Path(__file__).resolve().parents[2]
FOLDS_CSV = ROOT / "splits" / "player_folds.csv"
CV_FOLDS = (1, 2, 3, 4)
DEFAULT_PER_PLAYER = 32
TRAIN_SEED = 42  # training questions of split v: RandomState(TRAIN_SEED + v), as fit_rank_model
CHUNK = 8192  # questions packed at a time for the static features
BASE_RUNG = "r2_mlp24"  # the production model: base of every set rung and reference of every comparison
BASE_TOL = 1e-12
KEY_COLS = ["question_id", "fold", "player_id", "y", "n_games"]

# rung -> (vector rep, make(d_in), TrainCfg overrides); default sizes (Tasks 4-6): XAttnGRU d_hidden 256
# has 458,250 parameters at d_in 301 and CTM 437,514 (ratio 1.047), so no size change was needed
SET_RUNGS = {
    "r2_pooled_delta": ("staged", lambda d: PooledDelta(d), {}),
    "r3_null": ("staged", lambda d: QglobDelta(d), {}),
    "r3_deepsets": ("staged", lambda d: DeepSets(d), {}),
    "r3_settf": ("staged", lambda d: SetTransformer(d), {}),
    "r3_deepsets_whole": ("whole", lambda d: DeepSets(d), {}),
    "r4_xattn": ("staged", lambda d: XAttnGRU(d, d_hidden=256), {}),
    "r5_ctm": ("staged", lambda d: CTM(d), {}),
    "r5_ctm_minmax": ("staged", lambda d: CTM(d), {"loss": "ctm_minmax"}),
}
RUNGS = STATIC_RUNGS | SET_RUNGS
# report-only predictions of the same trained model: name -> predict_set_model keywords
EXTRAS = {
    "r4_xattn": {"T16": {"ticks": 16}},
    "r5_ctm": {"T16": {"ticks": 16}},
    "r5_ctm_minmax": {"T16": {"ticks": 16}, "certain": {"pick": "certain"}},
}

# gates: pre-specified (a, b) comparisons "a better than b"; all must meet every condition
GATES = {
    "G_rep": [("r2_staged", "r2_whole"), ("r3_deepsets", "r3_deepsets_whole")],
    "G_set": [("r3_settf", "r3_deepsets"), ("r3_settf", "r2_pooled_delta")],
    "G_iter": [("r4_xattn", "r3_settf"), ("r5_ctm", "r3_settf")],
    "G_sync": [("r5_ctm", "r4_xattn")],
}
# reported, never gated: per-game nonlinear encoding, training framework, re-calibration, the
# certain-tick pick, and 8 -> 32 training questions per player
REPORT_ONLY = [("r3_deepsets", "r2_pooled_delta"), ("r2_staged", "r2_pooled_delta"), ("r3_null", BASE_RUNG),
               ("r5_ctm_minmax@certain", "r5_ctm"), (f"{BASE_RUNG}@pp8", BASE_RUNG)]
N_BOOT = 5000
NB_VARIANCE = 2.33
MIN_DIFF = 0.005
N_SEEDS = 3
MIN_POSITIVE_FOLDS = 3
CONDITIONS = ("ci_lo_gt_0", "every_seed_gt_0", "folds_gt_0_3of4", "diff_ge_0.005")


# ---------------------------------------------------------------- inputs

def check_folds(folds):
    """Sorted tuple of distinct CV folds; fold 0 is refused."""
    folds = [int(f) for f in folds]
    if 0 in folds:
        raise ValueError("fold 0 is evaluated once (scripts/ladder_eval_fold0.py) and never enters the CV")
    if len(set(folds)) != len(folds) or len(folds) < 2:
        raise ValueError(f"need at least 2 distinct CV folds, got {folds}")
    return tuple(sorted(folds))


def load_players(folds, train_dir=TRAIN_DIR, folds_csv=FOLDS_CSV):
    """{fold: scoring.fold_players selection} of the CV folds (reads the training CSVs, ~30 s)."""
    folds = check_folds(folds)
    games, folds_df = player_games(train_dir), pd.read_csv(folds_csv)
    return {f: fold_players(games, folds_df, f) for f in folds}


def load_vectors(vectors_dir, folds):
    """The gamevec vectors of the folds merged into one dict (each fold{F}.npz read once, by name)
    and the fold of every game."""
    parts = []
    for f in folds:
        path = Path(vectors_dir) / f"fold{f}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} not built (scripts/ladder_build_vectors.py)")
        with np.load(path) as z:
            parts.append(dict(z))
    names = sorted(parts[0])
    if any(sorted(p) != names for p in parts):
        raise ValueError(f"vector files of folds {list(folds)} have different keys")
    vec = {k: np.concatenate([p[k] for p in parts]) for k in names}
    fold_of = np.concatenate([np.full(len(p["game_key"]), f, dtype=np.int64) for f, p in zip(folds, parts)])
    if len(np.unique(vec["game_key"])) != len(fold_of):
        raise ValueError(f"game keys repeat across the vectors of folds {list(folds)}")
    return vec, fold_of


def _check_players(players, vec, fold_of, folds):
    """The vectors of each fold must hold exactly the games its players contribute."""
    for f in folds:
        if f not in players:
            raise KeyError(f"no players for fold {f}")
        want = {k for ks in players[f]["game_keys"] for k in ks}
        have = set(vec["game_key"][fold_of == f].tolist())
        if want != have:
            raise ValueError(f"fold {f}: {len(have)} games in the vectors but {len(want)} selected "
                             f"(missing {len(want - have)}, extra {len(have - want)})")


def training_questions(players, v, folds, per_player):
    """Training questions of split v (fit_rank_model.train_set): one RandomState(42 + v), then
    sample_questions per training fold in ascending order; a `fold` column records the fold."""
    rng = np.random.RandomState(TRAIN_SEED + v)
    return pd.concat([sample_questions(players[f], per_player, rng).assign(fold=f) for f in sorted(folds) if f != v],
                     ignore_index=True)


def validation_questions(players, v):
    return pilot_questions(players[v]).assign(fold=v)


def _labels(qs):
    y = qs["rank"].map(RANK_TO_IDX)
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} questions without a known rank")
    return y.to_numpy(dtype=np.int64)


def _chunks(qs, vec, rep, counts):
    for i in range(0, len(qs), CHUNK):
        yield pack_questions(qs.iloc[i:i + CHUNK], vec, rep, counts)


def _features(qs, vec, kind, rep, counts, clip_k=None):
    """static_features of the questions, packed CHUNK at a time to bound memory; every kind is a
    function of each question alone, so the rows equal those of one packing of all questions."""
    return np.vstack([static_features(p, vec, kind, counts, clip_k) for p in _chunks(qs, vec, rep, counts)])


def _game_rows(qs, vec, rep, counts):
    """{"idx": per question its vec rows}: the part of a packing that fit_clip_k reads."""
    return {"idx": [ix for p in _chunks(qs, vec, rep, counts) for ix in p["idx"]]}


def _fit_predict(make, X_tr, y_tr, X_val):
    return make().fit(X_tr, y_tr).predict_proba(X_val)


def _suffix(per_player, counts):
    return ("" if per_player == DEFAULT_PER_PLAYER else f"_pp{per_player}") + ("" if counts else "_nocounts")


def cv_filename(rung, seed, per_player=DEFAULT_PER_PLAYER, counts=True):
    """{rung}[_pp{N}][_nocounts]_s{seed}.npz"""
    return f"{rung}{_suffix(per_player, counts)}_s{seed}.npz"


# ---------------------------------------------------------------- base logits

BASE_INFO = ("ce_train", "ce_val", "max_dev", "max_dev_unclipped", "n_clipped_rows", "n_train", "n_val")


def _questions_sha(*frames):
    h = hashlib.sha256()
    for qs in frames:
        for qid, keys in zip(qs["question_id"], qs["game_keys"]):
            h.update(f"{qid}:{';'.join(keys)}\n".encode())
        h.update(b"--\n")
    return h.hexdigest()


def _weighted_ce(logits, y):
    """Class-balanced CE of softmax(logits): sum w CE / sum w with rankmodel.class_weights."""
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    w = class_weights(y)
    return float((w * -logp[np.arange(len(y)), y]).sum() / w.sum())


def split_base(v, vec, train_q, val_q, counts=True, per_player=DEFAULT_PER_PLAYER, cache=None):
    """Base logits of split v for the set rungs: (train [N_train, 10], val [N_val, 10], info).

    Training rows: cross-fitted (each training fold from r2_mlp24 fitted on the other training folds);
    validation rows: r2_mlp24 fitted on all training folds, stacked ascending. The validation base must
    equal the r2_mlp24 static rung of the split: max |softmax(base) - p_static| <= 1e-12 on the rows
    whose static probabilities are all >= 1e-9 (below that the log clip of the base moves them); the
    max deviation over every row is recorded. info: class-balanced CE of the base on the training
    (cross-fitted) and validation questions, the deviations and the counts. With `cache` (a directory)
    the result is stored as base_v{v}[_pp{N}][_nocounts].npz and reused; a file made for other
    questions is refused.
    """
    sha = _questions_sha(train_q, val_q)
    path = None if cache is None else Path(cache) / f"base_v{v}{_suffix(per_player, counts)}.npz"
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as z:
            d = dict(z)
        if str(d["questions_sha256"]) != sha:
            raise ValueError(f"{path} is stale: it was made for other questions; delete it")
        return d["train"], d["val"], {k: d[k].item() for k in BASE_INFO}

    kind, rep, make = STATIC_RUNGS[BASE_RUNG]
    X, X_val = _features(train_q, vec, kind, rep, counts), _features(val_q, vec, kind, rep, counts)
    y, y_val = _labels(train_q), _labels(val_q)
    fold = train_q["fold"].to_numpy()
    if np.any(np.diff(fold) < 0):
        raise ValueError("training questions must be in ascending fold order")
    tr_folds = sorted(set(fold.tolist()))
    by_fold, base_val = crossfit_base_logits(make, {f: X[fold == f] for f in tr_folds}, {f: y[fold == f] for f in tr_folds},
                                             X_val)
    base = np.concatenate([by_fold[f] for f in tr_folds])
    p_static = _fit_predict(make, X, y, X_val)  # the static rung of this split
    dev = np.abs(_softmax(base_val) - p_static)
    unclipped = p_static.min(axis=1) >= BASE_CLIP
    info = {"ce_train": _weighted_ce(base, y), "ce_val": _weighted_ce(base_val, y_val), "max_dev": float(dev.max()),
            "max_dev_unclipped": float(dev[unclipped].max()) if unclipped.any() else 0.0,
            "n_clipped_rows": int((~unclipped).sum()), "n_train": len(y), "n_val": len(y_val)}
    if info["max_dev_unclipped"] > BASE_TOL:
        raise AssertionError(f"split {v}: validation base differs from the {BASE_RUNG} static proba by "
                             f"{info['max_dev_unclipped']:.3g} > {BASE_TOL}")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, train=base, val=base_val, y_train=y, y_val=y_val, questions_sha256=np.asarray(sha),
                            **{k: np.asarray(val) for k, val in info.items()})
        os.replace(tmp, path)
    return base, base_val, info


# ---------------------------------------------------------------- CV

def _static_split(rung, vec, train_q, val_q, counts):
    kind, rep, make = STATIC_RUNGS[rung]
    info, clip_k = {}, None
    if kind == "clip":  # the clip width comes from the training questions only
        clip_k = info["clip_k"] = fit_clip_k(_game_rows(train_q, vec, rep, counts), vec)
    X_tr, X_val = _features(train_q, vec, kind, rep, counts, clip_k), _features(val_q, vec, kind, rep, counts, clip_k)
    return {"": _fit_predict(make, X_tr, _labels(train_q), X_val)}, info


def _set_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, cache):
    rep, make, overrides = SET_RUNGS[rung]
    base, base_val, base_info = split_base(v, vec, train_q, val_q, counts, per_player, cache)
    cfg = replace(TrainCfg(seed=seed, device=device), **overrides)
    data = pack_questions(train_q, vec, rep, counts)
    fitted = fit_set_model(make, data, base, cfg)
    del data  # the training tokens (several GB) before packing the validation questions
    val = pack_questions(val_q, vec, rep, counts)
    out = {"": predict_set_model(fitted, val, base_val)}
    for name, kw in EXTRAS.get(rung, {}).items():
        out[name] = predict_set_model(fitted, val, base_val, **kw)
    return out, {"base": base_info, "train_loss": fitted.losses}


def _frame(val_q, v, probas):
    """Validation frame: KEY_COLS, then p0..p9 and score per prediction ("" first, then each extra X as @X)."""
    y = _labels(val_q)
    cols = {"question_id": val_q["question_id"].to_numpy(dtype=str), "fold": np.full(len(val_q), v, dtype=np.int64),
            "player_id": val_q["player_id"].to_numpy(dtype=str), "y": y, "n_games": val_q["n_games"].to_numpy(dtype=np.int64)}
    for name, P in probas.items():
        sfx = f"@{name}" if name else ""
        _, s = evaluate(P, y, val_q)
        cols.update({f"p{k}{sfx}": P[:, k] for k in range(N_CLASSES)})
        cols[f"score{sfx}"] = s
    return pd.DataFrame(cols)


def run_rung_cv(rung, vectors_dir, folds=CV_FOLDS, seed=0, per_player=DEFAULT_PER_PLAYER, counts=True, device="cuda",
                base_cache=None, players=None):
    """Leave-one-fold-out CV of one rung (seed: set models' initialisation and shuffling; static rungs
    are deterministic). Returns the validation questions of every fold with KEY_COLS, p0..p9, score
    (and @X columns for EXTRAS). attrs: rung, kind, seed, per_player, counts, folds, n_train per split
    and, by rung, clip_k (r1_clip) or base / train_loss (set rungs) per split.

    base_cache: directory for split_base's files (None: no caching). players: {fold: fold_players
    frame}, loaded with load_players when None.
    """
    folds = check_folds(folds)
    is_set = rung in SET_RUNGS
    if not is_set and rung not in STATIC_RUNGS:
        raise KeyError(f"unknown rung {rung!r}; known: {list(STATIC_RUNGS) + list(SET_RUNGS)}")
    vec, fold_of = load_vectors(vectors_dir, folds)
    players = load_players(folds) if players is None else players
    _check_players(players, vec, fold_of, folds)
    frames, per_split = [], {}
    for v in folds:
        train_q, val_q = training_questions(players, v, folds, per_player), validation_questions(players, v)
        if is_set:
            probas, info = _set_split(rung, v, vec, train_q, val_q, counts, per_player, seed, device, base_cache)
        else:
            probas, info = _static_split(rung, vec, train_q, val_q, counts)
        frames.append(_frame(val_q, v, probas))
        per_split[str(v)] = {**info, "n_train": len(train_q)}
    df = pd.concat(frames, ignore_index=True)
    df.attrs = {"rung": rung, "kind": "set" if is_set else "static", "seed": int(seed), "per_player": int(per_player),
                "counts": bool(counts), "folds": list(folds)}
    for key in ("n_train", "clip_k", "base", "train_loss"):
        if key in per_split[str(folds[0])]:
            df.attrs[key] = {v: d[key] for v, d in per_split.items()}
    return df


# ---------------------------------------------------------------- files

def _plain(x):
    """JSON-able copy: numpy scalars and arrays to Python, dict keys to str, non-finite floats to None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def to_json(obj, **kw):
    """Strict JSON (allow_nan=False) of obj after _plain, so NaN / inf become null."""
    return json.dumps(_plain(obj), allow_nan=False, **kw)


def save_cv(df, path):
    """Write a run_rung_cv frame and its attrs (as a JSON string) to an npz, atomically. String columns
    are fixed-width unicode, so np.load(path, allow_pickle=False) reads it back (see load_cv)."""
    arrays = {c: df[c].to_numpy() if pd.api.types.is_numeric_dtype(df[c]) else df[c].to_numpy(dtype=str)
              for c in df.columns}
    arrays["__meta__"] = np.asarray(to_json({**df.attrs, "columns": list(df.columns)}))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_cv(path):
    """(frame, meta) of a save_cv file; meta is also the frame's attrs."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        columns = meta.pop("columns")
        df = pd.DataFrame({c: z[c] for c in columns})
    df.attrs = dict(meta)
    return df, meta


# ---------------------------------------------------------------- summary

_RUN_FILE = re.compile(r".+_s\d+\.npz")


def _label(rung, per_player, counts, extra=None):
    return (rung + ("" if per_player == DEFAULT_PER_PLAYER else f"@pp{per_player}") + ("" if counts else "@nocounts")
            + (f"@{extra}" if extra else ""))


def _same_keys(a, b, what):
    for c in KEY_COLS:
        if not np.array_equal(np.asarray(a[c]).astype(str), np.asarray(b[c]).astype(str)):
            raise ValueError(f"{what}: the validation questions differ (column {c}); paired results need identical questions")


def _read_runs(cv_dir):
    """{label: {"rung", "kind", "per_player", "counts", "extra", "keys", "scores": {seed: [Q]}, "proba": {seed: [Q, 10]},
    "meta": {seed: meta}}} from the save_cv files of cv_dir; each extra column set is its own label."""
    runs = {}
    for path in sorted(Path(cv_dir).glob("*.npz")):
        if not _RUN_FILE.fullmatch(path.name) or path.name.startswith("base_"):
            continue
        df, meta = load_cv(path)
        rung, seed, pp, counts = meta["rung"], int(meta["seed"]), int(meta["per_player"]), bool(meta["counts"])
        if path.name != cv_filename(rung, seed, pp, counts):
            raise ValueError(f"{path.name}: the name does not match its contents ({cv_filename(rung, seed, pp, counts)})")
        extras = [None] + [c[len("score@"):] for c in df.columns if c.startswith("score@")]
        for extra in extras:
            label, sfx = _label(rung, pp, counts, extra), f"@{extra}" if extra else ""
            run = runs.setdefault(label, {"rung": rung, "kind": meta["kind"], "per_player": pp, "counts": counts, "extra": extra,
                                          "keys": df[KEY_COLS], "scores": {}, "proba": {}, "meta": {}})
            _same_keys(run["keys"], df, f"{label} seed {seed}")
            run["scores"][seed] = df[f"score{sfx}"].to_numpy(dtype=float)
            run["proba"][seed] = df[[f"p{k}{sfx}" for k in range(N_CLASSES)]].to_numpy(dtype=float)
            run["meta"][seed] = meta
    for run in runs.values():
        run["seeds"] = sorted(run["scores"])
        run["mean"] = np.mean([run["scores"][s] for s in run["seeds"]], axis=0)  # seed mean per question
    return runs


def _arrays(keys):
    return keys["y"].to_numpy(dtype=np.int64), keys["n_games"].to_numpy(dtype=np.int64), keys["fold"].to_numpy(dtype=np.int64)


def _per_fold(s, y, n, fold):
    return {int(f): balanced_mean_over_n(s[fold == f], y[fold == f], n[fold == f]) for f in np.unique(fold)}


def _rung_summary(run):
    y, n, fold = _arrays(run["keys"])
    s = run["mean"]
    per_seed = {seed: balanced_mean_over_n(run["scores"][seed], y, n) for seed in run["seeds"]}
    # per question, the mean over seeds of qeval.nll_brier (evaluate's definitions), then class-balanced
    nll, brier = (np.mean(v, axis=0) for v in zip(*(nll_brier(run["proba"][seed], y) for seed in run["seeds"])))
    out = {"rung": run["rung"], "kind": run["kind"], "per_player": run["per_player"], "counts": run["counts"],
           "extra": run["extra"], "report_only": run["extra"] is not None, "seeds": run["seeds"], "n_questions": len(y),
           "balanced_mean_over_n": balanced_mean_over_n(s, y, n),
           "balanced_by_n": {m: balanced_mean_over_n(s, y, n, sizes=(m,)) if (n == m).any() else None for m in SIZES},
           "per_fold": _per_fold(s, y, n, fold),
           "per_class": {LEVELS[k]: float(s[y == k].mean()) if (y == k).any() else None for k in range(N_CLASSES)},
           "nll_balanced": balanced(nll, y), "brier": balanced(brier, y),
           "per_seed": per_seed, "seed_sd": float(np.std(list(per_seed.values()), ddof=1)) if len(per_seed) > 1 else None}
    first = run["meta"][run["seeds"][0]]
    if "clip_k" in first:
        out["clip_k"] = first["clip_k"]
    if "train_loss" in first:  # last-epoch training loss per seed and split
        out["train_loss_last"] = {seed: {v: losses[-1] if losses else None
                                         for v, losses in run["meta"][seed]["train_loss"].items()}
                                  for seed in run["seeds"]}
    return out


def _seed_pairs(A, B):
    """Per-seed pairs (key, seed of a, seed of b). Only a static side (deterministic, seed 0) is reused
    against every seed of the other side; two set rungs pair the seeds both have, so a set rung that
    has run fewer seeds yields fewer pairs (and check_comparison counts each side's seeds)."""
    sa, sb = A["seeds"], B["seeds"]
    if A["kind"] == "static" and B["kind"] == "static":
        return [(sa[0], sa[0], sb[0])]
    if B["kind"] == "static":
        return [(s, s, sb[0]) for s in sa]
    if A["kind"] == "static":
        return [(s, sa[0], s) for s in sb]
    return [(s, s, s) for s in sorted(set(sa) & set(sb))]


def _compare(a, b, A, B, n_boot):
    """Paired comparison a - b on the pooled questions: the seed-mean point difference with its player
    bootstrap CI (qeval.paired_diff), the Nadeau-Bengio interval (the CI's distances from the point
    difference scaled by sqrt(2.33), i.e. variance x 2.33), per-fold and per-seed point differences."""
    _same_keys(A["keys"], B["keys"], f"{a} vs {b}")
    y, n, fold = _arrays(A["keys"])
    pid = A["keys"]["player_id"].to_numpy(dtype=str)
    d = paired_diff(A["mean"], B["mean"], y, pid, n, n=n_boot)
    diff, (lo, hi) = d["diff"], d["ci95"]
    k = math.sqrt(NB_VARIANCE)
    fa, fb = _per_fold(A["mean"], y, n, fold), _per_fold(B["mean"], y, n, fold)
    per_seed = {key: balanced_mean_over_n(A["scores"][s1], y, n) - balanced_mean_over_n(B["scores"][s2], y, n)
                for key, s1, s2 in _seed_pairs(A, B)}
    return {"a": a, "b": b, "diff": diff, "ci95": [lo, hi], "by_n": d["by_n"],
            "nb_ci95": [diff - k * (diff - lo), diff + k * (hi - diff)],
            "per_fold": {f: fa[f] - fb[f] for f in fa}, "per_seed": per_seed,
            "kind_a": A["kind"], "kind_b": B["kind"], "seeds_a": A["seeds"], "seeds_b": B["seeds"], "n_boot": n_boot}


def comparison_pairs(labels):
    """Pre-specified comparisons among the labels present: every label vs r2_mlp24, the gate pairs, the
    report-only pairs, and each @nocounts label vs its counts form (the game-length ablation)."""
    labels = list(labels)
    pairs = [(x, BASE_RUNG) for x in labels if x != BASE_RUNG]
    pairs += [p for gate in GATES.values() for p in gate] + REPORT_ONLY
    pairs += [(x, x.replace("@nocounts", "")) for x in labels if "@nocounts" in x]
    present = set(labels)
    return list(dict.fromkeys((a, b) for a, b in pairs if a in present and b in present))


def summarize(cv_dir, n_boot=N_BOOT):
    """Pooled CV summary of every save_cv file in cv_dir (see the module docstring), with the
    split_base diagnostics of its base files. Returned in its JSON form (string keys, non-finite
    values as None), so gates() reads it and cv_summary.json alike."""
    cv_dir = Path(cv_dir)
    runs = _read_runs(cv_dir)
    rungs = {label: _rung_summary(run) for label, run in runs.items()}
    comparisons = {f"{a} vs {b}": _compare(a, b, runs[a], runs[b], n_boot) for a, b in comparison_pairs(runs)}
    base = {}
    for path in sorted(cv_dir.glob("base_v*.npz")):
        if not path.name.endswith(".tmp.npz"):
            with np.load(path, allow_pickle=False) as z:
                base[path.stem] = {k: z[k].item() for k in BASE_INFO}
    folds = sorted({int(f) for run in runs.values() for f in run["keys"]["fold"].unique()})
    return _plain({"created": datetime.now().isoformat(timespec="seconds"), "cv_dir": str(cv_dir),
            "protocol": {"folds": folds, "sizes": list(SIZES), "train_questions": f"RandomState({TRAIN_SEED} + v)",
                         "n_boot": n_boot, "nb_variance_factor": NB_VARIANCE, "min_diff": MIN_DIFF, "n_seeds": N_SEEDS,
                         "min_positive_folds": f"{MIN_POSITIVE_FOLDS} of {len(CV_FOLDS)}"},
            "rungs": rungs, "comparisons": comparisons, "diagnostics": {"base": base}})


# ---------------------------------------------------------------- gates

def _gt(x, t):
    return x is not None and x > t


def check_comparison(c):
    """The four conditions of one comparison: (1) the seed-mean paired CI lower bound > 0, (2) every
    per-seed point difference > 0, where each set-rung side must have run >= 3 seeds (a static side is
    deterministic, seed 0, and is compared with each seed of the other side; two static rungs give
    one difference) and two set rungs need >= 3 seeds in common, (3) >= 3 of the 4 validation folds'
    point differences > 0, (4) the point difference >= 0.005. The comparison passes only with all
    four and a CI from >= N_BOOT bootstrap replicates (a summary made with fewer cannot decide)."""
    per_seed, per_fold = list(c["per_seed"].values()), list(c["per_fold"].values())
    sides = [(side, c[f"kind_{side}"], len(c[f"seeds_{side}"])) for side in ("a", "b")]
    short = [f"{side} is a set rung with {k} seed{'s' if k != 1 else ''}" for side, kind, k in sides
             if kind != "static" and k < N_SEEDS]
    need = N_SEEDS if any(kind != "static" for _, kind, _ in sides) else 1
    cond = {"ci_lo_gt_0": _gt(c["ci95"][0], 0.0),
            "every_seed_gt_0": not short and len(per_seed) >= need and all(_gt(d, 0.0) for d in per_seed),
            "folds_gt_0_3of4": len(per_fold) == len(CV_FOLDS) and sum(_gt(d, 0.0) for d in per_fold) >= MIN_POSITIVE_FOLDS,
            "diff_ge_0.005": c["diff"] is not None and c["diff"] >= MIN_DIFF}
    boot_ok = (c.get("n_boot") or 0) >= N_BOOT
    reasons = []
    if not cond["ci_lo_gt_0"]:
        reasons.append(f"CI lower bound {c['ci95'][0]} <= 0")
    if not cond["every_seed_gt_0"]:
        reasons.append(f"per-seed differences {per_seed} (need {need} seeds per set rung, all > 0"
                       + (f"; {', '.join(short)}" if short else "") + ")")
    if not cond["folds_gt_0_3of4"]:
        reasons.append(f"per-fold differences {per_fold} (need {MIN_POSITIVE_FOLDS} of {len(CV_FOLDS)} > 0)")
    if not cond["diff_ge_0.005"]:
        reasons.append(f"difference {c['diff']} < {MIN_DIFF}")
    if not boot_ok:
        reasons.append(f"CI from n_boot {c.get('n_boot')} < {N_BOOT} replicates: this summary cannot decide gates")
    return {"pass": all(cond.values()) and boot_ok, "missing": False, "conditions": cond, "n_boot_ok": boot_ok,
            "reasons": reasons, "diff": c["diff"], "ci95": c["ci95"]}


def gates(summary):
    """{gate: {"pass", "comparisons": {"a vs b": check_comparison result}}} for GATES; a comparison
    absent from the summary fails as missing."""
    out = {}
    for name, pairs in GATES.items():
        comps = {}
        for a, b in pairs:
            key = f"{a} vs {b}"
            c = summary.get("comparisons", {}).get(key)
            comps[key] = check_comparison(c) if c is not None else {
                "pass": False, "missing": True, "conditions": {k: False for k in CONDITIONS}, "n_boot_ok": False,
                "reasons": [f"no CV results for {key}"]}
        out[name] = {"pass": all(c["pass"] for c in comps.values()), "comparisons": comps}
    return out


def g_set_ready(summary):
    """(ok, reason): the iterative rungs (4)(5) run only when the r3 rungs and their controls are in
    the summary and G_set holds."""
    need = list(dict.fromkeys(x for pair in GATES["G_set"] for x in pair))
    missing = [x for x in need if x not in summary.get("rungs", {})]
    if missing:
        return False, (f"G_set cannot be checked: no CV results for {', '.join(missing)}; "
                       f"run them first, or pass --force to run r4/r5 anyway")
    g = gates(summary)["G_set"]
    if not g["pass"]:
        why = "; ".join(f"{k}: {', '.join(c['reasons'])}" for k, c in g["comparisons"].items() if not c["pass"])
        return False, f"G_set does not hold ({why}); pass --force to run r4/r5 anyway"
    return True, "G_set holds"
