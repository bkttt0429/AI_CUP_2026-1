#!/usr/bin/env python
"""Official test CSVs -> submission_rank.csv using the Human SL features and the saved L3/L4 model.

For every question: drop unusable games (wrong size, illegal replay, non-alternating colours, no
target moves), score the rest under the 18 rank profiles, build the question features, predict with
the saved model and the expected-score decision. Public questions are written first, then private,
and the file is validated before the script exits. Also writes a JSON report (games used / dropped
and why, passes, shard files, prediction counts, problems, exit code) and the per-question class
probabilities.

Scoring is sharded over GPUs (2026-10-25, item 1 of the to-do list). The task list -- every usable
game in input order -- is built identically by every process from the inputs alone; shard i of N
scores the tasks i, i+N, ... and writes <out stem>_shard{i}of{N}.npz next to --out: per game its
n_moves and 18 log p sums, per move the log p under the 18 profiles with its task and move index (the
input of later per-game vectors), and a JSON meta with a hash of the questions. The merge reads the N
files, checks that they were made from the same questions and sharding and that every task was scored
exactly once, then predicts, writes and validates. The single-GPU run is shard 0 of 1 followed by the
merge in the same process, so it writes the same files.

  python -I scripts/predict_submission.py ...                                  # one GPU
  bash scripts/nano4/shards.sh 8 python -I scripts/predict_submission.py ...   # 8 GPUs: adds --shard i --nshards 8
  python -I scripts/predict_submission.py --merge --nshards 8 ...              # then, on any node, no GPU

With --set-models (item 2) the decision comes from set models instead of the saved L3/L4 model: each
.pt + .json (aicup.setmodel_io) is a Δ-correction trained by scripts/train_set_model.py on top of the
paired base --set-base (a rank_model.json fitted on the same training questions; a sidecar's training
question hash must equal the base's, exit 2 otherwise). The merge reads the
shards' per-move log p, builds every game's staged and whole vectors (aicup.gamevec) and the packed
question sets (aicup.ladder_data.pack_questions), takes the base logits log(clip(base probabilities)),
runs every model and averages the probabilities first over the seeds of a rung, then over the rungs
(uniform), before the expected-score decision -- as the CV ensemble ens_top3 does. Three r3_pma seeds are
r3_pma; adding r4_xattn_t1 and r5_ctm makes ens_top3. A question whose set-model probabilities are not
finite falls back to the base's probabilities and is listed in the report (set_model_fallback_questions);
a question with no usable game still gets FALLBACK. Single-GPU and sharded runs give the same decisions,
not bit-identical vectors (TF32 and batch composition move log p by up to ~4e-3 nats).

With --l1b-heads (issue #9) the shard step also scores every position under each L1b heads file's 10 rank
heads (aicup.l1b; a fine-tuned trunk named in the sidecar is restored into its own model instance), one
trunk forward per model next to the 18 profile passes, and the shard file gains l1b_sums [n, K, 10] and
l1b_move_logp [M, K, 10] plus the models' names and sha256s. --fused-model (rank_model_fused.json,
scripts/fit_fused_model.py) then decides at the merge: the shards' L1b models must be exactly the
document's (state and trunk sha256), the fused MLP runs once per model on that model's fused features
(aicup.fused) and the probabilities are averaged before the expected-score decision; a question whose
fused probabilities are not finite gets --model's and is listed (fused_fallback_questions). --fused-model
and --set-models exclude each other; shards scored without --l1b-heads cannot feed a fused decision (exit 2).

--expect-rows / --expect-public / --expect-private (defaults: the official 800 = 400 + 400; 0 disables
one) are checked against the questions read, so a truncated input fails validation; a mock test passes
its own counts. A malformed input (a question whose num_games disagrees with its filled sgf columns)
stops the run before scoring; --lenient-inputs, the deadline-day escape hatch, reads such a question
from its non-empty columns instead and lists it in the report's input_warnings (every shard and the
merge must pass the flag alike). Exit codes: 0 ok; 1 the submission failed validation; 2 malformed
inputs, a model trained on other features, or missing / inconsistent shard files; 3 a question with no
usable game was filled with FALLBACK (--allow-fallback accepts that with a warning and exit 0); 4 the
inputs were read leniently with warnings (--allow-input-warnings accepts). The submission, the
probabilities and the report are written in every case the merge reaches; a shard run exits 0 once its
file is written, the merge decides the final code.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.fused import fused_predict, load_fused_model, pairing_order  # noqa: E402
from aicup.gamevec import SEGMENTS, WHOLE, game_vector  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.ladder_data import pack_questions  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qfeatures import FEATURE_NAMES, question_features  # noqa: E402
from aicup.rankmodel import _softmax, from_state  # noqa: E402
from aicup.setmodel_io import file_sha256, load_set_model, sidecar_path, summary as set_model_summary  # noqa: E402
from aicup.settrain import BASE_CLIP, predict_set_model  # noqa: E402
from aicup.submission import check_game, read_test, read_test_lenient, validate_submission, write_submission  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
FALLBACK = "1D"  # used only for a question with no usable game: an interior class keeps both neighbours
EXIT_OK, EXIT_INVALID, EXIT_INPUTS, EXIT_FALLBACK, EXIT_INPUT_WARNINGS = 0, 1, 2, 3, 4
DEFAULT_INPUTS = [str(ROOT / "data" / "tests" / "rank_prediction_test_public.csv"),
                  str(ROOT / "data" / "tests" / "rank_prediction_test_private.csv")]


def load_questions(inputs, lenient=False):
    """(questions of the input files in order, input warnings). Strict (the default): aicup.submission.read_test,
    ValueError on a malformed file and no warnings. lenient: read_test_lenient, a malformed question read
    from its non-empty sgf columns and one warning per such question."""
    questions, warnings = [], []
    for path in inputs:
        if lenient:
            qs, ws = read_test_lenient(path)
            warnings += ws
        else:
            qs = read_test(path)
        questions += qs
    return questions, warnings


def build_tasks(questions):
    """(tasks [(k, sgf, colour)] over the usable games in input order, owner [question index of task k],
    dropped [{question_id, game, error}], passes). Deterministic: the same in every process reading the
    same inputs, which is what lets shards score disjoint slices of it."""
    tasks, owner, dropped, passes = [], [], [], 0
    for qi, q in enumerate(questions):
        for gi, (sgf, color) in enumerate(q["games"]):
            ok, info = check_game(sgf, color)
            passes += info["passes"]
            if ok:
                owner.append(qi)
                tasks.append((len(tasks), sgf, color))
            else:
                dropped.append({"question_id": q["question_id"], "game": gi + 1, "error": info["error"]})
    return tasks, owner, dropped, passes


def questions_sha(inputs, questions):
    """sha256 over the input file names and every question's id and game count: the shards of one run
    must agree on it, so a merge never mixes files made from other inputs."""
    h = hashlib.sha256()
    for path in inputs:
        h.update(Path(path).name.encode() + b"\n")
    for q in questions:
        h.update(f"{q['question_id']}:{len(q['games'])}\n".encode())
    return h.hexdigest()


def shard_path(out, shard, nshards):
    """<out stem>_shard{shard}of{nshards}.npz next to `out`."""
    out = Path(out)
    return out.with_name(f"{out.stem}_shard{shard}of{nshards}.npz")


def l1b_models_meta(paths):
    """The shard meta's record of the L1b heads files: name (stem), path, sha256s and training folds from their sidecars
    (aicup.l1b.save_heads); ValueError when a sidecar is missing."""
    out = []
    for p in paths:
        side = sidecar_path(p)
        if not side.exists():
            raise ValueError(f"{p}: sidecar {side.name} is missing")
        meta = json.loads(side.read_text())
        out.append({"name": Path(p).stem, "path": str(p), "state_sha256": meta.get("state_sha256"), "trunk_file": meta.get("trunk_file"),
                    "trunk_sha256": meta.get("trunk_sha256"), "train_folds": meta.get("train_folds"), "levels": meta.get("levels")})
    return out


def score_tasks(tasks, workers, log=print, l1b_heads=()):
    """Score (k, sgf, colour) tasks on this GPU under the 18 profiles and, per L1b heads file, under its 10 rank
    heads: ({k: (n_moves, sums[18 + 10K])}, (move_logp [M, 18 + 10K], move_task [M], move_t [M]) or None, positions,
    wall seconds). Each heads file gets its own model instance (load_heads restores a fine-tuned trunk in place);
    the positions' features are computed once for every model."""
    import torch
    from aicup.humansl import load_humansl, meta_rows, score_positions as hsl_score
    from aicup.l1b import load_heads, score_positions as l1b_score
    from aicup.scoring import init_worker, score_stream
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    log(f"torch CPU threads: {torch.get_num_threads()} (scripts/nano4/shards.sh caps them per shard)")
    pool = mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda")
        l1b = []
        for p in l1b_heads:
            model_k, _ = load_humansl(str(CKPT), "cuda")
            heads_k, meta_k, warnings = load_heads(p, model_k, "cuda")
            for w in warnings:
                log(f"WARNING: {w}")
            l1b.append((model_k, heads_k))
        rows = meta_rows([(p, p) for p in PROFILES])

        def scorer(bins, globs, legal, target, colors, rows_):
            parts = [hsl_score(model, bins, globs, legal, target, colors, rows_)]
            parts += [l1b_score(m, h, bins, globs, legal, target, colors) for m, h in l1b]
            return np.concatenate(parts, axis=1)

        return score_stream(pool, model, tasks, rows, log_every=500, log=log, scorer=scorer if l1b else None)
    finally:
        pool.close()


def save_shard(path, shard, nshards, tasks, per_task, moves, n_pos, wall, inputs, sha, n_tasks_total, l1b_models=()):
    """Write a shard file atomically: task (the global indices scored), n_moves, sums [n, 18] (float64),
    move_logp [M, 18], move_task [M], move_t [M] and __meta__ (JSON). With l1b_models (K heads files, scored as
    columns 18.. of the sums and per-move log p) also l1b_sums [n, K, 10] (float64) and l1b_move_logp [M, K, 10]
    (float32), and the meta lists the models. Returns the meta."""
    P, L, K = len(PROFILES), len(LEVELS), len(l1b_models)
    idx = np.array([k for k, _, _ in tasks], dtype=np.int64)
    n_moves = np.array([per_task[k][0] for k in idx], dtype=np.int64)
    all_sums = np.stack([per_task[k][1] for k in idx]).astype(np.float64) if len(idx) else np.zeros((0, P + L * K))
    if all_sums.shape[1] != P + L * K:
        raise ValueError(f"scored {all_sums.shape[1]} columns per game, expected {P} profiles + {K} x {L} L1b heads")
    if moves is None:
        moves = (np.zeros((0, P + L * K), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
    meta = {"shard": shard, "nshards": nshards, "n_tasks_total": n_tasks_total, "n_tasks": int(len(idx)),
            "inputs": [str(p) for p in inputs], "questions_sha256": sha, "positions": int(n_pos), "scoring_wall_s": float(wall),
            "profiles": list(PROFILES), "ckpt": CKPT.name, "l1b_models": [dict(m) for m in l1b_models], "levels": list(LEVELS),
            "written": datetime.now().isoformat(timespec="seconds")}
    arrays = {"task": idx, "n_moves": n_moves, "sums": all_sums[:, :P], "move_logp": moves[0][:, :P], "move_task": moves[1], "move_t": moves[2]}
    if K:
        arrays["l1b_sums"] = all_sums[:, P:].reshape(len(idx), K, L)
        arrays["l1b_move_logp"] = moves[0][:, P:].reshape(len(moves[1]), K, L).astype(np.float32)
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, __meta__=np.asarray(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    return meta


def load_shards_l1b(out, nshards, with_moves=False):
    """([L1b model records], {k: l1b sums [K, 10]}, moves) from the shard files: the models (name, state and trunk
    sha256) must be the same in every shard; shards without L1b scores give ([], {}, None). moves is None, or with
    with_moves {k: (logp [n_moves, K, 10], t [n_moves])}."""
    models, per_task, moves = None, {}, {} if with_moves else None
    for i in range(nshards):
        path = shard_path(out, i, nshards)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            here = [(m.get("name"), m.get("state_sha256"), m.get("trunk_sha256")) for m in meta.get("l1b_models", [])]
            if models is None:
                models, key = meta.get("l1b_models", []), here
            elif here != key:
                raise ValueError(f"{path}: L1b models {here} differ from shard 0's {key}; every shard must be scored with the same heads")
            if not here:
                continue
            for k, s in zip(z["task"].tolist(), z["l1b_sums"]):
                per_task[k] = s.astype(np.float64)
            if with_moves:
                mt, logp, t = z["move_task"].astype(np.int64), z["l1b_move_logp"].astype(np.float64), z["move_t"].astype(np.int64)
                order = np.argsort(mt, kind="stable")
                mt, logp, t = mt[order], logp[order], t[order]
                tasks_here, starts = np.unique(mt, return_index=True)
                ends = np.append(starts[1:], len(mt))
                for k, a, b in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):
                    moves[k] = (logp[a:b], t[a:b])
    return models or [], per_task, moves


def fused_probs(questions, owner, per_task, l1b_per_task, mlp, order, rank_model):
    """({question index: probabilities [10]} for every question with a usable game, info): the fused MLP on each
    L1b model's fused features (shard columns `order`), averaged; a question whose averaged probabilities are not
    finite gets rank_model's (q24) probabilities and is listed in info["fused_fallback_questions"]."""
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    probs, fallback = {}, []
    for qi, ks in by_q.items():
        hsl = np.stack([per_task[k][1] for k in ks])
        n = np.array([per_task[k][0] for k in ks])
        missing = [k for k in ks if k not in l1b_per_task]
        if missing:
            raise ValueError(f"question {questions[qi]['question_id']}: {len(missing)} games have no L1b scores in the shards")
        p = fused_predict(mlp, hsl, np.stack([l1b_per_task[k] for k in ks]), n, order)
        if not np.isfinite(p).all():
            p = rank_model.predict_proba(question_features(hsl, n)[None, :])[0]
            fallback.append(questions[qi]["question_id"])
        probs[qi] = p
    return probs, {"fused_fallback_questions": fallback}


def load_shards(out, nshards, n_tasks_total, sha, with_moves=False):
    """({k: (n_moves, sums)}, [meta per shard], moves) from the nshards files of `out`. Every file must exist,
    be shard i of nshards made from the same questions (questions_sha256), and together they must score every
    task 0..n_tasks_total-1 exactly once; FileNotFoundError / ValueError otherwise. moves is None, or with
    with_moves {k: (logp [n_moves, 18] float64, t [n_moves])} -- each task's per-move log p and move indices,
    checked to count n_moves."""
    per_task, metas, moves = {}, [], {} if with_moves else None
    for i in range(nshards):
        path = shard_path(out, i, nshards)
        if not path.exists():
            raise FileNotFoundError(f"shard file {path} is missing ({nshards} shards expected)")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            if (meta["shard"], meta["nshards"], meta["questions_sha256"]) != (i, nshards, sha):
                raise ValueError(f"{path}: made for another sharding or other inputs (shard {meta['shard']} of {meta['nshards']}, "
                                 f"questions {meta['questions_sha256'][:12]}; expected {i} of {nshards}, {sha[:12]})")
            for k, n, s in zip(z["task"].tolist(), z["n_moves"].tolist(), z["sums"]):
                if k in per_task:
                    raise ValueError(f"task {k} was scored by two shards (shard {i} and an earlier one)")
                per_task[k] = (int(n), s)
            if with_moves:
                mt, logp, t = z["move_task"].astype(np.int64), z["move_logp"].astype(np.float64), z["move_t"].astype(np.int64)
                order = np.argsort(mt, kind="stable")
                mt, logp, t = mt[order], logp[order], t[order]
                tasks_here, starts = np.unique(mt, return_index=True)
                ends = np.append(starts[1:], len(mt))
                for k, a, b in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):
                    moves[k] = (logp[a:b], t[a:b])
                for k in z["task"].tolist():
                    got = len(moves[k][0]) if k in moves else 0
                    if got != per_task[k][0]:
                        raise ValueError(f"{path}: task {k} has {got} scored moves, n_moves says {per_task[k][0]}")
        metas.append(meta)
    missing = sorted(set(range(n_tasks_total)) - set(per_task))
    unknown = sorted(set(per_task) - set(range(n_tasks_total)))
    if missing or unknown:
        raise ValueError(f"the shards cover {len(per_task) - len(unknown)} of {n_tasks_total} tasks: {len(missing)} missing "
                         f"(e.g. {missing[:3]}), {len(unknown)} unknown (e.g. {unknown[:3]})")
    return per_task, metas, moves


def game_vectors(per_task, moves):
    """The vectors dict pack_questions reads, one row per task in index order: game_key (str(k)), staged [T, 300]
    and whole [T, 75] (aicup.gamevec.game_vector of the task's per-move log p), sums [T, 18] (float64) and
    n_moves [T]."""
    keys = sorted(per_task)
    staged = np.stack([game_vector(*moves[k], SEGMENTS) for k in keys]) if keys else np.zeros((0, 300), dtype=np.float32)
    whole = np.stack([game_vector(*moves[k], WHOLE) for k in keys]) if keys else np.zeros((0, 75), dtype=np.float32)
    return {"game_key": np.asarray([str(k) for k in keys], dtype=str), "staged": staged, "whole": whole,
            "sums": np.stack([np.asarray(per_task[k][1], dtype=np.float64) for k in keys]) if keys else np.zeros((0, len(PROFILES))),
            "n_moves": np.asarray([per_task[k][0] for k in keys], dtype=np.int64)}


def load_set_models(paths, device="cpu"):
    """[(Fitted, sidecar, path)] of the .pt files and the loader's warnings (aicup.setmodel_io.load_set_model)."""
    models, warnings = [], []
    for path in paths:
        fitted, meta, w = load_set_model(path, device)
        models.append((fitted, meta, str(path)))
        warnings += w
    return models, warnings


def check_pairing(models, base_doc, base_path):
    """The set models must be Δ-corrections of --set-base. A sidecar that records its training questions
    (questions_sha256, scripts/train_set_model.py) needs a base document with the same value: ValueError
    otherwise, also when the base has none (results/models/rank_model.json is not a paired base). Returns
    warnings: a sidecar's base_model_sha256 that is not the file's (the same base rewritten, or another file
    fitted on the same questions)."""
    warnings, sha = [], file_sha256(base_path)
    for _, meta, path in models:
        want = meta.get("questions_sha256")
        if want and base_doc.get("questions_sha256") != want:
            have = base_doc.get("questions_sha256")
            raise ValueError(f"{path}: trained on other questions than the base {base_path} (questions_sha256 {want[:12]} vs "
                             f"{have[:12] if have else 'none'}); --set-base must be the paired base its sidecar names ({meta.get('base_model')})")
        if meta.get("base_model_sha256") and meta["base_model_sha256"] != sha:
            warnings.append(f"{path}: its sidecar records the base file sha256 {meta['base_model_sha256'][:12]}, "
                            f"{base_path} has {sha[:12]} (the same questions; a rewritten base file)")
    return warnings


def average_set_models(frame, vec, X, base_model, models):
    """(P [Q, 10], info) of the set models on the questions of `frame` (question_id, player_id, n_games, game_keys
    into `vec`) with q24 features X [Q, 24]: the base logits are log(clip(base_model probabilities, BASE_CLIP));
    every model's predict_set_model gives probabilities, averaged first over the models of one rung (its seeds),
    then over the rungs. A question whose averaged probabilities are not finite gets the base's softmax instead
    and is listed in info["set_model_fallback_questions"]; info["rungs"] counts the models per rung."""
    base = np.log(np.clip(base_model.predict_proba(X), BASE_CLIP, None))
    packed = {rep: pack_questions(frame, vec, rep, True) for rep in {meta["rep"] for _, meta, _ in models}}
    per_rung = {}
    for fitted, meta, _ in models:
        per_rung.setdefault(meta["rung"], []).append(predict_set_model(fitted, packed[meta["rep"]], base))
    P = np.mean([np.mean(ps, axis=0) for ps in per_rung.values()], axis=0)
    bad = ~np.isfinite(P).all(axis=1)
    if bad.any():
        P[bad] = _softmax(base[bad])
    info = {"set_model_fallback_questions": frame["question_id"][bad].tolist(), "rungs": {r: len(ps) for r, ps in per_rung.items()}}
    return P, info


def question_frame(questions, owner, keys):
    """(frame for pack_questions, qis, by_q): the questions with a usable game, player_id = question_id, game_keys
    = keys[k] of their tasks k in task order; by_q[qi] lists those tasks."""
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    qis = sorted(by_q)
    frame = pd.DataFrame({"question_id": [questions[qi]["question_id"] for qi in qis],
                          "player_id": [questions[qi]["question_id"] for qi in qis],
                          "n_games": [len(by_q[qi]) for qi in qis], "game_keys": [[keys[k] for k in by_q[qi]] for qi in qis]})
    return frame, qis, by_q


def set_model_probs(questions, owner, per_task, moves, base_model, models):
    """({question index: probabilities [10]} for every question with a usable game, info) -- average_set_models
    over the per-game vectors of the shards (game_vectors; task k is game key str(k))."""
    frame, qis, by_q = question_frame(questions, owner, [str(k) for k in range(len(owner))])
    if not qis:
        return {}, {"set_model_fallback_questions": [], "rungs": {}}
    X = np.stack([question_features(np.stack([per_task[k][1] for k in by_q[qi]]), np.array([per_task[k][0] for k in by_q[qi]]))
                  for qi in qis])
    P, info = average_set_models(frame, game_vectors(per_task, moves), X, base_model, models)
    return dict(zip(qis, P)), info


def load_rank_model(path):
    """(saved JSON, model) of results/models/rank_model.json; ValueError when it was trained on other
    profiles or features than this pipeline computes."""
    saved = json.loads(Path(path).read_text())
    if saved["profiles"] != PROFILES or saved["features"] != FEATURE_NAMES:
        raise ValueError(f"{path}: the model was trained with different profiles or features than this pipeline computes")
    return saved, from_state(saved["state"])


def predict_questions(questions, owner, per_task, rank_model, set_probs=None):
    """(rows [(question_id, rank)], probs [[10 floats]] (NaN for a fallback question), fallback [question ids]):
    the expected-score decision on each question's probabilities -- set_probs[qi] when given (set_model_probs),
    else rank_model's on the question's q24 features; a question with no usable game gets FALLBACK."""
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(per_task[k])
    rows, probs, fallback = [], [], []
    for qi, q in enumerate(questions):
        games = by_q.get(qi, [])
        if not games:
            rows.append((q["question_id"], FALLBACK))
            probs.append([float("nan")] * len(LEVELS))
            fallback.append(q["question_id"])
            continue
        if set_probs is not None:
            p = set_probs[qi]
        else:
            x = question_features(np.stack([s for _, s in games]), np.array([n for n, _ in games]))
            p = rank_model.predict_proba(x[None, :])[0]
        rows.append((q["question_id"], LEVELS[expected_score_decision(p)]))
        probs.append([float(v) for v in p])
    return rows, probs, fallback


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="the test CSVs, public first")
    ap.add_argument("--model", default=str(ROOT / "results" / "models" / "rank_model.json"))
    ap.add_argument("--out", default=str(ROOT / "submission_rank.csv"))
    ap.add_argument("--report", default=None, help="default: <out stem>_report.json")
    ap.add_argument("--workers", type=int, default=10, help="CPU feature workers of the scorer")
    ap.add_argument("--shard", type=int, default=None, help="score only this shard (0-based) of --nshards and write its file")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="predict from the --nshards shard files already written; no scoring")
    ap.add_argument("--expect-rows", type=int, default=800, help="questions expected in the inputs (0: no check)")
    ap.add_argument("--expect-public", type=int, default=400, help="pub_ questions expected (0: no check)")
    ap.add_argument("--expect-private", type=int, default=400, help="priv_ questions expected (0: no check)")
    ap.add_argument("--allow-fallback", action="store_true",
                    help=f"exit 0 even when a question had no usable game and was set to {FALLBACK}")
    ap.add_argument("--lenient-inputs", action="store_true",
                    help="read a question whose num_games disagrees with its filled sgf columns from the non-empty columns, "
                         "with a warning in the report, instead of stopping (exit 4 unless --allow-input-warnings)")
    ap.add_argument("--allow-input-warnings", action="store_true", help="exit 0 despite --lenient-inputs warnings")
    ap.add_argument("--set-models", nargs="*", default=[], metavar="PT",
                    help="set models (.pt with .json sidecars, scripts/train_set_model.py) whose averaged probabilities decide")
    ap.add_argument("--set-base", default=None, help="the paired base of the set models (rank_model.json format); required with them")
    ap.add_argument("--set-device", default="cpu", help="device the set models run on")
    ap.add_argument("--l1b-heads", nargs="*", default=[], metavar="PT",
                    help="L1b heads files (.pt + .json, with a fine-tuned trunk when the sidecar names one) scored alongside the "
                         "18 profiles at the shard step; their sums go into the shard files for a fused decision")
    ap.add_argument("--fused-model", default=None,
                    help="rank_model_fused.json (scripts/fit_fused_model.py): decide with the fused MLP averaged over its L1b models, "
                         "which must be exactly the models the shards were scored with")
    a = ap.parse_args(argv)
    if a.set_models and not a.set_base:
        ap.error("--set-models needs --set-base, the paired base they were trained on")
    if a.fused_model and a.set_models:
        ap.error("--fused-model and --set-models exclude each other (one decision model)")
    if a.nshards < 1:
        ap.error("--nshards must be >= 1")
    if a.shard is not None and (a.merge or not 0 <= a.shard < a.nshards):
        ap.error("--shard needs 0 <= shard < --nshards and excludes --merge")
    if a.shard is None and not a.merge and a.nshards != 1:
        ap.error("--nshards > 1 needs --shard i (one process per shard) or --merge")
    out = Path(a.out)
    report_path = Path(a.report) if a.report else out.with_name(out.stem + "_report.json")
    t0 = time.time()

    try:
        questions, input_warnings = load_questions(a.inputs, a.lenient_inputs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    if input_warnings:
        print(f"WARNING: {len(input_warnings)} questions read leniently: " + "; ".join(input_warnings[:5])
              + (f" (and {len(input_warnings) - 5} more)" if len(input_warnings) > 5 else ""), file=sys.stderr, flush=True)
    tasks, owner, dropped, passes = build_tasks(questions)
    sha = questions_sha(a.inputs, questions)
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, "
          f"{len(dropped)} dropped, {passes} passes", flush=True)

    if not a.merge:
        shard = 0 if a.shard is None else a.shard
        mine = tasks[shard::a.nshards]
        try:
            l1b_models = l1b_models_meta(a.l1b_heads)
        except ValueError as e:
            print(e, file=sys.stderr)
            return EXIT_INPUTS
        print(f"scoring shard {shard} of {a.nshards}: {len(mine)} games" + (f", L1b heads {[m['name'] for m in l1b_models]}" if l1b_models else ""), flush=True)
        per_task, moves, n_pos, wall = score_tasks(mine, a.workers, l1b_heads=a.l1b_heads)
        path = shard_path(out, shard, a.nshards)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_shard(path, shard, a.nshards, mine, per_task, moves, n_pos, wall, a.inputs, sha, len(tasks), l1b_models)
        print(f"wrote {path}: {len(mine)} games, {n_pos} positions, {wall:.0f} s", flush=True)
        if a.shard is not None:
            return EXIT_OK  # one shard of a multi-GPU run; the merge is a separate invocation

    set_probs, set_info, set_models, set_warnings = None, {"set_model_fallback_questions": [], "rungs": {}}, [], []
    fused_doc, fused_info, l1b_models = None, {"fused_fallback_questions": []}, []
    try:
        per_task, shard_metas, moves = load_shards(out, a.nshards, len(tasks), sha, with_moves=bool(a.set_models))
        saved, rank_model = load_rank_model(a.model)
        l1b_models, l1b_per_task, _ = load_shards_l1b(out, a.nshards)
        if a.set_models:
            set_models, set_warnings = load_set_models(a.set_models, a.set_device)
            base_doc, set_base = load_rank_model(a.set_base)
            set_warnings += check_pairing(set_models, base_doc, a.set_base)
            set_probs, set_info = set_model_probs(questions, owner, per_task, moves, set_base, set_models)
        if a.fused_model:
            fused_doc, fused_mlp = load_fused_model(a.fused_model)
            order = pairing_order(fused_doc, l1b_models)
            set_probs, fused_info = fused_probs(questions, owner, per_task, l1b_per_task, fused_mlp, order, rank_model)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    for w in set_warnings:
        print(f"WARNING: {w}", file=sys.stderr, flush=True)
    rows, probs, fallback = predict_questions(questions, owner, per_task, rank_model, set_probs)
    write_submission(rows, out)
    expect_prefix = {p: n for p, n in (("pub_", a.expect_public), ("priv_", a.expect_private)) if n > 0}
    problems = validate_submission(out, [q["question_id"] for q in questions], a.expect_rows or None, expect_prefix or None)
    pd.DataFrame(probs, columns=[f"p_{lv}" for lv in LEVELS]).assign(question_id=[q["question_id"] for q in questions]) \
        .to_csv(out.with_name(out.stem + "_proba.csv"), index=False)
    code = (EXIT_INVALID if problems else EXIT_FALLBACK if fallback and not a.allow_fallback
            else EXIT_INPUT_WARNINGS if input_warnings and not a.allow_input_warnings else EXIT_OK)
    report = {"inputs": a.inputs, "model": a.model, "model_name": saved["name"], "mode": "merge" if a.merge else "single",
              "nshards": a.nshards, "shards": shard_metas, "questions": len(questions),
              "games_declared": int(sum(q["declared_games"] for q in questions)),
              "games_present": int(sum(len(q["games"]) for q in questions)),
              "games_used": len(tasks), "games_dropped": dropped, "passes": passes,
              "positions": int(sum(m["positions"] for m in shard_metas)),
              "scoring_wall_s": float(sum(m["scoring_wall_s"] for m in shard_metas)), "total_wall_s": time.time() - t0,
              "expectations": {"rows": a.expect_rows, "pub_": a.expect_public, "priv_": a.expect_private},
              "fallback_questions": fallback, "allow_fallback": a.allow_fallback,
              "lenient_inputs": a.lenient_inputs, "input_warnings": input_warnings, "allow_input_warnings": a.allow_input_warnings,
              "decision_model": "fused" if a.fused_model else "set_models" if a.set_models else "rank_model",
              "set_models": [{"path": path, **set_model_summary(meta)} for _, meta, path in set_models], "set_base": a.set_base,
              "set_rungs": set_info["rungs"], "set_model_warnings": set_warnings,
              "set_model_fallback_questions": set_info["set_model_fallback_questions"],
              "l1b_models": l1b_models,
              "fused_model": ({"path": a.fused_model, **{k: fused_doc.get(k) for k in ("name", "file_sha256", "train_folds", "per_player",
                                                                                         "questions_sha256", "l1b_models", "git_head")},
                               "n_models": len(fused_doc["l1b_models"])} if fused_doc else None),
              "fused_fallback_questions": fused_info["fused_fallback_questions"],
              "predicted_counts": dict(Counter(r for _, r in rows)), "problems": problems, "exit_code": code}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}", flush=True)
    if fused_info["fused_fallback_questions"]:
        print(f"WARNING: {len(fused_info['fused_fallback_questions'])} questions got {Path(a.model).name}'s probabilities because the "
              f"fused model's were not finite: {fused_info['fused_fallback_questions'][:5]}", file=sys.stderr, flush=True)
    if fallback:
        print(f"WARNING: {len(fallback)} questions had no usable game and were set to {FALLBACK}: {fallback[:5]}"
              + ("" if a.allow_fallback else f" (exit {EXIT_FALLBACK}; pass --allow-fallback to accept)"), file=sys.stderr, flush=True)
    if set_info["set_model_fallback_questions"]:
        print(f"WARNING: {len(set_info['set_model_fallback_questions'])} questions got the base's probabilities because the set "
              f"models' were not finite: {set_info['set_model_fallback_questions'][:5]}", file=sys.stderr, flush=True)
    if input_warnings and code == EXIT_INPUT_WARNINGS:
        print(f"WARNING: {len(input_warnings)} questions were read leniently (exit {EXIT_INPUT_WARNINGS}; pass "
              f"--allow-input-warnings to accept)", file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
