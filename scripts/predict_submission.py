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

--expect-rows / --expect-public / --expect-private (defaults: the official 800 = 400 + 400; 0 disables
one) are checked against the questions read, so a truncated input fails validation; a mock test passes
its own counts. Exit codes: 0 ok; 1 the submission failed validation; 2 malformed inputs, a model
trained on other features, or missing / inconsistent shard files; 3 a question with no usable game was
filled with FALLBACK (--allow-fallback accepts that with a warning and exit 0). The submission, the
probabilities and the report are written in every case the merge reaches.
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
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qfeatures import FEATURE_NAMES, question_features  # noqa: E402
from aicup.rankmodel import from_state  # noqa: E402
from aicup.submission import check_game, read_test, validate_submission, write_submission  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
FALLBACK = "1D"  # used only for a question with no usable game: an interior class keeps both neighbours
EXIT_OK, EXIT_INVALID, EXIT_INPUTS, EXIT_FALLBACK = 0, 1, 2, 3
DEFAULT_INPUTS = [str(ROOT / "data" / "tests" / "rank_prediction_test_public.csv"),
                  str(ROOT / "data" / "tests" / "rank_prediction_test_private.csv")]


def load_questions(inputs):
    """The questions of the input files, in order (aicup.submission.read_test; ValueError on a malformed file)."""
    return [q for path in inputs for q in read_test(path)]


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


def score_tasks(tasks, workers, log=print):
    """Score (k, sgf, colour) tasks on this GPU under the 18 profiles: ({k: (n_moves, sums[18])},
    (move_logp [M, 18], move_task [M], move_t [M]) or None, positions, wall seconds)."""
    import torch
    from aicup.humansl import load_humansl, meta_rows
    from aicup.scoring import init_worker, score_stream
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda")
        return score_stream(pool, model, tasks, meta_rows([(p, p) for p in PROFILES]), log_every=500, log=log)
    finally:
        pool.close()


def save_shard(path, shard, nshards, tasks, per_task, moves, n_pos, wall, inputs, sha, n_tasks_total):
    """Write a shard file atomically: task (the global indices scored), n_moves, sums [n, 18] (float64),
    move_logp [M, 18], move_task [M], move_t [M] and __meta__ (JSON). Returns the meta."""
    idx = np.array([k for k, _, _ in tasks], dtype=np.int64)
    n_moves = np.array([per_task[k][0] for k in idx], dtype=np.int64)
    sums = np.stack([per_task[k][1] for k in idx]).astype(np.float64) if len(idx) else np.zeros((0, len(PROFILES)))
    if moves is None:
        moves = (np.zeros((0, len(PROFILES)), dtype=np.float32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))
    meta = {"shard": shard, "nshards": nshards, "n_tasks_total": n_tasks_total, "n_tasks": int(len(idx)),
            "inputs": [str(p) for p in inputs], "questions_sha256": sha, "positions": int(n_pos), "scoring_wall_s": float(wall),
            "profiles": list(PROFILES), "ckpt": CKPT.name, "written": datetime.now().isoformat(timespec="seconds")}
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, task=idx, n_moves=n_moves, sums=sums, move_logp=moves[0], move_task=moves[1], move_t=moves[2],
                        __meta__=np.asarray(json.dumps(meta)))
    os.replace(tmp, path)
    return meta


def load_shards(out, nshards, n_tasks_total, sha):
    """({k: (n_moves, sums)}, [meta per shard]) from the nshards files of `out`. Every file must exist, be
    shard i of nshards made from the same questions (questions_sha256), and together they must score every
    task 0..n_tasks_total-1 exactly once; FileNotFoundError / ValueError otherwise."""
    per_task, metas = {}, []
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
        metas.append(meta)
    missing = sorted(set(range(n_tasks_total)) - set(per_task))
    unknown = sorted(set(per_task) - set(range(n_tasks_total)))
    if missing or unknown:
        raise ValueError(f"the shards cover {len(per_task) - len(unknown)} of {n_tasks_total} tasks: {len(missing)} missing "
                         f"(e.g. {missing[:3]}), {len(unknown)} unknown (e.g. {unknown[:3]})")
    return per_task, metas


def load_rank_model(path):
    """(saved JSON, model) of results/models/rank_model.json; ValueError when it was trained on other
    profiles or features than this pipeline computes."""
    saved = json.loads(Path(path).read_text())
    if saved["profiles"] != PROFILES or saved["features"] != FEATURE_NAMES:
        raise ValueError(f"{path}: the model was trained with different profiles or features than this pipeline computes")
    return saved, from_state(saved["state"])


def predict_questions(questions, owner, per_task, rank_model):
    """(rows [(question_id, rank)], probs [[10 floats]] (NaN for a fallback question), fallback [question ids]):
    the expected-score decision on the model's probabilities of each question's features; a question with no
    usable game gets FALLBACK."""
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
    a = ap.parse_args(argv)
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
        questions = load_questions(a.inputs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    tasks, owner, dropped, passes = build_tasks(questions)
    sha = questions_sha(a.inputs, questions)
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, "
          f"{len(dropped)} dropped, {passes} passes", flush=True)

    if not a.merge:
        shard = 0 if a.shard is None else a.shard
        mine = tasks[shard::a.nshards]
        print(f"scoring shard {shard} of {a.nshards}: {len(mine)} games", flush=True)
        per_task, moves, n_pos, wall = score_tasks(mine, a.workers)
        path = shard_path(out, shard, a.nshards)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_shard(path, shard, a.nshards, mine, per_task, moves, n_pos, wall, a.inputs, sha, len(tasks))
        print(f"wrote {path}: {len(mine)} games, {n_pos} positions, {wall:.0f} s", flush=True)
        if a.shard is not None:
            return EXIT_OK  # one shard of a multi-GPU run; the merge is a separate invocation

    try:
        per_task, shard_metas = load_shards(out, a.nshards, len(tasks), sha)
        saved, rank_model = load_rank_model(a.model)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    rows, probs, fallback = predict_questions(questions, owner, per_task, rank_model)
    write_submission(rows, out)
    expect_prefix = {p: n for p, n in (("pub_", a.expect_public), ("priv_", a.expect_private)) if n > 0}
    problems = validate_submission(out, [q["question_id"] for q in questions], a.expect_rows or None, expect_prefix or None)
    pd.DataFrame(probs, columns=[f"p_{lv}" for lv in LEVELS]).assign(question_id=[q["question_id"] for q in questions]) \
        .to_csv(out.with_name(out.stem + "_proba.csv"), index=False)
    code = EXIT_INVALID if problems else EXIT_FALLBACK if fallback and not a.allow_fallback else EXIT_OK
    report = {"inputs": a.inputs, "model": a.model, "model_name": saved["name"], "mode": "merge" if a.merge else "single",
              "nshards": a.nshards, "shards": shard_metas, "questions": len(questions),
              "games_declared": int(sum(q["declared_games"] for q in questions)),
              "games_present": int(sum(len(q["games"]) for q in questions)),
              "games_used": len(tasks), "games_dropped": dropped, "passes": passes,
              "positions": int(sum(m["positions"] for m in shard_metas)),
              "scoring_wall_s": float(sum(m["scoring_wall_s"] for m in shard_metas)), "total_wall_s": time.time() - t0,
              "expectations": {"rows": a.expect_rows, "pub_": a.expect_public, "priv_": a.expect_private},
              "fallback_questions": fallback, "allow_fallback": a.allow_fallback,
              "predicted_counts": dict(Counter(r for _, r in rows)), "problems": problems, "exit_code": code}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}", flush=True)
    if fallback:
        print(f"WARNING: {len(fallback)} questions had no usable game and were set to {FALLBACK}: {fallback[:5]}"
              + ("" if a.allow_fallback else f" (exit {EXIT_FALLBACK}; pass --allow-fallback to accept)"), file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
