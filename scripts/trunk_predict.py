#!/usr/bin/env python
"""Official test CSVs -> a second submission from the trunk + TCN ensemble (architecture v2 sections 4A / 4H: the C and T
arms of scripts/trunk_seq_check.py, 12 models, folds 1-4 CV 0.7258 against the fused MLP's 0.7026), built on top of the main
pipeline's shard files. Nothing here changes the main submission (predict_submission.py --merge): this is a separate output.

  extract  replay every usable game (predict_submission.py's tasks, in its order; the same check_game) and store the per-ply
           trunk vectors of the pristine humanv0 trunk exactly as scripts/trunk_extract.py stores them for the training folds
           (both players' plies, the fixed random projections to 64 + 64 dims, fp16), as parts under --vec; sharded over GPUs
           like the scoring step (--shard i --nshards N; parts resume per game).
  predict  per question and per L1b model of the shards (heads3_A, heads3_B): the 64 R features (fused_opp_diff_hd, from the
           shards' sums and the games' HD[]) and the per-ply tokens (trunk vectors + the shards' per-move Human SL log p of the
           target, L1b log p of the target and of the opponent + flags), standardised as each checkpoint was trained; each
           checkpoint's softmax, averaged over the checkpoints and the L1b models; the expected-score decision. A checkpoint whose
           sidecar says opp_hsl (section 4M M14, the ct6 models) reads tokens whose opponent plies carry the opponent's Human SL
           log p from the shards' 18-profile opponent pass (hsl_opp_move_logp), and with opp_hsl_r an R with the 39 opponent-HSL
           question features (103 columns); shards without that pass are refused for such checkpoints. A question
           without a usable game, or with non-finite probabilities, takes the main submission's row (--fallback-submission) or
           FALLBACK and is listed in the report. Exit codes as predict_submission.py: 0 ok, 1 the file failed validation, 2 bad
           inputs / shards / models, 3 a fallback row (--allow-fallback accepts), 4 lenient-input warnings (--allow-input-warnings).
  offline  the same checkpoints on a mock test built from one training fold (experiments/make_mock_test.py --fold f), through the
           training-side tables and trunk vectors (results/scores_hd, results/l1b/*_hd, results/trunk_vec/fold f): the reference
           of the consistency check of the test-side path.
  compare  two probability files side by side: max / mean |dp|, questions whose decisions differ, and with --answers the
           official score of each.

  sbatch -p 8gpus --gres=gpu:8 -c 96 -t 02:00:00 -J ctvec scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 \\
      python -I scripts/trunk_predict.py extract --inputs data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv \\
      --vec results/ct_submission/vec --workers 10
  sbatch -t 01:00:00 -J ctpred scripts/nano4/run.sbatch python -I scripts/trunk_predict.py predict --inputs ... --shards submission_rank.csv \\
      --nshards 8 --vec results/ct_submission/vec --models results/models/ct/all_*.pt --fallback-submission submission_rank.csv \\
      --out submission_rank_ct.csv
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES, fused_opp_diff_features  # noqa: E402
from aicup.handicap import question_hd_features_from_games  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.opphsl import N_OPP_HSL, opp_hsl_features, question_features as opp_hsl_question_features  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qeval import balanced_mean_over_n  # noqa: E402
from aicup.scoring import completed_parts, part_paths, resume_plan  # noqa: E402
from aicup.setmodel_io import file_sha256  # noqa: E402
from aicup.submission import validate_submission, write_submission  # noqa: E402
import predict_submission as ps  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

EXIT_OK, EXIT_INVALID, EXIT_INPUTS, EXIT_FALLBACK, EXIT_INPUT_WARNINGS = ps.EXIT_OK, ps.EXIT_INVALID, ps.EXIT_INPUTS, ps.EXIT_FALLBACK, ps.EXIT_INPUT_WARNINGS
FALLBACK = ps.FALLBACK
P_COLS = [f"p_{lv}" for lv in LEVELS]


# Section 4M M7 (2026-11-04): checkpoints trained on one question-size band (trunk_seq_check.py --train-sizes lo hi, sidecar
# train_sizes) vote only on questions of that many usable games; the others vote everywhere (votes_of / ensemble).


def task_key(k):
    """The vec parts' game_key of task k: a string, so the CSV round trip of the part tables keeps it."""
    return f"g{k:07d}"


def key_task(key):
    return int(key[1:])


# ---------------------------------------------------------------- extract

def extract(a):
    import trunk_extract as te
    from aicup.humansl import load_humansl
    from aicup.l1b import neutral_rows, trunk_features

    t0 = time.time()
    try:
        questions, warnings = ps.load_questions(a.inputs, a.lenient_inputs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    tasks, owner, dropped, _ = ps.build_tasks(questions)
    sha = ps.questions_sha(a.inputs, questions)
    mine = tasks[a.shard::a.nshards]
    print(f"{len(questions)} questions, {len(tasks)} usable games ({len(dropped)} dropped); shard {a.shard} of {a.nshards}: {len(mine)} games"
          + (f"; {len(warnings)} input warnings" if warnings else ""), flush=True)
    vec = Path(a.vec)
    vec.mkdir(parents=True, exist_ok=True)
    meta = {task_key(k): {"game_key": task_key(k), "task": k, "question_id": questions[owner[k]]["question_id"], "color": c} for k, _, c in mine}
    sgf_of = {task_key(k): (sgf, c) for k, sgf, c in mine}
    plan = resume_plan(list(meta), vec, a.shard, a.part_size)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    plies = 0
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        rows, proj_g, proj_l = neutral_rows(), None, None
        for n_part, (k, part_keys) in enumerate(plan):
            done = list(pool.imap(te.replay, [(q, *sgf_of[q]) for q in part_keys], chunksize=1))
            if proj_g is None:  # the projections of scripts/trunk_extract.py (seed PROJ_SEED), sized from the trunk's channels
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(),
                                                    torch.as_tensor(done[0][4][:1]).cuda().long(), rows)
                proj_g, proj_l = te.projections(trunk.shape[1], trunk.device)
            plies += te.write_part(vec, a.shard, k, done, meta, model, proj_g, proj_l, rows, a.batch)
            print(f"part {n_part + 1}/{len(plan)}: {plies} plies, {plies / max(time.time() - t0, 1e-9):.0f} plies/s", flush=True)
    finally:
        pool.close()
        pool.join()
    rec = {"inputs": [str(p) for p in a.inputs], "questions_sha256": sha, "n_tasks_total": len(tasks), "shard": a.shard, "nshards": a.nshards,
           "games": len(mine), "parts_this_run": len(plan), "plies_this_run": int(plies), "proj_seed": te.PROJ_SEED, "proj_dim": te.PROJ_DIM,
           "ckpt": te.CKPT.name, "lenient_inputs": a.lenient_inputs, "input_warnings": warnings, "wall_s": time.time() - t0,
           "written": datetime.now().isoformat(timespec="seconds")}
    (vec / f"extract_s{a.shard}of{a.nshards}.json").write_text(json.dumps(rec, indent=1))
    print(f"shard {a.shard}: {len(mine)} games, {plies} plies this run, {(time.time() - t0) / 60:.1f} min", flush=True)
    return EXIT_OK


# ---------------------------------------------------------------- predict

def vec_index(vec_dir):
    """The games of the vec parts in the ply order of trunk_seq_check.raw_tokens: [{key, task, start, n, target [n] bool, t [n]}],
    and the total number of plies. ValueError when a game's plies are not the contiguous 0..n-1 block the parts promise."""
    games, start = [], 0
    for s, k in completed_parts(vec_dir):
        csv_path, npz_path = part_paths(vec_dir, s, k)
        tab = pd.read_csv(csv_path, dtype={"game_key": str, "question_id": str, "color": str}, keep_default_na=False)
        with np.load(npz_path) as z:
            row, t, target = z["game_row"].astype(np.int64), z["t"].astype(np.int64), z["target"].astype(bool)
        for g, rec in enumerate(tab.to_dict("records")):
            idx = np.flatnonzero(row == g)
            if len(idx) != int(rec["n_plies"]) or not np.array_equal(t[idx], np.arange(len(idx))):
                raise ValueError(f"{npz_path}: game {rec['game_key']} has {len(idx)} plies stored, n_plies {rec['n_plies']}, or its t is not 0..n-1")
            games.append({"key": rec["game_key"], "task": int(rec["task"]), "start": start, "n": len(idx), "target": target[idx], "t": t[idx]})
            start += len(idx)
    return games, start


def load_opp_moves(shards_out, nshards):
    """{task: (logp [n_opp, K, 10] float64, t [n_opp])}: the opponent plies' L1b log p from the shard files (written since
    2026-11-04); ValueError when a shard lacks them."""
    moves = {}
    for i in range(nshards):
        path = ps.shard_path(shards_out, i, nshards)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            if not meta.get("l1b_opp_moves"):
                raise ValueError(f"{path}: no per-move L1b log p of the opponents (l1b_opp_moves): the shards must come from "
                                 "predict_submission.py of 2026-11-04 or later, scored with --l1b-heads and the opponent pass")
            mt, logp, t = z["opp_move_task"].astype(np.int64), z["l1b_opp_move_logp"].astype(np.float64), z["opp_move_t"].astype(np.int64)
            order = np.argsort(mt, kind="stable")
            mt, logp, t = mt[order], logp[order], t[order]
            tasks_here, starts = np.unique(mt, return_index=True)
            ends = np.append(starts[1:], len(mt))
            for k, s, e in zip(tasks_here.tolist(), starts.tolist(), ends.tolist()):
                moves[k] = (logp[s:e], t[s:e])
    return moves


def score_arrays(games, moves_hsl, moves_l1b, moves_opp, n_models, moves_opp_hsl=None):
    """[K, plies, N_SCORES] float16 in the ply order of `games` (vec_index): per L1b model k, the target plies hold the centred
    L1b log p of the target's moves and the opponent plies the centred L1b log p of the opponent's moves (columns 0-10), the
    target plies also the centred Human SL log p (columns 11-29) -- trunk_seq_check.build's layout; with moves_opp_hsl (the
    shards' 18-profile opponent pass, section 4M M14) the opponent plies hold the opponent's centred Human SL log p there too
    (build --opp-hsl's layout), else zeros. Every game's target plies must be the shards' Human SL and L1b moves and its other
    plies the opponent moves, ply for ply (ValueError otherwise)."""
    total = sum(g["n"] for g in games)
    sc = np.zeros((n_models, total, ts.N_SCORES), np.float32)
    bad = []
    for g in games:
        k, tgt = g["task"], g["target"]
        hl, ht = moves_hsl[k]
        ll, lt = moves_l1b[k]
        ol, ot = moves_opp.get(k, (np.zeros((0, n_models, len(LEVELS))), np.zeros(0, dtype=np.int64)))
        ohl, oht = (None, g["t"][~tgt]) if moves_opp_hsl is None else moves_opp_hsl.get(k, (np.zeros((0, len(PROFILES))), np.zeros(0, dtype=np.int64)))
        if not (np.array_equal(g["t"][tgt], ht) and np.array_equal(ht, lt) and np.array_equal(g["t"][~tgt], ot) and np.array_equal(g["t"][~tgt], oht)):
            bad.append(k)
            continue
        rows = g["start"] + np.arange(g["n"])
        for m in range(n_models):
            sc[m, rows[tgt], :len(LEVELS) + 1] = ts.centred(ll[:, m, :])
            sc[m, rows[~tgt], :len(LEVELS) + 1] = ts.centred(ol[:, m, :])
        sc[:, rows[tgt], len(LEVELS) + 1:] = ts.centred(hl)[None]
        if ohl is not None:
            sc[:, rows[~tgt], len(LEVELS) + 1:] = ts.centred(ohl)[None]
    if bad:
        raise ValueError(f"{len(bad)} games whose plies do not line up with the shards' moves, e.g. tasks {bad[:3]}")
    return sc.astype(np.float16)


def question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, task_games, m, opp_hsl_per_task=None):
    """[Q, 64]: the fused_opp_diff_hd features of the questions `qis` from shard model m's sums and the games' HD[] (what
    predict_submission.py's fused_probs computes for the _hd document, feature for feature); with opp_hsl_per_task (the shards'
    18-profile opponent pass: {k: (the opponent's n_moves, sums [18])}, section 4M M14) [Q, 103]: + the 39 opponent-HSL features."""
    rows = []
    width = len(FUSED_OPP_DIFF_HD_FEATURE_NAMES) + (N_OPP_HSL if opp_hsl_per_task is not None else 0)
    for qi in qis:
        ks = by_q[qi]
        hsl, n = np.stack([per_task[k][1] for k in ks]), np.array([per_task[k][0] for k in ks])
        l1b, opp = np.stack([l1b_per_task[k][m] for k in ks]), np.stack([opp_per_task[k][1][m] for k in ks])
        opp_n = np.array([opp_per_task[k][0] for k in ks])
        feats = [fused_opp_diff_features(hsl, l1b, n, opp, opp_n), question_hd_features_from_games([task_games[k] for k in ks])]
        if opp_hsl_per_task is not None:
            feats.append(opp_hsl_question_features(hsl, np.stack([opp_hsl_per_task[k][1] for k in ks]), np.array([opp_hsl_per_task[k][0] for k in ks])))
        rows.append(np.concatenate(feats))
    X = np.stack(rows) if rows else np.zeros((0, width))
    if X.shape[1] != width:
        raise ValueError(f"{X.shape[1]} R features computed, expected {width}")
    return X


def votes_of(side, n_games):
    """Section 4M M7 routing by question size: a checkpoint whose sidecar records train_sizes [lo, hi] votes only on questions of lo..hi
    games; one without votes everywhere. Returns (bool [Q], band label)."""
    sizes = side.get("train_sizes")
    if not sizes:
        return np.ones(len(n_games), dtype=bool), "all"
    lo, hi = int(sizes[0]), int(sizes[1])
    return (n_games >= lo) & (n_games <= hi), f"{lo}-{hi}"


def ensemble(ckpts, R_by_model, keys, X_raw_by_model, offsets, device, model_names=None):
    """(P [Q, 10] mean over the checkpoints x L1b models that vote on each question, {tag: P mean over that tag's checkpoints}, passes,
    {band: questions it votes on}): every checkpoint's softmax on the tokens standardised with its own token stats (one standardised
    copy per distinct stats) and its own R standardisation; votes_of routes by question size. A question without a vote is NaN.
    model_names (section 4M M9): the shards' L1b model names by index; a checkpoint whose sidecar lists l1b_heads only runs on those.
    X_raw_by_model {(m, oh): X} and R_by_model {(m, ohr): R} (section 4M M14): a checkpoint reads the tokens with the opponent plies'
    Human SL block filled when its sidecar says opp_hsl, and the R with the 39 opponent-HSL columns when it says opp_hsl_r;
    ValueError when the variant it needs was not built or R's width is not its n_r."""
    Q = len(keys)
    n_games = np.array([len(k) for k in keys], dtype=int)
    total, count, per_arm, n_arm, passes, routing = np.zeros((Q, len(LEVELS))), np.zeros(Q), {}, {}, 0, {}
    for m in sorted({mk[0] for mk in X_raw_by_model}):
        cache = {}
        for model, stats, (r_mean, r_std), side in ckpts:
            if model_names is not None and side.get("l1b_heads") and model_names[m] not in side["l1b_heads"]:
                continue  # this checkpoint's tokens came from other heads' log p
            oh, ohr = bool(side.get("opp_hsl")), bool(side.get("opp_hsl_r"))
            if (m, oh) not in X_raw_by_model or (m, ohr) not in R_by_model:
                raise ValueError(f"checkpoint {side.get('tag')} s{side.get('seed')} needs {'the opponent-HSL' if oh else 'the plain'} tokens and "
                                 f"{'the 103-column' if ohr else 'the 64-column'} R, which were not built here")
            X_raw = X_raw_by_model[(m, oh)]
            h = (hashlib.sha1(np.ascontiguousarray(stats[0], dtype=np.float32).tobytes() + np.ascontiguousarray(stats[1], dtype=np.float32).tobytes()).hexdigest(), oh)
            if h not in cache:
                cache = {h: torch.as_tensor(ts.standardise_tokens(X_raw.copy(), stats), device=device)}
            if R_by_model[(m, ohr)].shape[1] != len(r_mean):
                raise ValueError(f"checkpoint {side.get('tag')} s{side.get('seed')} takes {len(r_mean)} R features, {R_by_model[(m, ohr)].shape[1]} computed")
            R = ((R_by_model[(m, ohr)] - r_mean) / r_std).astype(np.float32)
            P = ts.predict(model, R, keys, cache[h], offsets, device) if Q else np.zeros((0, len(LEVELS)))
            v, band = votes_of(side, n_games)
            routing[band] = int(v.sum())
            total += P * v[:, None]
            count += v
            tag = side.get("tag") or side["arm"]
            per_arm[tag] = per_arm.get(tag, 0) + P * v[:, None]
            n_arm[tag] = n_arm.get(tag, 0) + v
            passes += 1
        del cache
    with np.errstate(invalid="ignore", divide="ignore"):
        P = total / count[:, None]
        arms = {tag: per_arm[tag] / n_arm[tag][:, None] for tag in per_arm}
    return P, arms, passes, routing


def predict(a):
    t0 = time.time()
    out = Path(a.out)
    report_path = Path(a.report) if a.report else out.with_name(out.stem + "_report.json")
    try:
        questions, input_warnings = ps.load_questions(a.inputs, a.lenient_inputs)
    except ValueError as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    tasks, owner, dropped, passes = ps.build_tasks(questions)
    sha = ps.questions_sha(a.inputs, questions)
    hd_counts = ps.handicap_counts(questions)
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, {len(dropped)} dropped; handicap HD[] "
          + " ".join(f"{k}:{v}" for k, v in hd_counts.items()), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if hd_counts.get("absent", 0) and not a.allow_missing_hd:
            raise ValueError(f"{hd_counts['absent']} of the games carry no HD[]; the checkpoints take the handicap features (zeros for an unknown "
                             "HD): pass --allow-missing-hd to go on, or keep the main submission")
        per_task, shard_metas, moves_hsl = ps.load_shards(a.shards, a.nshards, len(tasks), sha, with_moves=True)
        l1b_models, l1b_per_task, moves_l1b, opp_per_task = ps.load_shards_l1b(a.shards, a.nshards, with_moves=True)
        if not l1b_models:
            raise ValueError("the shards carry no L1b scores (the shard step needs --l1b-heads)")
        if opp_per_task is None:
            raise ValueError("the shards carry no opponent scores (the shard step must not pass --no-l1b-opponent)")
        moves_opp = load_opp_moves(a.shards, a.nshards)
        names = [m["name"] for m in l1b_models]
        use = [i for i, n in enumerate(names) if not a.l1b_names or n in a.l1b_names]
        missing = sorted(set(a.l1b_names or []) - set(names))
        if missing or not use:
            raise ValueError(f"--l1b-names {missing} are not among the shards' L1b models {names}")
        games, n_plies = vec_index(Path(a.vec))
        have = Counter(g["task"] for g in games)
        if sorted(have) != list(range(len(tasks))) or any(v != 1 for v in have.values()):
            dup = [k for k, v in have.items() if v > 1]
            lost = sorted(set(range(len(tasks))) - set(have))
            raise ValueError(f"the vec parts under {a.vec} hold {len(games)} games for {len(tasks)} tasks: {len(lost)} missing (e.g. {lost[:3]}), "
                             f"{len(dup)} stored twice (e.g. {dup[:3]}); run extract for every shard first")
        ckpts = [ts.load_checkpoint(p, device) for p in a.models]
        if not ckpts:
            raise ValueError("--models is empty")
        bad = [p for p, (_, stats, _, side) in zip(a.models, ckpts) if stats is None or side["arm"] == "R0"]
        if bad:
            raise ValueError(f"checkpoints without token statistics (R0 arm) cannot predict from tokens: {bad}")
        # section 4M M14: which token / R variants the checkpoints need (plain, or with the opponent's Human SL)
        need_x = {bool(side.get("opp_hsl")) for _, _, _, side in ckpts}
        need_r = {bool(side.get("opp_hsl_r")) for _, _, _, side in ckpts}
        opp_hsl_per_task, moves_opp_hsl = (ps.load_shards_opp_hsl(a.shards, a.nshards, with_moves=True) if (True in need_x or True in need_r) else (None, None))
        sc = score_arrays(games, moves_hsl, moves_l1b, moves_opp, len(l1b_models)) if False in need_x else None
        sc_oh = score_arrays(games, moves_hsl, moves_l1b, moves_opp, len(l1b_models), moves_opp_hsl) if True in need_x else None
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return EXIT_INPUTS
    print(f"shards: {len(tasks)} games, L1b models {names} (using {[names[i] for i in use]}); vec: {len(games)} games, {n_plies} plies; "
          f"{len(ckpts)} checkpoints ({Counter(s['arm'] for _, _, _, s in ckpts)}) on {device} ({time.time() - t0:.0f} s)", flush=True)
    vec = Path(a.vec)
    by_q = {}
    for k, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(k)
    qis = sorted(by_q)
    keys = [[task_key(k) for k in by_q[qi]] for qi in qis]
    task_games = {k: (sgf, c) for k, sgf, c in tasks}
    R_by_model, X_raw_by_model, offsets = {}, {}, None
    for m in use:
        for ohr in sorted(need_r):
            R_by_model[(m, ohr)] = question_r(qis, by_q, per_task, l1b_per_task, opp_per_task, task_games, m, opp_hsl_per_task if ohr else None)
        for oh, arr in ((False, sc), (True, sc_oh)):
            if arr is None:
                continue
            sc_path = vec / f"scores_{'oh_' if oh else ''}{names[m]}.npy"
            np.save(sc_path, arr[m])
            X_raw_by_model[(m, oh)], offsets = ts.raw_tokens([(vec, sc_path)])
    del sc, sc_oh
    P, per_arm, n_pass, routing = ensemble(ckpts, R_by_model, keys, X_raw_by_model, offsets, device, model_names=names)
    if n_pass == 0:
        print(f"no checkpoint accepts any of the shards' L1b models {names} (sidecar l1b_heads)", file=sys.stderr)
        return EXIT_INPUTS
    print(f"{n_pass} passes ({len(ckpts)} checkpoints x {len(use)} L1b models) over {len(qis)} questions; size routing {routing} ({time.time() - t0:.0f} s)", flush=True)
    fallback_rows = {}
    if a.fallback_submission:
        fb = pd.read_csv(a.fallback_submission, dtype=str, keep_default_na=False)
        fallback_rows = dict(zip(fb["question_id"], fb["pred_rank"]))
    pos = {qi: j for j, qi in enumerate(qis)}
    rows, probs, fallback, arm_rows = [], [], [], {arm: [] for arm in per_arm}
    for qi, q in enumerate(questions):
        j = pos.get(qi)
        if j is not None and np.isfinite(P[j]).all():
            rows.append((q["question_id"], LEVELS[expected_score_decision(P[j])]))
            probs.append([float(v) for v in P[j]])
            for arm in per_arm:
                arm_rows[arm].append(LEVELS[expected_score_decision(per_arm[arm][j])] if np.isfinite(per_arm[arm][j]).all() else None)
        else:
            rows.append((q["question_id"], fallback_rows.get(q["question_id"], FALLBACK)))
            probs.append([float("nan")] * len(LEVELS))
            fallback.append(q["question_id"])
            for arm in per_arm:
                arm_rows[arm].append(None)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(rows, out)
    expect_prefix = {p: n for p, n in (("pub_", a.expect_public), ("priv_", a.expect_private)) if n > 0}
    problems = validate_submission(out, [q["question_id"] for q in questions], a.expect_rows or None, expect_prefix or None)
    pd.DataFrame(probs, columns=P_COLS).assign(question_id=[q["question_id"] for q in questions]).to_csv(out.with_name(out.stem + "_proba.csv"), index=False)
    decided = [i for i, (qid, _) in enumerate(rows) if qid not in set(fallback)]
    voted = {arm: [i for i in decided if arm_rows[arm][i] is not None] for arm in per_arm}  # a size-routed arm votes on part of the questions
    agree_arm = {arm: float(np.mean([arm_rows[arm][i] == rows[i][1] for i in voted[arm]])) if voted[arm] else None for arm in per_arm}
    main_cmp = None
    if fallback_rows:
        same = [rows[i][1] == fallback_rows.get(rows[i][0]) for i in decided if rows[i][0] in fallback_rows]
        main_cmp = {"questions_compared": len(same), "same_decision_share": float(np.mean(same)) if same else None,
                    "differing": int(len(same) - sum(same))}
    code = (EXIT_INVALID if problems else EXIT_FALLBACK if fallback and not a.allow_fallback
            else EXIT_INPUT_WARNINGS if input_warnings and not a.allow_input_warnings else EXIT_OK)
    report = {"inputs": [str(p) for p in a.inputs], "shards": str(a.shards), "nshards": a.nshards, "questions_sha256": sha,
              "shard_metas": [{k: m.get(k) for k in ("shard", "n_tasks", "positions", "opp_positions", "l1b_opp_moves", "hsl_opponent", "written")} for m in shard_metas],
              "l1b_models": l1b_models, "l1b_models_used": [names[i] for i in use], "vec": str(a.vec), "vec_plies": int(n_plies),
              "vec_extract": [json.loads(p.read_text()) for p in sorted(vec.glob("extract_s*of*.json"))],
              "checkpoints": [{"path": str(p), "state_sha256": side.get("state_sha256"), "arm": side.get("arm"), "tag": side.get("tag"), "seed": side.get("seed"),
                               "val_fold": side.get("val_fold"), "train_folds": side.get("train_folds"), "train_sizes": side.get("train_sizes"), "l1b_heads": side.get("l1b_heads"),
                               "opp_hsl": bool(side.get("opp_hsl")), "opp_hsl_r": bool(side.get("opp_hsl_r")),
                               "best_inner": side.get("best_inner"), "git_head": side.get("git_head")}
                              for p, (_, _, _, side) in zip(a.models, ckpts)],
              "size_routing": routing, "arm_questions_voted": {arm: len(v) for arm, v in voted.items()},
              "passes": n_pass, "questions": len(questions), "games_declared": int(sum(q["declared_games"] for q in questions)),
              "games_present": int(sum(len(q["games"]) for q in questions)), "games_used": len(tasks), "games_dropped": dropped, "passes_in_games": passes,
              "handicap": hd_counts, "allow_missing_hd": a.allow_missing_hd,
              "features": "fused_opp_diff_hd + per-ply tokens" + (" (+ opponent HSL: section 4M M14)" if True in need_x or True in need_r else ""),
              "expectations": {"rows": a.expect_rows, "pub_": a.expect_public, "priv_": a.expect_private},
              "fallback_submission": a.fallback_submission, "fallback_questions": fallback, "allow_fallback": a.allow_fallback,
              "lenient_inputs": a.lenient_inputs, "input_warnings": input_warnings, "allow_input_warnings": a.allow_input_warnings,
              "predicted_counts": dict(Counter(r for _, r in rows)), "arm_agreement_with_ensemble": agree_arm, "vs_fallback_submission": main_cmp,
              "device": device, "total_wall_s": time.time() - t0, "problems": problems, "exit_code": code}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}; predicted {report['predicted_counts']}; "
          f"arm agreement {agree_arm}" + (f"; vs main submission: {main_cmp}" if main_cmp else ""), flush=True)
    if fallback:
        print(f"WARNING: {len(fallback)} questions took {'the fallback submission' if fallback_rows else FALLBACK}: {fallback[:5]}"
              + ("" if a.allow_fallback else f" (exit {EXIT_FALLBACK}; pass --allow-fallback to accept)"), file=sys.stderr, flush=True)
    if input_warnings and code == EXIT_INPUT_WARNINGS:
        print(f"WARNING: {len(input_warnings)} questions were read leniently (exit {EXIT_INPUT_WARNINGS}; pass --allow-input-warnings to accept)",
              file=sys.stderr, flush=True)
    return code


# ---------------------------------------------------------------- offline / compare

def offline(a):
    """The checkpoints on a mock test's questions through the training-side tables of their fold (the reference for compare)."""
    from aicup.handicap import read_hd_table
    from aicup.ladder import FOLDS_CSV
    from aicup.pilot import pilot_questions
    from aicup.scoring import fold_players, player_games, read_scored_games
    import hd_check
    import l1b_cv

    t0 = time.time()
    f = a.fold
    if f not in ts.FOLDS:
        raise SystemExit(f"fold {f} is not one of {ts.FOLDS} (fold 0 is not used here)")
    mock = Path(a.mock_dir)
    ans = pd.read_csv(mock / "answers.csv", dtype=str, keep_default_na=False)
    players = fold_players(player_games(), pd.read_csv(FOLDS_CSV), f)  # experiments/make_mock_test.py's construction
    qs = pilot_questions(players).set_index("question_id")
    unknown = [q for q in ans["source_question"] if q not in qs.index]
    if unknown:
        raise SystemExit(f"{len(unknown)} mock questions are not fold {f}'s pilot questions, e.g. {unknown[:3]}")
    sel = qs.loc[ans["source_question"]].reset_index()
    hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
    opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    X, y, n, _ = hd_check.features(hsl, l1b, opp, hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))], sel)
    R = X["fused_opp_diff_hd"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = [ts.load_checkpoint(p, device) for p in a.models]
    R_by = {(0, False): R}
    if any(side.get("opp_hsl_r") for _, _, _, side in ckpts):  # section 4M M14: the training-side opponent-HSL table of the fold
        R_by[(0, True)] = np.hstack([R, opp_hsl_features(hsl, read_scored_games(ts.OPP_HSL / f"fold{f}"), sel)])
    X_by, offsets = {}, None
    for oh in sorted({bool(side.get("opp_hsl")) for _, _, _, side in ckpts}):
        X_by[(0, oh)], offsets = ts.raw_tokens([(ts.VEC / f"fold{f}", ts.scores_path(a.cache, f, opp_hsl=oh))])
    keys = sel["game_keys"].tolist()
    print(f"{len(sel)} questions of fold {f}, {next(iter(X_by.values())).shape[0]} plies, {len(ckpts)} checkpoints on {device} ({time.time() - t0:.0f} s)", flush=True)
    P, per_arm, n_pass, _ = ensemble(ckpts, R_by, keys, X_by, offsets, device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(P, columns=P_COLS).assign(question_id=ans["question_id"].tolist()).to_csv(out, index=False)
    pred = np.array([expected_score_decision(p) for p in P])
    official = balanced_mean_over_n([score(p, t) for p, t in zip(pred, y)], y, n)
    print(f"wrote {out}: {n_pass} passes; official score on these {len(sel)} fold-{f} questions {official:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return EXIT_OK


def compare(a):
    """Two probability files (question_id + p_<level>) side by side; with --answers (rank, n_games) the official score of each."""
    A, B = (pd.read_csv(p, dtype={"question_id": str}).set_index("question_id") for p in (a.a, a.b))
    ids = [q for q in A.index if q in B.index]
    if len(ids) != len(A) or len(ids) != len(B):
        print(f"WARNING: {len(A)} / {len(B)} questions, {len(ids)} in both", flush=True)
    PA, PB = A.loc[ids, P_COLS].to_numpy(dtype=float), B.loc[ids, P_COLS].to_numpy(dtype=float)
    ok = np.isfinite(PA).all(axis=1) & np.isfinite(PB).all(axis=1)
    d = np.abs(PA[ok] - PB[ok])
    da, db = np.array([expected_score_decision(p) for p in PA[ok]]), np.array([expected_score_decision(p) for p in PB[ok]])
    rep = {"a": str(a.a), "b": str(a.b), "questions": len(ids), "compared": int(ok.sum()), "max_abs_dp": float(d.max()) if ok.any() else None,
           "mean_abs_dp": float(d.mean()) if ok.any() else None, "decisions_differ": int((da != db).sum()), "tolerance": a.tol,
           "pass": bool(ok.any() and d.max() <= a.tol)}
    if a.answers:
        ans = pd.read_csv(a.answers, dtype={"question_id": str, "rank": str}).set_index("question_id").loc[ids]
        y = ans["rank"].map({lv: i for i, lv in enumerate(LEVELS)}).to_numpy(dtype=int)[ok]
        n = ans["n_games"].to_numpy(dtype=int)[ok]
        for name, pred in (("score_a", da), ("score_b", db)):
            rep[name] = balanced_mean_over_n([score(p, t) for p, t in zip(pred, y)], y, n)
    print(json.dumps(rep, indent=1))
    if a.report:  # compare never touches --out (the submission path)
        Path(a.report).write_text(json.dumps(rep, indent=1))
    return EXIT_OK if rep["pass"] else EXIT_INVALID


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("extract", "predict", "offline", "compare"))
    ap.add_argument("--inputs", nargs="+", default=ps.DEFAULT_INPUTS, help="the test CSVs, public first (as predict_submission.py)")
    ap.add_argument("--lenient-inputs", action="store_true", help="as predict_submission.py: read a malformed question from its non-empty columns")
    ap.add_argument("--allow-input-warnings", action="store_true")
    ap.add_argument("--vec", default=str(ROOT / "results" / "ct_submission" / "vec"), help="extract: where the parts go; predict: where they are")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="extract: shards of the games; predict: the main run's shard files")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shards", default=str(ROOT / "submission_rank.csv"), help="predict: the main run's --out, next to which its shard files are")
    ap.add_argument("--models", nargs="*", default=[], help="predict / offline: the checkpoints (.pt + .json) of trunk_seq_check.py train")
    ap.add_argument("--l1b-names", nargs="*", default=None, help="predict: only these shard L1b models feed tokens and R (default: all)")
    ap.add_argument("--out", default=str(ROOT / "submission_rank_ct.csv"))
    ap.add_argument("--report", default=None, help="predict: default <out stem>_report.json; compare: where to write its JSON (optional)")
    ap.add_argument("--fallback-submission", default=None, help="predict: the main submission; its row is used for a question this path cannot decide")
    ap.add_argument("--allow-fallback", action="store_true")
    ap.add_argument("--allow-missing-hd", action="store_true", help="predict: go on although some games carry no HD[] (their HD features are zeros)")
    ap.add_argument("--expect-rows", type=int, default=800)
    ap.add_argument("--expect-public", type=int, default=400)
    ap.add_argument("--expect-private", type=int, default=400)
    ap.add_argument("--mock-dir", default=None, help="offline: a experiments/make_mock_test.py directory (answers.csv names the fold's pilot questions)")
    ap.add_argument("--fold", type=int, default=1, help="offline: the fold the mock test was built from")
    ap.add_argument("--cache", default=str(ROOT / "results/trunk_seq"), help="offline: trunk_seq_check's score arrays")
    ap.add_argument("--a", default=None, help="compare: a probability file")
    ap.add_argument("--b", default=None, help="compare: the other")
    ap.add_argument("--answers", default=None, help="compare: answers.csv of the mock test")
    ap.add_argument("--tol", type=float, default=1e-3, help="compare: the max |dp| that passes")
    a = ap.parse_args(argv)
    if a.nshards < 1 or not 0 <= a.shard < a.nshards:
        ap.error("--shard needs 0 <= shard < --nshards")
    if a.cmd in ("predict", "offline") and not a.models:
        ap.error(f"{a.cmd} needs --models")
    if a.cmd == "offline" and not a.mock_dir:
        ap.error("offline needs --mock-dir")
    if a.cmd == "compare" and not (a.a and a.b):
        ap.error("compare needs --a and --b")
    return {"extract": extract, "predict": predict, "offline": offline, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
