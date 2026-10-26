#!/usr/bin/env python
"""Check the per-game vectors the submission pipeline builds against the ladder's fold vectors (item 2).

The set models were trained on aicup.gamevec vectors that scripts/ladder_build_vectors.py built from the
fold scores; scripts/predict_submission.py builds the same vectors from its shard files (per-move log p).
This script matches every scored task of a run on games that are also in a fold vectors file -- a mock
test cut from that fold -- by n_moves and the 18 log p sums (the nearest game in max |Δ sum|, unique
within --sum-tol) and compares the 300 staged and 75 whole fields and log1p(n_moves) column by column.
It prints the max absolute difference per field group with the task where it occurs, and exits 1 when
any group exceeds --tol, a task has no match, or a match is ambiguous. Single-GPU and sharded scoring
differ by up to ~4e-3 nats per move (TF32, batch composition), so --tol is a tolerance, not zero; a
different normalisation, segment boundary or move filter shows as differences of order 0.1-1 in whole
groups. The count fields (n_eff, present) must agree exactly.

With --set-models and --set-base (the run's own) it also checks the probabilities: the offline path --
the run's questions packed over the fold file's vectors of their matched games, the base on the fold
file's sums, the same averaging (predict_submission.average_set_models) -- against the run's
<out stem>_proba.csv, max |Δp| per question within --proba-tol, and every decision equal to the run's
submission file. The mock test is cut from fold 0, which is training data of these models, so this is a
consistency check of the two paths, not a score.

  python -I scripts/check_game_vectors.py --inputs tests/fixtures/mock_fold0_40/rank_prediction_test_public.csv \\
      tests/fixtures/mock_fold0_40/rank_prediction_test_private.csv --out results/mock_fold0_40/submission_rank.csv \\
      --nshards 1 --vectors results/ladder/vectors/fold0.npz [--set-models results/models/set/r3_pma_s*.pt \\
      --set-base results/models/rank_model_f01234.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.gamevec import SEGMENTS, WHOLE, vec_names  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qfeatures import question_features  # noqa: E402
import predict_submission as ps  # noqa: E402

COUNT_STATS = ("n_eff", "present")


def match_tasks(vec, ref, sum_tol):
    """For every task (row of vec): the matching row of ref -- same n_moves, nearest in max |Δ sum| within
    sum_tol, no second candidate within sum_tol -- or None; returns (match [T] of int / -1, unmatched, ambiguous)."""
    match, unmatched, ambiguous = np.full(len(vec["n_moves"]), -1, dtype=np.int64), [], []
    ref_n, ref_sums = np.asarray(ref["n_moves"], dtype=np.int64), np.asarray(ref["sums"], dtype=np.float64)
    for i, (n, s) in enumerate(zip(vec["n_moves"].tolist(), np.asarray(vec["sums"], dtype=np.float64))):
        cand = np.flatnonzero(ref_n == n)
        if not len(cand):
            unmatched.append(i)
            continue
        d = np.abs(ref_sums[cand] - s).max(axis=1)
        order = np.argsort(d)
        if d[order[0]] > sum_tol:
            unmatched.append(i)
        elif len(order) > 1 and d[order[1]] <= sum_tol:
            ambiguous.append(i)
        else:
            match[i] = cand[order[0]]
    return match, unmatched, ambiguous


PER_STAGE_STATS = ("n_eff", "mean_surprisal", "present")  # one column per stage; the rest are per profile


def _groups(names):
    """{stat: column indices} of a vec_names list (s1_mean_12k -> "mean", s1_n_eff -> "n_eff")."""
    out = {}
    for i, name in enumerate(names):
        stat = name.split("_", 1)[1]
        if stat not in PER_STAGE_STATS:
            stat = stat.split("_", 1)[0]  # per-profile: mean/std/q10/q90
        out.setdefault(stat, []).append(i)
    return out


def compare(vec, ref, match):
    """Per rep and field group the max |Δ| over the matched tasks and the task where it occurs."""
    ok = match >= 0
    rows = {}
    for rep, segments in (("staged", SEGMENTS), ("whole", WHOLE)):
        a, b = np.asarray(vec[rep], dtype=np.float64)[ok], np.asarray(ref[rep], dtype=np.float64)[match[ok]]
        diff = np.abs(a - b)
        tasks = np.flatnonzero(ok)
        for stat, cols in _groups(vec_names(segments)).items():
            d = diff[:, cols]
            at = np.unravel_index(np.argmax(d), d.shape) if d.size else (0, 0)
            rows[f"{rep}.{stat}"] = {"max_abs_diff": float(d.max()) if d.size else 0.0, "task": int(tasks[at[0]]) if d.size else None,
                                     "column": vec_names(segments)[cols[at[1]]] if d.size else None, "count": stat in COUNT_STATS}
    n_a, n_b = np.log1p(np.asarray(vec["n_moves"], dtype=np.float64)[ok]), np.log1p(np.asarray(ref["n_moves"], dtype=np.float64)[match[ok]])
    rows["log1p_n_moves"] = {"max_abs_diff": float(np.abs(n_a - n_b).max()) if ok.any() else 0.0, "task": None, "column": None, "count": True}
    return rows


def offline_probs(questions, owner, ref, match, base_model, models):
    """The offline path (P [Q, 10], question ids, info): every question with a usable game packed over the fold
    file's vectors of its matched games (their keys, in task order), the base on the fold file's sums and
    n_moves, predict_submission.average_set_models. Needs every task matched."""
    if (match < 0).any():
        raise ValueError("offline probabilities need every task matched")
    frame, qis, by_q = ps.question_frame(questions, owner, [str(ref["game_key"][m]) for m in match])
    X = np.stack([question_features(np.asarray(ref["sums"])[match[by_q[qi]]], np.asarray(ref["n_moves"])[match[by_q[qi]]]) for qi in qis])
    P, info = ps.average_set_models(frame, ref, X, base_model, models)
    return P, frame["question_id"].tolist(), info


def compare_probs(P, qids, out, tol):
    """The offline P against the run's <out stem>_proba.csv and submission file: {"n_questions", "max_abs_diff",
    "question", "decisions_differ": [question ids], "tol", "pass"}."""
    out = Path(out)
    proba = pd.read_csv(out.with_name(out.stem + "_proba.csv")).set_index("question_id").loc[qids, [f"p_{lv}" for lv in LEVELS]].to_numpy()
    pred = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("question_id")["pred_rank"].loc[qids].tolist()
    diff = np.abs(P - proba).max(axis=1)
    decided = [LEVELS[expected_score_decision(p)] for p in P]
    differ = [q for q, d, r in zip(qids, decided, pred) if d != r]
    at = int(np.argmax(diff)) if len(diff) else None
    return {"n_questions": len(qids), "max_abs_diff": float(diff.max()) if len(diff) else 0.0, "question": qids[at] if at is not None else None,
            "decisions_differ": differ, "tol": tol, "pass": bool((len(diff) == 0 or diff.max() <= tol) and not differ)}


def compare_l1b(fold_dir, shard_models, l1b_per_task, ref, match, tol):
    """One l1b_score.py fold directory against the shards' L1b sums of the same games: (model name, row). The directory's
    meta_s*.json names the heads (state and trunk sha256); it must match exactly one shard model. Games the table
    lacks count as missing (and fail)."""
    import glob

    from aicup.scoring import read_scored_games
    metas = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(fold_dir) / "meta_s*.json")))]
    if not metas:
        raise ValueError("no meta_s*.json (not an l1b_score.py fold directory)")
    heads = metas[0].get("heads", {})
    key = (heads.get("state_sha256"), heads.get("trunk_sha256"))
    hits = [j for j, m in enumerate(shard_models) if (m.get("state_sha256"), m.get("trunk_sha256")) == key]
    if len(hits) != 1:
        raise ValueError(f"no shard model matches its heads sha256 {str(key[0])[:12]} ({len(hits)} matches)")
    j = hits[0]
    name = shard_models[j].get("name") or f"model{j}"
    table = read_scored_games(fold_dir).set_index("game_key")
    cols = [f"sum_{lv}" for lv in LEVELS]
    diffs, tasks, missing = [], [], 0
    for k in range(len(match)):
        gk = str(ref["game_key"][match[k]])
        if gk not in table.index:
            missing += 1
            continue
        diffs.append(np.abs(table.loc[gk, cols].to_numpy(dtype=np.float64) - l1b_per_task[k][j]))
        tasks.append(k)
    if diffs:
        d = np.stack(diffs)
        at = np.unravel_index(int(np.argmax(d)), d.shape)
        row = {"dir": str(fold_dir), "max_abs_diff": float(d.max()), "task": int(tasks[at[0]]), "level": LEVELS[at[1]]}
    else:
        row = {"dir": str(fold_dir), "max_abs_diff": 0.0, "task": None, "level": None}
    row.update({"matched": len(tasks), "missing": missing, "tol": tol, "pass": bool(missing == 0 and row["max_abs_diff"] <= tol)})
    return name, row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", nargs="+", default=ps.DEFAULT_INPUTS)
    ap.add_argument("--out", required=True, help="the run's --out (its shard files are read)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--vectors", required=True, help="a fold vectors file (scripts/ladder_build_vectors.py)")
    ap.add_argument("--tol", type=float, default=0.05, help="max |Δ| allowed in the float field groups")
    ap.add_argument("--sum-tol", type=float, default=0.5, help="max |Δ sum| for a game to match")
    ap.add_argument("--lenient-inputs", action="store_true")
    ap.add_argument("--set-models", nargs="*", default=[], metavar="PT", help="the run's set models: also check its probabilities")
    ap.add_argument("--set-base", default=None, help="the run's --set-base; required with --set-models")
    ap.add_argument("--set-device", default="cpu")
    ap.add_argument("--proba-tol", type=float, default=1e-3, help="max |Δp| allowed per question")
    ap.add_argument("--l1b-scores", nargs="*", default=[], metavar="FOLD_DIR",
                    help="l1b_score.py fold directories (e.g. results/l1b/scores3_A/fold0): the shards' per-game L1b sums are compared "
                         "with their tables, each directory matched to a shard model by the sha256 in its meta_s*.json")
    ap.add_argument("--l1b-tol", type=float, default=0.05, help="max |Δ sum| allowed per game and head")
    ap.add_argument("--report", default=None, help="write the result as JSON here too")
    a = ap.parse_args(argv)
    if a.set_models and not a.set_base:
        ap.error("--set-models needs --set-base, the run's own")
    questions, _ = ps.load_questions(a.inputs, a.lenient_inputs)
    tasks, owner, _, _ = ps.build_tasks(questions)
    per_task, _, moves = ps.load_shards(Path(a.out), a.nshards, len(tasks), ps.questions_sha(a.inputs, questions), with_moves=True)
    vec = ps.game_vectors(per_task, moves)
    with np.load(a.vectors, allow_pickle=False) as z:
        ref = {k: z[k] for k in ("game_key", "staged", "whole", "sums", "n_moves")}
    match, unmatched, ambiguous = match_tasks(vec, ref, a.sum_tol)
    rows = compare(vec, ref, match)
    bad_groups = [g for g, r in rows.items() if (r["max_abs_diff"] > 0 if r["count"] else r["max_abs_diff"] > a.tol)]
    result = {"tasks": len(tasks), "matched": int((match >= 0).sum()), "unmatched": unmatched, "ambiguous": ambiguous,
              "tol": a.tol, "sum_tol": a.sum_tol, "groups": rows, "failed_groups": bad_groups, "proba": None,
              "pass": not unmatched and not ambiguous and not bad_groups}
    print(f"{result['matched']} of {len(tasks)} tasks matched a game of {a.vectors} ({len(unmatched)} unmatched, {len(ambiguous)} ambiguous)")
    for g, r in rows.items():
        where = "" if r["task"] is None else f"  (task {r['task']}, {r['column']})"
        flag = "FAIL" if g in bad_groups else "ok"
        print(f"  {g:22s} max |Δ| {r['max_abs_diff']:.3e}  {flag}{where}")
    if a.l1b_scores:
        result["l1b"] = {}
        if unmatched or ambiguous:
            print("  l1b sums: not checked (unmatched or ambiguous tasks)")
            result["pass"] = False
        else:
            l1b_models, l1b_per_task, _ = ps.load_shards_l1b(Path(a.out), a.nshards)
            for d in a.l1b_scores:
                try:
                    name, row = compare_l1b(d, l1b_models, l1b_per_task, ref, match, a.l1b_tol)
                except ValueError as e:
                    print(f"  l1b {d}: {e}")
                    result["pass"] = False
                    bad_groups.append(f"l1b.{Path(d).parent.name}")
                    continue
                result["l1b"][name] = row
                if not row["pass"]:
                    bad_groups.append(f"l1b.{name}")
                    result["pass"] = False
                where = "" if row["task"] is None else f"  (task {row['task']}, {row['level']})"
                print(f"  {'l1b.' + name:22s} max |Δ sum| {row['max_abs_diff']:.3e}  {'ok' if row['pass'] else 'FAIL'}{where}  "
                      f"({row['matched']} games, {row['missing']} missing from {d})")
    if a.set_models:
        if unmatched or ambiguous:
            print("  set models: not checked (unmatched or ambiguous tasks)")
            result["pass"] = False
        else:
            models, warnings = ps.load_set_models(a.set_models, a.set_device)
            base_doc, base_model = ps.load_rank_model(a.set_base)
            warnings += ps.check_pairing(models, base_doc, a.set_base)
            for w in warnings:
                print(f"WARNING: {w}", file=sys.stderr, flush=True)
            P, qids, _ = offline_probs(questions, owner, ref, match, base_model, models)
            pr = compare_probs(P, qids, a.out, a.proba_tol)
            result["proba"] = {**pr, "models": a.set_models, "base": a.set_base, "warnings": warnings}
            result["pass"] = result["pass"] and pr["pass"]
            if not pr["pass"]:
                bad_groups.append("proba")
            print(f"  set models: {pr['n_questions']} questions, max |Δp| {pr['max_abs_diff']:.3e} (question {pr['question']}), "
                  f"{len(pr['decisions_differ'])} decisions differ  {'ok' if pr['pass'] else 'FAIL'}")
    print("PASS" if result["pass"] else f"FAIL: {', '.join(bad_groups) or ''}{' unmatched/ambiguous tasks' if unmatched or ambiguous else ''}")
    if a.report:
        Path(a.report).write_text(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
