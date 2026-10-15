#!/usr/bin/env python
"""Official test CSVs -> submission_rank.csv using the Human SL features and the saved L3/L4 model.

For every question: drop unusable games (wrong size, illegal replay, non-alternating
colours, no target moves), score the rest under the 18 rank profiles, build the question
features, predict with the saved model and the expected-score decision. Public questions are
written first, then private, and the file is validated before the script exits.
Also writes a JSON report (games used / dropped and why, passes, prediction counts) and the
per-question class probabilities.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qfeatures import FEATURE_NAMES, question_features  # noqa: E402
from aicup.rankmodel import from_state  # noqa: E402
from aicup.scoring import init_worker, score_stream  # noqa: E402
from aicup.submission import check_game, read_test, validate_submission, write_submission  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
FALLBACK = "1D"  # used only for a question with no usable game: an interior class keeps both neighbours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[str(ROOT / "data" / "tests" / "rank_prediction_test_public.csv"),
                                                    str(ROOT / "data" / "tests" / "rank_prediction_test_private.csv")])
    ap.add_argument("--model", default=str(ROOT / "results" / "models" / "rank_model.json"))
    ap.add_argument("--out", default=str(ROOT / "submission_rank.csv"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    report_path = Path(a.report) if a.report else out.with_name(out.stem + "_report.json")
    t0 = time.time()

    questions = [q for path in a.inputs for q in read_test(path)]
    saved = json.loads(Path(a.model).read_text())
    from aicup.humansl import PROFILES
    if saved["profiles"] != PROFILES or saved["features"] != FEATURE_NAMES:
        raise SystemExit("model was trained with different profiles or features")

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
    print(f"{len(questions)} questions, {sum(len(q['games']) for q in questions)} games, {len(tasks)} usable, "
          f"{len(dropped)} dropped, {passes} passes", flush=True)

    import torch
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    from aicup.humansl import load_humansl, meta_rows
    model, _ = load_humansl(str(CKPT), "cuda")
    per_task, _, n_pos, wall = score_stream(pool, model, tasks, meta_rows([(p, p) for p in PROFILES]), log_every=500)
    pool.close()

    by_q = {}
    for idx, qi in enumerate(owner):
        by_q.setdefault(qi, []).append(per_task[idx])
    rank_model = from_state(saved["state"])
    rows, probs, fallback = [], [], []
    for qi, q in enumerate(questions):
        games = by_q.get(qi, [])
        if not games:
            rows.append((q["question_id"], FALLBACK))
            probs.append([np.nan] * 10)
            fallback.append(q["question_id"])
            continue
        x = question_features(np.stack([s for _, s in games]), np.array([n for n, _ in games]))
        p = rank_model.predict_proba(x[None, :])[0]
        rows.append((q["question_id"], LEVELS[expected_score_decision(p)]))
        probs.append(p.tolist())
    write_submission(rows, out)
    problems = validate_submission(out, [q["question_id"] for q in questions])
    pd.DataFrame(probs, columns=[f"p_{lv}" for lv in LEVELS]).assign(question_id=[q["question_id"] for q in questions]) \
        .to_csv(out.with_name(out.stem + "_proba.csv"), index=False)
    report = {"inputs": a.inputs, "model": a.model, "model_name": saved["name"], "questions": len(questions),
              "games_declared": int(sum(q["declared_games"] for q in questions)), "games_present": int(sum(len(q["games"]) for q in questions)),
              "games_used": len(tasks), "games_dropped": dropped, "passes": passes, "positions": int(n_pos),
              "scoring_wall_s": wall, "total_wall_s": time.time() - t0, "fallback_questions": fallback,
              "predicted_counts": dict(Counter(r for _, r in rows)), "problems": problems}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"wrote {out} ({len(rows)} rows) and {report_path}; problems: {problems or 'none'}", flush=True)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
