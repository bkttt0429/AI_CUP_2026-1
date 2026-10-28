#!/usr/bin/env python
"""Compare a fused submission run's per-question probabilities and decisions with the offline fold-0 computation,
or with another run of the pipeline (issue #9, item 6).

The mock test's 40 questions are fold-0 pilot questions: tests/fixtures/mock_fold0_40/answers.csv maps each
question_id to its source_question. --offline is scripts/fit_fused_model.py's --fold0-proba table (per pilot
question the probabilities of hsl, fused_<model> and fused_avg; --set picks the column set, default fused_avg);
the run's <out stem>_proba.csv and submission file are read next to --submission. Every question must be in
both; max |Δp| must be within --tol and every decision (the run's pred_rank against the expected-score decision
on the offline probabilities) must agree, else exit 1. The default tolerance 1e-3 is the order of the per-move
log p differences between single-GPU and sharded scoring (up to 4e-3 nats per move) carried into the
probabilities. With --other instead of --offline, two runs of the pipeline (e.g. one GPU and 4 shards) are
compared on their own question ids.

  python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv \\
      --offline results/l1b/fold0_fused_proba.csv --answers tests/fixtures/mock_fold0_40/answers.csv --report results/mock_fold0_40_fused/fused_check.json
  python -I scripts/check_fused_submission.py --submission results/mock_fold0_40_fused/submission_rank.csv --other results/mock_fold0_40_fused4/submission_rank.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402

PROBA_COLS = [f"p_{lv}" for lv in LEVELS]


def read_run(submission):
    """(decisions {question_id: pred_rank}, probabilities DataFrame indexed by question_id) of one pipeline run."""
    submission = Path(submission)
    sub = pd.read_csv(submission, dtype=str, keep_default_na=False)
    proba = pd.read_csv(submission.with_name(submission.stem + "_proba.csv")).set_index("question_id")
    return dict(zip(sub["question_id"], sub["pred_rank"])), proba[PROBA_COLS]


def offline_table(offline, answers, column_set):
    """The offline probabilities re-indexed by the mock question ids through answers.csv's source_question."""
    ans = pd.read_csv(answers, dtype=str, keep_default_na=False)
    off = pd.read_csv(offline).set_index("question_id")
    cols = [f"{column_set}_p_{lv}" for lv in LEVELS]
    missing_cols = [c for c in cols if c not in off.columns]
    if missing_cols:
        raise ValueError(f"{offline} has no columns {missing_cols[:2]}... for --set {column_set}")
    missing = [s for s in ans["source_question"] if s not in off.index]
    if missing:
        raise ValueError(f"{len(missing)} source questions are not in the offline table {offline}, e.g. {missing[:3]}")
    table = off.loc[ans["source_question"].tolist(), cols].to_numpy(dtype=np.float64)
    return pd.DataFrame(table, columns=PROBA_COLS, index=ans["question_id"].tolist())


def compare(decisions, proba, reference, ref_decisions=None):
    """{"n_questions", "max_abs_diff", "question", "decisions_differ", ...}: the run against reference probabilities
    (same question ids); decisions compared with ref_decisions, or with the expected-score decision on the reference."""
    ids = list(proba.index)
    missing = [q for q in ids if q not in reference.index]
    if missing:
        raise ValueError(f"{len(missing)} questions are not in the reference, e.g. {missing[:3]}")
    P, R = proba.loc[ids].to_numpy(dtype=np.float64), reference.loc[ids].to_numpy(dtype=np.float64)
    finite = np.isfinite(P).all(axis=1) & np.isfinite(R).all(axis=1)
    diff = np.where(finite, np.abs(P - R).max(axis=1), 0.0)
    ref_dec = ref_decisions or {q: LEVELS[expected_score_decision(r)] for q, r in zip(ids, R)}
    differ = [q for q in ids if decisions.get(q) != ref_dec.get(q)]
    at = int(np.argmax(diff)) if len(diff) else None
    return {"n_questions": len(ids), "max_abs_diff": float(diff.max()) if len(diff) else 0.0, "question": ids[at] if at is not None else None,
            "decisions_differ": differ, "non_finite_questions": [q for q, f in zip(ids, finite) if not f]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--submission", required=True, help="the run's submission_rank.csv (its _proba.csv is read next to it)")
    ap.add_argument("--offline", default=None, help="fit_fused_model.py --fold0-proba table")
    ap.add_argument("--answers", default=str(ROOT / "tests" / "fixtures" / "mock_fold0_40" / "answers.csv"))
    ap.add_argument("--set", default="fused_avg",
                    help="offline column set to compare with (hsl, fused_A, fused_B, fused_avg; fused_opp_A, fused_opp_B, fused_opp_avg "
                         "in fit_fused_model.py --opp's table, for a rank_model_fused_opp.json run)")
    ap.add_argument("--other", default=None, help="another run's submission_rank.csv to compare with instead of --offline")
    ap.add_argument("--tol", type=float, default=1e-3, help="max |Δp| allowed per question")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)
    if bool(a.offline) == bool(a.other):
        ap.error("give exactly one of --offline and --other")
    try:
        decisions, proba = read_run(a.submission)
        if a.offline:
            reference, ref_dec, against = offline_table(a.offline, a.answers, a.set), None, f"offline {a.set} ({a.offline})"
        else:
            ref_dec, reference = read_run(a.other)
            against = f"run {a.other}"
        res = compare(decisions, proba, reference, ref_dec)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    res.update({"submission": str(a.submission), "against": against, "tol": a.tol,
                "pass": bool(res["max_abs_diff"] <= a.tol and not res["decisions_differ"] and not res["non_finite_questions"])})
    print(f"{res['n_questions']} questions vs {against}: max |Δp| {res['max_abs_diff']:.3e} (question {res['question']}), "
          f"{len(res['decisions_differ'])} decisions differ" + (f" {res['decisions_differ'][:5]}" if res["decisions_differ"] else "")
          + (f", {len(res['non_finite_questions'])} non-finite" if res["non_finite_questions"] else ""))
    print("PASS" if res["pass"] else f"FAIL (tolerance {a.tol:g})")
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
