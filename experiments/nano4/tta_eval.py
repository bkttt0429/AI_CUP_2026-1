"""8-symmetry TTA pilot evaluation on the fold-1 mock: decision change and |dp| of each symmetry vs symmetry 0, and the 8-way
probability average; questions that fell back (NaN probabilities, HD>=2 games whose standard handicap stones are not transformed
with the moves) in ANY symmetry are excluded from every comparison. The official test has no HD>=2 games.

  python -I logs/nano4/tta_eval.py --out results/r_variants/tta_pilot.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402

COLS = [f"p_{l}" for l in LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = pd.read_csv(ROOT / "results/mock_fold1_100/submission_rank_ct3_proba.csv")
    ans = pd.read_csv(ROOT / "data/mock_tests/fold1_100/answers.csv", dtype={"question_id": str, "rank": str}).set_index("question_id").loc[base["question_id"]]
    y = ans["rank"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
    Ps, names = [base[COLS].to_numpy()], [0]
    for k in range(1, 8):
        f = ROOT / f"results/mock_fold1_100_sym{k}/submission_rank_ct3_proba.csv"
        if f.exists():
            Ps.append(pd.read_csv(f)[COLS].to_numpy()); names.append(k)
    ok = np.all([np.isfinite(P).all(axis=1) for P in Ps], axis=0)
    rep = {"symmetries_present": names, "questions": int(len(y)), "questions_compared": int(ok.sum()), "symmetries": {}}
    P0 = Ps[0][ok]; yy = y[ok]
    d0 = np.array([expected_score_decision(p) for p in P0])
    sc = lambda d: float(np.mean([score(int(p), int(t)) for p, t in zip(d, yy)]))  # noqa: E731
    rep["score_sym0"] = sc(d0)
    for k, P in zip(names[1:], Ps[1:]):
        Pk = P[ok]; dk = np.array([expected_score_decision(p) for p in Pk])
        rep["symmetries"][str(k)] = {"decision_change": float((dk != d0).mean()), "mean_abs_dp": float(np.abs(Pk - P0).mean()), "max_abs_dp": float(np.abs(Pk - P0).max()), "score": sc(dk)}
    Pavg = np.mean([P[ok] for P in Ps], axis=0)
    davg = np.array([expected_score_decision(p) for p in Pavg])
    rep["score_avg"] = sc(davg)
    rep["avg_decision_change_vs_sym0"] = float((davg != d0).mean())
    rep["mean_decision_change"] = float(np.mean([v["decision_change"] for v in rep["symmetries"].values()])) if rep["symmetries"] else None
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
