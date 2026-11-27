"""TTA validation on fold 1's n=20 pilot questions as a mock test: the 8-symmetry probability average vs symmetry 0 (and each
single symmetry), paired player bootstrap (conditions: CI lower bound > 0 and diff >= 0.003 to deploy test-side TTA).

  python -I logs/nano4/tta_val_eval.py --root results/tta_val --out results/r_variants/tta_val.json
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
from aicup.qeval import paired_diff  # noqa: E402

COLS = [f"p_{l}" for l in LEVELS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/tta_val")
    ap.add_argument("--answers", default="data/mock_tests/fold1_n20/answers.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    ans = pd.read_csv(ROOT / a.answers, dtype={"question_id": str, "rank": str, "source_question": str})
    y = ans["rank"].map({l: i for i, l in enumerate(LEVELS)}).to_numpy()
    pid = ans["source_question"].str.split("_").str[1].to_numpy()  # "<rank>_<pid8>_n20_0"
    n = ans["n_games"].to_numpy()
    Ps, names = [], []
    for k in range(8):
        f = ROOT / a.root / f"sym{k}" / "submission_rank_ct_proba.csv"
        if f.exists():
            P = pd.read_csv(f).set_index("question_id").loc[ans["question_id"], COLS].to_numpy(); Ps.append(P); names.append(k)
    ok = np.all([np.isfinite(P).all(axis=1) for P in Ps], axis=0)
    rep = {"symmetries_present": names, "questions": int(len(y)), "compared": int(ok.sum()), "singles": {}}
    yy, pp, nn = y[ok], pid[ok], n[ok]
    scores = {}
    for k, P in zip(names, Ps):
        d = np.array([expected_score_decision(p) for p in P[ok]])
        scores[k] = np.array([score(int(a_), int(b)) for a_, b in zip(d, yy)])
        rep["singles"][str(k)] = float(scores[k].mean())
    Pavg = np.mean([P[ok] for P in Ps], axis=0)
    davg = np.array([expected_score_decision(p) for p in Pavg])
    s_avg = np.array([score(int(a_), int(b)) for a_, b in zip(davg, yy)])
    rep["avg_score"] = float(s_avg.mean())
    rep["mean_single_score"] = float(np.mean([v for v in rep["singles"].values()]))
    d0 = np.array([expected_score_decision(p) for p in Ps[0][ok]])
    rep["avg_decision_change_vs_sym0"] = float((davg != d0).mean())
    dd = paired_diff(s_avg, scores[0], yy, pp, nn, sizes=(20,), n=a.n_boot)
    rep["avg_vs_sym0"] = {**dd, "pass": bool(dd["ci95"][0] > 0 and dd["diff"] >= 0.003)}
    # the average of the 8 symmetries vs the mean single-symmetry score (what a random orientation would give)
    mean_single = np.mean([scores[k] for k in names], axis=0)
    dd2 = paired_diff(s_avg, mean_single, yy, pp, nn, sizes=(20,), n=a.n_boot)
    rep["avg_vs_mean_single"] = dd2
    print(json.dumps({k: v for k, v in rep.items() if k != "avg_vs_sym0"}, indent=1)[:1500])
    print("avg vs sym0:", f"{dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]", "PASS" if rep["avg_vs_sym0"]["pass"] else "no")
    print("avg vs mean single:", f"{dd2['diff']:+.4f} [{dd2['ci95'][0]:+.4f}, {dd2['ci95'][1]:+.4f}]")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
