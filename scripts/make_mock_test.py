#!/usr/bin/env python
"""Build official-format mock test files from one fold's scored players, with answers and,
when that fold's scores exist, the predictions the offline path gives for the same questions.

Outputs in --out-dir: rank_prediction_test_public.csv, rank_prediction_test_private.csv
(question_id pub_q_XXXX / priv_q_XXXX, num_games, sgf_1..20, color_1..20), answers.csv and
offline_predictions.csv (question_id, pred_rank) for checking predict_submission.py end to end.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.qfeatures import build_matrix  # noqa: E402
from aicup.rankmodel import from_state  # noqa: E402
from aicup.scoring import fold_players, load_games, player_games  # noqa: E402


def official_rows(qs, prefix, data):
    rows = []
    for j, q in enumerate(qs.itertuples(), start=1):
        r = {"question_id": f"{prefix}_q_{j:04d}", "num_games": len(q.game_keys)}
        for i in range(1, 21):
            key = q.game_keys[i - 1] if i <= len(q.game_keys) else None
            r[f"sgf_{i}"] = data[key][0] if key else ""
            r[f"color_{i}"] = data[key][1] if key else ""
        rows.append(r)
    cols = ["question_id", "num_games"] + [f"sgf_{i}" for i in range(1, 21)] + [f"color_{i}" for i in range(1, 21)]
    return pd.DataFrame(rows, columns=cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=str(ROOT / "results" / "models" / "rank_model.json"))
    ap.add_argument("--scores", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "mock_tests"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    players = fold_players(player_games(), pd.read_csv(ROOT / "splits" / "player_folds.csv"), a.fold)
    qs = pilot_questions(players)
    pick = qs.iloc[np.sort(np.random.RandomState(a.seed).choice(len(qs), size=a.n_questions, replace=False))].reset_index(drop=True)
    half = len(pick) // 2
    data = load_games([k for ks in pick["game_keys"] for k in ks])
    official_rows(pick.iloc[:half], "pub", data).to_csv(out / "rank_prediction_test_public.csv", index=False)
    official_rows(pick.iloc[half:], "priv", data).to_csv(out / "rank_prediction_test_private.csv", index=False)
    ids = [f"pub_q_{j:04d}" for j in range(1, half + 1)] + [f"priv_q_{j:04d}" for j in range(1, len(pick) - half + 1)]
    pd.DataFrame({"question_id": ids, "rank": pick["rank"], "source_question": pick["question_id"],
                  "n_games": pick["n_games"]}).to_csv(out / "answers.csv", index=False)
    print(f"wrote {len(pick)} mock questions ({half} public) to {out}")

    parts = sorted(glob.glob(str(Path(a.scores) / f"fold{a.fold}" / "games_s*_p*.csv")))
    if parts and Path(a.model).exists():
        games = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        X, _, _ = build_matrix(games, pick)
        model = from_state(json.loads(Path(a.model).read_text())["state"])
        pred = [LEVELS[expected_score_decision(p)] for p in model.predict_proba(X)]
        pd.DataFrame({"question_id": ids, "pred_rank": pred}).to_csv(out / "offline_predictions.csv", index=False)
        print(f"wrote offline predictions for comparison ({out / 'offline_predictions.csv'})")


if __name__ == "__main__":
    main()
