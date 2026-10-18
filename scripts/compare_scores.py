#!/usr/bin/env python
"""Check freshly scored games against a reference table, e.g. nano4 against the lab01 scores.

  python -I scripts/compare_scores.py results/nano4_smoke/<job>/fold0 results/scores/games_fold0.csv.gz

Prints the comparison (aicup.scoring.compare_scored_games) and exits 1 when no game matches,
a game is missing from the reference, a move count differs, or some profile's log p differs
by more than --tol nats per move.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.scoring import compare_scored_games, read_scored_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="fold directory written by score_folds.py")
    ap.add_argument("ref", help="reference games table (.csv or .csv.gz)")
    ap.add_argument("--tol", type=float, default=5e-3, help="largest allowed |delta log p| per move")
    a = ap.parse_args()
    res = compare_scored_games(read_scored_games(a.new), pd.read_csv(a.ref))
    print(json.dumps(res, indent=1))
    ok = (res["games"] > 0 and res["missing_in_ref"] == 0 and res["n_moves_mismatch"] == 0
          and res["max_abs_per_move"] <= a.tol)
    print("scores match the reference" if ok else f"scores differ from the reference (tolerance {a.tol} nats per move)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
