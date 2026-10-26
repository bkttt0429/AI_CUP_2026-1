#!/usr/bin/env python
"""Fold-0 check of the opponent-move features (issue #6, 2026-10-27; the user's decision: a check, no selection on it).

As scripts/l1b_fold0_check.py, with the opponent features: the MLPs are fitted on folds 1-4 (training questions
sample_questions --per-player per player, RandomState(42) over the folds ascending; folds 1-2 scored by B, 3-4 by
A, for the target's L1b features and the opponent's alike) and scored on fold 0's pilot questions with fold 0's
features from A, from B, and the mean of the two probabilities. Feature sets: hsl, fused (hsl + l1b), fused_opp
(fused + opp). Expected if the CV gain (+0.0191 over fused) is real: fused_opp above fused on fold 0 too.

  python -I scripts/l1b_opp_fold0_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/l1b/opp_fold0_check.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

TRAIN_FOLDS = (1, 2, 3, 4)
SETS = ("hsl", "fused", "fused_opp")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs=2, required=True, metavar=("A_DIR", "B_DIR"))
    ap.add_argument("--opp", nargs=2, required=True, metavar=("A_DIR", "B_DIR"))
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    src = {1: 1, 2: 1, 3: 0, 4: 0}  # index into (A, B): folds 1-2 from B, 3-4 from A
    data = {}
    for f in TRAIN_FOLDS:
        hsl, l1b = l1b_cv.load_fold(a.hsl, a.l1b[src[f]], f)
        opp = read_scored_games(Path(a.opp[src[f]]) / f"fold{f}")
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": l1b_cv.players_from_scores(hsl, folds_df)}
    rng = np.random.RandomState(l1b_cv.TRAIN_SEED)
    train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in TRAIN_FOLDS], ignore_index=True)
    cat = lambda key: pd.concat([data[f][key] for f in TRAIN_FOLDS], ignore_index=True)  # noqa: E731
    X_tr, y_tr, _ = oc.features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
    models = {s: MLPModel(hidden=32).fit(X_tr[s], y_tr) for s in SETS}
    hsl0 = read_scored_games(Path(a.hsl) / "fold0")
    val = l1b_cv.pilot_questions(l1b_cv.players_from_scores(hsl0, folds_df))
    X = {}
    for side, i in (("A", 0), ("B", 1)):
        _, l1b0 = l1b_cv.load_fold(a.hsl, a.l1b[i], 0)
        opp0 = read_scored_games(Path(a.opp[i]) / "fold0")
        X[side], y, n = oc.features(hsl0, l1b0, opp0, val)
    pid = val["player_id"].to_numpy(dtype=str)
    P = {"hsl": models["hsl"].predict_proba(X["A"]["hsl"])}
    for s in ("fused", "fused_opp"):
        pa, pb = models[s].predict_proba(X["A"][s]), models[s].predict_proba(X["B"][s])
        P.update({f"{s}_A": pa, f"{s}_B": pb, f"{s}_avg": (pa + pb) / 2})
    sc = {k: evaluate(p, y, val) for k, p in P.items()}
    s = {k: v[1] for k, v in sc.items()}
    out = {"train_questions": int(len(train_q)), "questions": int(len(y)), "players": int(len(set(pid))), "scores": {}, "comparisons": {}}
    for k in P:
        out["scores"][k] = {"balanced_mean_over_n": balanced_mean_over_n(s[k], y, n), "nll_balanced": sc[k][0]["nll_balanced"],
                            "balanced_by_n": {str(m): balanced_mean_over_n(s[k], y, n, sizes=(m,)) for m in SIZES},
                            "per_class": {LEVELS[c]: float(s[k][y == c].mean()) for c in range(len(LEVELS))}}
        h = out["scores"][k]
        print(f"{k:14s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  1D-3D " + " ".join(f"{h['per_class'][c]:.3f}" for c in ("1D", "2D", "3D")), flush=True)
    for x, z in (("fused_opp_avg", "fused_avg"), ("fused_opp_avg", "hsl"), ("fused_avg", "hsl")):
        d = paired_diff(s[x], s[z], y, pid, n, n=a.n_boot)
        out["comparisons"][f"{x} vs {z}"] = d
        print(f"{x} vs {z}: {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    out.update(environment("cpu"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
