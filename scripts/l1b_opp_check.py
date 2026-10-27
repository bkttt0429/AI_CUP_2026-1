#!/usr/bin/env python
"""Do the opponent's moves add evidence? Folds 1-4 CV of the fused features plus the L1b features of the opponent's moves.

Online games pair players of similar ranks, so the opponent's moves are a second, independent look at the target
player's rank. l1b_score.py --opponent scores the opponent's moves (the other colour) of the same evaluation games
under the same out-of-fold heads (A for folds 3-4, B for 1-2); question_features over those sums gives 16 more
features. Protocol as scripts/l1b_cv.py: validation pilot.pilot_questions, training questions sample_questions
(--per-player, RandomState(42 + v)), one MLPModel(hidden=32) fit per feature set and validation fold.

Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp (hsl 24 +
l1b 16 + opp 16) vs fused (hsl + l1b). Reported only: hsl_opp (hsl + opp) vs hsl, opp alone.

  python -I scripts/l1b_opp_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/l1b/opp_check.json
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
from aicup.qfeatures import build_matrix, sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402

SETS = ("hsl", "fused", "opp", "hsl_opp", "fused_opp")
PRE = [("fused_opp", "fused")]
REPORT = [("hsl_opp", "hsl"), ("fused", "hsl"), ("fused_opp", "hsl")]


def features(hsl, l1b, opp, questions):
    base, y, n = l1b_cv.feature_sets(hsl, l1b, questions)
    X_o, y2, _ = build_matrix(opp, questions, LEVELS)
    assert np.array_equal(y, y2)
    return {"hsl": base["hsl"], "fused": base["fused"], "opp": X_o, "hsl_opp": np.hstack([base["hsl"], X_o]),
            "fused_opp": np.hstack([base["fused"], X_o])}, y, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
        if set(opp["game_key"]) != set(hsl["game_key"]):
            raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players, "val": l1b_cv.pilot_questions(players)}
        print(f"fold {f}: {len(players)} players, {len(hsl)} games; opponent moves {int(opp['n_moves'].sum())} vs target "
              f"{int(hsl['n_moves'].sum())}", flush=True)
    rows, per_fold, nll = [], {s: [] for s in SETS}, {s: [] for s in SETS}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = features(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], val)
        sc = {}
        for s in SETS:
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, val)
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v}: " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS))},
                "per_fold": per_fold[s]} for s in SETS}

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= len(folds) - 1), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}
    for s, h in head.items():
        print(f"{s:10s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]) + "  1D-3D " + " ".join(f"{h['per_class'][c]:.3f}" for c in ("1D", "2D", "3D")))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "l1b_dirs": a.l1b, "opp_dirs": a.opp, "scores": head,
           "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
