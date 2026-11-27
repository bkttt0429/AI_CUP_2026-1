#!/usr/bin/env python
"""Folds 1-4 CV: does the question-level mean of per-game shape features add to fused_opp? (issue #21, the independent channel)

Issue #19 fed move sequences to encoders that corrected the L1b sums (base + Δ) and gained nothing at the question level,
although the per-game shape features of scripts/seq_diag.py carry a small, consistent rank signal. Here those features
(aicup.seqshape: std / autocorrelation / second-half-minus-first-half / slope of the per-move posterior-mean rank index, mean
and autocorrelation of the per-move surprisal; log_moves left out) are averaged over each question's games and appended to
the 56 fused_opp features: fused_opp_shape (+8, the target's moves) and fused_opp_shape2 (+16, the target's and the
opponent's). Protocol as scripts/l1b_opp_check.py (pilot validation questions, sample_questions per player with
RandomState(42 + v), MLPModel(hidden=32), player bootstrap); fused_opp must reproduce 0.6735. Pre-specified (conditions
(1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_shape vs fused_opp. Reported only:
fused_opp_shape2 vs fused_opp, fused_opp_shape2 vs fused_opp_shape. Fold 0 is refused.

  python -I scripts/shape_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/seq/shape_check.json
"""
import argparse
import sys
import time
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
from aicup.seqshape import SHAPE, fold_shape_table, question_shape  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("fused_opp", "fused_opp_shape", "fused_opp_shape2")
PRE = [("fused_opp_shape", "fused_opp")]
REPORT = [("fused_opp_shape2", "fused_opp"), ("fused_opp_shape2", "fused_opp_shape")]


def features(hsl, l1b, opp, shape_t, shape_o, questions):
    """The three feature sets of `questions`: fused_opp (56), + the target's shape means (64), + the opponent's (72)."""
    X, y, n = oc.features(hsl, l1b, opp, questions)
    S_t, S_o = question_shape(shape_t, questions), question_shape(shape_o, questions)
    return {"fused_opp": X["fused_opp"], "fused_opp_shape": np.hstack([X["fused_opp"], S_t]),
            "fused_opp_shape2": np.hstack([X["fused_opp"], S_t, S_o])}, y, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    try:
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            t0 = time.time()
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            shape_t, shape_o = fold_shape_table(Path(where[f]) / f"fold{f}"), fold_shape_table(Path(where_o[f]) / f"fold{f}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "shape_t": shape_t, "shape_o": shape_o, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions, shape tables in {time.time() - t0:.0f} s", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll, dims = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("shape_t"), cat("shape_o"), train_q)
        d = data[v]
        X_va, y_va, n_va = features(d["hsl"], d["l1b"], d["opp"], d["shape_t"], d["shape_o"], d["val"])
        sc = {}
        for s in SETS:
            dims[s] = int(X_tr[s].shape[1])
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])), "n_features": dims[s],
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in SETS}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}
    for s, h in head.items():
        print(f"{s:18s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp,
           "shape_features": list(SHAPE), "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
