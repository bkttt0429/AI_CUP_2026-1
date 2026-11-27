#!/usr/bin/env python
"""Folds 1-4 CV: do the strength head's question features add to fused_opp? (issue #14, option b)

Protocol as scripts/l1b_opp_check.py (validation pilot questions, training sample_questions with --per-player per
player and RandomState(42 + v), one MLPModel(hidden=32) fit per feature set and validation fold, player bootstrap).
Feature sets: hsl (24), fused_opp (56, the submission's), se (aicup.sefeat's 9 of the target's strength statistics),
hsl_se (24 + 9), fused_opp_se (56 + 9), fused_opp_se2 (56 + 9 + the opponent's 9). Pre-specified (conditions (1), (3),
(4)): fused_opp_se vs fused_opp and fused_opp_se2 vs fused_opp. Reported only: se vs hsl, hsl_se vs hsl, fused_opp vs hsl.
--se / --se-opp: scripts/se_score.py directories (each fold in exactly one; the same games as --hsl).

  python -I scripts/se_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --se results/l1b/se3 results/l1b/se3_B --se-opp results/l1b/se3_opp results/l1b/se3_B_opp --out results/l1b/se_check.json
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
from aicup.sefeat import SE_FEATURE_NAMES, build_se_matrix  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("hsl", "fused_opp", "se", "hsl_se", "fused_opp_se", "fused_opp_se2")
PRE = [("fused_opp_se", "fused_opp"), ("fused_opp_se2", "fused_opp")]
REPORT = [("se", "hsl"), ("hsl_se", "hsl"), ("fused_opp", "hsl")]


def features(hsl, l1b, opp, se, se_opp, questions):
    """{set: X} of SETS plus (y, n_games): l1b_opp_check.features' sets with the strength statistics appended."""
    base, y, n = oc.features(hsl, l1b, opp, questions)
    X_s, y2, _ = build_se_matrix(se, questions)
    X_o, y3, _ = build_se_matrix(se_opp, questions)
    assert np.array_equal(y, y2) and np.array_equal(y, y3)
    return {"hsl": base["hsl"], "fused_opp": base["fused_opp"], "se": X_s, "hsl_se": np.hstack([base["hsl"], X_s]),
            "fused_opp_se": np.hstack([base["fused_opp"], X_s]), "fused_opp_se2": np.hstack([base["fused_opp"], X_s, X_o])}, y, n


def _same_games(table, hsl, what, fold):
    if set(table["game_key"]) != set(hsl["game_key"]):
        raise ValueError(f"fold {fold}: the {what} scores cover other games than the Human SL scores")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories")
    ap.add_argument("--se", nargs="+", required=True, help="se_score.py directories (the target's moves)")
    ap.add_argument("--se-opp", nargs="+", required=True, help="se_score.py --opponent directories")
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
        where = {name: l1b_cv.locate_folds(dirs, folds) for name, dirs in (("l1b", a.l1b), ("opp", a.opp), ("se", a.se), ("se_opp", a.se_opp))}
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where["l1b"][f], f)
            tables = {k: _same_games(read_scored_games(Path(where[k][f]) / f"fold{f}"), hsl, k, f) for k in ("opp", "se", "se_opp")}
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, **tables, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll = [], {s: [] for s in SETS}, {s: [] for s in SETS}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("se"), cat("se_opp"), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = features(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], data[v]["se"], data[v]["se_opp"], val)
        sc = {}
        for s in SETS:
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, val)
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in SETS}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}
    for s, h in head.items():
        print(f"{s:14s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl,
           "dirs": {k: {str(f): str(d) for f, d in w.items()} for k, w in where.items()}, "se_features": SE_FEATURE_NAMES,
           "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
