#!/usr/bin/env python
"""Folds 1-4 CV: do five-fold cross-fit L1b models (80% of the players each) beat the A/B pair (40%)? And does ordinal
label smoothing of the MLP help? (issue #16)

Protocol as scripts/l1b_opp_check.py: validation pilot questions, training sample_questions (--per-player per player,
RandomState(42 + v)), the 56 fused_opp features of l1b_opp_check.features, one MLPModel(hidden=32) fit per feature set
and validation fold, player bootstrap. Feature sets: fused_opp (the A/B scores of --l1b/--opp; must reproduce
opp_check.json's 0.6735), fused_opp_x5 (the cross-fit scores of --xfit-l1b/--xfit-opp: each fold scored by the model that
left it out), fused_opp_ls and fused_opp_ls2 (A/B features, MLPModel(label_smoothing=0.1 / 0.2)), fused_opp_x5_ls (x5
features, label smoothing 0.1). Pre-specified (conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >=
0.005): fused_opp_x5 vs fused_opp and fused_opp_ls vs fused_opp. Reported only: fused_opp_ls2 vs fused_opp,
fused_opp_x5_ls vs fused_opp_x5, fused_opp_x5_ls vs fused_opp. Fold 0 is refused.

  python -I scripts/xfit_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --xfit-l1b results/l1b/scores5 --xfit-opp results/l1b/scores5_opp --out results/l1b/xfit_check.json
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

LS, LS2 = 0.1, 0.2
# set name -> (feature source: "base" = --l1b/--opp, "x5" = --xfit-l1b/--xfit-opp; label smoothing of the MLP)
SETS = {"fused_opp": ("base", 0.0), "fused_opp_x5": ("x5", 0.0), "fused_opp_ls": ("base", LS), "fused_opp_ls2": ("base", LS2),
        "fused_opp_x5_ls": ("x5", LS)}
PRE = [("fused_opp_x5", "fused_opp"), ("fused_opp_ls", "fused_opp")]
REPORT = [("fused_opp_ls2", "fused_opp"), ("fused_opp_x5_ls", "fused_opp_x5"), ("fused_opp_x5_ls", "fused_opp")]


def fused_opp_features(hsl, l1b, opp, questions):
    """(X [Q, 56], y, n_games): l1b_opp_check.features' fused_opp set."""
    sets, y, n = oc.features(hsl, l1b, opp, questions)
    return sets["fused_opp"], y, n


def load_source(hsl_dir, l1b_dirs, opp_dirs, folds, what):
    """{fold: {"l1b": table, "opp": table}} of one scoring source; the tables must cover the Human SL games."""
    where, where_o = l1b_cv.locate_folds(l1b_dirs, folds), l1b_cv.locate_folds(opp_dirs, folds)
    out = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, where[f], f)
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
        if set(opp["game_key"]) != set(hsl["game_key"]):
            raise ValueError(f"fold {f}: the {what} opponent scores in {where_o[f]} cover other games than {hsl_dir}")
        out[f] = {"l1b": l1b, "opp": opp, "l1b_dir": str(where[f]), "opp_dir": str(where_o[f])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="the A/B l1b_score.py directories (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="the A/B l1b_score.py --opponent directories")
    ap.add_argument("--xfit-l1b", nargs="+", required=True, help="the cross-fit l1b_score.py directories (fold k scored by the model that left it out)")
    ap.add_argument("--xfit-opp", nargs="+", required=True, help="the cross-fit --opponent directories")
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
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        src = {"base": load_source(a.hsl, a.l1b, a.opp, folds, "A/B"), "x5": load_source(a.hsl, a.xfit_l1b, a.xfit_opp, folds, "cross-fit")}
        data = {}
        for f in folds:
            hsl = read_scored_games(Path(a.hsl) / f"fold{f}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions; "
                  f"A/B from {src['base'][f]['l1b_dir']}, cross-fit from {src['x5'][f]['l1b_dir']}", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll = [], {s: [] for s in SETS}, {s: [] for s in SETS}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        val = data[v]["val"]
        cat_hsl = pd.concat([data[f]["hsl"] for f in folds if f != v], ignore_index=True)
        X_tr, X_va, y_tr, y_va, n_va = {}, {}, None, None, None
        for source in ("base", "x5"):
            cat = lambda key: pd.concat([src[source][f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
            X_tr[source], y, _ = fused_opp_features(cat_hsl, cat("l1b"), cat("opp"), train_q)
            X_va[source], yv, nv = fused_opp_features(data[v]["hsl"], src[source][v]["l1b"], src[source][v]["opp"], val)
            assert y_tr is None or (np.array_equal(y, y_tr) and np.array_equal(yv, y_va))
            y_tr, y_va, n_va = y, yv, nv
        sc = {}
        for s, (source, eps) in SETS.items():
            res, sc[s] = evaluate(MLPModel(hidden=32, label_smoothing=eps).fit(X_tr[source], y_tr).predict_proba(X_va[source]), y_va, val)
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS), flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s], "source": SETS[s][0], "label_smoothing": SETS[s][1]} for s in SETS}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT}}
    for s, h in head.items():
        print(f"{s:16s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]) + "  1D-3D " + " ".join(f"{h['per_class'].get(c, float('nan')):.3f}" for c in ("1D", "2D", "3D")))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl,
           "dirs": {source: {str(f): {"l1b": src[source][f]["l1b_dir"], "opp": src[source][f]["opp_dir"]} for f in folds} for source in src},
           "label_smoothing": {"ls": LS, "ls2": LS2}, "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
