#!/usr/bin/env python
"""Ordinal final layers on the fused_opp question features (strength-estimator comparison, option a): CORAL and the
pairwise Bradley-Terry score against the 10-class MLP, folds 1-4 CV with l1b_opp_check's protocol.

Protocol as scripts/l1b_opp_check.py: validation questions pilot.pilot_questions of fold v, training questions
qfeatures.sample_questions (--per-player per player, RandomState(42 + v)) of the other folds in ascending order,
features l1b_opp_check.features' --set (default fused_opp, 56), one deterministic fit per model and validation fold
(aicup.ordinal.MODELS: mlp = rankmodel.MLPModel(hidden=32), coral, bt), scores pooled over the four folds, player
bootstrap for the paired differences. The mlp row is l1b_opp_check's fused_opp fit and must reproduce its number
(opp_check.json: 0.6735). Pre-specified: coral vs mlp and bt vs mlp with the ladder's conditions (1) CI lower bound
> 0, (3) > 0 in >= len(folds) - 1 folds, (4) >= 0.005.

  python -I scripts/ordinal_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/l1b/ordinal_check.json
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
from aicup.ordinal import MODELS  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

BASELINE = "mlp"
PRE = [("coral", BASELINE), ("bt", BASELINE)]


def load_data(hsl_dir, l1b_dirs, opp_dirs, folds, folds_df, log=print):
    where, where_o = l1b_cv.locate_folds(l1b_dirs, folds), l1b_cv.locate_folds(opp_dirs, folds)
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(hsl_dir, where[f], f)
        opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
        if set(opp["game_key"]) != set(hsl["game_key"]):
            raise ValueError(f"fold {f}: the opponent scores cover other games than {hsl_dir}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players, "val": l1b_cv.pilot_questions(players)}
        log(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    return data, where, where_o


def run_cv(data, folds, per_player, feature_set, models, log=print):
    """Per validation fold one fit per model; returns the per-question scores table, per-fold metrics and NLLs."""
    rows, per_fold, nll = [], {m: [] for m in models}, {m: [] for m in models}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = oc.features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = oc.features(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], val)
        sc = {}
        for name in models:
            proba = MODELS[name]().fit(X_tr[feature_set], y_tr).predict_proba(X_va[feature_set])
            res, sc[name] = evaluate(proba, y_va, val)
            per_fold[name].append(res["balanced_mean_over_n"])
            nll[name].append(res["nll_balanced"])
        log(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{m} {per_fold[m][-1]:.4f}" for m in models), flush=True)
        rows.append(pd.DataFrame({"fold": v, "player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{m: sc[m] for m in models}}))
    return pd.concat(rows, ignore_index=True), per_fold, nll


def summarise(table, per_fold, nll, folds, models, n_boot):
    y, pid, n = table["y"].to_numpy(), table["player_id"].to_numpy(dtype=str), table["n"].to_numpy()
    head = {m: {"balanced_mean_over_n": balanced_mean_over_n(table[m].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[m])),
                "balanced_by_n": {str(s): balanced_mean_over_n(table[m].to_numpy(), y, n, sizes=(s,)) for s in SIZES},
                "per_class": {LEVELS[c]: float(table[m].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": [float(x) for x in per_fold[m]], "nll_per_fold": [float(x) for x in nll[m]]} for m in models}
    need = max(1, len(folds) - 1)
    comps = {}
    for x, z in PRE:
        if x not in models or z not in models:
            continue
        d = paired_diff(table[x].to_numpy(), table[z].to_numpy(), y, pid, n, n=n_boot)
        pf = [float(p - q) for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        comps[f"{x} vs {z}"] = {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}
    return head, comps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--set", default="fused_opp", choices=("hsl", "fused", "fused_opp"), help="feature set (l1b_opp_check.features)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    if BASELINE not in a.models:
        ap.error(f"--models must include the baseline {BASELINE}")
    try:
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data, where, where_o = load_data(a.hsl, a.l1b, a.opp, folds, folds_df)
        table, per_fold, nll = run_cv(data, folds, a.per_player, a.set, a.models)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    head, comps = summarise(table, per_fold, nll, folds, a.models, a.n_boot)
    for m, h in head.items():
        print(f"{m:6s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{s}: {h['balanced_by_n'][str(s)]:.4f}" for s in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]) + "  1D-3D " + " ".join(f"{h['per_class'].get(c, float('nan')):.3f}" for c in ("1D", "2D", "3D")))
    for k, c in comps.items():
        print(f"[pre-specified] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
              + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    out = {"folds": folds, "questions": int(len(table)), "per_player": a.per_player, "feature_set": a.set, "models": a.models,
           "l1b_dirs": {str(f): str(d) for f, d in where.items()}, "opp_dirs": {str(f): str(d) for f, d in where_o.items()}, "hsl_dir": a.hsl,
           "scores": head, "comparisons": comps, "min_diff": l1b_cv.MIN_DIFF, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
