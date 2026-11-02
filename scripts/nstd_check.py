#!/usr/bin/env python
"""Folds 1-4 CV: does standardising the fused_opp_diff features per question size n help? (issue #53)

Control: the current fused_opp_diff MLP (59 features, global standardisation inside MLPModel). Candidate fused_opp_diff_nstd: the
same features, MLP h32, training questions, seed and training length; only the standardisation is estimated per group of n
(5 ... 20, training players only; aicup.nstd.NStdMLP), the log n columns (log_games of the Human SL, L1b and opponent blocks)
kept on the global statistics, zero-variance columns on scale 1 as rankmodel does, and an n outside 5 ... 20 on the nearest
group. Protocol as scripts/hist_check.py (pilot questions, 32 training questions per player, player bootstrap). Reported first:
every group's training questions and per-column standard deviations (results JSON "groups", per validation split). Pre-specified
(conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_nstd vs fused_opp_diff; the gains
for n = 5 / 10 / 20 separately are reported with their own bootstrap intervals. Not in scope: re-weighting the training n mix,
one model per n, shrinkage, band-utility losses, temperature.

  python -I scripts/nstd_check.py --l1b results/l1b/scores3_hd results/l1b/scores3_B_hd --opp results/l1b/scores3_opp_hd results/l1b/scores3_B_opp_hd \\
      --hsl results/scores_hd --out results/seq/nstd_check.json
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
from aicup.fused import FUSED_OPP_DIFF_FEATURE_NAMES, question_diff_features  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.nstd import N_MAX, N_MIN, NStdMLP  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("fused_opp_diff", "fused_opp_diff_nstd")
PRE = [("fused_opp_diff_nstd", "fused_opp_diff")]
LOG_GAMES_COLS = [i for i, name in enumerate(FUSED_OPP_DIFF_FEATURE_NAMES) if name.endswith("log_games")]


def features(hsl, l1b, opp, questions):
    """(X [Q, 59], y, n) of the questions: the fused_opp_diff features."""
    X, y, n = oc.features(hsl, l1b, opp, questions)
    return np.hstack([X["fused_opp"], question_diff_features(l1b, opp, questions)]), y, n


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
    t0 = time.time()
    try:
        where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll, groups = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, n_tr = features(cat("hsl"), cat("l1b"), cat("opp"), train_q)
        d = data[v]
        X_va, y_va, n_va = features(d["hsl"], d["l1b"], d["opp"], d["val"])
        control = MLPModel(hidden=32).fit(X_tr, y_tr)
        nstd = NStdMLP(keep_global=LOG_GAMES_COLS, hidden=32).fit(X_tr, y_tr, n_tr)
        groups[str(v)] = nstd.group_table(FUSED_OPP_DIFF_FEATURE_NAMES)
        sc = {}
        for s, P in (("fused_opp_diff", control.predict_proba(X_va)), ("fused_opp_diff_nstd", nstd.predict_proba(X_va, n_va))):
            res, sc[s] = evaluate(P, y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions, n groups " + " ".join(f"{k}:{g['questions']}" for k, g in groups[str(v)].items()) + "): "
              + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS) + f"  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{s: sc[s] for s in SETS}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {s: {"balanced_mean_over_n": balanced_mean_over_n(t[s].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[s])), "n_features": int(X_tr.shape[1]),
                "balanced_by_n": {str(m): balanced_mean_over_n(t[s].to_numpy(), y, n, sizes=(m,)) for m in SIZES},
                "per_class": {LEVELS[c]: float(t[s].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()},
                "per_fold": per_fold[s]} for s in SETS}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        dd = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(dd["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(dd["diff"] >= l1b_cv.MIN_DIFF)}
        by_n = {}
        for m in SIZES:
            sel = n == m
            if sel.any():
                by_n[str(m)] = paired_diff(t[x].to_numpy()[sel], t[z].to_numpy()[sel], y[sel], pid[sel], n[sel], n=a.n_boot)
        return {**dd, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values()), "by_n": by_n}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE}}
    for s, h in head.items():
        print(f"{s:20s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for k, c in comps["pre_specified"].items():
        print(f"[pre_specified] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
              + "  by n " + " ".join(f"{m}: {c['by_n'][m]['diff']:+.4f} [{c['by_n'][m]['ci95'][0]:+.4f}, {c['by_n'][m]['ci95'][1]:+.4f}]" for m in c["by_n"])
              + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    g1 = groups[str(folds[0])]
    print(f"split {folds[0]} training questions per n: " + " ".join(f"{k}:{g['questions']}" for k, g in g1.items())
          + "; median column std n=5 / 10 / 20: " + " / ".join(f"{np.median(list(g1[str(m)]['std'].values())):.3f}" for m in SIZES if str(m) in g1))
    out = {"folds": folds, "per_player": a.per_player, "n_boot": a.n_boot, "min_diff": l1b_cv.MIN_DIFF, "n_groups": [N_MIN, N_MAX],
           "log_games_cols": LOG_GAMES_COLS, "feature_names": list(FUSED_OPP_DIFF_FEATURE_NAMES), "groups": groups, "scores": head, "comparisons": comps,
           "wall_s": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
