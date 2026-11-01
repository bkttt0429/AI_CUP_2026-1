#!/usr/bin/env python
"""Folds 1-4 CV: do heads that read the labelled player's rank off the OPPONENT's moves add to fused_opp_diff? (issue #33, route C)

The opponent channel is the only side information about the label that the data offer (METHOD_NOTES 2.18): servers pair by
rank, and fused_opp / fused_opp_diff both gain from the L1b heads applied to the opponent's moves. Those heads were trained
to predict the mover's own rank. Route C trains heads on the opponent's positions with the labelled player's rank as the
target (l1b_build_cache.py --opponent, l1b_train.py, l1b_score.py --opponent), so each head k gives log p(opponent's move |
position, partner's rank k) and their per-game sums are a direct estimate of the partner's label. Here the C heads' 16
question features (aicup.qfeatures.build_matrix over the --copp sums) are appended to the 59 fused_opp_diff features:
fused_opp_diff_c (75); fused_opp_diff_c2 (78) also appends the C diff features (mean(c pmi - own pmi), mean(c pmi),
std(c pmi)); fused_c_diff (59) replaces the L1b opponent block and the diff features with the C ones. Protocol as
scripts/l1b_opp_check.py (pilot validation questions, sample_questions per player with RandomState(42 + v),
MLPModel(hidden=32), player bootstrap); fused_opp_diff must reproduce 0.6916. Pre-specified (conditions (1) CI lower
bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_c vs fused_opp_diff. Reported only:
fused_opp_diff_c2 vs fused_opp_diff, fused_c_diff vs fused_opp_diff, fused_opp_diff_c2 vs fused_opp_diff_c; and the
channel strength at the player level over the validation folds' games (Spearman of the label with own / opp / c pmi, and
of the label's regression residual on own pmi with (opp - own) and (c - own)). Fold 0 is refused. The --copp fold
directories must come from l1b_score.py --opponent with opponent-side heads (their meta_s*.json say so).

  python -I scripts/opp_head_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --copp results/l1b/scores3c_A_opp results/l1b/scores3c_B_opp --out results/l1b/opp_head_check.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.fused import question_diff_features  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS, RANK_TO_IDX  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import build_matrix, sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import error_analysis as ea  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("fused_opp_diff", "fused_opp_diff_c", "fused_opp_diff_c2", "fused_c_diff")
PRE = [("fused_opp_diff_c", "fused_opp_diff")]
REPORT = [("fused_opp_diff_c2", "fused_opp_diff"), ("fused_c_diff", "fused_opp_diff"), ("fused_opp_diff_c2", "fused_opp_diff_c")]


def features(hsl, l1b, opp, copp, questions):
    """The four feature sets of `questions` (see the module docstring): 59, 75, 78 and 59 columns."""
    X, y, n = oc.features(hsl, l1b, opp, questions)
    C, y2, _ = build_matrix(copp, questions, LEVELS)
    assert np.array_equal(y, y2)
    D, Dc = question_diff_features(l1b, opp, questions), question_diff_features(l1b, copp, questions)
    base = np.hstack([X["fused_opp"], D])
    return {"fused_opp_diff": base, "fused_opp_diff_c": np.hstack([base, C]), "fused_opp_diff_c2": np.hstack([base, C, Dc]),
            "fused_c_diff": np.hstack([X["fused"], C, Dc])}, y, n


def check_c_scores(fold_dir):
    """ValueError unless every meta_s*.json of the fold says the opponent's moves were scored with opponent-side heads;
    returns the number of metas read (0: none found, the caller warns)."""
    metas = sorted(Path(fold_dir).glob("meta_s*.json"))
    for p in metas:
        m = json.loads(p.read_text())
        if m.get("side") != "opponent":
            raise ValueError(f"{p}: side {m.get('side')!r}; --copp must be l1b_score.py --opponent output")
        if (m.get("heads") or {}).get("side") != "opponent":
            raise ValueError(f"{p}: the heads' side is {(m.get('heads') or {}).get('side')!r}, not \"opponent\" (route C heads, l1b_build_cache.py --opponent)")
    return len(metas)


def channel_strength(data, folds):
    """Player-level Spearmans over the validation folds' games: label vs the mean own / opp / c pmi (posterior-mean rank index of a
    game's ten sums, scripts/error_analysis.py's game_pmi), the label's regression residual on own pmi vs (opp - own) and (c - own),
    and opp vs c."""
    rows = []
    for f in folds:
        d = data[f]
        l1b = d["l1b"]
        keys = l1b["game_key"].to_numpy()
        own, opp, c = ea.game_pmi(l1b), ea.game_pmi(d["opp"]), ea.game_pmi(d["copp"])
        rows.append(pd.DataFrame({"player_id": l1b["player_id"].to_numpy(dtype=str), "y": l1b["rank"].map(RANK_TO_IDX).to_numpy(dtype=float),
                                  "own": own[keys].to_numpy(), "opp": opp[keys].to_numpy(), "c": c[keys].to_numpy()}))
    t = pd.concat(rows, ignore_index=True).groupby("player_id")[["y", "own", "opp", "c"]].mean()
    sp = lambda a, b: float(pd.Series(np.asarray(a, dtype=float)).corr(pd.Series(np.asarray(b, dtype=float)), method="spearman"))  # noqa: E731
    y, own, opp, c = (t[k].to_numpy() for k in ("y", "own", "opp", "c"))
    slope, icpt = np.polyfit(own, y, 1)
    resid = y - (slope * own + icpt)
    return {"players": int(len(t)), "spearman_label_own_pmi": sp(y, own), "spearman_label_opp_pmi": sp(y, opp), "spearman_label_c_pmi": sp(y, c),
            "spearman_label_resid_on_own_vs_opp_minus_own": sp(resid, opp - own), "spearman_label_resid_on_own_vs_c_minus_own": sp(resid, c - own),
            "spearman_opp_c": sp(opp, c)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories under the L1b heads (each fold in exactly one)")
    ap.add_argument("--copp", nargs="+", required=True, help="l1b_score.py --opponent directories under the route-C heads (each fold in exactly one)")
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
        where, where_o, where_c = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds), l1b_cv.locate_folds(a.copp, folds)
        folds_df = pd.read_csv(a.folds_csv, dtype={"player_id": str})
        data = {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            copp = read_scored_games(Path(where_c[f]) / f"fold{f}")
            for name, t in (("opponent", opp), ("route-C opponent", copp)):
                if set(t["game_key"]) != set(hsl["game_key"]):
                    raise ValueError(f"fold {f}: the {name} scores cover other games than {a.hsl}")
            if check_c_scores(Path(where_c[f]) / f"fold{f}") == 0:
                print(f"WARNING: fold {f}: {where_c[f]} has no meta_s*.json; cannot confirm the route-C heads scored the opponent", file=sys.stderr)
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "copp": copp, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll, dims = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("copp"), train_q)
        d = data[v]
        X_va, y_va, n_va = features(d["hsl"], d["l1b"], d["opp"], d["copp"], d["val"])
        sc = {}
        for s in SETS:
            dims[s] = int(X_tr[s].shape[1])
            res, sc[s] = evaluate(MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s]), y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS) + f"  ({time.time() - t0:.0f} s)", flush=True)
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
    strength = channel_strength(data, folds)
    for s, h in head.items():
        print(f"{s:18s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"  # noqa: E731
    print(f"channel strength ({strength['players']} players): " + " ".join(f"{k} {fmt(v)}" for k, v in strength.items() if k != "players"))
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp, "copp_dirs": a.copp,
           "scores": head, "comparisons": comps, "channel_strength": strength, "min_diff": l1b_cv.MIN_DIFF, "seconds": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
