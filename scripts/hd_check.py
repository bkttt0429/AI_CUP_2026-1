#!/usr/bin/env python
"""Folds 1-4 CV: do the handicap (HD) features of dataset v1.1 add to fused_opp_diff? (issue #42, proposal (a))

The server's handicap is set from the two players' ranks (HD[1] 讓先 about one rank apart, HD[n] stones about n apart, HD[0]
equal), so it is label-side information like the opponent channel, but direct. aicup.handicap turns each question's games into
five means (hd_even, hd_sente, hd_stones, hd_gap_up, hd_gap_down; the signed gap seen from the labelled player uses the actual
colour, W = upper hand) that are appended to the 59 fused_opp_diff features (fused_opp_diff_hd, 64) and to the 56 fused_opp
features (fused_opp_hd, 61). Protocol as scripts/hist_check.py (pilot questions, 32 training questions per player, MLP h32,
player bootstrap): fused_opp must reproduce 0.6735 and fused_opp_diff 0.6916. Pre-specified (conditions (1) CI lower bound > 0,
(3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_hd vs fused_opp_diff. Reported only: fused_opp_hd vs fused_opp,
fused_opp_diff_hd vs fused_opp_hd, fused_opp_diff vs fused_opp. Diagnostics: the HD coverage of the scored games; every set's
score on the questions without any HD game / with 讓先 only / with a stones game; the Spearman correlation of fused_opp_diff's
residual (predicted - true rank index) with the question's signed gap. The HD table comes from scripts/hd_table.py (CPU); the
scores are the existing ones (no re-scoring): HD is read from the CSVs only.

  python -I scripts/hd_check.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --hd-table results/data/hd_table.csv --out results/seq/hd_check.json
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
from aicup.fused import question_diff_features  # noqa: E402
from aicup.handicap import HD_FEATURE_NAMES, hd_summary, question_hd_features, read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("fused_opp", "fused_opp_diff", "fused_opp_hd", "fused_opp_diff_hd")
PRE = [("fused_opp_diff_hd", "fused_opp_diff")]
REPORT = [("fused_opp_hd", "fused_opp"), ("fused_opp_diff_hd", "fused_opp_hd"), ("fused_opp_diff", "fused_opp")]
BUCKETS = ("no_hd", "sente_only", "stones")
COL_SENTE, COL_STONES, COL_UP, COL_DOWN = (HD_FEATURE_NAMES.index(n) for n in ("hd_sente", "hd_stones", "hd_gap_up", "hd_gap_down"))


def features(hsl, l1b, opp, hd, questions):
    """The four feature sets of `questions`: fused_opp (56), fused_opp_diff (59), fused_opp + HD (61), fused_opp_diff + HD (64); also the HD block."""
    X, y, n = oc.features(hsl, l1b, opp, questions)
    diff = np.hstack([X["fused_opp"], question_diff_features(l1b, opp, questions)])
    H = question_hd_features(hd, questions)
    return {"fused_opp": X["fused_opp"], "fused_opp_diff": diff, "fused_opp_hd": np.hstack([X["fused_opp"], H]), "fused_opp_diff_hd": np.hstack([diff, H])}, y, n, H


def bucket_of(H):
    """no_hd: no game with HD >= 1; stones: at least one HD >= 2 game; sente_only: HD = 1 games but no stones game."""
    stones, sente = H[:, COL_STONES] > 0, H[:, COL_SENTE] > 0
    return np.where(stones, "stones", np.where(sente, "sente_only", "no_hd"))


def _spearman(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True, help="l1b_score.py directories of the target's moves (each fold in exactly one)")
    ap.add_argument("--opp", nargs="+", required=True, help="l1b_score.py --opponent directories (each fold in exactly one)")
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--hd-table", required=True, help="scripts/hd_table.py's CSV")
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
        hd_all = read_hd_table(a.hd_table)
        data, coverage = {}, {}
        for f in folds:
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
            missing = set(hsl["game_key"]) - set(hd["game_key"])
            if missing:
                raise ValueError(f"fold {f}: {len(missing)} scored games are not in {a.hd_table}, e.g. {sorted(missing)[:3]}")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": l1b_cv.pilot_questions(players)}
            coverage[str(f)] = hd_summary(hd)
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions; HD>=1 {coverage[str(f)]['share_hd1']:.3f}, "
                  f"HD>=2 {coverage[str(f)]['share_hd2']:.3f}, absent {coverage[str(f)]['hd'].get('absent', 0)}", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll, dims, oof = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}, []
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, y_va, n_va, H_va = features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        sc = {}
        for s in SETS:
            dims[s] = int(X_tr[s].shape[1])
            P = MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s])
            res, sc[s] = evaluate(P, y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
            if s == "fused_opp_diff":
                oof.append(pd.DataFrame({"fold": v, "n_games": n_va, "y": y_va, "pred": [expected_score_decision(p) for p in P],
                                         "q_gap": H_va[:, COL_UP] - H_va[:, COL_DOWN]}))
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{s} {per_fold[s][-1]:.4f}" for s in SETS) + f"  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": d["val"]["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, "bucket": bucket_of(H_va), **{s: sc[s] for s in SETS}}))
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
    buckets = {b: {"questions": int((t["bucket"] == b).sum()), **{s: float(t.loc[t["bucket"] == b, s].mean()) if (t["bucket"] == b).any() else None for s in SETS}}
               for b in BUCKETS}
    o = pd.concat(oof, ignore_index=True)
    resid = o["pred"].to_numpy(dtype=float) - o["y"].to_numpy(dtype=float)
    residual = {"spearman_residual_vs_gap": _spearman(resid, o["q_gap"].to_numpy()),
                "by_n": {str(m): _spearman(resid[o["n_games"] == m], o.loc[o["n_games"] == m, "q_gap"].to_numpy()) for m in SIZES},
                "mean_residual_by_gap_sign": {k: float(resid[m].mean()) if m.any() else None
                                              for k, m in (("gap<0", o["q_gap"] < 0), ("gap=0", o["q_gap"] == 0), ("gap>0", o["q_gap"] > 0))}}
    for s, h in head.items():
        print(f"{s:18s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    for b, r in buckets.items():
        print(f"bucket {b:11s} {r['questions']:5d} questions  " + "  ".join(f"{s} {r[s]:.4f}" if r[s] is not None else f"{s} -" for s in SETS))
    print(f"fused_opp_diff residual vs signed gap: Spearman {residual['spearman_residual_vs_gap']:.3f}; by n "
          + " ".join(f"{m}: {residual['by_n'][str(m)]:.3f}" for m in SIZES) + "; mean residual " + " ".join(f"{k} {v:+.3f}" if v is not None else f"{k} -" for k, v in residual["mean_residual_by_gap_sign"].items()))
    out = {"folds": folds, "per_player": a.per_player, "n_boot": a.n_boot, "min_diff": l1b_cv.MIN_DIFF, "hd_table": a.hd_table, "hd_feature_names": HD_FEATURE_NAMES,
           "coverage": coverage, "scores": head, "comparisons": comps, "buckets": buckets, "residual": residual, "wall_s": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
