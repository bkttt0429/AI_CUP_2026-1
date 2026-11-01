#!/usr/bin/env python
"""Folds 1-4 CV: does the per-move score's whole distribution add to fused_opp_diff? (issue #29, the "distribution range" channel)

The user's proposal: score every move, then tell the ranks apart by the distribution of those per-move scores. The per-move
score we already have is pmi_t, the posterior-mean rank index of the move's ten L1b log p (aicup.seqshape.move_pmi); the
features of fused_opp_diff see its distribution only through sums (the mean). Here each game's histogram of pmi_t over ten
equal bins of [0, 9] (aicup.seqshape.game_pmi_hist, fractions of the game's moves) is averaged over the question's games
(aicup.seqshape.question_shape) and appended to the 59 fused_opp_diff features: fused_opp_diff_hist (+10, the target's
moves) and fused_opp_diff_hist2 (+20, the target's and the opponent's). Protocol as scripts/l1b_opp_check.py (pilot
validation questions, sample_questions per player with RandomState(42 + v), MLPModel(hidden=32), player bootstrap);
fused_opp must reproduce 0.6735 and fused_opp_diff 0.6916 (scripts/error_analysis.py's mlp_diff). Pre-specified
(conditions (1) CI lower bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): fused_opp_diff_hist vs fused_opp_diff.
Reported only: fused_opp_diff_hist2 vs fused_opp_diff, fused_opp_diff_hist2 vs fused_opp_diff_hist, fused_opp_diff vs
fused_opp (the anchor, +0.0181 in error_analysis.json).

Also reported, no decision rides on them: (i) the per-rank pooled histogram of pmi_t over the validation games (the
"distribution range" of each rank) with the overlap coefficient sum_i min(h_r, h_r+1) of adjacent ranks; (ii) the OOF
residual diagnostics of fused_opp and fused_opp_diff (scripts/error_analysis.py's player_table / residual_analysis /
opponent_channel): how much of the per-player mean residual still follows the opponent channel once the diff features
are in the model (the pre-check of the "opponent head" route, METHOD_NOTES 2.20). That per-player covariate is the mean
over ALL of the player's games, which the model never sees; so the same Spearmans are also given against the
QUESTION-level opp / own / (opp - own) pmi means the model did see (the diff columns of fused_opp_diff), per question,
by question size and per player (mean over the player's questions). A correlation that survives at the question level
is a missing model component; one that only exists against the all-games mean, and shrinks from n = 5 to n = 20, is
measurement noise of the question's estimate and is not recoverable from the question. Fold 0 is refused.

  python -I scripts/hist_check.py --l1b results/l1b/scores3 results/l1b/scores3_B \\
      --opp results/l1b/scores3_opp results/l1b/scores3_B_opp --out results/seq/hist_check.json
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
from aicup.fused import DIFF_FEATURE_NAMES, FUSED_OPP_FEATURE_NAMES, question_diff_features  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.seqshape import HIST, HIST_EDGES, fold_hist_table, question_shape  # noqa: E402
import error_analysis as ea  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

SETS = ("fused_opp", "fused_opp_diff", "fused_opp_diff_hist", "fused_opp_diff_hist2")
PRE = [("fused_opp_diff_hist", "fused_opp_diff")]
REPORT = [("fused_opp_diff_hist2", "fused_opp_diff"), ("fused_opp_diff_hist2", "fused_opp_diff_hist"), ("fused_opp_diff", "fused_opp")]
RESIDUAL_SETS = ("fused_opp", "fused_opp_diff")
RESIDUAL_COVARIATES = ("own_pmi", "opp_pmi", "opp_minus_own")
Q_COVARIATES = ("q_own_pmi", "q_opp_pmi", "q_opp_minus_own")  # the question-level pmi means the model saw (diff columns of fused_opp_diff)
COL_DIFF, COL_OPP = len(FUSED_OPP_FEATURE_NAMES) + DIFF_FEATURE_NAMES.index("opp_minus_own_pmi_mean"), len(FUSED_OPP_FEATURE_NAMES) + DIFF_FEATURE_NAMES.index("opp_pmi_mean")


def features(hsl, l1b, opp, hist_t, hist_o, questions):
    """The four feature sets of `questions`: fused_opp (56), fused_opp_diff (59), + the target's histogram means (69), + the opponent's (79)."""
    X, y, n = oc.features(hsl, l1b, opp, questions)
    diff = np.hstack([X["fused_opp"], question_diff_features(l1b, opp, questions)])
    H_t, H_o = question_shape(hist_t, questions, cols=HIST), question_shape(hist_o, questions, cols=HIST)
    return {"fused_opp": X["fused_opp"], "fused_opp_diff": diff, "fused_opp_diff_hist": np.hstack([diff, H_t]),
            "fused_opp_diff_hist2": np.hstack([diff, H_t, H_o])}, y, n


def rank_distribution(tables):
    """Per rank: the move-weighted pooled histogram of pmi_t over the games of `tables` (fold_hist_table frames), the number of
    games and moves; and the overlap coefficient sum_i min(h_r, h_r+1) of each adjacent pair of ranks (1 identical, 0 disjoint)."""
    t = pd.concat(tables, ignore_index=True)
    H = t[HIST].to_numpy(dtype=float) * t["n_moves"].to_numpy(dtype=float)[:, None]
    by_rank, pooled = {}, {}
    for lv in LEVELS:
        m = (t["rank"].to_numpy() == lv)
        moves = float(t["n_moves"].to_numpy(dtype=float)[m].sum())
        if m.any() and moves > 0:
            pooled[lv] = H[m].sum(axis=0) / moves
            by_rank[lv] = {"games": int(m.sum()), "moves": int(moves), "hist": pooled[lv].tolist()}
    overlap = {f"{a}|{b}": float(np.minimum(pooled[a], pooled[b]).sum()) for a, b in zip(LEVELS[:-1], LEVELS[1:]) if a in pooled and b in pooled}
    return {"bin_edges": HIST_EDGES.tolist(), "by_rank": by_rank, "adjacent_overlap": overlap}


def _sp(a, b):
    return float(pd.Series(np.asarray(a, dtype=float)).corr(pd.Series(np.asarray(b, dtype=float)), method="spearman"))


def question_level(oof):
    """Spearmans of the signed residual with the QUESTION-level pmi means the model saw (Q_COVARIATES): per question (all sizes and
    by question size), and per player (the player's mean residual against the mean of the covariate over the player's questions)."""
    e = oof["pred"].to_numpy() - oof["y"].to_numpy()
    n = oof["n_games"].to_numpy()
    out = {"spearman_e": {c: _sp(e, oof[c]) for c in Q_COVARIATES},
           "spearman_e_by_n": {str(m): {c: _sp(e[n == m], oof[c].to_numpy()[n == m]) for c in Q_COVARIATES} for m in SIZES if (n == m).sum() > 2}}
    g = oof.assign(e=e).groupby("player_id")
    mean_e = g["e"].mean()
    out["spearman_mean_e"] = {c: _sp(mean_e, g[c].mean().reindex(mean_e.index)) for c in Q_COVARIATES}
    return out


def residual_diagnostics(oof, data, folds_df):
    """error_analysis's per-player residual table of one model's OOF decisions: the opponent-channel Spearmans, the Spearman of the
    per-player mean signed residual with own / opponent pmi and their difference (means over ALL of the player's games), the
    between-player share of the residual variance; and question_level, the same against what the model saw."""
    players = ea.player_table(oof, data, folds_df)
    r = ea.residual_analysis(oof, players)
    return {"opponent_channel": ea.opponent_channel(players), "spearman_mean_e": {c: r["spearman_with_mean_e"][c] for c in RESIDUAL_COVARIATES},
            "spearman_abs_mean_e": {c: r["spearman_with_abs_mean_e"][c] for c in RESIDUAL_COVARIATES}, "between_player_share": r["between_player_share"],
            "exact": r["exact"], "within1": r["within1"], "question_level": question_level(oof)}


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
            t1 = time.time()
            hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
            opp = read_scored_games(Path(where_o[f]) / f"fold{f}")
            if set(opp["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the opponent scores cover other games than {a.hsl}")
            hist_t, hist_o = fold_hist_table(Path(where[f]) / f"fold{f}"), fold_hist_table(Path(where_o[f]) / f"fold{f}")
            if set(hist_t["game_key"]) != set(hsl["game_key"]) or set(hist_o["game_key"]) != set(hsl["game_key"]):
                raise ValueError(f"fold {f}: the per-move arrays cover other games than the sums")
            players = l1b_cv.players_from_scores(hsl, folds_df)
            data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hist_t": hist_t, "hist_o": hist_o, "players": players, "val": l1b_cv.pilot_questions(players)}
            print(f"fold {f}: {len(players)} players, {len(hsl)} games, {len(data[f]['val'])} validation questions, histogram tables in {time.time() - t1:.0f} s", flush=True)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return 2
    rows, per_fold, nll, dims, oof = [], {s: [] for s in SETS}, {s: [] for s in SETS}, {}, {s: [] for s in RESIDUAL_SETS}
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = features(cat("hsl"), cat("l1b"), cat("opp"), cat("hist_t"), cat("hist_o"), train_q)
        d = data[v]
        X_va, y_va, n_va = features(d["hsl"], d["l1b"], d["opp"], d["hist_t"], d["hist_o"], d["val"])
        sc = {}
        for s in SETS:
            dims[s] = int(X_tr[s].shape[1])
            P = MLPModel(hidden=32).fit(X_tr[s], y_tr).predict_proba(X_va[s])
            res, sc[s] = evaluate(P, y_va, d["val"])
            per_fold[s].append(res["balanced_mean_over_n"])
            nll[s].append(res["nll_balanced"])
            if s in oof:
                oof[s].append(pd.DataFrame({"question_id": d["val"]["question_id"].to_numpy(dtype=str), "rank": d["val"]["rank"].to_numpy(dtype=str),
                                            "player_id": d["val"]["player_id"].to_numpy(dtype=str), "n_games": n_va, "fold": v, "y": y_va,
                                            "pred": [expected_score_decision(p) for p in P], "q_opp_minus_own": X_va["fused_opp_diff"][:, COL_DIFF],
                                            "q_opp_pmi": X_va["fused_opp_diff"][:, COL_OPP],
                                            "q_own_pmi": X_va["fused_opp_diff"][:, COL_OPP] - X_va["fused_opp_diff"][:, COL_DIFF]}))
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
    dist = rank_distribution([data[f]["hist_t"] for f in folds])
    resid = {s: residual_diagnostics(pd.concat(oof[s], ignore_index=True), data, folds_df) for s in RESIDUAL_SETS}
    for s, h in head.items():
        print(f"{s:21s} ({h['n_features']:2d} features) {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n "
              + " ".join(f"{m}: {h['balanced_by_n'][str(m)]:.4f}" for m in SIZES) + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    print("pooled pmi histogram by rank (fractions of moves, bins " + " ".join(f"{e:.1f}" for e in HIST_EDGES) + "):")
    for lv, r in dist["by_rank"].items():
        print(f"  {lv:3s} {r['games']:6d} games {r['moves']:9d} moves  " + " ".join(f"{h:.3f}" for h in r["hist"]))
    print("adjacent-rank overlap: " + " ".join(f"{k} {v:.3f}" for k, v in dist["adjacent_overlap"].items()))
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"  # noqa: E731
    for s, r in resid.items():
        print(f"residuals of {s}: exact {r['exact']:.3f} within1 {r['within1']:.3f} between-player share {fmt(r['between_player_share'])}; "
              "spearman mean e vs " + " ".join(f"{c} {fmt(v)}" for c, v in r["spearman_mean_e"].items())
              + f"; opponent channel oof_mean_e_vs_opp_minus_own {fmt(r['opponent_channel']['spearman_oof_mean_e_vs_opp_minus_own'])}")
        q = r["question_level"]
        print("  question-level (what the model saw): e vs " + " ".join(f"{c} {fmt(v)}" for c, v in q["spearman_e"].items())
              + "; by n " + " ".join(f"{m}: {fmt(v['q_opp_minus_own'])}" for m, v in q["spearman_e_by_n"].items())
              + "; per player mean e vs " + " ".join(f"{c} {fmt(v)}" for c, v in q["spearman_mean_e"].items()))
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp,
           "hist_features": list(HIST), "bin_edges": HIST_EDGES.tolist(), "scores": head, "comparisons": comps, "rank_distribution": dist,
           "residual_diagnostics": resid, "min_diff": l1b_cv.MIN_DIFF, "seconds": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
