#!/usr/bin/env python
"""Why are questions wrong? Folds 1-4 OOF residual analysis of fused_opp plus two pre-specified model-side checks (issue #23).

Protocol as scripts/l1b_opp_check.py (pilot validation questions, sample_questions per player with RandomState(42 + v), player
bootstrap). Per validation fold four models are fitted on the other folds' questions: mlp (fused_opp 56, MLPModel(hidden=32);
must reproduce 0.6735; its OOF probabilities go to --proba-out), mlp_diff (fused_opp + 3 opponent-difference features: per
question the mean over its games of opponent pmi - own pmi, the mean and the std of the opponent pmi, pmi being the
posterior-mean rank index of a game's ten L1b sums; the server pairs players by their labels, so the opponents are the one
indirect look at the label side), gbm (fused_opp, xgboost multi:softprob with class-balanced weights: is there non-linear
structure the MLP misses?) and mlp_h128 (fused_opp, MLPModel(hidden=128): capacity). Pre-specified (conditions (1) CI lower
bound > 0, (3) > 0 in >= 3 of 4 folds, (4) >= 0.005): mlp_diff vs mlp, gbm vs mlp. Reported only: mlp_h128 vs mlp.

Residual analysis on mlp's OOF predictions (report only): (a) the signed error (expected-score decision minus the truth)
split into between- and within-player variance; players all right / all wrong against the independence expectation; the
all-wrong rate by training-game-count quintile and by rank; Spearman correlations of each player's mean residual (signed and
absolute) with the player's number of training games, share of Black games, mean game length, Human SL curve sharpness
(mean per-game softmax entropy over the 18 profiles, std of the per-game argmax) and the opponent-difference means; the
covariate means of all-wrong, all-right and other players. (b) the opponent channel: Spearman of the label with the mean own
pmi and with the mean opponent pmi, of the label's regression residual on own pmi with (opponent - own), and of the OOF
residual with (opponent - own). Fold 0 is refused.

  python -I scripts/error_analysis.py --l1b results/l1b/scores3 results/l1b/scores3_B --opp results/l1b/scores3_opp results/l1b/scores3_B_opp \\
      --out results/l1b/error_analysis.json --proba-out results/l1b/oof_fused_opp_proba.csv
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
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel, class_weights  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

MODELS = ("mlp", "mlp_diff", "gbm", "mlp_h128")
PRE = [("mlp_diff", "mlp"), ("gbm", "mlp")]
REPORT = [("mlp_h128", "mlp")]
DIFF_NAMES = ["opp_minus_own_pmi_mean", "opp_pmi_mean", "opp_pmi_std"]
COVARIATES = ["log_train_games", "frac_black", "mean_moves", "hsl_entropy", "hsl_argmax_std", "own_pmi", "opp_pmi", "opp_minus_own"]


def game_pmi(table, profiles=LEVELS):
    """Posterior-mean profile index of every game's summed log p, indexed by game_key."""
    return pd.Series(posterior_mean_index(table[sum_cols(profiles)].to_numpy(dtype=float)), index=table["game_key"].to_numpy())


def game_entropy(table, profiles=PROFILES):
    """Entropy (nats) of softmax over the profiles' summed log p, per game, indexed by game_key."""
    L = table[sum_cols(profiles)].to_numpy(dtype=float)
    p = np.exp(L - L.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return pd.Series(-(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1), index=table["game_key"].to_numpy())


def diff_features(pmi_t, pmi_o, questions):
    """[Q, 3]: mean(opp - own), mean(opp), std(opp) of the games' pmi."""
    out = np.zeros((len(questions), 3))
    for q, keys in enumerate(questions["game_keys"]):
        o, t = pmi_o[keys].to_numpy(dtype=float), pmi_t[keys].to_numpy(dtype=float)
        out[q] = [(o - t).mean(), o.mean(), o.std()]
    return out


def feature_sets(hsl, l1b, opp, questions):
    X, y, n = oc.features(hsl, l1b, opp, questions)
    D = diff_features(game_pmi(l1b), game_pmi(opp), questions)
    return {"fused_opp": X["fused_opp"], "fused_opp_diff": np.hstack([X["fused_opp"], D])}, y, n


def fit_gbm(X, y, seed=0, n_jobs=1):
    import xgboost as xgb
    m = xgb.XGBClassifier(objective="multi:softprob", n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", random_state=seed, n_jobs=n_jobs, verbosity=0)
    m.fit(X, y, sample_weight=class_weights(y))
    return m


def player_table(oof, data, folds_df):
    """Per player: residual summaries and covariates (training games, colour, length, Human SL sharpness, pmi means)."""
    oof = oof.copy()
    oof["e"] = oof["pred"] - oof["y"]
    oof["correct"] = oof["pred"] == oof["y"]
    oof["wrong"] = ~oof["correct"]
    acc = oof["correct"].mean()
    g = oof.groupby("player_id")
    t = pd.DataFrame({"y": g["y"].first(), "rank": g["rank"].first(), "n_questions": g.size(), "mean_e": g["e"].mean(), "all_right": g["correct"].all(), "all_wrong": g["wrong"].all(),
                      "expect_all_right": g.size().map(lambda k: acc ** k), "expect_all_wrong": g.size().map(lambda k: (1 - acc) ** k)})
    t["abs_mean_e"] = t["mean_e"].abs()
    n_train = folds_df.set_index("player_id")["n_games"].astype(float)
    t["log_train_games"] = np.log(n_train.reindex(t.index).to_numpy())
    per_game = []
    for f, d in data.items():
        l1b, hsl, opp = d["l1b"], d["hsl"], d["opp"]
        ent = game_entropy(hsl)
        hsl_i = hsl.set_index("game_key")
        argmax = pd.Series(hsl_i[sum_cols(PROFILES)].to_numpy(dtype=float).argmax(axis=1), index=hsl_i.index)
        own, oppm = game_pmi(l1b), game_pmi(opp)
        per_game.append(pd.DataFrame({"player_id": l1b["player_id"].to_numpy(), "black": (l1b["color"].to_numpy() == "B").astype(float), "n_moves": l1b["n_moves"].to_numpy(dtype=float),
                                      "entropy": ent[l1b["game_key"]].to_numpy(), "argmax": argmax[l1b["game_key"]].to_numpy(), "own": own[l1b["game_key"]].to_numpy(), "opp": oppm[l1b["game_key"]].to_numpy()}))
    pg = pd.concat(per_game, ignore_index=True).groupby("player_id")
    t["frac_black"] = pg["black"].mean().reindex(t.index)
    t["mean_moves"] = pg["n_moves"].mean().reindex(t.index)
    t["hsl_entropy"] = pg["entropy"].mean().reindex(t.index)
    t["hsl_argmax_std"] = pg["argmax"].std(ddof=0).reindex(t.index)
    t["own_pmi"] = pg["own"].mean().reindex(t.index)
    t["opp_pmi"] = pg["opp"].mean().reindex(t.index)
    t["opp_minus_own"] = t["opp_pmi"] - t["own_pmi"]
    return t.reset_index()


def residual_analysis(oof, players):
    e = oof["pred"].to_numpy() - oof["y"].to_numpy()
    m = oof.groupby("player_id")["pred"].transform("mean").to_numpy() - oof.groupby("player_id")["y"].transform("mean").to_numpy()
    ss_total = float(((e - e.mean()) ** 2).sum())
    ss_within = float(((e - m) ** 2).sum())
    out = {"questions": int(len(oof)), "players": int(len(players)), "exact": float((e == 0).mean()), "within1": float((np.abs(e) <= 1).mean()),
           "between_player_share": (ss_total - ss_within) / ss_total if ss_total > 0 else None,
           "all_right": {"observed": float(players["all_right"].mean()), "independent": float(players["expect_all_right"].mean())},
           "all_wrong": {"observed": float(players["all_wrong"].mean()), "independent": float(players["expect_all_wrong"].mean())}}
    q = pd.qcut(players["log_train_games"], 5, labels=False, duplicates="drop")
    if q.isna().all():  # every player has the same number of training games (synthetic data): one bin
        q = pd.Series(0, index=players.index)
    out["all_wrong_by_train_games_quintile"] = {str(int(k)): {"players": int((q == k).sum()), "all_wrong": float(players["all_wrong"][q == k].mean()), "all_right": float(players["all_right"][q == k].mean()),
                                                              "median_train_games": float(np.exp(players["log_train_games"][q == k].median()))} for k in sorted(q.dropna().unique())}
    out["by_rank"] = {LEVELS[k]: {"players": int((players["y"] == k).sum()), "all_wrong": float(players["all_wrong"][players["y"] == k].mean()), "all_right": float(players["all_right"][players["y"] == k].mean()),
                                  "mean_e": float(players["mean_e"][players["y"] == k].mean())} for k in range(len(LEVELS)) if (players["y"] == k).any()}
    out["spearman_with_mean_e"] = {c: float(players[["mean_e", c]].corr(method="spearman").iloc[0, 1]) for c in COVARIATES}
    out["spearman_with_abs_mean_e"] = {c: float(players[["abs_mean_e", c]].corr(method="spearman").iloc[0, 1]) for c in COVARIATES}
    groups = {"all_wrong": players["all_wrong"], "all_right": players["all_right"], "other": ~(players["all_wrong"] | players["all_right"])}
    out["covariate_means"] = {g: {c: float(players[c][mask].mean()) for c in COVARIATES} | {"players": int(mask.sum())} for g, mask in groups.items()}
    return out


def opponent_channel(players):
    y = players["y"].to_numpy(dtype=float)
    own, opp = players["own_pmi"].to_numpy(), players["opp_pmi"].to_numpy()
    sp = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))  # noqa: E731
    slope, icpt = np.polyfit(own, y, 1)
    resid = y - (slope * own + icpt)
    return {"spearman_label_own_pmi": sp(y, own), "spearman_label_opp_pmi": sp(y, opp), "spearman_label_resid_on_own_vs_opp_minus_own": sp(resid, opp - own),
            "spearman_oof_mean_e_vs_opp_minus_own": sp(players["mean_e"].to_numpy(), opp - own), "spearman_own_opp": sp(own, opp)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", required=True)
    ap.add_argument("--opp", nargs="+", required=True)
    ap.add_argument("--hsl", default=str(ROOT / "results" / "scores"))
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-player", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--no-gbm", action="store_true", help="skip the xgboost model (when xgboost is not installed)")
    ap.add_argument("--jobs", type=int, default=1, help="xgboost threads")
    ap.add_argument("--out", required=True)
    ap.add_argument("--proba-out", default=None, help="CSV of mlp's OOF probabilities per validation question")
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not evaluated here")
    models = [m for m in MODELS if not (a.no_gbm and m == "gbm")]
    try:
        if "gbm" in models:
            import importlib
            importlib.import_module("xgboost")  # fail early, before the folds are read
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
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        print(e, file=sys.stderr)
        return 2
    t0 = time.time()
    rows, per_fold, nll, oof = [], {m: [] for m in models}, {m: [] for m in models}, []
    for v in folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train_q = pd.concat([sample_questions(data[f]["players"], a.per_player, rng) for f in folds if f != v], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in folds if f != v], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _ = feature_sets(cat("hsl"), cat("l1b"), cat("opp"), train_q)
        val = data[v]["val"]
        X_va, y_va, n_va = feature_sets(data[v]["hsl"], data[v]["l1b"], data[v]["opp"], val)
        fits = {"mlp": lambda: MLPModel(hidden=32).fit(X_tr["fused_opp"], y_tr).predict_proba(X_va["fused_opp"]),
                "mlp_diff": lambda: MLPModel(hidden=32).fit(X_tr["fused_opp_diff"], y_tr).predict_proba(X_va["fused_opp_diff"]),
                "gbm": lambda: fit_gbm(X_tr["fused_opp"], y_tr, n_jobs=a.jobs).predict_proba(X_va["fused_opp"]),
                "mlp_h128": lambda: MLPModel(hidden=128).fit(X_tr["fused_opp"], y_tr).predict_proba(X_va["fused_opp"])}
        sc = {}
        for m in models:
            P = fits[m]()
            res, sc[m] = evaluate(P, y_va, val)
            per_fold[m].append(res["balanced_mean_over_n"])
            nll[m].append(res["nll_balanced"])
            if m == "mlp":
                o = pd.DataFrame({"question_id": val["question_id"].to_numpy(dtype=str), "rank": val["rank"].to_numpy(dtype=str), "player_id": val["player_id"].to_numpy(dtype=str),
                                  "n_games": n_va, "fold": v, "y": y_va, "pred": [expected_score_decision(p) for p in P]})
                for j, lv in enumerate(LEVELS):
                    o[f"fused_opp_p_{lv}"] = P[:, j]
                oof.append(o)
        print(f"fold {v} ({len(train_q)} training questions): " + "  ".join(f"{m} {per_fold[m][-1]:.4f}" for m in models) + f"  ({time.time() - t0:.0f} s)", flush=True)
        rows.append(pd.DataFrame({"player_id": val["player_id"].to_numpy(dtype=str), "y": y_va, "n": n_va, **{m: sc[m] for m in models}}))
    t = pd.concat(rows, ignore_index=True)
    y, pid, n = t["y"].to_numpy(), t["player_id"].to_numpy(dtype=str), t["n"].to_numpy()
    head = {m: {"balanced_mean_over_n": balanced_mean_over_n(t[m].to_numpy(), y, n), "nll_balanced": float(np.mean(nll[m])),
                "balanced_by_n": {str(k): balanced_mean_over_n(t[m].to_numpy(), y, n, sizes=(k,)) for k in SIZES},
                "per_class": {LEVELS[c]: float(t[m].to_numpy()[y == c].mean()) for c in range(len(LEVELS)) if (y == c).any()}, "per_fold": per_fold[m]} for m in models}
    need = max(1, len(folds) - 1)

    def compare(x, z):
        d = paired_diff(t[x].to_numpy(), t[z].to_numpy(), y, pid, n, n=a.n_boot)
        pf = [p - q for p, q in zip(per_fold[x], per_fold[z])]
        cond = {"1": bool(d["ci95"][0] > 0), "3": bool(sum(e > 0 for e in pf) >= need), "4": bool(d["diff"] >= l1b_cv.MIN_DIFF)}
        return {**d, "per_fold_diff": pf, "conditions": cond, "pass": all(cond.values())}

    comps = {"pre_specified": {f"{x} vs {z}": compare(x, z) for x, z in PRE if x in models}, "report_only": {f"{x} vs {z}": compare(x, z) for x, z in REPORT if x in models}}
    oof = pd.concat(oof, ignore_index=True)
    players = player_table(oof, data, folds_df)
    analysis = {"residuals": residual_analysis(oof, players), "opponent_channel": opponent_channel(players)}
    for m, h in head.items():
        print(f"{m:9s} {h['balanced_mean_over_n']:.4f}  NLL {h['nll_balanced']:.4f}  by n " + " ".join(f"{k}: {h['balanced_by_n'][str(k)]:.4f}" for k in SIZES)
              + "  folds " + " ".join(f"{x:.4f}" for x in h["per_fold"]))
    for g, cs in comps.items():
        for k, c in cs.items():
            print(f"[{g}] {k}: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  folds " + " ".join(f"{x:+.4f}" for x in c["per_fold_diff"])
                  + "  conditions " + " ".join(f"{q} {'✓' if v else '✗'}" for q, v in c["conditions"].items()) + f"  -> {'PASS' if c['pass'] else 'no'}")
    r = analysis["residuals"]
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"  # noqa: E731
    print(f"residuals: exact {r['exact']:.3f} within1 {r['within1']:.3f}; between-player share {fmt(r['between_player_share'])}; all right {r['all_right']['observed']:.3f} "
          f"(independent {r['all_right']['independent']:.3f}), all wrong {r['all_wrong']['observed']:.3f} ({r['all_wrong']['independent']:.3f})")
    print("spearman |mean e| vs covariates: " + " ".join(f"{c} {fmt(v)}" for c, v in r["spearman_with_abs_mean_e"].items()))
    print("all-wrong rate by training-games quintile: " + " ".join(f"q{k} {v['all_wrong']:.3f} (median {v['median_train_games']:.0f})" for k, v in r["all_wrong_by_train_games_quintile"].items()))
    oc_ = analysis["opponent_channel"]
    print("opponent channel: " + " ".join(f"{k} {fmt(v)}" for k, v in oc_.items()))
    out = {"folds": folds, "questions": int(len(y)), "per_player": a.per_player, "models": models, "diff_features": DIFF_NAMES, "covariates": COVARIATES,
           "hsl_dir": a.hsl, "l1b_dirs": a.l1b, "opp_dirs": a.opp, "scores": head, "comparisons": comps, **analysis, "min_diff": l1b_cv.MIN_DIFF,
           "seconds": time.time() - t0, **environment("cpu")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json(out, indent=1) + "\n")
    if a.proba_out:
        Path(a.proba_out).parent.mkdir(parents=True, exist_ok=True)
        oof.to_csv(a.proba_out, index=False)
    print(f"wrote {a.out}" + (f" and {a.proba_out}" if a.proba_out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
