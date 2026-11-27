"""Decomposing the error floor (user: 怎麼切分這個問題原子分析). Estimator: the 64-dim fused_opp_diff_hd MLP h32 (seeds 0-2 averaged),
trained on the other folds' questions; evaluated on every validation player's two DISJOINT 10-game halves (pilot_questions n=10,
j = 0 / 1), its 20-game question, and the two halves ordered by game id (early / late).
  A1 signed error e = E[rank] - label: between-player (persistent) vs within-player (sampling) variance; split-half r of e
  A2 shrinkage: slope of the player bias on the label; bias by class (edge classes D, 6D)
  A3 trend: per-game own pmi vs game-id order within a player (Spearman), early-half vs late-half error, relation to bias
  A4 how much of the persistent bias is explained by observables of the OTHER half (opponent gap, HD1 share, colour share,
     game length, the five style columns): 4-fold ridge, OOF R^2
  A5 confusion structure: adjacent vs far errors, posterior confidence on right / adjacent-wrong / far-wrong, persistence of
     the error direction across halves
  A6 label-noise reading: the score ceiling of a perfect model under symmetric adjacent label noise p, and the p implied by the
     confident persistent adjacent errors
  A7 kyu bands vs dan classes

  python -I logs/nano4/floor_decomp.py --out results/r_variants/floor_decomp.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS, expected_score_decision, score  # noqa: E402
from aicup.qfeatures import sample_questions, sum_cols  # noqa: E402
from aicup.rankmodel import MLPModel  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import hd_check  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from fp_diag import Fast  # noqa: E402
from pilot_stats import spearman  # noqa: E402
from r_variants import opening_keys  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def _fit(job):
    Xt, y, Xv, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    return MLPModel(hidden=32, seed=seed).fit(Xt, y).predict_proba(Xv)


def ridge_oof_r2(X, y, folds=4, lam=1.0, seed=0):
    """Out-of-fold R^2 of a ridge regression of y on standardised X (players shuffled into folds)."""
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)) % folds
    pred = np.zeros(len(y))
    for f in range(folds):
        tr, te = idx != f, idx == f
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    workers = max(1, min(len(os.sched_getaffinity(0)), 48))
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    hd_all = read_hd_table(ROOT / "results/data/hd_table.csv")
    lcols = sum_cols(LEVELS)
    data = {}
    for f in a.folds:
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        hd = hd_all[hd_all["game_key"].isin(set(hsl["game_key"]))]
        val = l1b_cv.pilot_questions(players)
        # order-based halves: the player's 20 games sorted by numeric game id (the dataset's row order), early 10 / late 10
        rows = []
        for p in players.itertuples():
            keys = list(p.game_keys)
            if len(keys) < 20:
                continue
            ordered = sorted(keys, key=lambda k: int(k.split(":g_")[1]))
            rows.append({"question_id": f"{p.rank}_{p.player_id[:8]}_early", "rank": p.rank, "player_id": p.player_id, "n_games": 10, "game_keys": ordered[:10]})
            rows.append({"question_id": f"{p.rank}_{p.player_id[:8]}_late", "rank": p.rank, "player_id": p.player_id, "n_games": 10, "game_keys": ordered[10:20]})
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": opp, "hd": hd, "players": players, "val": val, "ordered": pd.DataFrame(rows, columns=val.columns),
                   "opening": opening_keys(hsl["game_key"])}
        data[f]["fast"] = Fast.from_tables(hsl, l1b, data[f]["opening"])
        print(f"fold {f} loaded ({time.time() - t0:.0f} s)", flush=True)
    # OOF posteriors for the pilot questions and the ordered halves
    P_val, P_ord = {}, {}
    for v in a.folds:
        rng = np.random.RandomState(l1b_cv.TRAIN_SEED + v)
        train = [f for f in a.folds if f != v]
        train_q = pd.concat([sample_questions(data[f]["players"], 32, rng) for f in train], ignore_index=True)
        cat = lambda key: pd.concat([data[f][key] for f in train], ignore_index=True)  # noqa: E731
        X_tr, y_tr, _, _ = hd_check.features(cat("hsl"), cat("l1b"), cat("opp"), cat("hd"), train_q)
        d = data[v]
        X_va, _, _, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["val"])
        X_or, _, _, _ = hd_check.features(d["hsl"], d["l1b"], d["opp"], d["hd"], d["ordered"])
        Xt = X_tr["fused_opp_diff_hd"]
        Xv = np.vstack([X_va["fused_opp_diff_hd"], X_or["fused_opp_diff_hd"]])
        with mp.get_context("fork").Pool(min(workers, len(a.seeds))) as pool:
            probas = pool.map(_fit, [(Xt, y_tr, Xv, s) for s in a.seeds])
        P = np.mean(probas, axis=0)
        P_val[v], P_ord[v] = P[:len(d["val"])], P[len(d["val"]):]
        print(f"fold {v}: OOF posteriors for {len(d['val'])} pilot questions + {len(d['ordered'])} ordered halves ({time.time() - t0:.0f} s)", flush=True)
    # per-player table
    rows = []
    for v in a.folds:
        d, P = data[v], P_val[v]
        q = d["val"].reset_index(drop=True)
        q["E"] = (P * np.arange(10)).sum(axis=1)  # expected rank index from the probabilities (posterior_mean_index expects log scores)
        q["pred"] = [expected_score_decision(p) for p in P]
        q["pmax"] = P.max(axis=1)
        q["y"] = q["rank"].map(RANK)
        q["score"] = [score(int(p), int(t)) for p, t in zip(q["pred"], q["y"])]
        o = d["ordered"].reset_index(drop=True)
        o["E"] = (P_ord[v] * np.arange(10)).sum(axis=1)
        L, O = d["l1b"].set_index("game_key")[lcols], d["opp"].set_index("game_key")[lcols]
        colour = dict(zip(d["hsl"]["game_key"], d["hsl"]["color"]))
        nmoves = dict(zip(d["hsl"]["game_key"], d["hsl"]["n_moves"]))
        hdmap = dict(zip(d["hd"]["game_key"], d["hd"]["hd"]))
        for pid, g in q.groupby("player_id"):
            h = g[g.n_games == 10].sort_values("question_id")
            if len(h) != 2:
                continue
            full = g[g.n_games == 20]
            keys = list(d["players"].set_index("player_id").loc[pid, "game_keys"])
            # per-game own pmi vs game-id order (A3)
            per_game = posterior_mean_index(L.loc[keys].to_numpy(dtype=float))
            order = np.argsort(np.argsort([int(k.split(":g_")[1]) for k in keys]))
            slope = spearman(order, per_game)
            oe = o[o.player_id == pid].set_index("question_id")
            early = float(oe.loc[f"{g.iloc[0]['rank']}_{pid[:8]}_early", "E"]) if f"{g.iloc[0]['rank']}_{pid[:8]}_early" in oe.index else np.nan
            late = float(oe.loc[f"{g.iloc[0]['rank']}_{pid[:8]}_late", "E"]) if f"{g.iloc[0]['rank']}_{pid[:8]}_late" in oe.index else np.nan
            # observables of each half (A4)
            obs = []
            for ks in h["game_keys"]:
                ks = list(ks)
                own = posterior_mean_index(L.loc[ks].to_numpy(dtype=float).sum(axis=0))
                oppm = posterior_mean_index(O.loc[ks].to_numpy(dtype=float).sum(axis=0))
                fp = d["fast"].fp(ks)
                obs.append([oppm - own, np.mean([hdmap.get(k, 0) == 1 for k in ks]), np.mean([colour[k] == "B" for k in ks]), np.mean([nmoves[k] for k in ks]), *fp])
            y = int(g["y"].iloc[0])
            rows.append({"fold": v, "player_id": pid, "y": y, "e0": float(h["E"].iloc[0]) - y, "e1": float(h["E"].iloc[1]) - y,
                         "e20": (float(full["E"].iloc[0]) - y) if len(full) else np.nan, "pred0": int(h["pred"].iloc[0]), "pred1": int(h["pred"].iloc[1]),
                         "pmax0": float(h["pmax"].iloc[0]), "pmax1": float(h["pmax"].iloc[1]), "score0": float(h["score"].iloc[0]), "score1": float(h["score"].iloc[1]),
                         "trend_rho": slope, "e_early": early - y if np.isfinite(early) else np.nan, "e_late": late - y if np.isfinite(late) else np.nan,
                         "obs0": obs[0], "obs1": obs[1]})
    T = pd.DataFrame(rows)
    rep = {"players": int(len(T))}
    e0, e1, y = T["e0"].to_numpy(), T["e1"].to_numpy(), T["y"].to_numpy()
    e = np.concatenate([e0, e1]); b = (e0 + e1) / 2
    # A1
    var_total = e.var(); var_between = b.var() - 0.5 * np.var(e0 - e1) / 2  # unbiased between-player variance (mean of 2 halves minus half the within)
    var_within = 0.5 * np.mean((e0 - e1) ** 2)
    r = float(np.corrcoef(e0, e1)[0, 1])
    rep["A1"] = {"var_total": float(var_total), "var_within_sampling": float(var_within), "var_between_persistent": float(var_total - var_within), "share_persistent": float((var_total - var_within) / var_total),
                 "split_half_r": r, "rmse_half": float(np.sqrt(np.mean(e ** 2))), "rmse_20": float(np.sqrt(np.nanmean(T["e20"] ** 2))),
                 "by_class_share_persistent": {LEVELS[c]: float(1 - 0.5 * np.mean((e0[y == c] - e1[y == c]) ** 2) / np.concatenate([e0[y == c], e1[y == c]]).var()) for c in range(10)}}
    print(f"\n== A1 signed error (E[rank] - label), 10-game halves of {len(T)} players: var {var_total:.3f} = persistent {var_total - var_within:.3f} ({rep['A1']['share_persistent']:.1%}) + sampling {var_within:.3f}; split-half r {r:.3f}; RMSE half {rep['A1']['rmse_half']:.3f}, 20 games {rep['A1']['rmse_20']:.3f}")
    print("   persistent share by class: " + " ".join(f"{k} {v_:.2f}" for k, v_ in rep["A1"]["by_class_share_persistent"].items()))
    # A2
    A = np.column_stack([np.ones(len(b)), y]); beta = np.linalg.lstsq(A, b, rcond=None)[0]
    rep["A2"] = {"bias_slope_on_label": float(beta[1]), "bias_by_class": {LEVELS[c]: float(b[y == c].mean()) for c in range(10)}, "bias_sd_by_class": {LEVELS[c]: float(b[y == c].std()) for c in range(10)}}
    print(f"\n== A2 shrinkage: slope of the player bias on the label {beta[1]:+.3f} per class; bias by class " + " ".join(f"{k} {v_:+.2f}" for k, v_ in rep["A2"]["bias_by_class"].items()))
    # A3
    tr = T["trend_rho"].to_numpy(dtype=float)
    ok = np.isfinite(tr)
    el = (T["e_late"] - T["e_early"]).to_numpy(dtype=float)
    rep["A3"] = {"trend_rho_mean": float(np.nanmean(tr)), "trend_rho_sd": float(np.nanstd(tr)), "share_positive": float((tr[ok] > 0).mean()),
                 "corr_trend_bias": spearman(tr[ok], b[ok]), "late_minus_early_error_mean": float(np.nanmean(el)), "late_minus_early_se": float(np.nanstd(el) / np.sqrt(np.isfinite(el).sum())),
                 "share_late_higher": float(np.nanmean(el > 0)), "corr_late_minus_early_with_bias": spearman(el[np.isfinite(el)], b[np.isfinite(el)])}
    print(f"\n== A3 trend over game-id order: per-player Spearman(order, per-game pmi) mean {rep['A3']['trend_rho_mean']:+.3f} (sd {rep['A3']['trend_rho_sd']:.3f}), {rep['A3']['share_positive']:.1%} positive; corr(trend, bias) {rep['A3']['corr_trend_bias']:+.3f}; "
          f"late-half E minus early-half E {rep['A3']['late_minus_early_error_mean']:+.4f} (se {rep['A3']['late_minus_early_se']:.4f}), {rep['A3']['share_late_higher']:.1%} of players later higher")
    # A4
    OBS = ["opp_minus_own_pmi", "hd1_share", "black_share", "mean_target_moves", "pmi_std", "curve_pairdist", "first1_agree", "first3_agree", "nmoves_cv"]
    X01 = np.array([o for o in T["obs1"]]); X10 = np.array([o for o in T["obs0"]])
    r2_cross = float(np.mean([ridge_oof_r2(X01, e0), ridge_oof_r2(X10, e1)]))
    r2_cross_y = float(np.mean([ridge_oof_r2(np.column_stack([X01, y]), e0), ridge_oof_r2(np.column_stack([X10, y]), e1)]))
    r2_y = float(np.mean([ridge_oof_r2(y[:, None].astype(float), e0), ridge_oof_r2(y[:, None].astype(float), e1)]))
    uni = {c: spearman(X01[:, j], e0) for j, c in enumerate(OBS)}
    rep["A4"] = {"oof_r2_observables_other_half": r2_cross, "oof_r2_observables_plus_label": r2_cross_y, "oof_r2_label_only": r2_y, "ceiling_r2_persistent": r, "univariate_spearman_with_error": uni}
    print(f"\n== A4 persistent bias explained by the other half's observables: OOF R^2 {r2_cross:.3f} (label only {r2_y:.3f}, observables + label {r2_cross_y:.3f}); the persistent share itself is {r:.3f}")
    print("   univariate Spearman with the error: " + " ".join(f"{k} {v_:+.2f}" for k, v_ in uni.items()))
    # A5
    adj0 = (np.abs(T["pred0"] - y) == 1); far0 = np.abs(T["pred0"] - y) > 1; hit0 = T["pred0"] == y
    both_wrong = (T["pred0"] != y) & (T["pred1"] != y)
    same_dir = both_wrong & (np.sign(T["pred0"] - y) == np.sign(T["pred1"] - y))
    conf_adj_persist = same_dir & adj0 & (np.abs(T["pred1"] - y) == 1) & (T["pmax0"] > 0.5) & (T["pmax1"] > 0.5)
    rep["A5"] = {"hit": float(hit0.mean()), "adjacent": float(adj0.mean()), "far": float(far0.mean()), "pmax_hit": float(T.loc[hit0, "pmax0"].mean()), "pmax_adjacent": float(T.loc[adj0, "pmax0"].mean()), "pmax_far": float(T.loc[far0, "pmax0"].mean()),
                 "both_halves_wrong": float(both_wrong.mean()), "same_direction_given_both_wrong": float(same_dir.sum() / max(both_wrong.sum(), 1)), "confident_persistent_adjacent_share": float(conf_adj_persist.mean())}
    print(f"\n== A5 confusion (half 0): hit {rep['A5']['hit']:.3f} adjacent {rep['A5']['adjacent']:.3f} far {rep['A5']['far']:.3f}; pmax hit {rep['A5']['pmax_hit']:.2f} / adjacent {rep['A5']['pmax_adjacent']:.2f} / far {rep['A5']['pmax_far']:.2f}; "
          f"both halves wrong {rep['A5']['both_halves_wrong']:.3f}, same direction given both wrong {rep['A5']['same_direction_given_both_wrong']:.2f}; confident persistent adjacent errors {rep['A5']['confident_persistent_adjacent_share']:.3f}")
    # A6
    p_grid = np.linspace(0, 0.5, 51)
    ceiling = [(1 - p) * 1.0 + p * np.exp(-1) for p in p_grid]
    p_impl = rep["A5"]["confident_persistent_adjacent_share"]
    rep["A6"] = {"ceiling_vs_p": dict(zip([f"{p:.2f}" for p in p_grid], ceiling)), "p_implied_lower_bound": p_impl, "ceiling_at_implied_p": float((1 - p_impl) + p_impl * np.exp(-1))}
    print(f"\n== A6 label-noise reading: a perfect model under adjacent label noise p scores (1-p) + p/e; confident persistent adjacent errors give p >= {p_impl:.3f} -> ceiling <= {rep['A6']['ceiling_at_implied_p']:.3f} on those terms")
    # A7
    kyu = y <= 3
    rep["A7"] = {"kyu": {"hit": float(hit0[kyu].mean()), "bias_sd": float(b[kyu].std()), "persistent_share": float(1 - 0.5 * np.mean((e0[kyu] - e1[kyu]) ** 2) / np.concatenate([e0[kyu], e1[kyu]]).var())},
                 "dan": {"hit": float(hit0[~kyu].mean()), "bias_sd": float(b[~kyu].std()), "persistent_share": float(1 - 0.5 * np.mean((e0[~kyu] - e1[~kyu]) ** 2) / np.concatenate([e0[~kyu], e1[~kyu]]).var())}}
    print(f"\n== A7 kyu (D-A) hit {rep['A7']['kyu']['hit']:.3f} bias sd {rep['A7']['kyu']['bias_sd']:.3f} persistent {rep['A7']['kyu']['persistent_share']:.2f} | dan hit {rep['A7']['dan']['hit']:.3f} bias sd {rep['A7']['dan']['bias_sd']:.3f} persistent {rep['A7']['dan']['persistent_share']:.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    T.drop(columns=["obs0", "obs1"]).to_csv(Path(a.out).with_suffix(".players.csv"), index=False)
    print("written", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
