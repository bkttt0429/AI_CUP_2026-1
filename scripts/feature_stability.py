#!/usr/bin/env python
"""Report only: how much does each question feature move when the SAME player is observed on a disjoint set of games?

Per fold f (1-4) every scored player has 20 games; they are split into two disjoint 10-game questions (and the first 5 of each
half into two 5-game questions). The features of both halves are put on the training-period scale (mean / std of the other three
folds' pilot questions of the same size; the TSFEL block is standardised and PCA-16 fitted there too, never on fold f) and compared
per feature: W_j = 0.5 * mean_p (z_pA - z_pB)^2 (GPT's sensitivity to the game sample), the between-player variance of the player
mean, their ratio, and the split-half correlation corr_p(z_pA, z_pB) (1 = the feature is a property of the player, 0 = of the sample).
Blocks: base 59 (hsl 24, l1b 16, opp 16, diff 3), shape 8, TSFEL PCA-16, and with --hd-table the HD 5 of aicup.handicap (issue #42).
Fold 0 refused; nothing is trained or scored.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.fused import question_diff_features  # noqa: E402
from aicup.l1bcv import players_from_scores  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.pilot import pilot_questions  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from aicup.seqshape import SHAPE, fold_shape_table, question_shape  # noqa: E402
from aicup.seqtsfel import feature_columns, load_tsfel_table, pca_apply, pca_fit, question_tsfel, standardise_impute  # noqa: E402
import l1b_cv  # noqa: E402
import l1b_opp_check as oc  # noqa: E402

BLOCKS = {"hsl24": slice(0, 24), "l1b16": slice(24, 40), "opp16": slice(40, 56), "diff3": slice(56, 59), "base59": slice(0, 59)}


def raw(hsl, l1b, opp, tsfel, shape, cols, q):
    X, y, n = oc.features(hsl, l1b, opp, q)
    return np.hstack([X["fused_opp"], question_diff_features(l1b, opp, q)]), question_tsfel(tsfel, q, cols), question_shape(shape, q, cols=SHAPE)


def halves(players, n, rng):
    rows_a, rows_b = [], []
    for p in players.itertuples():
        keys = list(p.game_keys)
        idx = rng.permutation(len(keys))
        if len(keys) < 2 * n:
            continue
        for rows, sel in ((rows_a, idx[:n]), (rows_b, idx[n:2 * n])):
            rows.append({"question_id": f"{p.player_id[:8]}_{n}", "rank": p.rank, "player_id": p.player_id, "n_games": n, "game_keys": [keys[i] for i in sel]})
    return pd.DataFrame(rows_a), pd.DataFrame(rows_b)


def stats(za, zb, names):
    w = 0.5 * np.mean((za - zb) ** 2, axis=0)
    between = np.var(0.5 * (za + zb), axis=0, ddof=1)
    corr = np.array([np.corrcoef(za[:, j], zb[:, j])[0, 1] if np.std(za[:, j]) > 0 and np.std(zb[:, j]) > 0 else np.nan for j in range(za.shape[1])])
    return pd.DataFrame({"feature": names, "W": w, "between_var": between, "ratio": w / np.maximum(between, 1e-12), "split_half_corr": corr})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--l1b", nargs="+", default=["results/l1b/scores3", "results/l1b/scores3_B"])
    ap.add_argument("--opp", nargs="+", default=["results/l1b/scores3_opp", "results/l1b/scores3_B_opp"])
    ap.add_argument("--hsl", default="results/scores")
    ap.add_argument("--tsfel-cache", required=True)
    ap.add_argument("--hd-table", default=None, help="scripts/hd_table.py's CSV (dataset v1.1 HD[n]); adds the hd block of aicup.handicap")
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    folds = sorted(set(a.folds))
    if 0 in folds:
        ap.error("fold 0 is not used here")
    t0 = time.time()
    hd_all, hd_names = None, []
    if a.hd_table:
        from aicup.handicap import HD_FEATURE_NAMES, question_hd_features, read_hd_table  # noqa: E402
        hd_all, hd_names = read_hd_table(a.hd_table), list(HD_FEATURE_NAMES)
    where, where_o = l1b_cv.locate_folds(a.l1b, folds), l1b_cv.locate_folds(a.opp, folds)
    folds_df = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})
    data = {}
    for f in folds:
        hsl, l1b = l1b_cv.load_fold(a.hsl, where[f], f)
        data[f] = {"hsl": hsl, "l1b": l1b, "opp": read_scored_games(Path(where_o[f]) / f"fold{f}"), "players": players_from_scores(hsl, folds_df),
                   "tsfel": load_tsfel_table(Path(a.tsfel_cache) / f"fold{f}.npz"), "shape": fold_shape_table(Path(where[f]) / f"fold{f}")}
        print(f"fold {f}: {len(data[f]['players'])} players ({time.time() - t0:.0f} s)", flush=True)
    cols = sorted(set().union(*[set(feature_columns(d["tsfel"])) for d in data.values()]))
    base_names = [f"base_{i}" for i in range(59)]
    extra_blocks = [("shape", list(SHAPE)), ("tsfel", [f"pc{i + 1}" for i in range(16)])] + ([("hd", hd_names)] if hd_all is not None else [])
    tables = []
    for f in folds:
        others = [g for g in folds if g != f]
        cat = lambda key: pd.concat([data[g][key] for g in others], ignore_index=True)  # noqa: E731
        H, L, O, T, S = cat("hsl"), cat("l1b"), cat("opp"), cat("tsfel"), cat("shape")
        pilot_tr = pd.concat([pilot_questions(data[g]["players"]) for g in others], ignore_index=True)
        rng = np.random.RandomState(a.seed + f)
        d = data[f]
        for n in (10, 5):
            q_tr = pilot_tr[pilot_tr["n_games"] == n].reset_index(drop=True)
            b_tr, t_tr, s_tr = raw(H, L, O, T, S, cols, q_tr)
            z_tr, _, _ = standardise_impute(t_tr, t_tr)
            pca = pca_fit(z_tr, 16)
            p_tr = pca_apply(pca, z_tr)
            scale = {k: (x.mean(axis=0), x.std(axis=0) + 1e-12) for k, x in (("base", b_tr), ("shape", s_tr), ("tsfel", p_tr))}
            if hd_all is not None:
                h_tr = question_hd_features(hd_all, q_tr)
                scale["hd"] = (h_tr.mean(axis=0), h_tr.std(axis=0) + 1e-12)
            qa, qb = halves(d["players"], n, rng)
            za, zb = {}, {}
            for side, q in (("a", qa), ("b", qb)):
                b, t, s = raw(d["hsl"], d["l1b"], d["opp"], d["tsfel"], d["shape"], cols, q)
                _, zt, _ = standardise_impute(t_tr, t)
                p = pca_apply(pca, zt)
                z = {"base": (b - scale["base"][0]) / scale["base"][1], "shape": (s - scale["shape"][0]) / scale["shape"][1], "tsfel": (p - scale["tsfel"][0]) / scale["tsfel"][1]}
                if hd_all is not None:
                    z["hd"] = (question_hd_features(hd_all, q) - scale["hd"][0]) / scale["hd"][1]
                (za if side == "a" else zb).update(z)
            for blk, names in [("base", base_names)] + extra_blocks:
                st = stats(za[blk], zb[blk], names)
                st.insert(0, "n", n)
                st.insert(0, "fold", f)
                st.insert(0, "block", blk)
                tables.append(st)
            med = {blk: np.nanmedian(stats(za[blk], zb[blk], names)["split_half_corr"]) for blk, names in [("base", base_names)] + extra_blocks}
            print(f"fold {f} n={n}: {len(qa)} players split; median split-half corr " + " ".join(f"{blk} {v:.3f}" for blk, v in med.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    t = pd.concat(tables, ignore_index=True)
    summary = {}
    for n in (10, 5):
        summary[str(n)] = {}
        for blk, sl in (("base59", BLOCKS["base59"]), ("hsl24", BLOCKS["hsl24"]), ("l1b16", BLOCKS["l1b16"]), ("opp16", BLOCKS["opp16"]), ("diff3", BLOCKS["diff3"])):
            sub = t[(t["block"] == "base") & (t["n"] == n)]
            feats = base_names[sl]
            sub = sub[sub["feature"].isin(feats)]
            summary[str(n)][blk] = {"median_split_half_corr": float(sub["split_half_corr"].median()), "mean_ratio_W_over_between": float(sub["ratio"].mean()), "mean_W": float(sub["W"].mean())}
        for blk, _ in extra_blocks:
            sub = t[(t["block"] == blk) & (t["n"] == n)]
            summary[str(n)][blk] = {"median_split_half_corr": float(sub["split_half_corr"].median()), "mean_ratio_W_over_between": float(sub["ratio"].mean()), "mean_W": float(sub["W"].mean()),
                                    "per_feature_corr": sub.groupby("feature")["split_half_corr"].mean().round(3).to_dict()}
        print(f"n={n}: " + "  ".join(f"{blk} corr {v['median_split_half_corr']:.3f} W/between {v['mean_ratio_W_over_between']:.2f}" for blk, v in summary[str(n)].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_json({"folds": folds, "seed": a.seed, "hd_table": a.hd_table, "summary": summary, "seconds": time.time() - t0, **environment("cpu")}, indent=1) + "\n")
    t.to_csv(str(Path(a.out).with_suffix(".csv")), index=False)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
