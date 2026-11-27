"""Section 4M M25 evaluation: searched strong-net move quality (results/ai_search/fold1/shard*.csv) on M23's 600 fold-1 players.
Same statistics as logs/nano4/ai_quality_eval.py: per-feature Spearman with the label and partial correlation given the oh ensemble's E;
an OOF ridge composite of the searched features and its partial given E (pre-registered continue condition, as M23: CI lower bound > 0
and >= 0.08); additionally the composite's partial given E AND M23's no-search composite (what search adds over no search).
  python -I logs/nano4/ai_search_eval.py --root results/ai_search/fold1 --nosearch results/ai_quality/fold1 --out results/r_variants/ai_search_pilot.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.metric import LEVELS  # noqa: E402
from ai_quality_eval import FEATS as FEATS_NS, ridge_oof  # noqa: E402
from oppview_eval import boot, partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
FEATS = ["t_loss_mean", "t_loss_median", "t_loss_q90", "t_gt1", "t_gt3", "t_gt8", "t_wloss_mean", "t_wgt10", "t_s_agree", "t_s_top3",
         "o_loss_mean", "o_loss_median", "o_gt3", "o_gt8", "o_wloss_mean", "o_s_agree", "o_s_top3"]


def load(root):
    return pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(f"{root}/shard[0-9]*.csv")) if "_human" not in f], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--nosearch", required=True); ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    S, N = load(a.root), load(a.nosearch)
    print(f"searched: {len(S)} games, {S.player_id.nunique()} players; answered turns {S.turns_answered.sum()} of {S.turns_expected.sum()}")
    gs = S.groupby("player_id").mean(numeric_only=True); gn = N.groupby("player_id").mean(numeric_only=True)
    y_all = S.groupby("player_id")["rank"].first().map(RANK)
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    ids = [p for p in gs.index if p in E.index and p in gn.index]
    gs, gn = gs.loc[ids], gn.loc[ids]; y = y_all.loc[ids].to_numpy(float); Ev = E.loc[ids].to_numpy(float); n = len(ids)
    print(f"{n} players with an oh-ensemble E and both feature sets")
    rep = {"players": n, "features": {}}
    for c in FEATS:
        if c not in gs.columns:
            continue
        x = gs[c].to_numpy(float); ok = np.isfinite(x)
        pe = partial(x[ok], y[ok], [Ev[ok]]); ci = boot(lambda idx: partial(x[ok][idx], y[ok][idx], [Ev[ok][idx]]), int(ok.sum()))
        rep["features"][c] = {"spearman_y": spearman(x[ok], y[ok]), "partial_given_E": pe, "ci95": ci}
        print(f"{c:16s} Spearman y {rep['features'][c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    fill = lambda X: np.where(np.isfinite(X), X, np.nanmean(X, axis=0, keepdims=True))  # noqa: E731
    Xs = fill(gs[[c for c in FEATS if c in gs.columns]].to_numpy(float)); Xn = fill(gn[[c for c in FEATS_NS if c in gn.columns]].to_numpy(float))
    cs, cn, cb = ridge_oof(Xs, y), ridge_oof(Xn, y), ridge_oof(np.column_stack([Xs, Xn]), y)
    est = partial(cs, y, [Ev]); ci = boot(lambda idx: partial(cs[idx], y[idx], [Ev[idx]]), n)
    inc = partial(cs, y, [Ev, cn]); ci_inc = boot(lambda idx: partial(cs[idx], y[idx], [Ev[idx], cn[idx]]), n)
    rep["composite_search"] = {"spearman_y": spearman(cs, y), "partial_given_E": est, "ci95": ci, "partial_given_E_and_nosearch": inc, "ci95_inc": ci_inc}
    rep["composite_nosearch"] = {"spearman_y": spearman(cn, y), "partial_given_E": partial(cn, y, [Ev])}
    rep["composite_both"] = {"spearman_y": spearman(cb, y), "partial_given_E": partial(cb, y, [Ev])}
    print(f"search composite ({Xs.shape[1]} features): Spearman y {rep['composite_search']['spearman_y']:.3f}  partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  partial|E,no-search {inc:+.3f} [{ci_inc[0]:+.3f}, {ci_inc[1]:+.3f}]")
    print(f"no-search composite (M23 features): Spearman y {rep['composite_nosearch']['spearman_y']:.3f}  partial|E {rep['composite_nosearch']['partial_given_E']:+.3f}; both: partial|E {rep['composite_both']['partial_given_E']:+.3f}")
    rep["continue"] = bool(ci[0] > 0 and est >= 0.08); print(f"continue (partial search composite | E: CI lower > 0 and >= 0.08): {'YES' if rep['continue'] else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
