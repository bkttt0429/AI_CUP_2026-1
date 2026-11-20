"""Section 4M M23 evaluation: strong-net move-quality features (results/ai_quality/fold1/shard*.csv; humanv0 9d-judge control from
results/quality/fold1 on the same games) at the player level (600 fold-1 players, 20 games): Spearman with the label, partial correlation
given the oh ensemble's E (and given own HSL pmi), an OOF ridge composite of the own + opponent AI features and its partial given E, the
residual against the oh bias. Continue condition (pre-registered): partial(composite | E) CI lower bound > 0 and >= 0.08.
  python -I logs/nano4/ai_quality_eval.py --root results/ai_quality/fold1 --fold 1 --out results/r_variants/ai_quality_pilot.json
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
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from oppview_eval import boot, partial, resid_on  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
COLS = [f"sum_{p}" for p in PROFILES]
FEATS = ["t_ai_logp", "t_ai_agree", "t_ai_entropy", "t_ai_ploss", "t_ai_ploss_lt3", "t_ai_ploss_lt6", "t_loss_mean", "t_loss_median", "t_gt3", "t_gt8", "t_wloss_mean",
         "o_ai_logp", "o_ai_agree", "o_ai_ploss", "o_loss_mean", "o_gt3", "o_wloss_mean"]


def ridge_oof(X, y, k=5, lam=1.0, seed=0):
    f = np.random.RandomState(seed).randint(0, k, len(y)); pred = np.zeros(len(y))
    for i in range(k):
        tr, te = f != i, f == i; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        A = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd]); w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ w
    return pred


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True); a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(f"{a.root}/shard[0-9]*.csv")) if "_human" not in f], ignore_index=True)
    print(f"{len(T)} games, {T.player_id.nunique()} players; per-game sd of t_ai_logp {T.t_ai_logp.std():.3f}, agree mean {T.t_ai_agree.mean():.3f}, ploss mean {T.t_ai_ploss.mean():.3f}")
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}").set_index("game_key"); oh = read_scored_games(ROOT / "results/scores_opp_hd" / f"fold{a.fold}").set_index("game_key")
    T["own_pmi"] = posterior_mean_index(hsl.loc[T.game_key, COLS].to_numpy(float)); T["oppo"] = posterior_mean_index(oh.loc[T.game_key, COLS].to_numpy(float))
    hq = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in glob.glob(str(ROOT / "results/quality" / f"fold{a.fold}" / "quality_s*_p*.csv"))], ignore_index=True).set_index("game_key") if glob.glob(str(ROOT / "results/quality" / f"fold{a.fold}" / "quality_s*_p*.csv")) else None
    if hq is not None:
        for c in ("t_loss_mean", "t_gt3", "t_wloss_mean", "o_loss_mean"):
            T[f"h_{c}"] = hq[c].reindex(T.game_key).to_numpy()
    g = T.groupby("player_id").mean(numeric_only=True); g["y"] = T.groupby("player_id")["rank"].first().map(RANK)
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    g["E"] = E.reindex(g.index); g = g.dropna(subset=["E"]); g["bias"] = g.E - g.y; n = len(g); y = g.y.to_numpy(float)
    print(f"{n} players with an oh-ensemble E")
    rep = {"players": n, "features": {}}
    for c in FEATS + [c for c in g.columns if c.startswith("h_")]:
        if c not in g.columns or g[c].isna().all():
            continue
        x = g[c].to_numpy(float); ok = np.isfinite(x)
        xv, yv, Ev, hv, ov = x[ok], y[ok], g.E.to_numpy()[ok], g.own_pmi.to_numpy()[ok], g.oppo.to_numpy()[ok]
        pe = partial(xv, yv, [Ev]); cie = boot(lambda idx: partial(xv[idx], yv[idx], [Ev[idx]]), len(xv))
        rep["features"][c] = {"spearman_y": spearman(xv, yv), "partial_given_E": pe, "ci95": cie, "partial_given_own_oppo": partial(xv, yv, [hv, ov])}
        print(f"{c:16s} Spearman y {rep['features'][c]['spearman_y']:+.3f}  partial|E {pe:+.3f} [{cie[0]:+.3f}, {cie[1]:+.3f}]  partial|own,oppo {rep['features'][c]['partial_given_own_oppo']:+.3f}")
    X = g[[c for c in FEATS if c in g.columns]].to_numpy(float); X = np.nan_to_num(X, nan=np.nanmean(X, axis=0)[None, :].repeat(len(X), 0)[np.isnan(X)] if np.isnan(X).any() else 0.0) if np.isnan(X).any() else X
    comp = ridge_oof(X, y); Ev = g.E.to_numpy()
    est = partial(comp, y, [Ev]); ci = boot(lambda idx: partial(comp[idx], y[idx], [Ev[idx]]), n)
    rep["composite"] = {"spearman_y": spearman(comp, y), "partial_given_E": est, "ci95": ci, "partial_given_own_oppo": partial(comp, y, [g.own_pmi.to_numpy(), g.oppo.to_numpy()]),
                        "resid_vs_bias": spearman(resid_on(comp, np.column_stack([g.own_pmi, g.oppo])), g.bias.to_numpy())}
    print(f"OOF ridge composite of {X.shape[1]} AI features: Spearman y {rep['composite']['spearman_y']:.3f}  partial|E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  partial|own,oppo {rep['composite']['partial_given_own_oppo']:+.3f}  resid vs bias {rep['composite']['resid_vs_bias']:+.3f}")
    rep["continue"] = bool(ci[0] > 0 and est >= 0.08); print(f"continue (partial composite | E: CI lower > 0 and >= 0.08): {'YES' if rep['continue'] else 'no'}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
