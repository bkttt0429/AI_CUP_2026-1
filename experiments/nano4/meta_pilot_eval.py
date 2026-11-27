"""Section 4M M18 evaluation: per setting (kgs control, fox, tygem, kgs_byo) the per-game posterior-mean index over its profiles, the
fit (mean per-move log p of the actual moves under the best profile: a better-matched metadata explains the moves better), game-level
and player-level adjacent-class AUC (per-game evidence sharpness), Spearman with the label, and the partial correlation of the setting's
player-level pmi with the label given the KGS pmi and the oh ensemble's E. Sanity: the kgs control must reproduce results/scores_hd.
Continue condition (pre-registered): game-level adjacent AUC >= kgs + 0.01, or partial(pmi_setting | pmi_kgs, E) CI lower > 0 and >= 0.08.
  python -I logs/nano4/meta_pilot_eval.py --root results/meta_pilot/fold1 --fold 1 --out results/r_variants/meta_pilot.json
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
from aicup.evidence import auc, posterior_mean_index  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from meta_pilot_score import SETTINGS  # noqa: E402
from oppview_eval import boot, partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def adjacent_auc(x, y):
    return float(np.mean([auc(x[y == c + 1], x[y == c]) for c in range(len(LEVELS) - 1) if (y == c).any() and (y == c + 1).any()]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(f"{a.root}/shard*.csv"))], ignore_index=True)
    yg = T["rank"].map(RANK).to_numpy(); n = T["n_moves"].to_numpy(dtype=float)
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}").set_index("game_key").loc[T.game_key]
    prod = hsl[[f"sum_{p}" for p in PROFILES]].to_numpy(dtype=float)
    kgs = T[[f"ll_kgs_{p}" for p in PROFILES]].to_numpy(dtype=float)
    rep = {"games": int(len(T)), "control_max_abs_diff_vs_production": float(np.abs(kgs - prod).max()), "settings": {}}
    print(f"{len(T)} games; kgs control vs production sums: max |diff| {rep['control_max_abs_diff_vs_production']:.4f}")
    # where does the control differ from production? by the game's HD (the production tables re-scored HD >= 2 games with the stones placed)
    from aicup.handicap import read_hd_table
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"].reindex(T.game_key).fillna(-1).to_numpy()
    dmax = np.abs(kgs - prod).max(axis=1)
    rep["control_diff_by_hd"] = {str(int(h)): {"games": int((hd == h).sum()), "median_max_abs_diff": float(np.median(dmax[hd == h])), "share_gt_1": float((dmax[hd == h] > 1).mean())} for h in np.unique(hd)}
    print("control |diff| by HD:", {k: (v["games"], round(v["median_max_abs_diff"], 4), round(v["share_gt_1"], 3)) for k, v in rep["control_diff_by_hd"].items()})
    rep["production_game_adjacent_auc"] = adjacent_auc(posterior_mean_index(prod), yg)
    print(f"production table (own = opp = p, HD-aware) game adj-AUC {rep['production_game_adjacent_auc']:.4f}")
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta["n_games"] == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    pl = T.groupby("player_id").agg(y=("rank", "first")); pl["y"] = pl["y"].map(RANK); pl["E"] = E.reindex(pl.index)
    pmi_kgs_g = posterior_mean_index(kgs)
    for name, (_, _, _, _, profs) in SETTINGS.items():
        L = T[[f"ll_{name}_{p}" for p in profs]].to_numpy(dtype=float)
        pmi_g = posterior_mean_index(L); pmi18_g = posterior_mean_index(L[:, :len(PROFILES)])
        fit = (L / n[:, None]).max(axis=1)  # per-move log p under the best profile
        g = pd.DataFrame({"player_id": T.player_id, "pmi": pmi_g, "pmi18": pmi18_g, "fit": fit}).groupby("player_id").mean().reindex(pl.index)
        ok = pl["E"].notna().to_numpy()
        yp = pl["y"].to_numpy(dtype=float)
        rec = {"fit_mean_logp_best_profile": float(fit.mean()), "game_adjacent_auc": adjacent_auc(pmi_g, yg), "game_adjacent_auc_18": adjacent_auc(pmi18_g, yg),
               "player_adjacent_auc": adjacent_auc(g["pmi"].to_numpy(), pl["y"].to_numpy()), "player_spearman": spearman(g["pmi"].to_numpy(), yp),
               "share_best_profile_above_6d": float((L.argmax(axis=1) >= len(PROFILES)).mean()) if len(profs) > len(PROFILES) else 0.0}
        if name != "kgs":
            gk = pd.DataFrame({"player_id": T.player_id, "k": pmi_kgs_g}).groupby("player_id").mean().reindex(pl.index)["k"].to_numpy()
            x = g["pmi"].to_numpy()
            for tag, Z in (("|kgs", [gk]), ("|kgs,E", [gk, pl["E"].to_numpy()])):
                m = ok if "E" in tag else np.ones(len(x), bool)
                xv, yv, Zv = x[m], yp[m], [z[m] for z in Z]
                est = partial(xv, yv, Zv); ci = boot(lambda idx: partial(xv[idx], yv[idx], [z[idx] for z in Zv]), len(xv))
                rec[f"partial{tag}"] = {"estimate": est, "ci95": ci}
        rep["settings"][name] = rec
        extra = "" if name == "kgs" else f"  partial|kgs {rec['partial|kgs']['estimate']:+.3f} [{rec['partial|kgs']['ci95'][0]:+.3f}, {rec['partial|kgs']['ci95'][1]:+.3f}]  partial|kgs,E {rec['partial|kgs,E']['estimate']:+.3f} [{rec['partial|kgs,E']['ci95'][0]:+.3f}, {rec['partial|kgs,E']['ci95'][1]:+.3f}]"
        print(f"{name:8s} fit {rec['fit_mean_logp_best_profile']:.4f}  game adj-AUC {rec['game_adjacent_auc']:.4f} (18 prof {rec['game_adjacent_auc_18']:.4f})  player adj-AUC {rec['player_adjacent_auc']:.4f}  Spearman {rec['player_spearman']:.3f}  best>6d {rec['share_best_profile_above_6d']:.3f}{extra}")
    base = rep["settings"]["kgs"]["game_adjacent_auc"]
    rep["continue"] = {name: bool(r["game_adjacent_auc"] >= base + 0.01 or (name != "kgs" and r["partial|kgs,E"]["ci95"][0] > 0 and r["partial|kgs,E"]["estimate"] >= 0.08)) for name, r in rep["settings"].items() if name != "kgs"}
    print("continue:", rep["continue"])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
