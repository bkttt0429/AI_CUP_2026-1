"""Section 4M M26 evaluation (rules / komi): per arm (cn75 production control, cnk, jp) the per-game posterior-mean index over the 18
profiles, fit (mean per-move log p of the actual moves under the best profile: inputs that match the games better explain the moves
better), game- and player-level adjacent-class AUC, Spearman with the label, and partial(pmi_arm | pmi_cn75, E) with E the oh ensemble;
the paired player-bootstrap CI of the game-level AUC difference arm - cn75 (report only). Sanity: cn75 must reproduce results/scores_hd.
Continue condition (pre-registered, as M18): game-level adjacent AUC >= cn75 + 0.01, or partial(pmi_arm | pmi_cn75, E) CI lower > 0 and >= 0.08.
  python -I logs/nano4/rules_pilot_eval.py --root results/rules_pilot/fold1 --fold 1 --out results/r_variants/rules_pilot.json
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
from meta_pilot_eval import adjacent_auc  # noqa: E402
from oppview_eval import boot, partial  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
ARMS = ("cn75", "cnk", "jp")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pd.concat([pd.read_csv(f, dtype={"player_id": str, "game_key": str}) for f in sorted(glob.glob(f"{a.root}/shard*.csv"))], ignore_index=True)
    yg = T["rank"].map(RANK).to_numpy(); n = T["n_moves"].to_numpy(dtype=float)
    prod = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}").set_index("game_key").loc[T.game_key][[f"sum_{p}" for p in PROFILES]].to_numpy(dtype=float)
    L = {arm: T[[f"ll_{arm}_{p}" for p in PROFILES]].to_numpy(dtype=float) for arm in ARMS}
    rep = {"games": int(len(T)), "control_max_abs_diff_vs_production": float(np.abs(L["cn75"] - prod).max()),
           "control_median_abs_diff_vs_production": float(np.median(np.abs(L["cn75"] - prod))), "arms": {}}
    for arm in ("cnk", "jp"):
        d = L[arm] - L["cn75"]; per_move = d / np.maximum(n, 1)[:, None]
        rep[f"{arm}_vs_cn75_per_move_logp_diff"] = {"mean": float(per_move.mean()), "mean_abs": float(np.abs(per_move).mean()), "q99_abs": float(np.quantile(np.abs(per_move), 0.99)),
                                                    "corr_game_sums": float(np.corrcoef(L[arm].ravel(), L["cn75"].ravel())[0, 1])}
    print(f"{len(T)} games; cn75 vs production sums: max |diff| {rep['control_max_abs_diff_vs_production']:.4f}, median {rep['control_median_abs_diff_vs_production']:.5f}")
    meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{a.fold}.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{a.fold}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS)); E = meta[meta["n_games"] == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    pl = T.groupby("player_id").agg(y=("rank", "first")); pl["y"] = pl["y"].map(RANK); pl["E"] = E.reindex(pl.index)
    yp = pl["y"].to_numpy(dtype=float); ok = pl["E"].notna().to_numpy()
    pmi = {arm: posterior_mean_index(L[arm]) for arm in ARMS}
    gpm = {arm: pd.DataFrame({"player_id": T.player_id, "x": pmi[arm]}).groupby("player_id").mean().reindex(pl.index)["x"].to_numpy() for arm in ARMS}
    pid = T.player_id.to_numpy(); players = np.unique(pid); rows_of = {p: np.flatnonzero(pid == p) for p in players}
    for arm in ARMS:
        fit = (L[arm] / np.maximum(n, 1)[:, None]).max(axis=1)
        rec = {"fit_mean_logp_best_profile": float(fit.mean()), "game_adjacent_auc": adjacent_auc(pmi[arm], yg), "player_adjacent_auc": adjacent_auc(gpm[arm], pl["y"].to_numpy()),
               "player_spearman": spearman(gpm[arm], yp)}
        if arm != "cn75":
            x, Z = gpm[arm][ok], [gpm["cn75"][ok], pl["E"].to_numpy()[ok]]
            est = partial(x, yp[ok], Z); ci = boot(lambda idx: partial(x[idx], yp[ok][idx], [z[idx] for z in Z]), int(ok.sum()))
            rec["partial|cn75,E"] = {"estimate": est, "ci95": ci}
            rng = np.random.default_rng(0); diffs = []
            for _ in range(500):
                idx = np.concatenate([rows_of[p] for p in rng.choice(players, len(players))])
                diffs.append(adjacent_auc(pmi[arm][idx], yg[idx]) - adjacent_auc(pmi["cn75"][idx], yg[idx]))
            rec["game_auc_diff_vs_cn75"] = {"estimate": rec["game_adjacent_auc"] - adjacent_auc(pmi["cn75"], yg), "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]}
        rep["arms"][arm] = rec
        extra = "" if arm == "cn75" else (f"  AUC diff {rec['game_auc_diff_vs_cn75']['estimate']:+.4f} [{rec['game_auc_diff_vs_cn75']['ci95'][0]:+.4f}, {rec['game_auc_diff_vs_cn75']['ci95'][1]:+.4f}]"
                                          f"  partial|cn75,E {est:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        print(f"{arm:5s} fit {rec['fit_mean_logp_best_profile']:.4f}  game adj-AUC {rec['game_adjacent_auc']:.4f}  player adj-AUC {rec['player_adjacent_auc']:.4f}  Spearman {rec['player_spearman']:.3f}{extra}")
    base = rep["arms"]["cn75"]["game_adjacent_auc"]
    rep["continue"] = {arm: bool(rep["arms"][arm]["game_adjacent_auc"] >= base + 0.01 or (rep["arms"][arm]["partial|cn75,E"]["ci95"][0] > 0 and rep["arms"][arm]["partial|cn75,E"]["estimate"] >= 0.08)) for arm in ("cnk", "jp")}
    print("per-move log p change vs cn75:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in rep.items() if k.endswith("_per_move_logp_diff")})
    print("continue:", rep["continue"])
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
