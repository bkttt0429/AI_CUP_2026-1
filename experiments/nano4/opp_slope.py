"""Does a player play 'up' against stronger opponents? Per validation player (folds 1-4, 20 games): the within-player slope of the
per-game own L1b pmi on the per-game opponent pmi, its distribution by rank, and its relation to the player's persistent error
(floor_decomp players.csv). Also the between-player relation: mean opponent gap vs error, and the share of games vs stronger opponents.
  python -I logs/nano4/opp_slope.py --out results/r_variants/opp_slope.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import l1b_cv  # noqa: E402
import trunk_seq_check as ts  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    folds_df = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str})
    bias = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str})
    bias["b"] = (bias["e0"] + bias["e1"]) / 2
    bmap = dict(zip(bias["player_id"], bias["b"]))
    lcols = sum_cols(LEVELS)
    rows = []
    for f in (1, 2, 3, 4):
        hsl, l1b = l1b_cv.load_fold(ts.HSL, ROOT / "results/l1b" / ts.L1B[f], f)
        opp = read_scored_games(ROOT / "results/l1b" / ts.OPP[f] / f"fold{f}")
        players = l1b_cv.players_from_scores(hsl, folds_df)
        L, O = l1b.set_index("game_key"), opp.set_index("game_key")
        own = pd.Series(posterior_mean_index(L[lcols].to_numpy(dtype=float)), index=L.index)
        oppi = pd.Series(posterior_mean_index(O[lcols].to_numpy(dtype=float)), index=O.index)
        for p in players.itertuples():
            ks = list(p.game_keys)
            if len(ks) < 10:
                continue
            x, y = oppi.loc[ks].to_numpy(), own.loc[ks].to_numpy()
            if x.std() < 1e-6:
                continue
            slope = float(np.polyfit(x, y, 1)[0]); r = float(np.corrcoef(x, y)[0, 1])
            rows.append({"player_id": p.player_id, "y": RANK[p.rank], "slope": slope, "r_within": r, "gap_mean": float((x - y).mean()), "share_stronger_opp": float((x > y).mean()),
                         "own_mean": float(y.mean()), "opp_mean": float(x.mean()), "bias": bmap.get(p.player_id, np.nan), "n": len(ks)})
    T = pd.DataFrame(rows)
    ok = np.isfinite(T["bias"])
    rep = {"players": int(len(T)), "within_player_slope_mean": float(T["slope"].mean()), "within_player_slope_median": float(T["slope"].median()), "share_slope_positive": float((T["slope"] > 0).mean()),
           "within_player_r_mean": float(T["r_within"].mean()), "slope_by_rank": [float(T.loc[T.y == k, "slope"].mean()) for k in range(10)],
           "corr_slope_bias": spearman(T.loc[ok, "slope"], T.loc[ok, "bias"]), "corr_gapmean_bias": spearman(T.loc[ok, "gap_mean"], T.loc[ok, "bias"]),
           "corr_sharestronger_bias": spearman(T.loc[ok, "share_stronger_opp"], T.loc[ok, "bias"]), "corr_gapmean_rank": spearman(T["gap_mean"], T["y"]),
           "gap_mean_by_rank": [float(T.loc[T.y == k, "gap_mean"].mean()) for k in range(10)]}
    # the difficulty common cause: how much of the within-player r survives when both pmi are measured on different plies?  (not available here) -> report only
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
