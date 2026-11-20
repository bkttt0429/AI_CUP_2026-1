"""Can the data tell whether the label is a per-player MODE / snapshot (rank varies across the player's games) or the rank at game time
(constant)? Folds 1-4 players with 80 scored games, L1b per-game expected ranks, opponents calibrated against the own side's shrinkage
(per-class mean of own estimates by label, inverted by interpolation). Dan players (1D-5D: one class = one server rank).
T1  HD[1] anchors: a constant-rank player meets r-1 as the upper hand (anchor = opp + 1 = r) and r+1 as the lower hand (anchor = opp - 1 = r);
    a player whose rank moved between r and r+1 meets r when he is r+1 (upper, anchor r+1) and r+1 when he is r (lower, anchor r): so
    mean(anchor_upper - anchor_lower) is 0 under constant rank and > 0 (the hoverer share) under mode / snapshot labels; even-game anchors
    lie in between. Player bootstrap CIs.
T2  within-player sd of the per-game anchor vs |E - y| and vs |anchor mean - y| (a rank mixture inflates the anchor sd of the biased players).
T3  game_id order: anchor(late 40) - anchor(early 40) and whether |anchor - y| shrinks late (snapshot at extraction => later games closer).
  python -I logs/nano4/label_mode_test.py --out results/r_variants/label_mode_test.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.handicap import read_hd_table  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from anchor_kalman import per_game  # noqa: E402
from pilot_stats import spearman  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}


def boot_mean(x, rng, B=1000):
    x = np.asarray(x, float); v = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return float(x.mean()), [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    G = pd.concat([per_game(f, hd) for f in (1, 2, 3, 4)], ignore_index=True)
    cnt = G.groupby("player_id").size(); G = G[G.player_id.isin(cnt[cnt >= 80].index)].copy()
    G["gid"] = G.game_key.str.split("g_").str[1].astype(int)
    # calibration of the L1b expected rank against the label, from the own side
    m = G.groupby("y")["e_own"].mean().sort_index(); xs, ys = m.to_numpy(), m.index.to_numpy(float)
    cal = lambda e: np.interp(e, xs, ys, left=ys[0] + (e - xs[0]) if False else None) if False else np.interp(e, xs, ys)  # noqa: E731
    G["opp_cal"] = cal(G.e_opp.to_numpy()); G["own_cal"] = cal(G.e_own.to_numpy())
    print("own-side calibration map (label -> mean own estimate):", {LEVELS[int(k)]: round(v, 2) for k, v in m.items()})
    rng = np.random.default_rng(0); rep = {"players": int(G.player_id.nunique())}
    Es = []
    for v in (1, 2, 3, 4):
        meta = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
        P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val{v}/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-x40-oh", "T-x40-oh") for s in (0, 1, 2)], axis=0)
        meta["E"] = P @ np.arange(10); Es.append(meta[meta.n_games == 20].drop_duplicates("player_id")[["player_id", "E"]])
    E = pd.concat(Es).set_index("player_id")["E"]
    # T1: dan players
    D = G[G.y.between(RANK["1D"], RANK["5D"])]
    rows = []
    for pid, s in D.groupby("player_id"):
        even = s[s.hd == 0]; up = s[(s.hd == 1) & (s.color == "W")]; lo = s[(s.hd == 1) & (s.color == "B")]
        rows.append({"player_id": pid, "y": s.y.iat[0], "n_even": len(even), "n_up": len(up), "n_lo": len(lo),
                     "a_even": even.opp_cal.mean() if len(even) else np.nan, "a_up": up.opp_cal.mean() + 1 if len(up) else np.nan, "a_lo": lo.opp_cal.mean() - 1 if len(lo) else np.nan,
                     "own": s.own_cal.mean(), "anchor_sd": (s.opp_cal + np.where(s.hd == 0, 0, np.where(s.color == "W", 1, -1) * s.hd)).std(),
                     "own_sd": s.own_cal.std(), "anchor_mean": (s.opp_cal + np.where(s.hd == 0, 0, np.where(s.color == "W", 1, -1) * s.hd)).mean()})
    T = pd.DataFrame(rows); T["E"] = E.reindex(T.player_id).to_numpy(); T["bias"] = T.E - T.y
    both = T[(T.n_up >= 2) & (T.n_lo >= 2)]
    rep["T1"] = {"dan_players": int(len(T)), "players_with_both_hd1_sides": int(len(both)),
                 "upper_minus_lower": boot_mean((both.a_up - both.a_lo).to_numpy(), rng), "upper_minus_even": boot_mean((both.a_up - both.a_even).dropna().to_numpy(), rng),
                 "even_minus_lower": boot_mean((both.a_even - both.a_lo).dropna().to_numpy(), rng),
                 "upper_minus_label": boot_mean((T[T.n_up >= 2].a_up - T[T.n_up >= 2].y).to_numpy(), rng), "lower_minus_label": boot_mean((T[T.n_lo >= 2].a_lo - T[T.n_lo >= 2].y).to_numpy(), rng),
                 "even_minus_label": boot_mean((T[T.n_even >= 5].a_even - T[T.n_even >= 5].y).to_numpy(), rng), "own_minus_label": boot_mean((T.own - T.y).to_numpy(), rng),
                 "by_label_upper_minus_lower": {LEVELS[int(c)]: boot_mean((g.a_up - g.a_lo).to_numpy(), rng) for c, g in both.groupby("y") if len(g) >= 20}}
    t = rep["T1"]
    print(f"T1 dan players {t['dan_players']}, with >= 2 HD1 games on each side {t['players_with_both_hd1_sides']}: anchor_upper - anchor_lower = {t['upper_minus_lower'][0]:+.3f} {t['upper_minus_lower'][1]}  "
          f"(upper - even {t['upper_minus_even'][0]:+.3f}, even - lower {t['even_minus_lower'][0]:+.3f}); vs label: upper {t['upper_minus_label'][0]:+.3f}, even {t['even_minus_label'][0]:+.3f}, lower {t['lower_minus_label'][0]:+.3f}, own {t['own_minus_label'][0]:+.3f}")
    print("   by label upper - lower: " + " ".join(f"{k}:{v[0]:+.2f}[{v[1][0]:+.2f},{v[1][1]:+.2f}]" for k, v in t["by_label_upper_minus_lower"].items()))
    # T2
    ok = T.E.notna()
    rep["T2"] = {"spearman_anchor_sd_vs_abs_bias": spearman(T.anchor_sd[ok], T.bias[ok].abs()), "spearman_own_sd_vs_abs_bias": spearman(T.own_sd[ok], T.bias[ok].abs()),
                 "spearman_anchor_sd_vs_abs_anchor_err": spearman(T.anchor_sd, (T.anchor_mean - T.y).abs()), "spearman_anchor_sd_vs_abs_own_err": spearman(T.anchor_sd, (T.own - T.y).abs()),
                 "spearman_own_sd_vs_abs_own_err": spearman(T.own_sd, (T.own - T.y).abs())}
    print("T2 within-player sd vs |error|: " + "  ".join(f"{k} {v:+.3f}" for k, v in rep["T2"].items()))
    # T3 game-id order (all 80-game players, all classes)
    rows3 = []
    for pid, s in G.groupby("player_id"):
        s = s.sort_values("gid"); h = len(s) // 2
        anc = s.opp_cal + np.where(s.hd == 0, 0, np.where(s.color == "W", 1, -1) * s.hd); y = s.y.iat[0]
        rows3.append({"late_minus_early_anchor": anc.iloc[h:].mean() - anc.iloc[:h].mean(), "late_minus_early_own": s.own_cal.iloc[h:].mean() - s.own_cal.iloc[:h].mean(),
                      "abs_err_late_minus_early_anchor": abs(anc.iloc[h:].mean() - y) - abs(anc.iloc[:h].mean() - y), "rho_anchor_gid": spearman(np.arange(len(s)), anc.to_numpy())})
    R3 = pd.DataFrame(rows3)
    rep["T3"] = {k: boot_mean(R3[k].dropna().to_numpy(), rng) for k in R3.columns}
    print("T3 game-id order (late 40 - early 40): " + "  ".join(f"{k} {v[0]:+.3f} [{v[1][0]:+.3f}, {v[1][1]:+.3f}]" for k, v in rep["T3"].items()))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
