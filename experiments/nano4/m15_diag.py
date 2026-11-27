"""Section 4M M15 diagnostics (2026-11-05; user: 繼續研究還有潛力、值得挖的訊號), folds 1-4, every scored player's 20 games:
D1 profile clipping: share of games whose best profile is an edge (12k or 6d) for the target's and for the opponent's moves, by class
D2 colour in even games (HD = 0): P(target plays W) by class; within-class mean of (opponent pmi - own pmi) for W vs B, pooled Spearman;
   HD = 1 as the sanity check of the sign convention (W = the upper hand must have own - opponent > 0)
D3 HD as a per-game correction: over HD >= 1 games, Spearman and slope of (own pmi - opponent pmi) on the signed gap; within-player sd of
   the opponent pmi vs of the implied rank (opponent pmi + gap) for players with >= 5 HD games
D4 persistent bias (E[rank] - label on the n = 20 pilot question) of the x40 ensemble vs the oh ensemble (C/T seeds 0-2): sd, their
   correlation, and each one's correlation with the question's mean (opponent pmi - own pmi), mean signed gap and even-game W share

  python -I logs/nano4/m15_diag.py --out results/r_variants/m15_diag.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aicup.evidence import posterior_mean_index  # noqa: E402
from aicup.handicap import read_hd_table, signed_gap  # noqa: E402
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
import trunk_seq_check as ts  # noqa: E402

COLS = [f"sum_{p}" for p in PROFILES]
RANK = {lv: i for i, lv in enumerate(LEVELS)}
ES = ROOT / "results/trunk_seq/es"


def spearman(a, b):
    a, b = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def games_table(f, hd):
    hsl = read_scored_games(ts.HSL / f"fold{f}")
    oh = read_scored_games(ts.OPP_HSL / f"fold{f}")
    g = hsl[["game_key", "player_id", "rank", "color", "n_moves", *COLS]].merge(
        oh[["game_key", *COLS]].rename(columns={c: c + "_o" for c in COLS}), on="game_key")
    own, opp = g[COLS].to_numpy(dtype=float), g[[c + "_o" for c in COLS]].to_numpy(dtype=float)
    g["own_pmi"], g["opp_pmi"] = posterior_mean_index(own), posterior_mean_index(opp)
    g["own_arg"], g["opp_arg"] = own.argmax(axis=1), opp.argmax(axis=1)
    g["hd"] = g["game_key"].map(hd)
    g["gap"] = signed_gap(g["hd"].to_numpy(), g["color"].to_numpy())
    g["y"] = g["rank"].map(RANK)
    g["fold"] = f
    return g


def ensemble_bias(v, arms, seeds=(0, 1, 2)):
    m = pd.DataFrame(json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"])
    P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in arms for s in seeds], axis=0)
    E = P @ np.arange(len(LEVELS))
    m["E"] = E
    m["y"] = m["rank"].map(RANK)
    m["bias"] = m["E"] - m["y"]
    return m[m["n_games"] == 20].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hd = read_hd_table(ROOT / "results/data/hd_table.csv").drop_duplicates("game_key").set_index("game_key")["hd"]
    G = pd.concat([games_table(f, hd) for f in a.folds], ignore_index=True)
    print(f"{len(G)} games, {G['player_id'].nunique()} players; HD known {G['hd'].notna().mean():.3f}, HD>=1 share {(G['hd'] >= 1).mean():.3f}", flush=True)
    rep = {"games": int(len(G))}
    # D1 clipping
    d1 = {}
    for lv in LEVELS:
        s = G[G["rank"] == lv]
        d1[lv] = {"own_edge_low": float((s["own_arg"] == 0).mean()), "own_edge_high": float((s["own_arg"] == len(PROFILES) - 1).mean()),
                  "opp_edge_low": float((s["opp_arg"] == 0).mean()), "opp_edge_high": float((s["opp_arg"] == len(PROFILES) - 1).mean()),
                  "own_pmi": float(s["own_pmi"].mean()), "opp_pmi": float(s["opp_pmi"].mean()), "opp_pmi_gt_16_5": float((s["opp_pmi"] > 16.5).mean()),
                  "opp_pmi_lt_0_5": float((s["opp_pmi"] < 0.5).mean()), "games": int(len(s))}
    rep["D1_clipping"] = d1
    print("D1 clipping (share of games with the best profile at the 12k / 6d edge; own | opponent), mean pmi own | opp:")
    for lv in LEVELS:
        x = d1[lv]
        print(f"  {lv:2s}: own {x['own_edge_low']:.3f}/{x['own_edge_high']:.3f}  opp {x['opp_edge_low']:.3f}/{x['opp_edge_high']:.3f}  pmi {x['own_pmi']:.2f} | {x['opp_pmi']:.2f}  opp pmi>16.5 {x['opp_pmi_gt_16_5']:.3f}")
    # D2 colour
    G["d"] = G["opp_pmi"] - G["own_pmi"]
    d2 = {}
    for name, mask in (("even", G["hd"] == 0), ("hd1", G["hd"] == 1)):
        s = G[mask].copy()
        s["d_dm"] = s["d"] - s.groupby("rank")["d"].transform("mean")
        isw = (s["color"] == "W").to_numpy(dtype=float)
        by = {lv: {"p_w": float((t["color"] == "W").mean()), "d_W": float(t.loc[t["color"] == "W", "d"].mean()), "d_B": float(t.loc[t["color"] == "B", "d"].mean()), "games": int(len(t))}
              for lv, t in s.groupby("rank")}
        d2[name] = {"games": int(len(s)), "p_w": float(isw.mean()), "spearman_isW_vs_oppminusown_within_class": spearman(isw, s["d_dm"].to_numpy()),
                    "mean_d_W_minus_B_within_class": float(s.loc[s["color"] == "W", "d_dm"].mean() - s.loc[s["color"] == "B", "d_dm"].mean()), "by_class": by}
        print(f"D2 {name}: {len(s)} games, P(W) {isw.mean():.3f}; within-class Spearman(isW, opp-own) {d2[name]['spearman_isW_vs_oppminusown_within_class']:+.3f}; "
              f"mean(opp-own | W) - mean(opp-own | B) within class {d2[name]['mean_d_W_minus_B_within_class']:+.3f}")
        print("   by class P(W): " + " ".join(f"{lv}:{by[lv]['p_w']:.2f}" for lv in LEVELS if lv in by))
    rep["D2_colour"] = d2
    # D3 HD as a per-game correction
    h = G[G["hd"] >= 1]
    sl = np.polyfit(h["gap"].to_numpy(), (-h["d"]).to_numpy(), 1)[0] if len(h) > 10 else float("nan")
    d3 = {"hd_games": int(len(h)), "spearman_gap_vs_ownminusopp": spearman(h["gap"].to_numpy(), (-h["d"]).to_numpy()), "slope_ownminusopp_on_gap": float(sl),
          "mean_ownminusopp_gap_pos": float((-h.loc[h["gap"] > 0, "d"]).mean()), "mean_ownminusopp_gap_neg": float((-h.loc[h["gap"] < 0, "d"]).mean())}
    cnt = G.groupby("player_id")["hd"].apply(lambda x: int((x >= 1).sum()))
    many = cnt[cnt >= 5].index
    sub = G[G["player_id"].isin(many)]
    sd_opp = sub.groupby("player_id")["opp_pmi"].std()
    sd_imp = sub.assign(imp=sub["opp_pmi"] + sub["gap"]).groupby("player_id")["imp"].std()
    d3.update({"players_ge5_hd": int(len(many)), "within_player_sd_opp_pmi": float(sd_opp.mean()), "within_player_sd_implied": float(sd_imp.mean()),
               "share_players_sd_reduced": float((sd_imp < sd_opp).mean())})
    rep["D3_hd_correction"] = d3
    print(f"D3: {len(h)} HD>=1 games; Spearman(gap, own-opp) {d3['spearman_gap_vs_ownminusopp']:+.3f}, slope {sl:+.3f} profiles per rank of gap; "
          f"own-opp | gap>0 {d3['mean_ownminusopp_gap_pos']:+.2f}, | gap<0 {d3['mean_ownminusopp_gap_neg']:+.2f}; "
          f"{len(many)} players with >=5 HD games: within-player sd opp pmi {d3['within_player_sd_opp_pmi']:.3f} vs implied {d3['within_player_sd_implied']:.3f} "
          f"(reduced for {d3['share_players_sd_reduced']:.2f})")
    # D4 bias of the two ensembles
    per_game = G.set_index("game_key")
    rows = []
    for v in a.folds:
        bx = ensemble_bias(v, ["C-x40", "T-x40"])
        bo = ensemble_bias(v, ["C-x40-oh", "T-x40-oh"])
        assert (bx["player_id"] == bo["player_id"]).all()
        feats = []
        for keys in bx["game_keys"]:
            s = per_game.loc[list(keys)]
            even = s[s["hd"] == 0]
            feats.append((float(s["d"].mean()), float(s["gap"].mean()), float((even["color"] == "W").mean()) if len(even) else 0.5, float(s["own_pmi"].mean())))
        F = np.array(feats)
        rows.append(pd.DataFrame({"fold": v, "player_id": bx["player_id"], "y": bx["y"], "bias_x40": bx["bias"], "bias_oh": bo["bias"],
                                  "oppminusown": F[:, 0], "gap_mean": F[:, 1], "w_share_even": F[:, 2], "own_pmi": F[:, 3]}))
    B = pd.concat(rows, ignore_index=True)
    d4 = {"players": int(len(B))}
    for k in ("bias_x40", "bias_oh"):
        d4[k] = {"sd": float(B[k].std()), "mean_abs": float(B[k].abs().mean()),
                 "spearman_vs_oppminusown": spearman(B[k], B["oppminusown"]), "spearman_vs_gap_mean": spearman(B[k], B["gap_mean"]),
                 "spearman_vs_w_share_even": spearman(B[k], B["w_share_even"]),
                 "spearman_vs_oppminusown_within_class": spearman(B[k] - B.groupby("y")[k].transform("mean"), B["oppminusown"] - B.groupby("y")["oppminusown"].transform("mean"))}
    d4["corr_bias_x40_oh"] = float(np.corrcoef(B["bias_x40"], B["bias_oh"])[0, 1])
    d4["sd_ratio_oh_over_x40"] = float(B["bias_oh"].std() / B["bias_x40"].std())
    rep["D4_bias"] = d4
    print(f"D4: {len(B)} players (n=20): bias sd x40 {d4['bias_x40']['sd']:.3f} -> oh {d4['bias_oh']['sd']:.3f} (ratio {d4['sd_ratio_oh_over_x40']:.3f}), corr(bias_x40, bias_oh) {d4['corr_bias_x40_oh']:.3f}")
    for k in ("bias_x40", "bias_oh"):
        x = d4[k]
        print(f"   {k}: Spearman vs opp-own {x['spearman_vs_oppminusown']:+.3f} (within class {x['spearman_vs_oppminusown_within_class']:+.3f}), vs mean gap {x['spearman_vs_gap_mean']:+.3f}, vs even-game W share {x['spearman_vs_w_share_even']:+.3f}")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    B.to_csv(Path(a.out).with_suffix(".players.csv"), index=False)
    print("written", a.out)


if __name__ == "__main__":
    main()
