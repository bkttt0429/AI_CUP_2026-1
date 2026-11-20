# M51 follow-up: (i) do training players appear under more than one rank label? (ii) within-question heterogeneity (game_argmax_std,
# opp_game_argmax_std, opp_pmi_std) test vs validation within question size, quantiles; (iii) the raw within-size deltas.
import numpy as np, pandas as pd, sys
sys.path.insert(0, "src")
from aicup.fused import FUSED_OPP_DIFF_HD_FEATURE_NAMES as R_NAMES
t = pd.read_csv("results/data/hd_table.csv", dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False, usecols=["game_key", "player_id", "rank"])
per_level = t.groupby("rank")["player_id"].nunique(); distinct = t["player_id"].nunique()
pr = t.drop_duplicates(["player_id", "rank"]).groupby("player_id")["rank"].agg(list)
multi = pr[pr.map(len) > 1]
print(f"players: distinct {distinct}, sum over levels {int(per_level.sum())}; players under >= 2 ranks {len(multi)} ({len(multi) / distinct:.1%})")
print("players per level: " + " ".join(f"{k} {v}" for k, v in per_level.items()))
if len(multi):
    LV = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; idx = {k: i for i, k in enumerate(LV)}
    spans = multi.map(lambda rs: max(idx[r] for r in rs) - min(idx[r] for r in rs))
    print("rank span of multi-rank players: " + " ".join(f"span {s}: {int((spans == s).sum())}" for s in sorted(spans.unique())))
    print("examples: " + "; ".join(f"{p}: {rs}" for p, rs in list(multi.items())[:5]))
    g = t[t["player_id"].isin(multi.index)].groupby(["player_id", "rank"]).size()
    print(f"games of multi-rank players: {int(g.sum())} of {len(t)} ({g.sum() / len(t):.1%})")
test = np.load("results/trunk_seq/es/m51_test_inputs.npz", allow_pickle=False); R_test = test["R"].mean(0); n_test = test["n_games"].astype(int)
import json
m = json.load(open("results/trunk_seq/r_val1_f0_x40.json"))["val"]; n_val = np.array(m["n_games"]); R_val = np.load("results/trunk_seq/r_val1_f0_x40.npz")["X_va"].astype(np.float64)
for col in ("game_argmax_std", "opp_game_argmax_std", "opp_pmi_std", "pmi_sum", "opp_pmi_mean", "peak_interp", "hd_sente", "hd_gap_up", "hd_gap_down"):
    j = R_NAMES.index(col)
    for s in (5, 10, 20):
        a, b = R_test[n_test == s, j], R_val[n_val == s, j]
        qa, qb = np.quantile(a, [0.1, 0.5, 0.9]), np.quantile(b, [0.1, 0.5, 0.9])
        print(f"{col:20s} n={s:2d}: test mean {a.mean():.3f} q10/50/90 {qa[0]:.3f}/{qa[1]:.3f}/{qa[2]:.3f} | val mean {b.mean():.3f} q {qb[0]:.3f}/{qb[1]:.3f}/{qb[2]:.3f} | delta/sd {(a.mean() - b.mean()) / (b.std() + 1e-9):+.2f}")
