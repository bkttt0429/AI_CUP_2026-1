"""Section 4M M64 (2026-11-09): is the persistent (player-level) bias aligned with the position of the player's games in the level file?
A3 found the later 10 games of a player are estimated 0.088 levels lower than the earlier 10 (game_id order carries something). If
the files are time-ordered and the server's rank scale drifts over time, a player's mean file position would predict the persistent
bias (e0+e1)/2 from floor_decomp. Diagnostic only: test questions carry no ids, so nothing here can be used for prediction.
  python -I logs/nano4/m64_time_axis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

fd = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str})
fd["bias"] = (fd.e0 + fd.e1) / 2
rows = []
for lv in LEVELS:
    df = pd.read_csv(ROOT / f"data/training/train_{lv}.csv", usecols=["player_id", "game_id"], dtype=str)
    df["pos"] = df.game_id.str.replace("g_", "", regex=False).astype(int) / len(df)
    g = df.groupby("player_id").pos
    agg = pd.DataFrame({"pos_mean": g.mean(), "pos_sd": g.std(), "pos_min": g.min(), "pos_max": g.max(), "n_games": g.size()})
    agg["level"] = lv; rows.append(agg.reset_index())
    # contiguity: are a player's games adjacent in the file?  median gap between consecutive game ids of the same player, in rows
    df["idx"] = df.game_id.str.replace("g_", "", regex=False).astype(int)
    gaps = df.sort_values("idx").groupby("player_id").idx.diff().dropna()
    print(f"{lv}: {len(df)} games, {df.player_id.nunique()} players; per-player pos sd mean {agg.pos_sd.mean():.3f} (uniform-random would be ~0.29); "
          f"consecutive-game gap median {gaps.median():.0f} rows, share of gap==1 {float((gaps == 1).mean()):.3f}")
pos = pd.concat(rows)
m = fd.merge(pos, on="player_id", how="inner")
print(f"merged {len(m)} of {len(fd)} validation players")
out = {}
r_all = spearmanr(m.pos_mean, m.bias); print(f"ALL: Spearman(bias, mean file position) {r_all.statistic:+.3f} p {r_all.pvalue:.2g}")
out["all"] = {"rho": float(r_all.statistic), "p": float(r_all.pvalue)}
# within level (the bias has a label-dependent shrinkage component, A2, so compare within level)
for lv in LEVELS:
    s = m[m.level == lv]
    r = spearmanr(s.pos_mean, s.bias); r20 = spearmanr(s.pos_mean, s.e20)
    dec = pd.qcut(s.pos_mean, 5, labels=False, duplicates="drop"); prof = s.groupby(dec).bias.mean()
    print(f"  {lv:3s} n {len(s):4d} rho(bias, pos) {r.statistic:+.3f} p {r.pvalue:.2g} | rho(e20, pos) {r20.statistic:+.3f} | bias by position quintile " + " ".join(f"{v:+.2f}" for v in prof))
    out[lv] = {"n": int(len(s)), "rho_bias": float(r.statistic), "p": float(r.pvalue), "rho_e20": float(r20.statistic), "quintile_bias": [float(v) for v in prof]}
# pooled within-level: rank bias within level, then Spearman
m["bias_rk"] = m.groupby("level").bias.rank(pct=True); m["pos_rk"] = m.groupby("level").pos_mean.rank(pct=True)
rw = spearmanr(m.pos_rk, m.bias_rk); print(f"pooled within-level Spearman {rw.statistic:+.3f} p {rw.pvalue:.2g}; R^2 of a per-level linear fit of bias on pos: "
      f"{np.mean([np.corrcoef(m[m.level==lv].pos_mean, m[m.level==lv].bias)[0,1]**2 for lv in LEVELS]):.4f}")
out["pooled_within_level"] = {"rho": float(rw.statistic), "p": float(rw.pvalue)}
# does the player's own position spread (how far apart their games are in the file) relate to |bias|?
rs = spearmanr(m.pos_sd.fillna(0), m.bias.abs()); print(f"Spearman(|bias|, player's position spread) {rs.statistic:+.3f} p {rs.pvalue:.2g}")
out["abs_bias_vs_spread"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
Path(ROOT / "results/r_variants/m64_time_axis.json").write_text(json.dumps(out, indent=1)); print("written")
