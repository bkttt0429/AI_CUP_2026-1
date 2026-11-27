"""Early look at M25 (not the pre-registered test): on the 40 games of the throughput checks, how much does search change the per-game
quality numbers relative to M23's no-search features of the same games, and how much do repeated searched runs agree (search noise)?
  python -I logs/nano4/ai_search_peek.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
B = {Path(f).stem: pd.read_csv(f, dtype={"game_key": str}).set_index("game_key") for f in sorted(glob.glob(str(ROOT / "results/ai_search/fold1/bench40_*.csv")))}
N = pd.concat([pd.read_csv(f, dtype={"game_key": str}) for f in sorted(glob.glob(str(ROOT / "results/ai_quality/fold1/shard[0-9]*.csv")))]).set_index("game_key")
names = list(B); base = B[names[-1]]; keys = [k for k in base.index if k in N.index]
print(f"runs {names}; {len(keys)} of {len(base)} games also in M23; players {base.loc[keys, 'player_id'].nunique()}, ranks {sorted(set(base.loc[keys, 'rank']))}")
sp = lambda a, b: pd.Series(a).rank().corr(pd.Series(b).rank())  # noqa: E731
for c in ("t_loss_mean", "t_loss_median", "t_gt3", "t_gt8", "t_wloss_mean", "o_loss_mean", "o_gt3"):
    s, n = base.loc[keys, c].to_numpy(float), N.loc[keys, c].to_numpy(float)
    print(f"{c:14s} searched mean {np.nanmean(s):7.3f}  no-search mean {np.nanmean(n):7.3f}  Spearman(search, no-search) {sp(s, n):+.3f}")
s, n = base.loc[keys, "t_s_agree"].to_numpy(float), N.loc[keys, "t_ai_agree"].to_numpy(float)
print(f"{'agree':14s} searched best = played {np.nanmean(s):.3f}  policy top-1 = played {np.nanmean(n):.3f}  Spearman {sp(s, n):+.3f}")
if len(names) > 1:
    for c in ("t_loss_mean", "t_gt3", "t_s_agree"):
        a, b = B[names[0]].loc[keys, c].to_numpy(float), B[names[-1]].loc[keys, c].to_numpy(float)
        print(f"repeat runs {names[0]} vs {names[-1]} {c}: Spearman {sp(a, b):+.3f}, mean |diff| {np.nanmean(np.abs(a - b)):.4f}")
