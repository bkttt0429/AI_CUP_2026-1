# Why does fused_opp_seq_sum differ from fused_opp? Compare the sum encoder's tables with the L1b tables they replace, and the features.
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/u2507115/BK_AICUP_2026-1")
for d in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(d))
from aicup.metric import LEVELS
from aicup.scoring import read_scored_games
import l1b_cv
where = l1b_cv.locate_folds(["results/l1b/scores3", "results/l1b/scores3_B"], [1, 2, 3, 4])
cols = [f"sum_{lv}" for lv in LEVELS]
for f in (1, 2, 3, 4):
    a = read_scored_games(Path(where[f]) / f"fold{f}")
    b = read_scored_games(Path("results/seq/sum") / f"fold{f}")
    same_order = list(a["game_key"]) == list(b["game_key"])
    m = a.merge(b, on="game_key", suffixes=("_a", "_b"))
    d = m[[c + "_b" for c in cols]].to_numpy(float) - m[[c + "_a" for c in cols]].to_numpy(float)
    print(f"fold {f}: same order {same_order}; games {len(a)} {len(b)} merged {len(m)}; diff abs mean {np.abs(d).mean():.2e} max {np.abs(d).max():.2e} mean {d.mean():+.2e}; "
          f"n_moves equal {(m['n_moves_a'] == m['n_moves_b']).all()}; columns a {list(a.columns)[:6]} b {list(b.columns)[:6]}", flush=True)
