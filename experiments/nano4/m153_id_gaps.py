"""M153 (2026-11-15, quick look, report only): are the game ids of each rank file complete? Missing ids = games removed after the
numbering (the other copy of a game between two listed players, or held-out players' games)."""
import glob
import sys

import numpy as np
import pandas as pd

for f in sorted(glob.glob("data/training/train_*.csv")):
    d = pd.read_csv(f, usecols=["player_id", "game_id", "rank", "color"], dtype=str); ids = d.game_id.str.replace("g_", "").astype(int).to_numpy(); order_ok = bool((np.diff(ids) > 0).all())
    full = np.zeros(ids.max() + 1, bool); full[ids] = True; miss = np.flatnonzero(~full[1:]) + 1
    runs = np.split(miss, np.flatnonzero(np.diff(miss) > 1) + 1) if len(miss) else []; rl = np.array([len(r) for r in runs]) if runs else np.array([0])
    # players: contiguous blocks of ids?
    first = d.groupby("player_id", sort=False).game_id.agg(["first", "last", "size"]); blk = (first["last"].str.replace("g_", "").astype(int) - first["first"].str.replace("g_", "").astype(int) + 1)
    print(f"{f.split('/')[-1]}: rows {len(d)}, players {d.player_id.nunique()}, ids {ids.min()}..{ids.max()}, ids increasing in file order {order_ok}, missing ids {len(miss)} ({len(miss) / ids.max():.3f}) in {len(runs)} runs; run lengths: max {rl.max()}, quartiles {np.quantile(rl, [0.5, 0.9, 0.99]).round(1).tolist()}, runs >= 5: {int((rl >= 5).sum())}, >= 20: {int((rl >= 20).sum())}; "
          f"players whose ids form one span without foreign ids {(blk == first['size']).mean():.3f}; span minus own games (missing inside a player's span): mean {float((blk - first['size']).mean()):.2f}, share of players with any {(blk > first['size']).mean():.3f}; colour B share {(d.color == 'B').mean():.3f}", flush=True)
    if f.endswith("train_1D.csv"):
        g = d.assign(i=ids).groupby("player_id", sort=False).i.agg(["min", "max", "size"]).reset_index(); print("   first players of 1D (first id, last id, games):", g.head(8).values.tolist())
        # the gaps between consecutive players' spans
        gaps = g["min"].to_numpy()[1:] - g["max"].to_numpy()[:-1] - 1; print(f"   gaps between consecutive players' spans: zero in {(gaps == 0).mean():.3f}, quartiles of the non-zero ones {np.quantile(gaps[gaps > 0], [0.25, 0.5, 0.75, 0.95]).tolist() if (gaps > 0).any() else '-'}; negative (interleaved) {(gaps < 0).mean():.3f}")
