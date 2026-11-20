"""Section 4M M70 step 1 (2026-11-10): an era marker read from the moves. Early 3-3 invasion = within the first 30 plies of an HD[0]
game, a move on a 3-3 point whose adjacent 4-4 point holds an opponent stone while the mover has no stone in that 7x7 corner (captures
ignored: none this early). Rare in amateur play before 2017, common after AlphaGo. Reports (design doc M70):
 (1) rate by level and by who played it; (2) by decile of the position in the level file; (3) per-player rate (>= 30 HD0 games):
 quantiles, split-half correlation against the binomial-noise expectation; (4) test Public / Private vs training;
 (5) within-level Spearman with the persistent bias (floor_decomp players.csv), target-side / opponent-side / either.
  python -I logs/nano4/m70_era_pilot.py
"""
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src")]
from aicup.metric import LEVELS  # noqa: E402

MOVE = re.compile(r";([BW])\[([a-s])([a-s])\]")
CORNERS = {(2, 2): (3, 3), (2, 16): (3, 15), (16, 2): (15, 3), (16, 16): (15, 15)}
T = 30


def flags(sgf, color):
    """(either, by target, by opponent, ply of the first one or -1, first move of the game is a 4-4) for one game."""
    stones = {}; tgt = opp = 0; first = -1
    ms = MOVE.findall(sgf, 0, 4000)[:T]
    for t, (c, a, b) in enumerate(ms):
        x, y = ord(a) - 97, ord(b) - 97
        if (x, y) in CORNERS:
            sx, sy = CORNERS[(x, y)]
            other = "W" if c == "B" else "B"
            if stones.get((sx, sy)) == other:
                cx, cy = (0 if x < 9 else 12), (0 if y < 9 else 12)
                if not any(col == c and cx <= px <= cx + 6 and cy <= py <= cy + 6 for (px, py), col in stones.items()):
                    if first < 0:
                        first = t
                    if c == color:
                        tgt = 1
                    else:
                        opp = 1
        stones[(x, y)] = c
    star = 0
    if ms:
        x, y = ord(ms[0][1]) - 97, ord(ms[0][2]) - 97
        star = int((x, y) in CORNERS.values())
    return int(tgt or opp), tgt, opp, first, star


def one_level(lv):
    df = pd.read_csv(ROOT / f"data/training/train_{lv}.csv", dtype=str)
    df = df[df.sgf_content.str.contains("HD[0]", regex=False)].copy()
    f = np.array([flags(s, c) for s, c in zip(df.sgf_content, df.color)])
    out = df[["player_id", "game_id"]].copy(); out["level"] = lv
    out["any"], out["tgt"], out["opp"], out["first"], out["star"] = f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4]
    out["pos"] = out.game_id.str.replace("g_", "", regex=False).astype(int) / 100000.0
    return out


def main():
    with mp.Pool(10) as pool:
        G = pd.concat(pool.map(one_level, LEVELS), ignore_index=True)
    rep = {"hd0_games": int(len(G)), "rate_any": float(G["any"].mean())}
    print(f"{len(G)} HD0 training games; early 3-3 invasion in {G['any'].mean():.4f} of games (target side {G.tgt.mean():.4f}, opponent side {G.opp.mean():.4f}); first move a 4-4: {G.star.mean():.3f}")
    print("(1) by level: " + " ".join(f"{lv} {G[G.level == lv]['any'].mean():.3f}" for lv in LEVELS))
    rep["by_level"] = {lv: float(G[G.level == lv]["any"].mean()) for lv in LEVELS}
    print("(2) by decile of file position (all levels pooled): " + " ".join(f"{v:.3f}" for v in G.groupby(pd.cut(G.pos, 10, labels=False))["any"].mean()))
    for lv in ("D", "A", "3D", "6D"):
        s = G[G.level == lv]; print(f"    {lv}: " + " ".join(f"{v:.3f}" for v in s.groupby(pd.cut(s.pos, 10, labels=False))["any"].mean()))
    # (3) per-player heterogeneity
    rng = np.random.default_rng(20261110)
    G["half"] = rng.integers(0, 2, len(G))
    pp = G.groupby("player_id").agg(n=("any", "size"), rate=("any", "mean"), tgt=("tgt", "mean"), opp=("opp", "mean"), level=("level", "first"), pos=("pos", "mean"))
    big = pp[pp.n >= 30]
    q = big.rate.quantile([0, .1, .25, .5, .75, .9, .99, 1]).round(3).to_dict()
    h = G[G.player_id.isin(big.index)].groupby(["player_id", "half"])["any"].mean().unstack()
    r_half = float(h.corr().iloc[0, 1])
    # binomial-noise expectation of the split-half r if every player of a level shared that level's rate
    sim = G[G.player_id.isin(big.index)].copy(); sim["p"] = sim.level.map(rep["by_level"]); sim["fake"] = (rng.random(len(sim)) < sim.p).astype(int)
    hs = sim.groupby(["player_id", "half"])["fake"].mean().unstack(); r_null = float(hs.corr().iloc[0, 1])
    hw = G[G.player_id.isin(big.index)].copy(); hw["lvl_mean"] = hw.groupby("level")["any"].transform("mean"); hw["c"] = hw["any"] - hw.lvl_mean
    hh = hw.groupby(["player_id", "half"])["c"].mean().unstack(); r_within = float(hh.corr().iloc[0, 1])
    print(f"(3) players with >= 30 HD0 games: {len(big)}; per-player rate quantiles {q}; share of players with rate 0: {(big.rate == 0).mean():.3f}, > 0.2: {(big.rate > 0.2).mean():.3f}")
    print(f"    split-half r of the per-player rate {r_half:.3f} (level-centred {r_within:.3f}); if all players of a level shared one rate: {r_null:.3f}")
    rep["per_player"] = {"players": int(len(big)), "quantiles": q, "split_half_r": r_half, "split_half_r_level_centred": r_within, "null_r": r_null}
    # opponent-side vs target-side rates of the same player (does the player's ENVIRONMENT share the habit?)
    rr = spearmanr(big.tgt, big.opp); print(f"    per-player target-side vs opponent-side rate: Spearman {rr.statistic:+.3f}")
    rep["per_player"]["tgt_vs_opp_rho"] = float(rr.statistic)
    # (4) test
    for half in ("public", "private"):
        t = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{half}.csv", dtype=str)
        vals = []
        for _, row in t.iterrows():
            for k in range(1, int(row.num_games) + 1):
                s = row.get(f"sgf_{k}")
                if isinstance(s, str) and "HD[0]" in s:
                    vals.append(flags(s, row.get(f"color_{k}", "B")))
        v = np.array(vals); print(f"(4) test {half}: {len(v)} HD0 games, early 3-3 invasion {v[:, 0].mean():.4f} (target {v[:, 1].mean():.4f}, opponent {v[:, 2].mean():.4f}); first move 4-4 {v[:, 4].mean():.3f}")
        rep[f"test_{half}"] = {"games": int(len(v)), "rate_any": float(v[:, 0].mean()), "star": float(v[:, 4].mean())}
    print(f"    training reference with equal weight per level: {np.mean(list(rep['by_level'].values())):.4f}")
    # (5) persistent bias
    fd = pd.read_csv(ROOT / "results/r_variants/floor_decomp.players.csv", dtype={"player_id": str}); fd["bias"] = (fd.e0 + fd.e1) / 2
    m = fd.merge(pp.reset_index(), on="player_id"); m = m[m.n >= 20]
    print(f"(5) {len(m)} validation players with >= 20 HD0 games; within-level Spearman of the per-player rate with the persistent bias:")
    rep["bias"] = {}
    for col in ("rate", "tgt", "opp"):
        m["a"] = m.groupby("level")[col].rank(pct=True); m["b"] = m.groupby("level").bias.rank(pct=True)
        r = spearmanr(m.a, m.b); by = {lv: float(spearmanr(m[m.level == lv][col], m[m.level == lv].bias).statistic) for lv in LEVELS}
        print(f"    {col:5s} pooled within-level rho {r.statistic:+.3f} (p {r.pvalue:.2g}); by level " + " ".join(f"{lv} {v:+.2f}" for lv, v in by.items()))
        rep["bias"][col] = {"rho": float(r.statistic), "p": float(r.pvalue), "by_level": by}
    r = spearmanr(m.rate, m.e20); print(f"    (rate vs raw e20, not level-centred: {r.statistic:+.3f}; rate vs label index: {spearmanr(m.rate, m.y).statistic:+.3f})")
    Path(ROOT / "results/r_variants/m70_era_pilot.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")


if __name__ == "__main__":
    main()
