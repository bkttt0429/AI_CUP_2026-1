#!/usr/bin/env python
"""Section 4M M160 (2026-11-15, report only): held-out identification accuracy by the player's activity (design doc M160).

M111 took the per-player held-out top-1 (one game of each player) as the recognition rate of opponents and got "70% of the
opponents are labelled players, 40% of them recognised". Opponents appear in proportion to their activity, and the identity
networks draw players uniformly (256 players x 8 games per episode): a busy player's prototype is sharp and his games are seen
rarely (not memorised). All-players networks (seed 7 = 40,000 steps; seeds 0, 1, 3 = 12,000 steps; the four averaged) on the
3,244 held-out own games: top-1 / top-5 by the player's number of training games, each band's share of the training rows, the
row-weighted top-1 (= the expected recognition rate of an opponent drawn by activity) and the implied share of labelled opponents.

  python -I logs/nano4/m160_top1_by_activity.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd(); RV = ROOT / "results/r_variants"; BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; MASS = {"SP2e~m15": 0.2258, "SPAL~m15": 0.2460}


def main():
    q = np.load(ROOT / "results/m84_cache/queries.npz"); players = q["players"].astype(str); C = len(players); Q, H = int(q["Q"]), int(q["H"]); truth = q["held_truth"].astype(int)
    pf = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id"); ng = pf.n_games.astype(int).reindex(players).to_numpy(); rep = {}
    share_rows = np.array([ng[(ng >= lo) & (ng < hi)].sum() for lo, hi in BANDS]) / ng.sum(); share_pl = np.array([((ng >= lo) & (ng < hi)).mean() for lo, hi in BANDS]); band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in ng[truth]])
    print(f"players {C}, training rows {int(ng.sum())}; held-out own games {H} of {len(np.unique(truth))} players; their players' games: quartiles " + " / ".join(f"{x:.0f}" for x in np.quantile(ng[truth], [0.1, 0.25, 0.5, 0.75, 0.9])))
    print("band (training games of the player): share of the players | share of the training rows | held-out games in the band")
    for k, (lo, hi) in enumerate(BANDS):
        print(f"   {lo:5d}-{'' if hi > 10 ** 8 else hi - 1:<5}: {share_pl[k]:.3f} | {share_rows[k]:.3f} | {int((band == k).sum())}")
    S_all = {}
    for name, tag in (("seed 7, 40,000 steps", "_s7"), ("seed 0, 12,000 steps", ""), ("seed 1", "_s1"), ("seed 3", "_s3")):
        f = ROOT / f"results/m120/out_-1{tag}.npz"
        if not f.exists():
            print(f"({f.name} missing)"); continue
        z = np.load(f); pos = {int(g): i for i, g in enumerate(z["qidx"])}; ix = np.array([pos[2 * Q + i] for i in range(H)]); S_all[name] = z["qvec"][ix].astype(np.float64) @ z["protos"].astype(np.float64).T
    if len(S_all) > 1:
        S_all["average of the " + str(len(S_all))] = sum(S_all.values()) / len(S_all)
    for name, S in S_all.items():
        r = (S > S[np.arange(H), truth][:, None]).sum(1); t1 = np.array([float((r[band == k] == 0).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))]); t5 = np.array([float((r[band == k] < 5).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))])
        ok = np.isfinite(t1); w1 = float((share_rows[ok] * t1[ok]).sum() / share_rows[ok].sum()); w5 = float((share_rows[ok] * t5[ok]).sum() / share_rows[ok].sum())
        # a finer estimate: logistic curve of top-1 on log(games), applied to every player with row weights
        x = np.log(ng[truth]); X = np.column_stack([np.ones(H), x, x * x]); y = (r == 0).astype(float); w = np.zeros(3)
        for _ in range(50):
            p = 1 / (1 + np.exp(-X @ w)); w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(3), X.T @ (p - y))
        xa = np.log(np.maximum(ng, 1)); pa = 1 / (1 + np.exp(-np.column_stack([np.ones(C), xa, xa * xa]) @ w)); wfit = float((pa * ng).sum() / ng.sum())
        rep[name] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top1_by_band": t1.tolist(), "top5_by_band": t5.tolist(), "row_weighted_top1": w1, "row_weighted_top5": w5, "row_weighted_top1_fit": wfit}
        print(f"\n{name}: held-out top-1 {rep[name]['top1']:.3f}, top-5 {rep[name]['top5']:.3f} (one game per player, unweighted)")
        print("   top-1 by band: " + "  ".join(f"{lo}+ {t1[k]:.3f}" for k, (lo, hi) in enumerate(BANDS)) + " | top-5: " + "  ".join(f"{t5[k]:.3f}" for k in range(len(BANDS))))
        print(f"   weighted by the training rows: top-1 {w1:.3f} (bands) / {wfit:.3f} (logistic curve on log games), top-5 {w5:.3f}")
        print("   implied share of labelled opponents = recognised share / weighted top-1: " + ", ".join(f"{k} {v:.4f} -> {v / w1:.2f} (bands) / {v / wfit:.2f} (curve)" for k, v in MASS.items()) + "   [the recognised shares are those of the FUSED identifiers; a single network's weighted top-1 is lower than the fused one, so these shares are upper bounds]")
    (RV / "m160_top1_by_activity.json").write_text(json.dumps({"bands": BANDS, "share_rows": share_rows.tolist(), "share_players": share_pl.tolist(), "nets": rep}, indent=1)); print("\nwritten results/r_variants/m160_top1_by_activity.json")


if __name__ == "__main__":
    main()
