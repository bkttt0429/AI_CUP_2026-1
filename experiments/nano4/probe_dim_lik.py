"""Is the extra rank information of wider trunk projections additive to the likelihood features?  Fold 1 players: per-game HSL
18-profile mean log p + L1b 10-head mean log p (28 cols) with trunk means projected to 0 / 64 / 128 / 256 dims; player-level
logistic probe (mean over the 20 games), class-balanced official score, paired bootstrap over players.
  python -I logs/nano4/probe_dim_lik.py --root results/probe_dim/fold1 --fold 1 --out results/r_variants/probe_dim_lik.json
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
from aicup.humansl import PROFILES  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.qfeatures import sum_cols  # noqa: E402
from aicup.scoring import read_scored_games  # noqa: E402
from probe_dim import RANK, balanced, official, oof_probs, paired_boot, project  # noqa: E402
import trunk_extract as te  # noqa: E402
import trunk_seq_check as ts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--out", required=True); a = ap.parse_args()
    torch.set_num_threads(12)
    parts = [np.load(f, allow_pickle=False) for f in sorted(glob.glob(str(Path(a.root) / "shard*.npz")))]
    key = np.concatenate([p["game_key"] for p in parts]); pid = np.concatenate([p["player_id"] for p in parts]); rank = np.concatenate([p["rank"] for p in parts])
    Gm = np.concatenate([p["g_mean"] for p in parts]); Lm = np.concatenate([p["l_mean"] for p in parts]); n_own = np.concatenate([p["n_own"] for p in parts])
    ok = n_own > 0; key, pid, rank, Gm, Lm = key[ok], pid[ok], rank[ok], Gm[ok], Lm[ok]
    hsl = read_scored_games(ROOT / "results/scores_hd" / f"fold{a.fold}").set_index("game_key"); l1b = read_scored_games(ROOT / "results/l1b" / ts.L1B[a.fold] / f"fold{a.fold}").set_index("game_key")
    hcols, lcols = [f"sum_{p}" for p in PROFILES], sum_cols(LEVELS)
    have = np.array([k in hsl.index and k in l1b.index for k in key]); key, pid, rank, Gm, Lm = key[have], pid[have], rank[have], Gm[have], Lm[have]
    H = hsl.loc[key]; L = l1b.loc[key]
    lik = np.concatenate([H[hcols].to_numpy(float) / H["n_moves"].to_numpy(float)[:, None], L[lcols].to_numpy(float) / L["n_moves"].to_numpy(float)[:, None]], axis=1)
    y_game = np.array([RANK[r] for r in rank]); up, gidx = np.unique(pid, return_inverse=True); y_pl = np.zeros(len(up), int); y_pl[gidx] = y_game
    rep = {"players": int(len(up)), "games": int(len(key)), "arms": {}}; S = {}
    for name, d in (("lik", None), ("lik+64", 64), ("lik+128", 128), ("lik+256", 256), ("lik+full", 0)):
        X = lik if d is None else np.concatenate([lik, project(Gm, Lm, d, te.PROJ_SEED)], axis=1)
        Xp = np.zeros((len(up), X.shape[1])); np.add.at(Xp, gidx, X); Xp /= np.bincount(gidx)[:, None]
        P = oof_probs(Xp, y_pl, np.arange(len(up))); s, pred = official(P, y_pl); S[name] = s
        rep["arms"][name] = {"balanced": balanced(s, y_pl), "exact": float((pred == y_pl).mean()), "within1": float((np.abs(pred - y_pl) <= 1).mean()), "dims": int(X.shape[1])}
        print(name, json.dumps(rep["arms"][name]), flush=True)
    rep["pairs"] = {}
    for a_, b_ in (("lik+64", "lik"), ("lik+128", "lik+64"), ("lik+256", "lik+64"), ("lik+full", "lik+64"), ("lik+256", "lik")):
        rep["pairs"][f"{a_} vs {b_}"] = {"diff": balanced(S[a_], y_pl) - balanced(S[b_], y_pl), "ci95": paired_boot(S[a_], S[b_], y_pl)}
        print(a_, "vs", b_, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in rep["pairs"][f"{a_} vs {b_}"].items()}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
