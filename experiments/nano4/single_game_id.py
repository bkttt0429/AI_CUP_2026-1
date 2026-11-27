"""Can one game identify its player among ~1,600 candidates by style? (user 2026-11-06: 開局習慣可以把同類玩家找出來). Fold-1 players'
per-game full-channel trunk means (results/probe_dim/fold1, 384 board-pooled + 384 played-point dims): leave-one-game-out retrieval of the
player by cosine similarity to the other players' centroids (each player's 19 remaining games). Reports top-1 / top-5 / top-20 accuracy among
1,592 players and the share of games whose best match is confident (margin). If single-game identification is weak here, matching a test
opponent to one of 8,007 training players (to read off an exact rank) is out of reach.
  python -I logs/nano4/single_game_id.py --out results/r_variants/single_game_id.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path.cwd()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    G, L, pid, keys = [], [], [], []
    for f in sorted(glob.glob(str(ROOT / "results/probe_dim/fold1/shard*.npz"))):
        z = np.load(f, allow_pickle=False); G.append(z["g_mean"]); L.append(z["l_mean"]); pid.append(z["player_id"].astype(str)); keys.append(z["game_key"].astype(str))
    X = np.hstack([np.vstack(G), np.vstack(L)]).astype(np.float64); pid = np.concatenate(pid)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    players, inv = np.unique(pid, return_inverse=True); n_p = len(players)
    sums = np.zeros((n_p, X.shape[1])); cnt = np.bincount(inv, minlength=n_p)
    np.add.at(sums, inv, X)
    rep = {"games": int(len(X)), "players": int(n_p), "dims": int(X.shape[1])}
    for name, feat in (("board+point 768d", slice(0, X.shape[1])), ("board 384d", slice(0, 384)), ("point 384d", slice(384, 768))):
        Xf = X[:, feat]; S = sums[:, feat]
        cent = (S[inv] - Xf) / np.maximum(cnt[inv] - 1, 1)[:, None]  # leave-one-out own centroid
        C_all = S / cnt[:, None]
        Cn = C_all / (np.linalg.norm(C_all, axis=1, keepdims=True) + 1e-9); xn = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-9)
        sim = xn @ Cn.T  # [games, players]
        own = (xn * (cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9))).sum(1)
        sim[np.arange(len(Xf)), inv] = own  # replace the own-player similarity by the leave-one-out version
        order = np.argsort(-sim, axis=1)
        rank_of_true = np.argmax(order == inv[:, None], axis=1)
        top = sim[np.arange(len(Xf)), order[:, 0]]; second = sim[np.arange(len(Xf)), order[:, 1]]; margin = top - second
        conf = margin > np.quantile(margin, 0.95)
        rep[name] = {"top1": float((rank_of_true == 0).mean()), "top5": float((rank_of_true < 5).mean()), "top20": float((rank_of_true < 20).mean()), "median_rank_of_true": float(np.median(rank_of_true) + 1),
                     "top1_among_5pct_most_confident": float((rank_of_true[conf] == 0).mean()), "chance_top1": 1.0 / n_p}
        print(f"{name}: top-1 {rep[name]['top1']:.4f} (chance {1/n_p:.4f}), top-5 {rep[name]['top5']:.4f}, top-20 {rep[name]['top20']:.4f}, median rank of true player {rep[name]['median_rank_of_true']:.0f}; top-1 among the 5% most confident {rep[name]['top1_among_5pct_most_confident']:.3f}")
    Path(a.out).write_text(json.dumps(rep, indent=1)); print("written", a.out)


if __name__ == "__main__":
    main()
