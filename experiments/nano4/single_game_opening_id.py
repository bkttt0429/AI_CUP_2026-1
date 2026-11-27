"""Single-game identification with competition 1's opening model (user 2026-11-06: 認出來就可以推分數). Pool: a seeded sample of training
players with >= 30 Black games (cap 60 Black games each); per player 5 held-out Black games are the queries, the opening model (depth 12,
symmetry) and the style model are fitted on the rest. Reports top-1 / top-5 / top-20 among the pool and the precision of the 10% most
confident queries (margin). Opponent probe: in the pool players' WHITE games the opponent plays Black; its opening is scored against the pool;
the share of opponent games whose best-vs-second margin exceeds the 99th percentile of the impostor null (held-out queries scored with the
true player excluded) minus 1% estimates the identifiable training-player opponents inside the pool (scale by 8,007 / pool for all players).
  python -I logs/nano4/single_game_opening_id.py --players 2000 --out results/r_variants/single_game_opening_id.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(Path.home() / "BK_AICUP_2026-2/src")]
from aicup.scoring import TRAIN_DIR, load_games  # noqa: E402
from aicup_pid.opening import OpeningModel, parse  # noqa: E402
from aicup_pid.style import StyleModel, game_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--players", type=int, default=2000); ap.add_argument("--cap", type=int, default=60); ap.add_argument("--queries", type=int, default=5)
    ap.add_argument("--out", required=True); a = ap.parse_args(); t0 = time.time()
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color"]) for p in sorted(Path(TRAIN_DIR).glob("train_*.csv"))], ignore_index=True)
    rows["game_key"] = rows["rank"] + ":" + rows["game_id"]
    black = rows[rows.color == "B"]; nb = black.groupby("player_id").size(); eligible = nb[nb >= 30].index.to_numpy()
    rng = np.random.default_rng(0); pool = rng.choice(eligible, a.players, replace=False)
    B = black[black.player_id.isin(pool)].groupby("player_id").head(a.cap)
    W = rows[(rows.color == "W") & rows.player_id.isin(pool)].groupby("player_id").head(10)  # the opponent plays Black here
    print(f"{len(eligible)} players with >= 30 Black games; pool {len(pool)}; Black games {len(B)}, White games (opponent as Black) {len(W)} ({time.time() - t0:.0f} s)", flush=True)
    sgfs = load_games(B.game_key.tolist() + W.game_key.tolist())
    moves = {k: parse(v[0]) for k, v in sgfs.items()}
    print(f"parsed {len(moves)} games ({time.time() - t0:.0f} s)", flush=True)
    fit_keys, query_keys = {}, []
    for pid, g in B.groupby("player_id"):
        ks = g.game_key.tolist(); rng.shuffle(ks); query_keys += [(pid, k) for k in ks[:a.queries]]; fit_keys[pid] = ks[a.queries:]
    opening = OpeningModel(depth=12, alpha=2.0, beta=5.0, alpha_full=1.0).fit({p: [moves[k] for k in ks] for p, ks in fit_keys.items()})
    feats = {k: game_features(moves[k]) for k in list(B.game_key) + list(W.game_key)}
    style = StyleModel(shrink=0.1).fit({p: np.stack([feats[k] for k in ks]) for p, ks in fit_keys.items()})
    idx = {p: i for i, p in enumerate(opening.players)}
    print(f"models fitted ({time.time() - t0:.0f} s)", flush=True)
    rep = {"pool": len(pool), "queries": len(query_keys)}
    for name, scorer in (("opening", lambda k: opening.log_scores([moves[k]])), ("style", lambda k: style.log_scores(feats[k][None, :])), ("opening+style", lambda k: opening.log_scores([moves[k]]) + style.log_scores(feats[k][None, :]))):
        ranks, margins, wrong_margin = [], [], []
        for pid, k in query_keys:
            s = scorer(k); i = idx[pid]; order = np.argsort(-s); ranks.append(int(np.argmax(order == i)))
            top2 = s[order[:2]]; margins.append(float(top2[0] - top2[1]))
            s2 = s.copy(); s2[i] = -np.inf; o2 = np.argsort(-s2); wrong_margin.append(float(s2[o2[0]] - s2[o2[1]]))  # impostor null
        ranks, margins, wrong_margin = np.array(ranks), np.array(margins), np.array(wrong_margin)
        conf = margins >= np.quantile(margins, 0.9)
        rep[name] = {"top1": float((ranks == 0).mean()), "top5": float((ranks < 5).mean()), "top20": float((ranks < 20).mean()), "median_rank": float(np.median(ranks) + 1),
                     "top1_among_10pct_most_confident": float((ranks[conf] == 0).mean()), "null_margin_q99": float(np.quantile(wrong_margin, 0.99))}
        print(f"{name:14s}: top-1 {rep[name]['top1']:.3f} (chance {1/len(pool):.4f}), top-5 {rep[name]['top5']:.3f}, top-20 {rep[name]['top20']:.3f}, median rank {rep[name]['median_rank']:.0f}; top-1 among the 10% most confident {rep[name]['top1_among_10pct_most_confident']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        if name == "opening":
            # opponents (Black side of the pool players' White games) against the pool
            om = []
            for pid, k in zip(W.player_id, W.game_key):
                s = opening.log_scores([moves[k]]); s[idx[pid]] = -np.inf; o = np.argsort(-s); om.append(float(s[o[0]] - s[o[1]]))
            om = np.array(om); q99 = rep[name]["null_margin_q99"]
            rep["opponents_opening"] = {"games": int(len(om)), "share_margin_above_null_q99": float((om > q99).mean()), "excess_over_1pct": float((om > q99).mean() - 0.01),
                                        "implied_identifiable_share_all_players": float(((om > q99).mean() - 0.01) * 8007 / len(pool))}
            print(f"opponents (Black in the pool players' White games, {len(om)} games): margin above the impostor q99 {rep['opponents_opening']['share_margin_above_null_q99']:.4f} (excess {rep['opponents_opening']['excess_over_1pct']:+.4f}; scaled to all 8,007 players {rep['opponents_opening']['implied_identifiable_share_all_players']:+.4f})", flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)


if __name__ == "__main__":
    main()
