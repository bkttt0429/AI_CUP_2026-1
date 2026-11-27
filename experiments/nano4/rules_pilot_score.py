"""Section 4M M26 (2026-11-06; user: 「試試看日本規則，畢竟數據是日本的」): the rules and komi humanv0 sees. Production replays every game
under KataGo's RULES_CHINESE with whiteKomi 7.5 (a feature-extraction setting: rules and komi are global inputs of the net). The source
server (Nihon Ki-in 幽玄の間) plays Japanese rules, komi 6.5 in even games (HD0) and none in 讓先 / handicap games (Q13). Three arms on
the same fold-1 class-stratified sample as M18 / M23 (60 players per class, 20 games, rng 20261105), the TARGET's moves, production
metadata (aicup.humansl.make_profile_meta: KGS, 1200 s + 30 s x 5, 2020-03), 18 profiles:
  cn75  RULES_CHINESE, whiteKomi 7.5 (production; must reproduce results/scores_hd)
  cnk   RULES_CHINESE, whiteKomi 6.5 (HD0) / 0 (HD >= 1): komi only
  jp    RULES_JAPANESE (territory scoring, seki tax), whiteKomi 6.5 / 0: the server's rules
Per game the summed log p of the target's moves per (arm, profile) -> results/rules_pilot/fold1/shard{i}.csv (rules_pilot_eval.py).
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/rules_pilot_score.py --fold 1 --per-class 60 --out results/rules_pilot
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import aicup.pilot as pilot  # noqa: E402
import aicup.scoring as sc  # noqa: E402
from aicup.humansl import PROFILES, load_humansl, make_profile_meta  # noqa: E402
from aicup.scoring import fold_players, load_games, player_games, score_stream  # noqa: E402
from aicup.sgf import handicap  # noqa: E402
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402

CKPT = ROOT / "models/katago/b18c384nbt-humanv0.ckpt"
ARMS = ("cn75", "cnk", "jp")
_ARM = None
_ORIG_GP = pilot.game_positions


def rules_for(sgf, arm):
    hd = handicap(sgf) or 0
    if arm == "cn75":
        r = dict(GameState.RULES_CHINESE); r["whiteKomi"] = 7.5
    elif arm == "cnk":
        r = dict(GameState.RULES_CHINESE); r["whiteKomi"] = 6.5 if hd == 0 else 0.0
    elif arm == "jp":
        r = dict(GameState.RULES_JAPANESE); r["whiteKomi"] = 6.5 if hd == 0 else 0.0
    else:
        raise ValueError(arm)
    return r


def game_positions_rules(sgf, color, features, handicap=True):
    """aicup.pilot.game_positions with this worker's arm's rules and komi (start_state is patched for the one call)."""
    rules, orig = rules_for(sgf, _ARM), pilot.start_state
    pilot.start_state = lambda stones=(), _r=None: orig(stones, rules)
    try:
        return _ORIG_GP(sgf, color, features, handicap)
    finally:
        pilot.start_state = orig


def init_arm(config, arm):
    global _ARM
    sc.init_worker(config); _ARM = arm; sc.game_positions = game_positions_rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--per-class", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=64)
    a = ap.parse_args()
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, a.fold, max_games=20); players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:a.per_class]] for _, grp in players.groupby("rank")])
    games = [(p.rank, p.player_id, k) for p in pick.itertuples() for k in p.game_keys][a.shard::a.nshards]
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_humansl(str(CKPT), "cuda"); model.eval()
    rows = np.zeros((len(PROFILES), 2, 192), dtype=np.float32)
    for s, p in enumerate(PROFILES):
        m = make_profile_meta(p, p)
        for c, pla in enumerate((Board.BLACK, Board.WHITE)):
            rows[s, c] = m.get_metadata_row(nextPlayer=pla, boardArea=361)
    recs = {g[2]: {"game_key": g[2], "player_id": g[1], "rank": g[0]} for g in games}; t0 = time.time()
    for arm in ARMS:
        pool = mp.get_context("fork").Pool(a.workers, initializer=init_arm, initargs=(config, arm))
        try:
            for p0 in range(0, len(games), a.part_size):
                part = games[p0:p0 + a.part_size]; data = load_games([g[2] for g in part])
                tasks = [(i, data[k][0], data[k][1]) for i, (_, _, k) in enumerate(part)]
                per_task, _, n_pos, wall = score_stream(pool, model, tasks, rows, log_every=10 ** 9)
                for i, (_, _, k) in enumerate(part):
                    n, sums = per_task[i]; r = recs[k]; r["target_color"] = data[k][1]; r[f"n_moves_{arm}"] = int(n)
                    r.update({f"ll_{arm}_{p}": float(sums[s]) for s, p in enumerate(PROFILES)})
            print(f"arm {arm} done: {len(games)} games, {time.time() - t0:.0f} s total", flush=True)
        finally:
            pool.close(); pool.join()
    df = pd.DataFrame(list(recs.values())); df["n_moves"] = df["n_moves_cn75"]
    df.to_csv(out_dir / f"shard{a.shard}.csv", index=False); print(f"saved {len(df)} games, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
