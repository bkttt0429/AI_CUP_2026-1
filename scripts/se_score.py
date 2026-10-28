#!/usr/bin/env python
"""Per-game strength statistics of a trained strength head over the evaluation games of folds (issue #14, option b).

scripts/l1b_score.py's layout and rules: the games of scoring.fold_players (<= --max-games per player, >= --min-games),
resumable parts, <out>/fold{F}/games_s{shard}_p{k}.csv (game_key, player_id, rank, color, n_moves and aicup.sefeat's
SE_STATS: mean, std, 10 / 50 / 90 % quantiles and the three stage means of the strength scores of the side's moves)
with moves_s{shard}_p{k}.npz (s [M] float32, game_row, t), and meta_s{shard}.json (the head's sidecar summary with its
state and trunk sha256, the side scored). --opponent scores the other colour's moves. The head must be out-of-fold for
the folds scored (its sidecar's train_folds; --allow-seen for diagnostics, into a *_seen directory only). Shard with
--shard i --nshards N (scripts/nano4/shards.sh).

  bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_A.pt --folds 3 4 0 --out results/l1b/se3 --workers 10
  bash scripts/nano4/shards.sh 8 python -I scripts/se_score.py --head results/l1b/se_A.pt --folds 3 4 0 --out results/l1b/se3_opp --opponent --workers 10
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.scoring import fold_players, load_games, part_paths, player_games, resume_plan, score_stream  # noqa: E402
from aicup.sefeat import SE_STATS, game_stats  # noqa: E402
from aicup.strength import load_strength_head, strength_scores  # noqa: E402
import l1b_score as ls  # noqa: E402

HEAD_SUMMARY = ("train_folds", "max_games", "best_epoch", "inner_bt_init", "inner_bt_best", "inner_pair_accuracy_best", "n_params",
                "state_sha256", "trunk_file", "trunk_sha256", "code_sha256", "git_head", "git_dirty", "humanv0_sha256", "model_source")
COLUMNS = ["game_key", "player_id", "rank", "color", "n_moves", *SE_STATS]


def score_fold(fold, players, out, shard, nshards, part_size, pool, scorer, meta_head, log=print, opponent=False):
    """Score one fold's pending parts into out/fold{F}; returns (games in shard, positions scored, seconds)."""
    games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][shard::nshards]
    info = {g[2]: g for g in games}
    fold_dir = Path(out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    plan = resume_plan([g[2] for g in games], fold_dir, shard, part_size)
    pending = sum(len(keys) for _, keys in plan)
    log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(games) - pending} already done, "
        f"{pending} to score in {len(plan)} parts", flush=True)
    t0, n_pos = time.time(), 0
    if plan:
        data = load_games([k for _, keys in plan for k in keys])
        for j, (k, keys) in enumerate(plan):
            side = {key: (("W" if data[key][1] == "B" else "B") if opponent else data[key][1]) for key in keys}
            tasks = [(i, data[key][0], side[key]) for i, key in enumerate(keys)]
            per_task, arrays, n, wall = score_stream(pool, None, tasks, None, log_every=10**9, scorer=scorer)
            if arrays is not None:
                s, task_idx, ts = arrays[0][:, 0].astype(np.float32), arrays[1], arrays[2]
            else:
                s, task_idx, ts = np.zeros(0, np.float32), np.zeros(0, np.int32), np.zeros(0, np.int16)
            recs = []
            for i, key in enumerate(keys):
                m = task_idx == i
                recs.append({"game_key": key, "player_id": info[key][1], "rank": info[key][0], "color": side[key], "n_moves": int(per_task[i][0]),
                             **game_stats(s[m], ts[m])})
            csv_path, npz_path = part_paths(fold_dir, shard, k)
            tmp_csv, tmp_npz = csv_path.with_suffix(".tmp.csv"), npz_path.with_name(npz_path.stem + ".tmp.npz")
            np.savez_compressed(tmp_npz, s=s, game_row=task_idx, t=ts)
            pd.DataFrame(recs, columns=COLUMNS).to_csv(tmp_csv, index=False)
            os.replace(tmp_npz, npz_path)
            os.replace(tmp_csv, csv_path)
            n_pos += n
            log(f"  fold {fold} part {j + 1}/{len(plan)} (index {k}): {len(keys)} games, {n} positions, {n / max(wall, 1e-9):.0f} positions/s", flush=True)
    secs = time.time() - t0
    meta = {"fold": fold, "shard": shard, "nshards": nshards, "part_size": part_size, "games": len(games), "positions": int(n_pos),
            "scoring_wall_s": secs, "stats": list(SE_STATS), "head": meta_head, "side": "opponent" if opponent else "target",
            "finished": datetime.now().isoformat(timespec="seconds"), **environment("cpu")}
    (fold_dir / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")
    return len(games), n_pos, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head", required=True, help="strength head .pt (scripts/se_train.py)")
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--ckpt", default=str(ls.CKPT))
    ap.add_argument("--model-config", default=None, help="random KataGo config instead of --ckpt (tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--allow-seen", action="store_true")
    ap.add_argument("--opponent", action="store_true", help="score the opponent's moves (the other colour) instead of the target player's")
    a = ap.parse_args(argv)
    model = ls.load_trunk(a.ckpt, a.model_config, "cpu")  # on the CPU until the pool is forked
    try:
        head, meta, warnings = load_strength_head(a.head, model, "cpu")
        ls.check_unseen(meta, a.folds, a.allow_seen, a.out)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr, flush=True)
    meta_head = {"path": str(a.head), **{k: meta.get(k) for k in HEAD_SUMMARY}}
    all_games, folds_df = player_games(), pd.read_csv(a.folds_csv)
    pool = ls.make_pool(a.workers, model.config)
    model = model.to(a.device)
    head = head.to(a.device)

    def scorer(bins, globs, legal, target, colors, rows):
        return strength_scores(model, head, bins, globs, target, colors)[:, None]

    try:
        for fold in a.folds:
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=a.min_games)
            n_games, n_pos, secs = score_fold(fold, players, a.out, a.shard, a.nshards, a.part_size, pool, scorer, meta_head, opponent=a.opponent)
            print(f"fold {fold} done: {n_games} games, {n_pos} positions scored, {secs / 60:.1f} min", flush=True)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
