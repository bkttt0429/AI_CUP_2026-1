#!/usr/bin/env python
"""Score the evaluation games of folds under the L1b rank heads, in scripts/score_folds.py's layout.

Plan 2026-10-25-l1b-rank-heads, Task 5. The games are scoring.fold_players' (<= --max-games per player
with >= --min-games, the defaults 20 / 5 being results/scores' selection), so every fold scored here
has the same games as results/scores/fold{F}/ and the two can be compared move by move. Output per
fold: <out>/fold{F}/games_s{shard}_p{k:03d}.csv (game_key, player_id, rank, color, n_moves, sum_<D..6D>)
and moves_s{shard}_p{k:03d}.npz (logp [M, 10] float32, game_row, t), written atomically and resumable
per game (scoring.resume_plan); plus <out>/fold{F}/meta_s{shard}.json with the heads' sidecar summary,
the environment and the counts.

The heads must not have seen the fold: a fold in the sidecar's train_folds is refused (exit 2) unless
--allow-seen, and then only into an --out whose name contains "_seen" (diagnostics only).

  python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 0 --out results/l1b/scores --workers 10
  bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_B.pt --folds 1 2 0 --out results/l1b/scores_B
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from aicup.l1b import load_heads, score_positions  # noqa: E402
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.scoring import fold_players, init_worker, load_games, part_paths, player_games, resume_plan, score_stream  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
HEADS_SUMMARY = ("train_folds", "max_games", "best_epoch", "inner_nll_init", "inner_nll_best", "stopped_early", "n_params",
                 "state_sha256", "code_sha256", "git_head", "git_dirty", "humanv0_sha256", "model_source")


def load_trunk(ckpt, model_config, device):
    import torch
    if model_config:
        from katago.train import modelconfigs
        from katago.train.model_pytorch import Model
        torch.manual_seed(0)
        model = Model(modelconfigs.config_of_name[model_config], 19)
        model.initialize()
        return model.to(device).eval()
    from aicup.humansl import load_humansl
    return load_humansl(str(ckpt), device)[0]


def make_pool(workers, config):
    """The CPU feature pool; fork it before anything touches CUDA (PyTorch does not support forking a CUDA process)."""
    return mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(config,))


def check_unseen(meta, folds, allow_seen, out):
    """ValueError when the heads were trained on one of `folds` (unless allow_seen into a *_seen directory), or when the
    sidecar does not record train_folds at all (a cache without meta.json): out-of-fold cannot be shown."""
    if meta.get("train_folds") is None:
        raise ValueError("the heads' sidecar has no train_folds (the cache had no meta.json?); cannot show the features are out-of-fold")
    seen = sorted(set(meta.get("train_folds")) & set(folds))
    if seen and not allow_seen:
        raise ValueError(f"the heads' train_folds {meta.get('train_folds')} include {seen}; their features would not be out-of-fold "
                         "(--allow-seen for diagnostics only)")
    if allow_seen and "_seen" not in Path(out).name:
        raise ValueError(f"--allow-seen writes only into a directory whose name contains '_seen', not {out}")


def score_fold(fold, players, out, shard, nshards, part_size, pool, scorer, meta_heads, log=print):
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
            tasks = [(i, data[key][0], data[key][1]) for i, key in enumerate(keys)]
            per_task, arrays, n, wall = score_stream(pool, None, tasks, None, log_every=10**9, scorer=scorer)
            logp, task_idx, ts = arrays if arrays is not None else (np.zeros((0, len(LEVELS)), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int16))
            recs = [{"game_key": key, "player_id": info[key][1], "rank": info[key][0], "color": data[key][1], "n_moves": per_task[i][0],
                     **{f"sum_{lv}": float(s) for lv, s in zip(LEVELS, per_task[i][1])}} for i, key in enumerate(keys)]
            csv_path, npz_path = part_paths(fold_dir, shard, k)
            tmp_csv, tmp_npz = csv_path.with_suffix(".tmp.csv"), npz_path.with_name(npz_path.stem + ".tmp.npz")
            np.savez_compressed(tmp_npz, logp=logp.astype(np.float32), game_row=task_idx, t=ts)
            pd.DataFrame(recs).to_csv(tmp_csv, index=False)
            os.replace(tmp_npz, npz_path)
            os.replace(tmp_csv, csv_path)
            n_pos += n
            log(f"  fold {fold} part {j + 1}/{len(plan)} (index {k}): {len(keys)} games, {n} positions, {n / max(wall, 1e-9):.0f} positions/s", flush=True)
    secs = time.time() - t0
    meta = {"fold": fold, "shard": shard, "nshards": nshards, "part_size": part_size, "games": len(games), "positions": int(n_pos),
            "scoring_wall_s": secs, "levels": list(LEVELS), "heads": meta_heads, "finished": datetime.now().isoformat(timespec="seconds"),
            **environment("cpu")}
    (fold_dir / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")
    return len(games), n_pos, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heads", required=True)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--model-config", default=None, help="random KataGo config instead of --ckpt (tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--allow-seen", action="store_true")
    a = ap.parse_args(argv)
    model = load_trunk(a.ckpt, a.model_config, "cpu")  # on the CPU until the pool is forked
    try:
        heads, meta, warnings = load_heads(a.heads, model, "cpu")
        check_unseen(meta, a.folds, a.allow_seen, a.out)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr, flush=True)
    meta_heads = {"path": str(a.heads), **{k: meta.get(k) for k in HEADS_SUMMARY}}
    all_games, folds_df = player_games(), pd.read_csv(a.folds_csv)
    pool = make_pool(a.workers, model.config)
    model = model.to(a.device)
    heads = heads.to(a.device)

    def scorer(bins, globs, legal, target, colors, rows):
        return score_positions(model, heads, bins, globs, legal, target, colors)

    try:
        for fold in a.folds:
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=a.min_games)
            n_games, n_pos, secs = score_fold(fold, players, a.out, a.shard, a.nshards, a.part_size, pool, scorer, meta_heads)
            print(f"fold {fold} done: {n_games} games, {n_pos} positions scored, {secs / 60:.1f} min", flush=True)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
