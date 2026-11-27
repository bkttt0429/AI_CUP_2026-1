#!/usr/bin/env python
"""Move quality from humanv0's value and score heads: per-game point and win-rate losses of both players (idea #2, 2026-10-27).

For every evaluation game (scoring.fold_players: <= --max-games per player, >= --min-games; the games of
results/scores) every position before a move -- both colours -- goes through humanv0 under the 9d-vs-9d profile
(a strong human judge), bf16 on CUDA, no search. From the side to move: win = softmax(value)[win], lead =
pred_lead (points). The move at position t (by the side to move there) loses lead(t) + lead(t+1) points (lead(t+1)
is the opponent's view after the move) and win(t) - (1 - win(t+1)) win rate; the last move of a game has no
following position and is skipped. Per game and per side (target = the player of the question, opp = the
opponent) the table stores: moves judged, mean / median / q90 point loss, shares of moves losing > 1, 3, 8 points,
mean win-rate loss, share losing > 0.1, mean point loss by stage (moves 1-40, 41-100, 101+), and the target's
lead at the last position. humanv0 never saw this data, so the features need no out-of-fold split.

Output: <out>/fold{F}/quality_s{shard}_p{k:03d}.csv (one row per game), written atomically per part, resumable;
meta_s{shard}.json. Shard with --shard i --nshards N (scripts/nano4/shards.sh).

  bash scripts/nano4/shards.sh 8 python -I scripts/quality_score.py --folds 1 2 --out results/quality --workers 10
"""
import argparse
import json
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
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402
from aicup.scoring import fold_players, init_worker, load_games, player_games, score_stream  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
JUDGE = "9d"
STAGES = ((0, 40), (40, 100), (100, 10**6))
COLS = ["n", "loss_mean", "loss_median", "loss_q90", "gt1", "gt3", "gt8", "wloss_mean", "wgt10", "loss_s1", "loss_s2", "loss_s3"]


def side_stats(t, loss, wloss):
    """The COLS statistics of one side's judged moves (t = move index, losses clipped to [-10, 40] points)."""
    if len(loss) == 0:
        return {c: (0 if c == "n" else np.nan) for c in COLS}
    loss = np.clip(loss, -10, 40)
    out = {"n": int(len(loss)), "loss_mean": float(loss.mean()), "loss_median": float(np.median(loss)), "loss_q90": float(np.quantile(loss, 0.9)),
           "gt1": float((loss > 1).mean()), "gt3": float((loss > 3).mean()), "gt8": float((loss > 8).mean()),
           "wloss_mean": float(np.clip(wloss, -1, 1).mean()), "wgt10": float((wloss > 0.1).mean())}
    for j, (lo, hi) in enumerate(STAGES, 1):
        m = (t >= lo) & (t < hi)
        out[f"loss_s{j}"] = float(loss[m].mean()) if m.any() else np.nan
    return out


def game_quality(tb, vb, tw, vw, target):
    """Per-game stats from the black-to-move positions (move index tb, values vb [n, 2] = win, lead) and the white ones."""
    t = np.concatenate([tb, tw]).astype(np.int64)
    v = np.concatenate([vb, vw]).astype(np.float64)
    side = np.concatenate([np.zeros(len(tb), np.int8), np.ones(len(tw), np.int8)])
    order = np.argsort(t, kind="stable")
    t, v, side = t[order], v[order], side[order]
    nxt = np.searchsorted(t, t + 1)
    has = (nxt < len(t)) & (t[np.minimum(nxt, len(t) - 1)] == t + 1)
    j = np.minimum(nxt, len(t) - 1)
    loss = v[:, 1] + v[j, 1]
    wloss = v[:, 0] - (1.0 - v[j, 0])
    tgt = 0 if target == "B" else 1
    row = {}
    for name, s in (("t", tgt), ("o", 1 - tgt)):
        m = has & (side == s)
        row.update({f"{name}_{c}": x for c, x in side_stats(t[m], loss[m], wloss[m]).items()})
    last = int(np.argmax(t)) if len(t) else None
    row["t_end_lead"] = float(v[last, 1] if side[last] == tgt else -v[last, 1]) if last is not None else np.nan
    row["n_positions"] = int(len(t))
    return row


def make_scorer(model, device):
    import torch
    from aicup.humansl import meta_rows
    rows = torch.as_tensor(meta_rows([(JUDGE, JUDGE)])[0], device=device)  # [2, 192], row c = colour c to move

    def scorer(bins, globs, legal, target, colors, rows_):
        out = np.empty((len(target), 2), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(target), 256):
                b = torch.as_tensor(np.asarray(bins[i:i + 256]), device=device).float()
                g = torch.as_tensor(np.asarray(globs[i:i + 256]), device=device).float()
                c = torch.as_tensor(np.asarray(colors[i:i + 256]), device=device).long()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    outs = model(b, g, input_meta=rows[c])
                head = model.postprocess_output(outs)[0]
                win = torch.softmax(head[1].float(), dim=1)[:, 0]  # value_logits {win, loss, noresult}
                lead = head[10].float()  # pred_lead [N], points for the side to move (postprocess_output's order)
                out[i:i + len(c)] = torch.stack([win, lead], dim=1).cpu().numpy()
        return out
    return scorer


def score_fold(fold, players, out, shard, nshards, part_size, pool, scorer, log=print):
    games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][shard::nshards]
    info = {g[2]: g for g in games}
    fold_dir = Path(out) / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in fold_dir.glob(f"quality_s{shard}_p*.csv"):
        done |= set(pd.read_csv(p, usecols=["game_key"], dtype=str)["game_key"])
    used = [int(p.stem.split("_p")[1]) for p in fold_dir.glob(f"quality_s{shard}_p*.csv")]
    todo = [g[2] for g in games if g[2] not in done]
    log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(games) - len(todo)} already done", flush=True)
    t0, n_pos, k = time.time(), 0, (max(used) + 1 if used else 0)
    for start in range(0, len(todo), part_size):
        keys = todo[start:start + part_size]
        data = load_games(keys)
        tasks = []
        for i, key in enumerate(keys):
            tasks += [(2 * i, data[key][0], "B"), (2 * i + 1, data[key][0], "W")]
        per_task, arrays, n, wall = score_stream(pool, None, tasks, None, log_every=10**9, scorer=scorer)
        vals, tid, ts = arrays if arrays is not None else (np.zeros((0, 2), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int16))
        recs = []
        for i, key in enumerate(keys):
            mb, mw = tid == 2 * i, tid == 2 * i + 1
            recs.append({"game_key": key, "player_id": info[key][1], "rank": info[key][0], "color": data[key][1],
                         **game_quality(ts[mb], vals[mb], ts[mw], vals[mw], data[key][1])})
        path = fold_dir / f"quality_s{shard}_p{k:03d}.csv"
        tmp = path.with_suffix(".tmp.csv")
        pd.DataFrame(recs).to_csv(tmp, index=False)
        os.replace(tmp, path)
        n_pos += n
        k += 1
        log(f"  fold {fold} part {k}: {len(keys)} games, {n} positions, {n / max(wall, 1e-9):.0f} positions/s", flush=True)
    meta = {"fold": fold, "shard": shard, "nshards": nshards, "games": len(games), "positions": int(n_pos), "seconds": time.time() - t0,
            "judge_profile": f"rank_{JUDGE}", "columns": COLS, "finished": datetime.now().isoformat(timespec="seconds"), **environment("cpu")}
    (fold_dir / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")
    return len(games), n_pos


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-games", type=int, default=20)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    import torch
    from aicup.humansl import load_humansl
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))  # fork before CUDA
    model, _ = load_humansl(str(CKPT), a.device)
    scorer = make_scorer(model, a.device)
    all_games, folds_df = player_games(), pd.read_csv(FOLDS_CSV)
    try:
        for fold in a.folds:
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=a.min_games)
            g, n = score_fold(fold, players, a.out, a.shard, a.nshards, a.part_size, pool, scorer)
            print(f"fold {fold} done: {g} games, {n} positions", flush=True)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
