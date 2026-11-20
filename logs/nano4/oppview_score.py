"""Opponent-view pilot (section 4M M13, 2026-11-05): score the OPPONENT's actual moves under a 2-D humanv0 metadata grid
(opponent rank r_o, target rank r_t).  The production HSL scoring moved both ranks together (own = opp = p) and only scored the
target's moves, so "how does the opponent's play depend on the target's displayed rank" was never isolated.  Per game: the summed
log p of the opponent's moves for each (r_o, r_t), plus the summed KL between adjacent r_t at each r_o (a distribution-level
sensitivity control that ignores the actual move).  Fold 1, a class-stratified sample of players, 20 games each.
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/oppview_score.py --fold 1 --per-class 60 --out results/oppview
Section 4M M17 (--target): the same grid on the TARGET's own moves with the metadata (own = target at r_t, opp = opponent at r_o):
the target's likelihood conditioned on the opponent's rank instead of own = opp = p.  Same players, same games.
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/oppview_score.py --fold 1 --per-class 60 --target --out results/oppview_tgt
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
from aicup.humansl import load_humansl, meta_rows  # noqa: E402
from aicup.scoring import fold_players, init_worker, load_games, player_games, score_stream  # noqa: E402

CKPT = ROOT / "models/katago/b18c384nbt-humanv0.ckpt"
R_O = ["6k", "3k", "1k", "2d", "4d", "6d"]  # the mover (= the opponent)
R_T = ["6k", "4k", "2k", "1d", "2d", "3d", "4d", "5d", "6d"]  # the target player's hypothesised rank
SPECS = [(ro, rt) for ro in R_O for rt in R_T]  # meta_rows: (own = mover, opp = other side)


def make_scorer(model, batch=64):
    device = next(model.parameters()).device
    n_o, n_t = len(R_O), len(R_T); S, K = n_o * n_t, n_o * (n_t - 1)

    def scorer(bins, globs, legal, target, colors, rows_):
        rows_t = torch.as_tensor(rows_, dtype=torch.float32, device=device)
        n = len(target); out = np.empty((n, S + K), np.float32)
        with torch.no_grad():
            for i in range(0, n, batch):
                j = min(n, i + batch)
                b = torch.as_tensor(bins[i:j], device=device).float(); g = torch.as_tensor(globs[i:j], device=device).float()
                illegal = ~torch.as_tensor(legal[i:j], device=device); tgt = torch.as_tensor(target[i:j], device=device).long()[:, None]
                col = torch.as_tensor(colors[i:j], device=device).long()
                lps = []
                for s in range(S):
                    outputs = model(b, g, input_meta=rows_t[s, col])
                    logits = model.postprocess_output(outputs)[0][0][:, 0, :].float()
                    lp = torch.log_softmax(logits.masked_fill(illegal, float("-inf")), dim=1); lps.append(lp)
                    out[i:j, s] = lp.gather(1, tgt)[:, 0].cpu().numpy()
                k = 0
                for o in range(n_o):
                    for t in range(n_t - 1):
                        a, c = lps[o * n_t + t], lps[o * n_t + t + 1]
                        kl = torch.where(illegal, torch.zeros_like(a), a.exp() * (a - c)).sum(1)
                        out[i:j, S + k] = kl.cpu().numpy(); k += 1
        return out
    return scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--per-class", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--part-size", type=int, default=64)
    ap.add_argument("--target", action="store_true", help="section 4M M17: score the TARGET's moves with own = target at r_t, opp = opponent at r_o")
    a = ap.parse_args()
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, a.fold, max_games=20)
    players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:a.per_class]] for _, grp in players.groupby("rank")])
    games = [(p.rank, p.player_id, k) for p in pick.itertuples() for k in p.game_keys][a.shard::a.nshards]
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda"); model.eval()
        rows = meta_rows([(rt, ro) for ro, rt in SPECS] if a.target else SPECS); scorer = make_scorer(model, a.batch)  # (own = mover, opp)
        recs, t0, S = [], time.time(), len(SPECS)
        n_parts = (len(games) + a.part_size - 1) // a.part_size
        for p0 in range(0, len(games), a.part_size):
            part = games[p0:p0 + a.part_size]; data = load_games([g[2] for g in part])
            tasks = [(i, data[k][0], data[k][1] if a.target else ("W" if data[k][1] == "B" else "B")) for i, (_, _, k) in enumerate(part)]  # the OPPONENT's moves (or, --target, the target's)
            per_task, _, n_pos, wall = score_stream(pool, model, tasks, rows, scorer=scorer, log_every=10 ** 9)
            for i, (rank, pid, k) in enumerate(part):
                n, sums = per_task[i]
                rec = {"game_key": k, "player_id": pid, "rank": rank, "target_color": data[k][1], ("n_own_moves" if a.target else "n_opp_moves"): int(n)}
                rec.update({f"ll_{ro}_{rt}": float(sums[s]) for s, (ro, rt) in enumerate(SPECS)})
                kk = 0
                for ro in R_O:
                    for t in range(len(R_T) - 1):
                        rec[f"kl_{ro}_{R_T[t]}_{R_T[t + 1]}"] = float(sums[S + kk]); kk += 1
                recs.append(rec)
            print(f"part {p0 // a.part_size + 1}/{n_parts}: {n_pos} positions in {wall:.0f} s, {time.time() - t0:.0f} s total", flush=True)
        pd.DataFrame(recs).to_csv(out_dir / f"shard{a.shard}.csv", index=False)
        print(f"saved {len(recs)} games, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
