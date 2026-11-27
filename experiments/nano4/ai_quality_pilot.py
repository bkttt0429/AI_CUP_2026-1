"""Section 4M M23 (2026-11-05, user: 試試看): objective move quality from a STRONG non-human KataGo net (kata1-b18c384nbt-s9996604416,
models/katago/kata1_b18c384nbt/, external model, to be disclosed) as per-move evidence orthogonal to humanv0. Same fold-1 class-stratified
sample as M13 / M17 / M18 (60 players per class, 20 games). Every position before a move, both colours, bf16, no search: the strong policy's
log p of the played move, top-1 agreement, policy entropy, the policy "loss" (log p played - log p best), win probability and score lead
for the side to move; per game and per side: means / shares, point loss (lead_t + lead_{t+1}) statistics as scripts/quality_score.py.
Output: results/ai_quality/fold1/shard{i}.csv.
  bash scripts/nano4/shards.sh 8 python -I logs/nano4/ai_quality_pilot.py --fold 1 --per-class 60 --out results/ai_quality
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
from aicup.humansl import load_humansl  # noqa: E402
from aicup.scoring import fold_players, init_worker, load_games, player_games, score_stream  # noqa: E402
import quality_score as qs  # noqa: E402

CKPT = ROOT / "models/katago/kata1_b18c384nbt/kata1-b18c384nbt-s9996604416-d4316597426/model.ckpt"
HUMAN_CKPT = ROOT / "models/katago/b18c384nbt-humanv0.ckpt"


def make_scorer(model, device, batch=256):
    has_meta = model.get_has_metadata_encoder()

    def scorer(bins, globs, legal, target, colors, rows_):
        out = np.empty((len(target), 6), dtype=np.float32)  # logp_played, agree, entropy, ploss, win, lead
        with torch.no_grad():
            for i in range(0, len(target), batch):
                j = min(len(target), i + batch)
                b = torch.as_tensor(np.asarray(bins[i:j]), device=device).float(); g = torch.as_tensor(np.asarray(globs[i:j]), device=device).float()
                illegal = ~torch.as_tensor(np.asarray(legal[i:j]), device=device); tgt = torch.as_tensor(np.asarray(target[i:j]), device=device).long()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    outs = model(b, g, input_meta=torch.zeros(j - i, 192, device=device)) if has_meta else model(b, g)
                head = model.postprocess_output(outs)[0]
                logits = head[0][:, 0, :].float().masked_fill(illegal, float("-inf"))
                lp = torch.log_softmax(logits, dim=1); p = lp.exp()
                lp_t = lp.gather(1, tgt[:, None])[:, 0]; best = lp.max(dim=1).values
                ent = -(p * torch.where(illegal, torch.zeros_like(lp), lp)).sum(1)
                win = torch.softmax(head[1].float(), dim=1)[:, 0]; lead = head[10].float()
                out[i:j] = torch.stack([lp_t, (lp.argmax(1) == tgt).float(), ent, lp_t - best, win, lead], dim=1).cpu().numpy()
        return out
    return scorer


def side_policy_stats(a):
    """Means of the strong policy's per-move quantities for one side: [n, 6] -> dict."""
    if len(a) == 0:
        return {"ai_logp": np.nan, "ai_agree": np.nan, "ai_entropy": np.nan, "ai_ploss": np.nan, "ai_ploss_lt3": np.nan, "ai_ploss_lt6": np.nan}
    return {"ai_logp": float(a[:, 0].mean()), "ai_agree": float(a[:, 1].mean()), "ai_entropy": float(a[:, 2].mean()), "ai_ploss": float(a[:, 3].mean()),
            "ai_ploss_lt3": float((a[:, 3] < -3).mean()), "ai_ploss_lt6": float((a[:, 3] < -6).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1); ap.add_argument("--per-class", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100); ap.add_argument("--human", action="store_true", help="use humanv0 (9d judge) instead of the strong net, as a control")
    a = ap.parse_args()
    ckpt = HUMAN_CKPT if a.human else CKPT
    config = torch.load(str(ckpt), map_location="cpu", weights_only=False)["config"]
    all_games, folds = player_games(), pd.read_csv(ROOT / "splits/player_folds.csv")
    players = fold_players(all_games, folds, a.fold, max_games=20); players = players[players["game_keys"].map(len) == 20]
    rng = np.random.default_rng(20261105)
    pick = pd.concat([grp.iloc[rng.permutation(len(grp))[:a.per_class]] for _, grp in players.groupby("rank")])
    games = [(p.rank, p.player_id, k) for p in pick.itertuples() for k in p.game_keys][a.shard::a.nshards]
    info = {g[2]: g for g in games}
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(ckpt), "cuda"); model.eval()
        print(f"model {ckpt.name}: metadata encoder {model.get_has_metadata_encoder()}", flush=True)
        scorer = make_scorer(model, "cuda")
        recs, t0 = [], time.time(); n_parts = (len(games) + a.part_size - 1) // a.part_size
        for p0 in range(0, len(games), a.part_size):
            keys = [g[2] for g in games[p0:p0 + a.part_size]]; data = load_games(keys)
            tasks = []
            for i, key in enumerate(keys):
                tasks += [(2 * i, data[key][0], "B"), (2 * i + 1, data[key][0], "W")]
            per_task, arrays, n_pos, wall = score_stream(pool, None, tasks, None, log_every=10 ** 9, scorer=scorer)
            vals, tid, ts = arrays
            for i, key in enumerate(keys):
                mb, mw = tid == 2 * i, tid == 2 * i + 1
                tgt_col = data[key][1]
                rec = {"game_key": key, "player_id": info[key][1], "rank": info[key][0], "color": tgt_col}
                rec.update(qs.game_quality(ts[mb], vals[mb][:, 4:6], ts[mw], vals[mw][:, 4:6], tgt_col))  # point / win-rate losses (t_* own, o_* opponent)
                own, opp = (mb, mw) if tgt_col == "B" else (mw, mb)
                rec.update({f"t_{k}": v for k, v in side_policy_stats(vals[own]).items()}); rec.update({f"o_{k}": v for k, v in side_policy_stats(vals[opp]).items()})
                recs.append(rec)
            print(f"part {p0 // a.part_size + 1}/{n_parts}: {n_pos} positions in {wall:.0f} s ({n_pos / max(wall, 1e-9):.0f}/s), {time.time() - t0:.0f} s total", flush=True)
        pd.DataFrame(recs).to_csv(out_dir / f"shard{a.shard}{'_human' if a.human else ''}.csv", index=False)
        print(f"saved {len(recs)} games, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
