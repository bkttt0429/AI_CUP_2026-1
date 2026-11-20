"""Is the 64-dim random projection of the trunk the bottleneck?  Per-game means (over the target player's plies) of the FULL
pristine humanv0 trunk: board-pooled [C] and played-point [C] vectors, for one fold's players (20 games each), without any
projection; the production 64-dim vectors are linear in these means, so the probe (probe_dim.py) can project them post hoc.
Sharded like trunk_extract:  bash scripts/nano4/shards.sh 8 python -I logs/nano4/probe_dim_extract.py --fold 1 --out results/probe_dim
"""
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import trunk_extract as te  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100); ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if a.fold == 0:
        raise SystemExit("fold 0 is not used here")
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds_df = te.player_games(), pd.read_csv(te.FOLDS_CSV)
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    out_dir = Path(a.out) / f"fold{a.fold}"; out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, _ = te.load_humansl(str(te.CKPT), "cuda"); model.eval()
        rows = te.neutral_rows()
        players = te.fold_players(all_games, folds_df, a.fold, max_games=20, min_games=5)
        meta = {k: (r.player_id, r.rank) for r in players.itertuples() for k in r.game_keys}
        keys = sorted(meta)[a.shard::a.nshards]
        G, Lc, N, K = [], [], [], []
        t0 = time.time()
        for p0 in range(0, len(keys), a.part_size):
            part = keys[p0:p0 + a.part_size]
            sgf = te.load_games(part)
            done = list(pool.imap(te.replay, [(q, *sgf[q]) for q in part], chunksize=1))
            bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done])
            pos = np.concatenate([d[3] for d in done]); mover = np.concatenate([d[4] for d in done]); target = np.concatenate([d[5] for d in done])
            lens = np.array([len(d[3]) for d in done]); game_row = np.repeat(np.arange(len(done)), lens)
            n = len(pos); g_all, l_all = None, None
            with torch.no_grad():
                for i in range(0, n, a.batch):
                    j = min(n, i + a.batch)
                    b, g = torch.as_tensor(bins[i:j], device="cuda"), torch.as_tensor(globs[i:j], device="cuda")
                    c = torch.as_tensor(mover[i:j], device="cuda").long()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        trunk, _, _, _ = te.trunk_features(model, b, g, c, rows)
                    trunk = trunk.float()
                    if g_all is None:
                        C = trunk.shape[1]; g_all, l_all = np.zeros((n, C), np.float32), np.zeros((n, C), np.float32)
                    p = torch.as_tensor(pos[i:j], device="cuda").long(); on = p < te.PASS_POS
                    local = torch.zeros(j - i, C, device="cuda")
                    local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
                    g_all[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(); l_all[i:j] = local.cpu().numpy()
            w = target.astype(np.float32)
            cnt = np.bincount(game_row, weights=w, minlength=len(done))
            gm = np.zeros((len(done), C), np.float32); lm = np.zeros((len(done), C), np.float32)
            np.add.at(gm, game_row, g_all * w[:, None]); np.add.at(lm, game_row, l_all * w[:, None])
            ok = cnt > 0
            gm[ok] /= cnt[ok, None]; lm[ok] /= cnt[ok, None]
            G.append(gm); Lc.append(lm); N.append(cnt); K.extend(d[0] for d in done)
            print(f"part {p0 // a.part_size + 1}/{(len(keys) + a.part_size - 1) // a.part_size}: {n} plies, {time.time() - t0:.0f} s", flush=True)
        G, Lc, N = np.concatenate(G), np.concatenate(Lc), np.concatenate(N)
        pid = np.array([meta[k][0] for k in K]); rank = np.array([meta[k][1] for k in K])
        np.savez(out_dir / f"shard{a.shard}.npz", game_key=np.array(K), player_id=pid, rank=rank, g_mean=G, l_mean=Lc, n_own=N)
        print(f"saved {len(K)} games, C={G.shape[1]}, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
