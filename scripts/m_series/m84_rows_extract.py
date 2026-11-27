"""Section 4M M84 (2026-11-11): the per-game-side trunk rows (mean and standard deviation of the 64 + 64 per-ply trunk vectors, as
m82_cond_id.py's rows stage) of every training row that has NO stored trunk vectors yet (results/trunk_vec, _x40, _x80 hold about 405k of
the 1,000,000 games). Same network, projection, replay and fp16 rounding as scripts/trunk_extract.py; the per-ply vectors are not kept
(disk), only the two 256-dim rows per game. Resumes per part.
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I logs/nano4/m84_rows_extract.py --job 0 --njobs 3 --workers 10
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import m81_tactic_id as m81  # noqa: E402
import trunk_extract as te  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402

OUT = ROOT / "results/m84_rowsp"  # second run: also the phase means (the first run's partial files stay in results/m84_rows, unused)
VEC = [ROOT / "results/trunk_vec", ROOT / "results/trunk_vec_x40", ROOT / "results/trunk_vec_x80"]


def safe_replay(task):
    try:
        return te.replay(task)
    except Exception as e:  # a record the replay cannot follow: no row for this game
        return (task[0], None, repr(e)[:80])


def side_rows(X, row, target, t, n_g):
    """Per game and side (0 = the target's plies, 1 = the opponent's): the 256-dim row (mean, std over the side's plies), the ply
    count, and the phase means [384] = the mean vector over the side's plies with t < 60, 60 <= t < 150, t >= 150 (zeros if none)."""
    out = np.zeros((2, n_g, 256), np.float32); ph = np.zeros((2, n_g, 384), np.float16); cnt = np.zeros((2, n_g), np.int32)
    for side, m in enumerate((target, ~target)):
        r = row[m]; order = np.argsort(r, kind="stable"); xs = X[m][order]; n = np.bincount(r, minlength=n_g); nz = n > 0
        if not nz.any():
            continue
        starts = (np.cumsum(n) - n)[nz]; s1 = np.add.reduceat(xs, starts, axis=0); s2 = np.add.reduceat(xs * xs, starts, axis=0); mean = s1 / n[nz][:, None]
        out[side][nz] = np.hstack([mean, np.sqrt(np.maximum(s2 / n[nz][:, None] - mean * mean, 0.0))]); cnt[side] = n
        for k, pm in enumerate((t < 60, (t >= 60) & (t < 150), t >= 150)):
            mm = m & pm; r = row[mm]; n = np.bincount(r, minlength=n_g); nz = n > 0
            if nz.any():
                order = np.argsort(r, kind="stable"); ph[side][nz, 128 * k:128 * (k + 1)] = np.add.reduceat(X[mm][order], (np.cumsum(n) - n)[nz], axis=0) / n[nz][:, None]
    return out, ph, cnt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--job", type=int, default=0); ap.add_argument("--njobs", type=int, default=1); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--part", type=int, default=200); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); rows = m81.load_rows(); have = set()
    for d in VEC:
        for p in d.glob("fold*/games_s*_p*.csv"):
            have.update(pd.read_csv(p, usecols=["game_key"], dtype=str)["game_key"])
    todo = np.array(sorted(set(rows.game_key) - have)); mine = todo[a.job::a.njobs][a.shard::a.nshards]; mine = mine[:a.limit] if a.limit else mine
    sgf = dict(zip(rows.game_key, zip(rows.sgf_content, rows.color))); del rows
    print(f"{len(have)} games have vectors, {len(todo)} do not; this shard: {len(mine)} ({time.time() - t0:.0f} s)", flush=True)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda"); model.eval(); nrows = neutral_rows(); proj = None; plies = 0; skipped = 0; t1 = time.time()
        for k, start in enumerate(range(0, len(mine), a.part)):
            path = OUT / f"rows_j{a.job}of{a.njobs}_s{a.shard}of{a.nshards}_p{k:05d}.npz"
            if path.exists():
                continue
            keys = mine[start:start + a.part]; res = list(pool.imap(safe_replay, [(q, *sgf[q]) for q in keys], chunksize=2)); done = [d for d in res if d[1] is not None and len(d[3]) > 0]; skipped += len(res) - len(done)
            if not done:
                continue
            if proj is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)
                proj = te.projections(trunk.shape[1], trunk.device)
            lens = np.array([len(d[3]) for d in done]); bins = np.concatenate([d[1] for d in done]); globs = np.concatenate([d[2] for d in done]); pos = np.concatenate([d[3] for d in done])
            mover = np.concatenate([d[4] for d in done]); target = np.concatenate([d[5] for d in done]); g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch)
            out, ph, cnt = side_rows(np.hstack([g, l]).astype(np.float64), np.repeat(np.arange(len(done)), lens), target, np.concatenate([np.arange(n) for n in lens]), len(done))
            tmp = path.with_name(path.name + ".tmp.npz"); np.savez(tmp, key=np.array([d[0] for d in done]), T=out[0], O=out[1], PT=ph[0], PO=ph[1], nT=cnt[0], nO=cnt[1]); os.replace(tmp, path); plies += int(lens.sum())
            if k % 20 == 0:
                print(f"part {k + 1}/{(len(mine) + a.part - 1) // a.part}: {plies} plies, {plies / (time.time() - t1):.0f} plies/s, skipped {skipped}", flush=True)
        print(f"done: {len(mine)} games, {plies} plies, skipped {skipped}, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
