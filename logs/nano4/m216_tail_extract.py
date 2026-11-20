"""Section 4M M216 (2026-11-17): plies 256-383 of the training games that m90_ply_extract.py stored only up to ply 255, so that
the identity network can read games up to 384 plies. Same network, projection, replay and quantisation as m90_ply_extract.py
(results/m90/glstats.npz, int8 = round((x - mu) / sd * 20) clipped to +-127).

Games: those whose stored length in results/m90_ply{0,1,2} is 256 (= the game has at least 256 plies). Per game the stored plies
are t = 255 .. min(length, 384) - 1: ply 255 is stored a second time as an overlap check for the loader (point, mover, vector);
a game of exactly 256 plies holds only that one ply. Per part: key, LEN2 (= min(length, 384)), OFF (row offsets, n + 1), POS, MOV,
GLQ [rows, 128].
  --list                                writes the game list (results/m216/tail_keys.npy) from the m90 parts (CPU)
  sbatch --gres=gpu:8 -c 96 ... bash scripts/nano4/shards.sh 8 python -I logs/nano4/m216_tail_extract.py --workers 10
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
STATS = ROOT / "results/m90/glstats.npz"; OUT = Path("/work/u2507115/m216_tail"); LIST = ROOT / "results/m216/tail_keys.npy"
T_OLD, T_NEW, Q = 256, 384, 20.0


def truncated_keys(path):
    with np.load(path) as z:
        return z["key"].astype(str)[z["LEN"] >= T_OLD]


def safe_replay(task):
    import trunk_extract as te
    try:
        return te.replay(task)
    except Exception as e:  # a record the replay cannot follow: left out (the loader reports it as a game without a tail)
        return (task[0], None, repr(e)[:80])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--part", type=int, default=200); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--limit", type=int, default=0, help="smoke: only this many games of the shard")
    ap.add_argument("--out", default=str(OUT)); ap.add_argument("--list-file", default=str(LIST)); ap.add_argument("--first-files", type=int, default=0, help="--list for a smoke run: only the first N m90 parts (the ones m88_seqid.load_games(smoke=True) reads)")
    a = ap.parse_args(); t0 = time.time(); out_dir = Path(a.out); list_file = Path(a.list_file)
    if a.list:
        files = [f for k in (0, 1, 2) for f in sorted((ROOT / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]; files = files[:a.first_files] if a.first_files else files
        with mp.get_context("fork").Pool(a.workers) as pool:
            keys = np.sort(np.concatenate(pool.map(truncated_keys, files, chunksize=8)))
        assert len(set(keys.tolist())) == len(keys), "duplicate game keys in the m90 parts"; list_file.parent.mkdir(parents=True, exist_ok=True); np.save(list_file, keys)
        print(f"{len(files)} m90 parts; {len(keys)} games stored with length {T_OLD} (truncated or exactly {T_OLD} plies); list written to {list_file} ({time.time() - t0:.0f} s)", flush=True); return
    import torch
    import m81_tactic_id as m81
    import trunk_extract as te
    from aicup.humansl import load_humansl
    from aicup.l1b import neutral_rows, trunk_features
    st = np.load(STATS); mu, sd = st["mu"], st["sd"]; out_dir.mkdir(parents=True, exist_ok=True); todo = np.load(list_file); mine = todo[a.shard::a.nshards]; mine = mine[:a.limit] if a.limit else mine
    rows = m81.load_rows(); sgf = dict(zip(rows.game_key, zip(rows.sgf_content, rows.color))); del rows
    print(f"{len(todo)} games to extend; this shard: {len(mine)} ({time.time() - t0:.0f} s)", flush=True)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]; pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda"); model.eval(); nrows = neutral_rows(); proj = None; plies = 0; skipped = 0; short = 0; t1 = time.time()
        for k, start in enumerate(range(0, len(mine), a.part)):
            path = out_dir / f"tail_s{a.shard}of{a.nshards}_p{k:05d}.npz"
            if path.exists():
                continue
            keys = mine[start:start + a.part]; res = list(pool.map(safe_replay, [(q, *sgf[q]) for q in keys], chunksize=2)); done = [d for d in res if d[1] is not None and len(d[3]) >= T_OLD]; skipped += sum(d[1] is None for d in res); short += sum(d[1] is not None and len(d[3]) < T_OLD for d in res)
            if not done:
                continue
            if proj is None:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(), torch.as_tensor(done[0][4][:1]).cuda().long(), nrows)
                proj = te.projections(trunk.shape[1], trunk.device)
            len2 = np.array([min(len(d[3]), T_NEW) for d in done]); sl = [slice(T_OLD - 1, n) for n in len2]; cnt = len2 - (T_OLD - 1); off = np.concatenate([[0], np.cumsum(cnt)])
            bins = np.concatenate([d[1][s] for d, s in zip(done, sl)]); globs = np.concatenate([d[2][s] for d, s in zip(done, sl)]); pos = np.concatenate([d[3][s] for d, s in zip(done, sl)]); mover = np.concatenate([d[4][s] for d, s in zip(done, sl)])
            g, l = te.vectors(model, bins, globs, pos, mover, proj[0], proj[1], nrows, a.batch)
            q = np.clip(np.rint((np.hstack([g, l]).astype(np.float32) - mu) / sd * Q), -127, 127).astype(np.int8)
            tmp = path.with_name(path.name + ".tmp.npz"); np.savez(tmp, key=np.array([d[0] for d in done]), LEN2=len2.astype(np.int16), OFF=off.astype(np.int64), POS=pos.astype(np.int16), MOV=mover.astype(np.int8), GLQ=q); os.replace(tmp, path); plies += int(cnt.sum())
            if k % 20 == 0:
                print(f"part {k + 1}/{(len(mine) + a.part - 1) // a.part}: {plies} stored plies, {plies / (time.time() - t1):.0f} plies/s, replay failures {skipped}, shorter than {T_OLD} on replay {short}", flush=True)
        print(f"done: {len(mine)} games, {plies} stored plies, replay failures {skipped}, shorter than {T_OLD} on replay {short}, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close(); pool.join()


if __name__ == "__main__":
    main()
