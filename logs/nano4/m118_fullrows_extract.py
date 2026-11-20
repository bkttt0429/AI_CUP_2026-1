#!/usr/bin/env python
"""M118: per-game-side statistics of the UNPROJECTED trunk vectors for the game sides this project's opponent identification needs
and the player-identification session's extraction (target side of the training games of folds 1-4) does not cover.

  --set test     both sides of the 10,000 test games (key t:<index in m81.load_tests()>)
  --set queries  the opponent side of the validation query games (results/m84_cache/queries.npz [0:Q))
  --set fold0    the target side of the training games of fold 0's players (candidates' own games)
  --set check    the target side of the first 400 games of the peer's sample (compared with its rows by --verify)

Per game side, in the layout of the peer's `full` rows (its scripts/pid_trunk_extract.py --full, aggregate_full): mean / std of
the 384-dim board-pooled vector over the side's plies, mean / std of the 384-dim played-point vector over the side's non-pass
plies (1,536 dims, fp16); for the check set also the usual 256-dim projected row of the same plies (its `aggregate`). Same replay, network,
neutral rows and projection as scripts/trunk_extract.py. Parts of 200 games, <out>/<set>/part_p<k>.npz; any number of workers may
run at once (a part is claimed with a lock file), and a run resumes per part (--clear-locks after a killed run).

  CUDA_VISIBLE_DEVICES=2 python -I logs/nano4/m118_fullrows_extract.py --set test queries fold0 --workers 10
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import m81_tactic_id as m81  # noqa: E402
import trunk_extract as te  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402

OUT = ROOT / "results/m118_fullrows"  # /home: /work is nearly full (2026-11-14 06:55)
KEEP_RP = ("check",)  # the 256-dim projected row is stored for the check set only (disk)
PEER = Path("/work/u2507115/pid_v2/fullrows")
C = 384
SIDES = {"test": "TO", "queries": "O", "fold0": "T", "check": "T"}


def replay_side(task):
    """scripts/trunk_extract.py's replay, for the plies of the wanted sides only: the game is played through as there, but the
    input features of the other side's positions are never computed (they are the replay's cost)."""
    key, sgf, color, sides = task
    try:
        stones, moves = te.setup_and_moves(sgf)
        gs = te.start_state(stones, te.RULES)
        bins, globs, pos, mover, target = [], [], [], [], []
        for c, x, y in moves:
            pla = te.Board.BLACK if c == "B" else te.Board.WHITE
            loc = te.Board.PASS_LOC if x is None else gs.board.loc(x, y)
            is_t = c == color
            if sides == "TO" or (sides == "T") == is_t:
                b, g = gs.get_input_features(te._FEATURES)
                bins.append(b[0].astype(np.uint8))
                globs.append(g[0].astype(np.float32))
                pos.append(te.PASS_POS if x is None else te._FEATURES.loc_to_tensor_pos(loc, gs.board))
                mover.append(0 if c == "B" else 1)
                target.append(is_t)
            gs.play(pla, loc)
        if not pos:
            return (key, None, "no plies of the wanted side")
        return key, np.stack(bins), np.stack(globs), np.asarray(pos, np.int16), np.asarray(mover, np.int8), np.asarray(target, bool)
    except Exception as e:  # a record the replay cannot follow: no row for this game
        return (key, None, repr(e)[:80])


def tasks_of(name, rows):
    """[(key, sgf, colour)] of one set, sorted by key (test: by index)."""
    if name == "test":
        return [(f"t:{i:05d}", g[0], g[1]) for i, (_q, g) in enumerate(m81.load_tests())]
    if name == "queries":
        q = np.load(ROOT / "results/m84_cache/queries.npz")
        want = set(q["qkey"][: int(q["Q"])].astype(str).tolist())
    elif name == "fold0":
        folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str})
        want = set(rows.game_key[rows.player_id.isin(set(folds.player_id[folds.fold == 0]))])
    elif name == "check":
        want = set((PEER / "keys.txt").read_text().split()[:400])
    else:
        raise ValueError(name)
    sub = rows[rows.game_key.isin(want)].sort_values("game_key")
    if len(sub) != len(want):
        raise ValueError(f"{name}: {len(want)} keys wanted, {len(sub)} training rows found")
    return list(zip(sub.game_key, sub.sgf_content, sub.color))


def full_vectors(model, bins, globs, pos, mover, rows, batch):
    """[n, 384] float32 board-pooled and played-point trunk vectors without the projection (a pass: zero played-point vector)."""
    n = len(pos)
    out_g, out_l = np.empty((n, C), np.float32), np.empty((n, C), np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b, g = torch.as_tensor(bins[i:j]).cuda(), torch.as_tensor(globs[i:j]).cuda()
            c = torch.as_tensor(mover[i:j]).cuda().long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)
            trunk = trunk.float()
            if trunk.shape[1] != C:
                raise ValueError(f"trunk has {trunk.shape[1]} channels, expected {C}")
            p = torch.as_tensor(pos[i:j]).cuda().long()
            on = p < te.PASS_POS
            local = torch.zeros(j - i, C, device=trunk.device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            out_g[i:j], out_l[i:j] = trunk.mean(dim=(2, 3)).cpu().numpy(), local.cpu().numpy()
    return out_g, out_l


def stats(g, l, on):
    """[4 d]: mean / std of g over the plies, mean / std of l over the non-pass plies (zeros when there are none)."""
    d = g.shape[1]
    out = np.zeros(4 * d, np.float32)
    if len(g):
        out[0:d], out[d:2 * d] = g.mean(axis=0), g.std(axis=0)
        li = l[on]
        if len(li):
            out[2 * d:3 * d], out[3 * d:4 * d] = li.mean(axis=0), li.std(axis=0)
    return out


def claim(path):
    try:
        os.close(os.open(str(path) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def verify():
    """The check set against the peer's rows of the same games."""
    mine = {}
    for f in sorted((OUT / "check").glob("part_p*.npz")):
        z = np.load(f)
        for k, u, r in zip(z["key"].astype(str), z["T_full"].astype(np.float32), z["T_rp"]):
            mine[k] = (u, r)
    n, worst_u, worst_r, mean_u = 0, 0.0, 0.0, []
    for f in sorted((PEER / "rows").glob("*.npz")):
        z = np.load(f)
        for k, r, u in zip(z["keys"].astype(str), z["feats"], z["full"].astype(np.float32)):
            if k in mine:
                du, dr = np.abs(mine[k][0] - u), np.abs(mine[k][1] - r)
                worst_u, worst_r = max(worst_u, float(du.max())), max(worst_r, float(dr.max()))
                mean_u.append(float(du.mean()))
                n += 1
        if n >= len(mine):
            break
    print(f"verify: {n} of {len(mine)} check games found in the peer's rows; unprojected rows max |diff| {worst_u:.5f} (mean {np.mean(mean_u):.6f}), projected rows max |diff| {worst_r:.5f}")
    return 0 if n == len(mine) and worst_u < 0.02 and worst_r < 0.02 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", nargs="+", choices=tuple(SIDES), default=["test", "queries", "fold0"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--part", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--clear-locks", action="store_true", help="remove the lock files left by a killed run first (no other worker may be running)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return verify()
    t0 = time.time()
    rows = m81.load_rows() if any(s != "test" for s in a.set) else None
    todo = {s: tasks_of(s, rows) for s in a.set}
    del rows
    print(", ".join(f"{s}: {len(v)} games (sides {SIDES[s]})" for s, v in todo.items()) + f" ({time.time() - t0:.0f} s)", flush=True)
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(te.CKPT), "cuda")
        model.eval()
        nrows = neutral_rows()
        proj_g, proj_l = (p.float().cpu().numpy() for p in te.projections(C, torch.device("cuda")))
        for name, tasks in todo.items():
            out = OUT / name
            out.mkdir(parents=True, exist_ok=True)
            if a.clear_locks:
                for f in out.glob("*.lock"):
                    f.unlink()
            sides = SIDES[name]
            n_parts = (len(tasks) + a.part - 1) // a.part
            games = plies = skipped = 0
            t1 = time.time()
            for k in range(n_parts):
                path = out / f"part_p{k:05d}.npz"
                if path.exists() or not claim(path):
                    continue
                part = tasks[k * a.part:(k + 1) * a.part]
                res = list(pool.imap(replay_side, [(*t, sides) for t in part], chunksize=2))
                done = [d for d in res if d[1] is not None and len(d[3]) > 0]
                skipped += len(res) - len(done)
                sel = [np.ones(len(d[3]), bool) for d in done]  # replay_side returned the wanted plies only
                lens = [int(m.sum()) for m in sel]
                cols = {"key": np.array([d[0] for d in done])}
                if sum(lens):
                    bins = np.concatenate([d[1][m] for d, m in zip(done, sel)])
                    globs = np.concatenate([d[2][m] for d, m in zip(done, sel)])
                    pos = np.concatenate([d[3][m] for d, m in zip(done, sel)])
                    mover = np.concatenate([d[4][m] for d, m in zip(done, sel)])
                    tgt = np.concatenate([d[5][m] for d, m in zip(done, sel)])
                    fg, fl = full_vectors(model, bins, globs, pos, mover, nrows, a.batch)
                    pg, pl = (fg @ proj_g).astype(np.float16).astype(np.float32), (fl @ proj_l).astype(np.float16).astype(np.float32)
                    on = pos < te.PASS_POS
                    starts = np.r_[0, np.cumsum(lens)]
                    for s in sides:
                        U = np.zeros((len(done), 4 * C), np.float32)
                        R = np.zeros((len(done), 4 * te.PROJ_DIM), np.float32)
                        cnt = np.zeros(len(done), np.int16)
                        for i in range(len(done)):
                            sl = slice(starts[i], starts[i + 1])
                            m = tgt[sl] if s == "T" else ~tgt[sl]
                            U[i] = stats(fg[sl][m], fl[sl][m], on[sl][m])
                            R[i] = stats(pg[sl][m], pl[sl][m], on[sl][m])
                            cnt[i] = int(m.sum())
                        if not (np.isfinite(U).all() and np.isfinite(R).all()):
                            raise ValueError(f"{name} part {k}: non-finite rows")
                        cols[f"{s}_full"], cols[f"n{s}"] = U.astype(np.float16), cnt
                        if name in KEEP_RP:
                            cols[f"{s}_rp"] = R
                else:
                    for s in sides:
                        cols[f"{s}_full"], cols[f"n{s}"] = np.zeros((len(done), 4 * C), np.float16), np.zeros(len(done), np.int16)
                tmp = path.with_name(path.name + ".tmp.npz")
                np.savez(tmp, **cols)
                os.replace(tmp, path)
                os.unlink(str(path) + ".lock")
                games += len(part)
                plies += sum(lens)
                if games % (10 * a.part) == 0 or k == n_parts - 1:
                    el = time.time() - t1
                    print(f"{name} part {k + 1}/{n_parts}: this worker {games} games, {games / el:.1f} games/s, {plies / el:.0f} plies/s, skipped {skipped} ({time.time() - t0:.0f} s)", flush=True)
            print(f"{name}: this worker did {games} games, skipped {skipped} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
