#!/usr/bin/env python
"""Engineering probe for the trunk + TCN + whole-question design: 100 fold-1 games, every ply. Report only; fold 0 refused.

Measures replay and trunk throughput, bytes per ply, whether the played-move vector indexes the right board point,
how the stored per-move scores line up with plies, and the cost of a TCN forward/backward with the gradient reaching it.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402
from aicup.pilot import start_state  # noqa: E402
from aicup.scoring import load_games  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
RULES = dict(GameState.RULES_CHINESE, whiteKomi=7.5)
_FEATURES = None


def init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def replay(task):
    """Inputs before every ply of one game, in play order."""
    key, sgf, color = task
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, RULES)
    P = _FEATURES.pass_pos
    bins, globs, pos, mover, is_target = [], [], [], [], []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        b, g = gs.get_input_features(_FEATURES)
        bins.append(b[0].astype(np.uint8))
        globs.append(g[0].astype(np.float32))
        pos.append(P if x is None else _FEATURES.loc_to_tensor_pos(loc, gs.board))
        mover.append(0 if c == "B" else 1)
        is_target.append(c == color)
        gs.play(pla, loc)
    return key, np.stack(bins), np.stack(globs), np.asarray(pos), np.asarray(mover), np.asarray(is_target)


class Block(torch.nn.Module):
    """One dilated kernel-3 convolution per block, so five blocks with dilations 1..16 see 63 plies."""

    def __init__(self, d, dilation):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.conv = torch.nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + torch.relu(self.conv(self.norm(x).transpose(1, 2))).transpose(1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1b", default="results/l1b/scores3_B_hd")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--n-games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.fold == 0:
        ap.error("fold 0 is not used here")
    fold_dir = ROOT / a.l1b / f"fold{a.fold}"
    games = pd.concat([pd.read_csv(p) for p in sorted(fold_dir.glob("games_s*_p000.csv"))], ignore_index=True)
    games = games.sample(n=a.n_games, random_state=7).reset_index(drop=True)
    sgf = load_games(games["game_key"].tolist())
    tasks = [(k, *sgf[k]) for k in games["game_key"]]
    rep = {"fold": a.fold, "games": len(tasks)}

    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    model, _ = load_humansl(str(CKPT), "cuda")
    model.eval()
    t0 = time.time()
    done = pool.map(replay, tasks, chunksize=4)
    pool.close()
    rep["replay_s"] = time.time() - t0
    bins = np.concatenate([d[1] for d in done])
    globs = np.concatenate([d[2] for d in done])
    pos = np.concatenate([d[3] for d in done])
    mover = np.concatenate([d[4] for d in done])
    target = np.concatenate([d[5] for d in done])
    lens = np.array([len(d[3]) for d in done])
    n, passes = len(pos), pos == 361
    rep.update(plies=int(n), plies_per_game_mean=float(lens.mean()), plies_per_game_p50=float(np.median(lens)), plies_per_game_max=int(lens.max()),
               target_share=float(target.mean()), passes=int(passes.sum()), input_mb_per_1k_plies=float((bins[:1000].nbytes + globs[:1000].nbytes) / 1e6))

    # Fixed, seeded projections: both are reproducible and carry no label information.
    gen = torch.Generator().manual_seed(20261103)
    proj_g = (torch.randn(384, 64, generator=gen) / 384 ** 0.5).cuda()
    proj_l = (torch.randn(384, 64, generator=gen) / 384 ** 0.5).cuda()
    rows = neutral_rows()
    vec_g, vec_l, raw_var = np.empty((n, 64), np.float16), np.empty((n, 64), np.float16), []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, a.batch):
            j = min(n, i + a.batch)
            b, g = torch.as_tensor(bins[i:j]).cuda(), torch.as_tensor(globs[i:j]).cuda()
            c = torch.as_tensor(mover[i:j]).cuda().long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)
            trunk = trunk.float()
            p = torch.as_tensor(pos[i:j]).cuda().long()
            on = p < 361
            local = torch.zeros(j - i, trunk.shape[1], device=trunk.device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            pooled = trunk.mean(dim=(2, 3))
            if i == 0:
                rep["trunk_channels"] = int(trunk.shape[1])
                raw_var = [float(pooled.std(0).mean()), float(local[on].std(0).mean())]
            vec_g[i:j] = (pooled @ proj_g).cpu().numpy().astype(np.float16)
            vec_l[i:j] = (local @ proj_l).cpu().numpy().astype(np.float16)
    rep["trunk_s"] = time.time() - t0
    rep["trunk_plies_per_s_one_gpu"] = n / rep["trunk_s"]
    rep["trunk_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    rep["pooled_local_channel_sd"] = raw_var
    rep["vectors_finite"] = bool(np.isfinite(vec_g).all() and np.isfinite(vec_l).all())
    rep["vector_sd"] = [float(vec_g.std(0).mean()), float(vec_l.std(0).mean())]

    # Index check: the point played at ply t must hold a stone in the input of ply t + 1 unless it was captured at once.
    start = 0
    hit, tot = 0, 0
    for L in lens:
        for t in range(start, start + L - 1):
            if pos[t] < 361:
                y, x = pos[t] // 19, pos[t] % 19
                hit += int(bins[t + 1, 1, y, x] or bins[t + 1, 2, y, x])
                tot += 1
        start += L
    rep["played_point_holds_stone_next_ply"] = hit / tot

    # Stored scores: does a stored row's t count target moves or plies, and do the counts match this replay?
    z = {}
    for p in sorted(fold_dir.glob("games_s*_p000.csv")):
        s = pd.read_csv(p, usecols=["game_key"])["game_key"].tolist()
        with np.load(str(p).replace("games_", "moves_").replace(".csv", ".npz")) as f:
            row, tt = f["game_row"], f["t"]
        for i, k in enumerate(s):
            z[k] = tt[row == i]
    matches, as_index, as_ply = 0, 0, 0
    for k, tgt in zip(games["game_key"], [d[5] for d in done]):
        t = z[k]
        plies = np.flatnonzero(tgt)
        matches += int(len(t) == len(plies))
        as_index += int(len(t) == len(plies) and np.array_equal(t, np.arange(len(plies))))
        as_ply += int(len(t) == len(plies) and np.array_equal(t, plies))
    rep.update(stored_count_matches=matches, stored_t_is_target_index=as_index, stored_t_is_ply=as_ply)

    # TCN cost on real tokens: a 20-game question, gradient back to the first block.
    d_in = 64 + 64 + 28 + 6
    x = np.concatenate([vec_g, vec_l], axis=1).astype(np.float32)
    keep = int(np.percentile(lens, 95))
    tokens = torch.zeros(20, keep, d_in).cuda()
    start = 0
    for g_i, L in enumerate(lens[:20]):
        m = min(int(L), keep)
        tokens[g_i, :m, :128] = torch.as_tensor(x[start:start + m]).cuda()
        start += int(L)
    proj, blocks = torch.nn.Linear(d_in, 64).cuda(), torch.nn.ModuleList([Block(64, 2 ** k) for k in range(5)]).cuda()
    head = torch.nn.Linear(128, 10).cuda()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steps = 50
    for _ in range(steps):
        h = proj(tokens)
        for blk in blocks:
            h = blk(h)
        game = torch.cat([h.mean(1), h.std(1)], dim=1)
        loss = torch.nn.functional.cross_entropy(head(game.mean(0, keepdim=True)), torch.tensor([3]).cuda())
        loss.backward()
    wall = time.time() - t0
    rep["tcn"] = {"question_games": 20, "plies_per_game_padded": keep, "steps": steps, "s_per_question": wall / steps,
                  "params": sum(p.numel() for p in proj.parameters()) + sum(p.numel() for p in blocks.parameters()),
                  "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                  "first_block_grad_norm": float(blocks[0].conv.weight.grad.norm()), "receptive_field_plies": 1 + 2 * sum(2 ** k for k in range(5))}
    rep["bytes_per_ply"] = {"vectors_fp16": 256, "with_28_scores_and_16_index": 328}

    # Measured, not assumed: what the real file costs on disk.
    tmp = Path(a.out).with_suffix(".npz")
    np.savez(tmp, g=vec_g, l=vec_l, pos=pos.astype(np.int16), mover=mover.astype(np.int8), target=target)
    rep["npz_bytes_per_ply"] = tmp.stat().st_size / n
    tmp.unlink()
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
