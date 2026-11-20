#!/usr/bin/env python
"""Per-ply trunk vectors of the scored games: the pristine humanv0 trunk, neutral metadata, both players' plies.

Each ply stores the board-pooled and the played-point trunk vector, each through a fixed random projection to 64 dimensions
(fp16), taken from the position before the move; a pass has no point, so its played-point vector is zero. The games are the
selection of scripts/score_folds.py (up to 20 per player), replayed with the handicap fix at whiteKomi 7.5, so the plies line
up with the per-move scores by (game_key, t). Parts resume per game, as the scoring runs. Fold 0 is refused.

  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 1 2 3 4 --out results/trunk_vec --workers 10
Section 4M M3 (2026-11-04): --trunk <stage-2 heads .pt> swaps in that run's fine-tuned trunk (l1b.load_heads, sha256-checked) with
the same projection; the out-of-fold rule of the L1b scores applies (folds 1, 2 with heads3_B, folds 3, 4 with heads3_A), and each
fold directory records its trunk in trunk.json and refuses another one:
  bash scripts/nano4/shards.sh 8 python -I scripts/trunk_extract.py --folds 1 2 --trunk results/l1b/heads3_B.pt --out results/trunk_vec_ft
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "scripts"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from katago.game.board import Board  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import load_heads, neutral_rows, trunk_features  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.pilot import start_state  # noqa: E402
from aicup.scoring import fold_players, load_games, part_paths, player_games, read_game_keys, restrict_players, resume_plan  # noqa: E402
from aicup.sgf import setup_and_moves  # noqa: E402

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"
RULES = dict(GameState.RULES_CHINESE, whiteKomi=7.5)
PROJ_SEED = 20261103
PROJ_DIM = 64
PASS_POS = 361
_FEATURES = None
QUANT = None  # section 4M M135 (--quant8): (mu [128], sd [128]); g and l are then stored as int8 in units of sd / 20 (keys gq, lq)


def trunk_info(heads_path):
    """What a fold directory records about the trunk its vectors came from: the pristine checkpoint, or a stage-2 run's heads file
    with its fine-tuned trunk's sha256 (from the sidecar)."""
    if heads_path is None:
        return {"trunk": "pristine", "ckpt": CKPT.name}
    hp = Path(heads_path)
    meta = json.loads(hp.with_suffix(".json").read_text())
    if not meta.get("trunk_file"):
        raise SystemExit(f"{hp}: not a stage-2 run (no fine-tuned trunk in its sidecar)")
    return {"trunk": "fine-tuned", "heads": hp.name, "trunk_file": meta["trunk_file"], "trunk_sha256": meta["trunk_sha256"]}


def check_trunk_meta(fold_dir, info):
    """trunk.json of a fold directory: written when missing, compared when present (SystemExit on a different trunk), so a
    directory never mixes vectors of two trunks. A directory with parts but no trunk.json is the pristine trunk's."""
    fold_dir = Path(fold_dir)
    path = fold_dir / "trunk.json"
    if path.exists():
        old = json.loads(path.read_text())
    elif any(fold_dir.glob("games_s*_p*.csv")):
        old = {"trunk": "pristine", "ckpt": CKPT.name}
    else:
        old = None
    if old is not None and old != info:
        raise SystemExit(f"{fold_dir}: holds vectors of trunk {old}, not {info}")
    if old is None or not path.exists():
        path.write_text(json.dumps(info, indent=1))


def init_worker(config):
    global _FEATURES
    from katago.game.features import Features
    _FEATURES = Features(config, 19)


def replay(task):
    """Inputs before every ply of one game, in play order."""
    idx, sgf, color = task
    stones, moves = setup_and_moves(sgf)
    gs = start_state(stones, RULES)
    bins, globs, pos, mover, target = [], [], [], [], []
    for c, x, y in moves:
        pla = Board.BLACK if c == "B" else Board.WHITE
        loc = Board.PASS_LOC if x is None else gs.board.loc(x, y)
        b, g = gs.get_input_features(_FEATURES)
        bins.append(b[0].astype(np.uint8))
        globs.append(g[0].astype(np.float32))
        pos.append(PASS_POS if x is None else _FEATURES.loc_to_tensor_pos(loc, gs.board))
        mover.append(0 if c == "B" else 1)
        target.append(c == color)
        gs.play(pla, loc)
    return idx, np.stack(bins), np.stack(globs), np.asarray(pos, np.int16), np.asarray(mover, np.int8), np.asarray(target, bool)


def projections(channels, device):
    gen = torch.Generator().manual_seed(PROJ_SEED)
    scale = channels ** 0.5
    return (torch.randn(channels, PROJ_DIM, generator=gen) / scale).to(device), (torch.randn(channels, PROJ_DIM, generator=gen) / scale).to(device)


def vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, batch):
    """[n, 64] fp16 pooled and played-point vectors of a stack of positions."""
    device = proj_g.device
    n = len(pos)
    out_g, out_l = np.empty((n, PROJ_DIM), np.float16), np.empty((n, PROJ_DIM), np.float16)
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b, g = torch.as_tensor(bins[i:j], device=device), torch.as_tensor(globs[i:j], device=device)
            c = torch.as_tensor(mover[i:j], device=device).long()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                trunk, _, _, _ = trunk_features(model, b, g, c, rows)
            trunk = trunk.float()
            p = torch.as_tensor(pos[i:j], device=device).long()
            on = p < PASS_POS
            local = torch.zeros(j - i, trunk.shape[1], device=device)
            local[on] = trunk[on.nonzero()[:, 0], :, p[on] // 19, p[on] % 19]
            out_g[i:j] = (trunk.mean(dim=(2, 3)) @ proj_g).cpu().numpy().astype(np.float16)
            out_l[i:j] = (local @ proj_l).cpu().numpy().astype(np.float16)
    return out_g, out_l


def write_part(fold_dir, shard, k, done, meta, model, proj_g, proj_l, rows, batch):
    bins = np.concatenate([d[1] for d in done])
    globs = np.concatenate([d[2] for d in done])
    pos = np.concatenate([d[3] for d in done])
    mover = np.concatenate([d[4] for d in done])
    target = np.concatenate([d[5] for d in done])
    lens = np.array([len(d[3]) for d in done])
    g, l = vectors(model, bins, globs, pos, mover, proj_g, proj_l, rows, batch)
    if not (np.isfinite(g).all() and np.isfinite(l).all()):
        raise ValueError(f"part {k}: non-finite vector")
    game_row = np.repeat(np.arange(len(done), dtype=np.int32), lens)
    t = np.concatenate([np.arange(n, dtype=np.int16) for n in lens])
    csv_path, npz_path = part_paths(fold_dir, shard, k)
    tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
    if QUANT is not None:  # section 4M M135: int8 in units of sd / 20 around mu (the reader restores g, l: trunk_seq_check.part_gl)
        mu, sd = QUANT
        q8 = lambda x, m, s: np.clip(np.rint((x.astype(np.float32) - m) / s * 20.0), -127, 127).astype(np.int8)  # noqa: E731
        np.savez(tmp, gq=q8(g, mu[:PROJ_DIM], sd[:PROJ_DIM]), lq=q8(l, mu[PROJ_DIM:], sd[PROJ_DIM:]), pos=pos, mover=mover, target=target, game_row=game_row, t=t)
    else:
        np.savez(tmp, g=g, l=l, pos=pos, mover=mover, target=target, game_row=game_row, t=t)
    os.replace(tmp, npz_path)
    table = pd.DataFrame([{**meta[d[0]], "n_plies": int(n)} for d, n in zip(done, lens)])
    table.to_csv(csv_path.with_name(csv_path.name + ".tmp.csv"), index=False)
    os.replace(csv_path.with_name(csv_path.name + ".tmp.csv"), csv_path)
    return int(lens.sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=100)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))
    ap.add_argument("--trunk", default=None, help="section 4M M3: a stage-2 heads .pt whose fine-tuned trunk replaces the pristine one")
    ap.add_argument("--max-games", type=int, default=20, help="scoring.fold_players' cap per player (section 4M M8: 40 with --game-keys)")
    ap.add_argument("--game-keys", default=None, help="only these games of each fold's selection (a CSV with a game_key column)")
    ap.add_argument("--allow-fold0", action="store_true", help="section 4M (fold 0 joins the submission's training players, the user's decision)")
    ap.add_argument("--quant8", default=None, help="section 4M M135: an npz with mu, sd [128] (results/m90/glstats.npz); g and l are stored as int8 in units of sd / 20, the statistics are copied to <fold dir>/quant8.npz")
    a = ap.parse_args(argv)
    if 0 in a.folds and not a.allow_fold0:
        ap.error("fold 0 is not used here (--allow-fold0 for the submission's training set)")
    only = read_game_keys(a.game_keys) if a.game_keys else None
    config = torch.load(CKPT, map_location="cpu", weights_only=True)["config"]
    all_games, folds_df = player_games(), pd.read_csv(a.folds_csv)
    pool = mp.get_context("fork").Pool(a.workers, initializer=init_worker, initargs=(config,))
    try:
        model, _ = load_humansl(str(CKPT), "cuda")
        info = trunk_info(a.trunk)
        if a.quant8:  # section 4M M135
            global QUANT
            st = np.load(a.quant8)
            QUANT = (st["mu"].astype(np.float32), st["sd"].astype(np.float32))
            if QUANT[0].shape != (2 * PROJ_DIM,) or QUANT[1].shape != (2 * PROJ_DIM,) or not (QUANT[1] > 0).all():
                raise SystemExit(f"--quant8 {a.quant8}: mu and sd must be [{2 * PROJ_DIM}] with sd > 0")
            info = {**info, "quant8": "sd/20"}
        if a.trunk:
            _, side, warnings = load_heads(a.trunk, model, "cuda")  # loads the fine-tuned trunk into `model` in place (sha256-checked)
            for w in warnings:
                print(f"warning: {w}", flush=True)
            print(f"trunk: fine-tuned, {Path(a.trunk).name} -> {side['trunk_file']} (sha256 {side['trunk_sha256'][:12]})", flush=True)
        model.eval()
        proj_g, proj_l = None, None
        rows = neutral_rows()
        for fold in a.folds:
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=5)
            if only is not None:
                players = restrict_players(players, only)
            meta = {k: {"game_key": k, "player_id": r.player_id, "rank": r.rank} for r in players.itertuples() for k in r.game_keys}
            keys = sorted(meta)[a.shard::a.nshards]
            fold_dir = Path(a.out) / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            check_trunk_meta(fold_dir, info)
            if QUANT is not None and not (fold_dir / "quant8.npz").exists():
                np.savez(fold_dir / f"quant8.tmp{a.shard}.npz", mu=QUANT[0], sd=QUANT[1])
                os.replace(fold_dir / f"quant8.tmp{a.shard}.npz", fold_dir / "quant8.npz")
            plan = resume_plan(keys, fold_dir, a.shard, a.part_size)
            t0, plies = time.time(), 0
            for n_part, (k, part_keys) in enumerate(plan):
                sgf = load_games(part_keys)
                tasks = [(q, *sgf[q]) for q in part_keys]
                for q in part_keys:
                    meta[q]["color"] = sgf[q][1]
                done = list(pool.imap(replay, tasks, chunksize=1))
                if proj_g is None:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        trunk, _, _, _ = trunk_features(model, torch.as_tensor(done[0][1][:1]).cuda(), torch.as_tensor(done[0][2][:1]).cuda(),
                                                        torch.as_tensor(done[0][4][:1]).cuda().long(), rows)
                    proj_g, proj_l = projections(trunk.shape[1], trunk.device)
                plies += write_part(fold_dir, a.shard, k, done, meta, model, proj_g, proj_l, rows, a.batch)
                print(f"fold {fold} part {n_part + 1}/{len(plan)}: {plies} plies, {plies / (time.time() - t0):.0f} plies/s", flush=True)
            print(f"fold {fold} done: {len(keys)} games in this shard, {plies} plies this run, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        pool.close()
        pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
