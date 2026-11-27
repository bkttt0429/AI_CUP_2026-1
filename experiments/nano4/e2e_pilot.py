#!/usr/bin/env python
"""Section 4M M34 / M35 (2026-11-06; user: 「讓模型自己學我們都沒辦法解釋的特徵」→「還沒試的都試試看」): a game-level pilot in which the
features come from the humanv0 trunk itself instead of the fixed random 64-dim projections of the production tokens.

  prepare  replay the games once (CPU pool) into DIR (packbits input planes, globals, played point, mover, target flag):
           training = each player of folds 0, 2, 3, 4 with its first --games-per-player (4) games of the 20-game selection by sorted
           game_key; evaluation = the M25 pilot's 600 fold-1 players, 12,000 games (results/ai_search_moves/fold1/shard*_games.csv)
  train    one arm, one seed, one GPU; the trunk (b18c384nbt-humanv0, neutral metadata, eval mode) feeds a per-ply head:
             mean    [board mean 384, played point 384] -> Linear(768, 128) -> ReLU (a learned projection; trunk frozen)
             attn    (M34) 1x1 conv 384 -> 96, four softmax spatial-attention heads over the 19x19 board -> 4 x 96, + the played
                     point's 96 -> Linear(480, 128) -> ReLU (trunk frozen)
             ft      (M35) mean, with the trunk's last block and its final norm trainable at lr 1e-5 (everything else frozen)
             ftattn  attn + the same fine-tuning
           per game: mean and std over the target's plies and over the opponent's -> Linear(512, 64) -> ReLU -> Dropout -> 10 classes;
           class-balanced game-level cross-entropy, 2 epochs, 16 games a batch, AdamW (head lr 1e-3, wd 1e-4), cosine; then the
           log-softmax of every evaluation game
  stats    each arm (seed-averaged and per seed): the player's 20 game log-softmaxes summed -> posterior -> expected rank s;
           partial(s, y | E) with E the ct7 (M22) val1 members' expected rank (C-f0-x40-oh, T-f0-x40-oh, seeds 0-2), bootstrap 95% CI;
           Spearman(s, y); the game-level adjacent-rank AUC. Continue (pre-registered): partial CI lower > 0 and >= 0.08.

Section 4M M36 (2026-11-06): the attn arm cross-fitted (prepare --train-folds 1 2 --games-per-player 8 --no-eval; train --save-head),
then `games` lists every game of the R caches' questions and `score` writes each game's log-softmax under heads A and B (the trunk runs
once per ply, both heads read it); the consumer picks B for folds 1-2, A for folds 3-4, their mean for fold 0 and the test set.
  python -I logs/nano4/e2e_pilot.py games --out results/r_variants/attn_games.csv
  python -I logs/nano4/e2e_pilot.py score --data - --games results/r_variants/attn_games.csv --heads A.pt B.pt --out results/attn_scores --shard 0 --nshards 32

  python -I logs/nano4/e2e_pilot.py prepare --data /dev/shm/e2e --workers 88
  python -I logs/nano4/e2e_pilot.py train --data /dev/shm/e2e --arm attn --seed 0 --out results/r_variants/e2e
  python -I logs/nano4/e2e_pilot.py stats --data /dev/shm/e2e --out-dir results/r_variants/e2e --out results/r_variants/e2e_pilot.json
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "logs/nano4")]
import trunk_extract as te  # noqa: E402  (replay, init_worker, CKPT)
from aicup.humansl import load_humansl  # noqa: E402
from aicup.l1b import neutral_rows, trunk_features  # noqa: E402
from aicup.ladder import FOLDS_CSV  # noqa: E402
from aicup.metric import LEVELS  # noqa: E402
from aicup.rankmodel import class_weights  # noqa: E402
from aicup.scoring import fold_players, load_games, player_games  # noqa: E402

RANK = {lv: i for i, lv in enumerate(LEVELS)}
TRAIN_FOLDS = (0, 2, 3, 4)
ARMS = ("mean", "attn", "ft", "ftattn")
PASS = te.PASS_POS
D_PLY = 128


# ---------------------------------------------------------------- prepare
def game_sets(gpp, smoke, train_folds=TRAIN_FOLDS, with_eval=True):
    all_games, folds_df = player_games(), pd.read_csv(FOLDS_CSV)
    train = []
    for f in train_folds:
        for r in fold_players(all_games, folds_df, f, max_games=20, min_games=5).itertuples():
            train += [(k, str(r.player_id), RANK[r.rank]) for k in sorted(r.game_keys)[:gpp]]
    evl = []
    if with_eval:
        ev = pd.concat([pd.read_csv(p, dtype={"player_id": str, "game_key": str}) for p in sorted(glob.glob(str(ROOT / "results/ai_search_moves/fold1/shard*_games.csv")))])
        ev = ev.drop_duplicates("game_key")
        evl = [(k, p, RANK[r]) for k, p, r in zip(ev["game_key"], ev["player_id"], ev["rank"])]
    if smoke:
        train, evl = train[::97][:160], evl[:120]
    return train, evl


def replay_split(pool, rows, name, out):
    """Replay rows [(game_key, player_id, label)] and write <name>_*.npy plus <name>_meta.json into out."""
    keys = [k for k, _, _ in rows]
    sgf = load_games(keys)
    tasks = [(i, *sgf[k]) for i, k in enumerate(keys)]
    bins, globs, pos, mover, target, lens = [], [], [], [], [], []
    t0 = time.time()
    for n_done, (i, b, g, p, m, tg) in enumerate(pool.imap(te.replay, tasks, chunksize=4)):
        assert i == n_done
        if n_done < 50:
            assert b.max() <= 1, "input planes are not binary"
        bins.append(np.packbits(b.reshape(len(b), -1), axis=1)); globs.append(g); pos.append(p); mover.append(m); target.append(tg); lens.append(len(p))
        if (n_done + 1) % 2000 == 0:
            print(f"{name}: {n_done + 1}/{len(tasks)} games, {sum(lens)} plies, {sum(lens) / (time.time() - t0):.0f} plies/s", flush=True)
    C = int(b.shape[1])
    np.save(out / f"{name}_bins.npy", np.concatenate(bins)); np.save(out / f"{name}_globs.npy", np.concatenate(globs).astype(np.float32))
    np.save(out / f"{name}_pos.npy", np.concatenate(pos).astype(np.int16)); np.save(out / f"{name}_mover.npy", np.concatenate(mover).astype(np.int8))
    np.save(out / f"{name}_target.npy", np.concatenate(target).astype(bool))
    off = np.r_[0, np.cumsum(lens)].astype(np.int64)
    np.save(out / f"{name}_off.npy", off)
    (out / f"{name}_meta.json").write_text(json.dumps({"keys": keys, "player_id": [p for _, p, _ in rows], "label": [int(y) for _, _, y in rows], "planes": C, "train_folds": TRAIN_FOLDS_USED,
                                                         "plies": int(off[-1]), "colors": [sgf[k][1] for k in keys]}))
    print(f"{name}: {len(keys)} games, {off[-1]} plies, {C} planes ({time.time() - t0:.0f} s)", flush=True)


TRAIN_FOLDS_USED = list(TRAIN_FOLDS)


def prepare(a):
    global TRAIN_FOLDS_USED
    TRAIN_FOLDS_USED = [int(f) for f in a.train_folds]
    out = Path(a.data); out.mkdir(parents=True, exist_ok=True)
    train, evl = game_sets(a.games_per_player, a.smoke, tuple(a.train_folds), not a.no_eval)
    print(f"training games {len(train)} ({len({p for _, p, _ in train})} players, folds {tuple(a.train_folds)}); evaluation games {len(evl)} ({len({p for _, p, _ in evl})} players)", flush=True)
    assert not ({p for _, p, _ in train} & {p for _, p, _ in evl}), "an evaluation player is among the training players"
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    try:
        replay_split(pool, train, "train", out)
        if evl:
            replay_split(pool, evl, "eval", out)
    finally:
        pool.close(); pool.join()
    return 0


# ---------------------------------------------------------------- model
class MemSplit:
    """A replayed part held in memory, with Split's attributes (section 4M M36 scoring)."""

    def __init__(self, done, C):
        self.bins = np.concatenate([np.packbits(d[1].reshape(len(d[1]), -1), axis=1) for d in done])
        self.globs = np.concatenate([d[2] for d in done]).astype(np.float32)
        self.pos = np.concatenate([d[3] for d in done]).astype(np.int16); self.mover = np.concatenate([d[4] for d in done]).astype(np.int8)
        self.target = np.concatenate([d[5] for d in done]).astype(bool)
        self.off = np.r_[0, np.cumsum([len(d[3]) for d in done])].astype(np.int64); self.C = C


class Split:
    def __init__(self, data, name):
        d = Path(data)
        self.bins = np.load(d / f"{name}_bins.npy", mmap_mode="r"); self.globs = np.load(d / f"{name}_globs.npy", mmap_mode="r")
        self.pos = np.load(d / f"{name}_pos.npy"); self.mover = np.load(d / f"{name}_mover.npy"); self.target = np.load(d / f"{name}_target.npy")
        self.off = np.load(d / f"{name}_off.npy"); self.meta = json.loads((d / f"{name}_meta.json").read_text())
        self.label = np.asarray(self.meta["label"], int); self.C = int(self.meta["planes"])


class Head(torch.nn.Module):
    def __init__(self, arm, c):
        super().__init__()
        self.attn = arm in ("attn", "ftattn")
        if self.attn:
            self.u, self.att = torch.nn.Conv2d(c, 96, 1), torch.nn.Conv2d(c, 4, 1)
            self.ply = torch.nn.Linear(4 * 96 + 96, D_PLY)
        else:
            self.ply = torch.nn.Linear(2 * c, D_PLY)
        self.game = torch.nn.Sequential(torch.nn.Linear(4 * D_PLY, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 10))

    def ply_features(self, trunk, mask, pos):
        P = trunk.shape[0]
        on = pos < PASS
        pc = torch.where(on, pos, torch.zeros_like(pos))
        rows, cols, ar = pc // 19, pc % 19, torch.arange(P, device=trunk.device)
        if self.attn:
            u = self.u(trunk)
            w = self.att(trunk).masked_fill(mask == 0, float("-inf")).flatten(2).softmax(-1)  # [P, 4, 361]
            pooled = torch.einsum("phs,pcs->phc", w, u.flatten(2)).flatten(1)
            x = torch.cat([pooled, u[ar, :, rows, cols] * on[:, None]], dim=1)
        else:
            g = (trunk * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp(min=1.0)
            x = torch.cat([g, trunk[ar, :, rows, cols] * on[:, None]], dim=1)
        return torch.relu(self.ply(x))

    def pool(self, f, game_idx, target, n_games):
        grp = game_idx * 2 + (~target).long()
        G, dev = 2 * n_games, f.device
        cnt = torch.zeros(G, device=dev).index_add_(0, grp, torch.ones(len(grp), device=dev)).clamp(min=1.0)[:, None]
        s1 = torch.zeros(G, f.shape[1], device=dev).index_add_(0, grp, f)
        s2 = torch.zeros(G, f.shape[1], device=dev).index_add_(0, grp, f * f)
        m = s1 / cnt
        sd = ((s2 / cnt - m * m).clamp(min=0.0) + 1e-6).sqrt()
        return self.game(torch.cat([m.view(n_games, -1), sd.view(n_games, -1)], dim=1))


def unpack(bits, C):
    shifts = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    return ((bits.unsqueeze(-1) >> shifts) & 1).flatten(1)[:, :C * 361].view(-1, C, 19, 19).float()


def game_logits(model, head, S, games, rows, device, grad_trunk, grad_head, chunk=2048):
    idx = np.concatenate([np.arange(S.off[g], S.off[g + 1]) for g in games])
    gi = torch.as_tensor(np.repeat(np.arange(len(games)), [S.off[g + 1] - S.off[g] for g in games]), device=device)
    feats = []
    for lo in range(0, len(idx), chunk):
        sl = idx[lo:lo + chunk]
        b = unpack(torch.as_tensor(np.asarray(S.bins[sl]), device=device), S.C)
        g = torch.as_tensor(np.asarray(S.globs[sl]), device=device)
        c = torch.as_tensor(S.mover[sl], device=device).long()
        with torch.set_grad_enabled(grad_trunk), torch.autocast("cuda", dtype=torch.bfloat16):
            trunk, mask, _, _ = trunk_features(model, b, g, c, rows)
        with torch.set_grad_enabled(grad_head), torch.autocast("cuda", dtype=torch.bfloat16):
            feats.append(head.ply_features(trunk, mask, torch.as_tensor(S.pos[sl], device=device).long()).float())
    with torch.set_grad_enabled(grad_head):
        return head.pool(torch.cat(feats), gi, torch.as_tensor(S.target[idx], device=device), len(games))


def train(a):
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    device = torch.device("cuda")
    tr = Split(a.data, "train")
    ev = Split(a.data, "eval") if (Path(a.data) / "eval_meta.json").exists() else None
    model, _ = load_humansl(str(te.CKPT), device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ft = a.arm in ("ft", "ftattn")
    trunk_params = []
    if ft:  # M35: the last trunk block and the final trunk norm learn; everything else stays as humanv0
        for m in (model.blocks[-1], model.norm_trunkfinal):
            m.requires_grad_(True)
        trunk_params = [p for p in model.parameters() if p.requires_grad]
    c = int(trunk_features(model, unpack(torch.as_tensor(np.asarray(tr.bins[:1]), device=device), tr.C), torch.as_tensor(np.asarray(tr.globs[:1]), device=device),
                           torch.zeros(1, dtype=torch.long, device=device), neutral_rows())[0].shape[1])
    head = Head(a.arm, c).to(device)
    rows = neutral_rows()
    w_all = torch.as_tensor(class_weights(tr.label), dtype=torch.float32, device=device)
    y_all = torch.as_tensor(tr.label, device=device)
    groups = [{"params": list(head.parameters()), "lr": a.lr}] + ([{"params": trunk_params, "lr": a.trunk_lr}] if ft else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    n_games = len(tr.label)
    steps = a.epochs * (n_games // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(1.0, t / max(1, steps)))))
    print(f"arm {a.arm} seed {a.seed}: {n_games} training games, {tr.off[-1]} plies; head params {sum(p.numel() for p in head.parameters())}, "
          f"trainable trunk params {sum(p.numel() for p in trunk_params)}; {steps} steps", flush=True)
    t0, step, run = time.time(), 0, 0.0
    head.train()
    for epoch in range(a.epochs):
        perm = rng.permutation(n_games)
        for i in range(0, n_games - a.batch + 1, a.batch):
            games = perm[i:i + a.batch]
            logits = game_logits(model, head, tr, games, rows, device, grad_trunk=ft, grad_head=True)
            gt = torch.as_tensor(games, device=device)
            ce = torch.nn.functional.cross_entropy(logits, y_all[gt], reduction="none")
            loss = (ce * w_all[gt]).sum() / w_all[gt].sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()) + trunk_params, 5.0)
            opt.step(); sched.step(); step += 1; run += loss.item()
            if step % a.log_every == 0:
                print(f"arm {a.arm} seed {a.seed} step {step}/{steps} (epoch {epoch + 1}): loss {run / a.log_every:.4f}  {time.time() - t0:.0f} s  "
                      f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
                run = 0.0
    head.eval()
    if a.save_head:  # section 4M M36: the trained head for `score`
        Path(a.save_head).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": a.arm, "c": c, "seed": a.seed, "epochs": a.epochs, "train_folds": tr.meta.get("train_folds"), "state": head.state_dict(),
                    "n_games": n_games, "ckpt": str(te.CKPT.name)}, a.save_head)
        print(f"saved head {a.save_head}", flush=True)
    if ev is None:
        print(f"arm {a.arm} seed {a.seed}: done in {time.time() - t0:.0f} s (no evaluation split)", flush=True)
        return 0
    out = []
    with torch.no_grad():
        for i in range(0, len(ev.label), 32):
            games = np.arange(i, min(i + 32, len(ev.label)))
            out.append(torch.log_softmax(game_logits(model, head, ev, games, rows, device, False, False).float(), dim=1).cpu().numpy())
    lp = np.concatenate(out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / f"{a.arm}_s{a.seed}.npz", logp=lp.astype(np.float32), keys=np.asarray(ev.meta["keys"]))
    acc = float((lp.argmax(1) == ev.label).mean())
    print(f"arm {a.arm} seed {a.seed}: done in {time.time() - t0:.0f} s; evaluation games {len(lp)}, game-level accuracy {acc:.4f}", flush=True)
    return 0


# ---------------------------------------------------------------- M36: games and score
def games_cmd(a):
    """Every game of the R caches' questions (r_val{v}_f0_x40, r_val{v}, v = 1-4), of every player's 20-game selection and of the x40 keys, with its player's fold."""
    keys = set()
    for v in (1, 2, 3, 4):
        for tag in (f"r_val{v}_f0_x40", f"r_val{v}"):
            m = json.load(open(ROOT / f"results/trunk_seq/{tag}.json"))
            for part in ("train", "val"):
                for ks in m[part]["game_keys"]:
                    keys.update(ks)
    all_games = player_games()
    folds_df = pd.read_csv(FOLDS_CSV)
    for f in (0, 1, 2, 3, 4):  # every game of each player's 20-game selection (r_variants draws its own training questions from them)
        for ks in fold_players(all_games, folds_df, f, max_games=20, min_games=5)["game_keys"]:
            keys.update(ks)
    keys.update(pd.read_csv(ROOT / "results/data/x40_keys.csv", dtype={"game_key": str})["game_key"])  # games 21-40 (section 4M M8)
    g = all_games.drop_duplicates("game_key").set_index("game_key")
    folds = pd.read_csv(FOLDS_CSV, dtype={"player_id": str}).set_index("player_id")["fold"]
    df = pd.DataFrame({"game_key": sorted(keys)})
    df["player_id"] = g["player_id"].astype(str).reindex(df["game_key"]).to_numpy()
    df["fold"] = folds.reindex(df["player_id"]).to_numpy()
    assert df["fold"].notna().all(), "a game without a fold"
    df["fold"] = df["fold"].astype(int)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} games; by fold " + str(df["fold"].value_counts().sort_index().to_dict()) + f" -> {a.out}", flush=True)
    return 0


def score_cmd(a):
    """Each game's log-softmax under every head in --heads (one trunk pass per ply), in parts under --out: part_<shard>_<k>.npz."""
    device = torch.device("cuda")
    keys_all = pd.read_csv(a.games, dtype={"game_key": str})["game_key"].tolist()
    keys = keys_all[a.shard::a.nshards]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model, _ = load_humansl(str(te.CKPT), device); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    heads = []
    for hp in a.heads:
        h = torch.load(hp, map_location="cpu", weights_only=False)
        head = Head(h["arm"], int(h["c"])); head.load_state_dict(h["state"]); heads.append(head.to(device).eval())
    rows = neutral_rows()
    config = torch.load(te.CKPT, map_location="cpu", weights_only=True)["config"]
    pool = mp.get_context("fork").Pool(a.workers, initializer=te.init_worker, initargs=(config,))
    t0, plies = time.time(), 0
    try:
        for k, lo in enumerate(range(0, len(keys), a.part_size)):
            path = out / f"part_{a.shard:03d}of{a.nshards:03d}_{k:04d}.npz"
            if path.exists():
                continue
            part = keys[lo:lo + a.part_size]
            sgf = load_games(part)
            done = list(pool.imap(te.replay, [(i, *sgf[q]) for i, q in enumerate(part)], chunksize=4))
            S = MemSplit(done, int(done[0][1].shape[1]))
            lps = [[] for _ in heads]
            with torch.no_grad():
                for i in range(0, len(part), 32):
                    games = np.arange(i, min(i + 32, len(part)))
                    idx = np.concatenate([np.arange(S.off[g], S.off[g + 1]) for g in games])
                    gi = torch.as_tensor(np.repeat(np.arange(len(games)), [S.off[g + 1] - S.off[g] for g in games]), device=device)
                    feats = [[] for _ in heads]
                    for c0 in range(0, len(idx), 2048):
                        sl = idx[c0:c0 + 2048]
                        b = unpack(torch.as_tensor(S.bins[sl], device=device), S.C)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            trunk, mask, _, _ = trunk_features(model, b, torch.as_tensor(S.globs[sl], device=device), torch.as_tensor(S.mover[sl], device=device).long(), rows)
                            for j, h in enumerate(heads):
                                feats[j].append(h.ply_features(trunk, mask, torch.as_tensor(S.pos[sl], device=device).long()).float())
                    tg = torch.as_tensor(S.target[idx], device=device)
                    for j, h in enumerate(heads):
                        lps[j].append(torch.log_softmax(h.pool(torch.cat(feats[j]), gi, tg, len(games)).float(), dim=1).cpu().numpy())
            np.savez(path, keys=np.asarray(part), **{f"logp{j}": np.concatenate(l).astype(np.float32) for j, l in enumerate(lps)}, heads=np.asarray([str(h) for h in a.heads]))
            plies += int(S.off[-1])
            print(f"shard {a.shard}/{a.nshards} part {k + 1}/{math.ceil(len(keys) / a.part_size)}: {plies} plies, {plies / (time.time() - t0):.0f} plies/s", flush=True)
    finally:
        pool.close(); pool.join()
    print(f"shard {a.shard} done: {len(keys)} games ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


# ---------------------------------------------------------------- stats
def adjacent_auc(score, y):
    aucs = []
    for r in range(9):
        a, b = score[y == r], score[y == r + 1]
        if len(a) < 10 or len(b) < 10:
            continue
        allv = np.r_[a, b]; rk = pd.Series(allv).rank().to_numpy()
        aucs.append((rk[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b)))
    return float(np.mean(aucs)) if aucs else float("nan")


def stats(a):
    from oppview_eval import boot, partial
    from pilot_stats import spearman
    ev = Split(a.data, "eval")
    keys, pid, lab = np.asarray(ev.meta["keys"]), np.asarray(ev.meta["player_id"]), ev.label
    meta = pd.DataFrame(json.load(open(ROOT / "results/trunk_seq/r_val1.json"))["val"])
    P = np.mean([np.load(ROOT / f"results/trunk_seq/es/val1/{arm}_s{s}.npz")["proba"].astype(float) for arm in ("C-f0-x40-oh", "T-f0-x40-oh") for s in (0, 1, 2)], axis=0)
    meta["E"] = P @ np.arange(len(LEVELS))
    E = meta[meta.n_games == 20].drop_duplicates("player_id").set_index("player_id")["E"]
    players = sorted(set(pid) & set(E.index.astype(str)))
    E = E.copy(); E.index = E.index.astype(str)
    rep = {"players": len(players), "games": int(len(keys)), "arms": {}}
    if len(players) < 30:
        print(f"only {len(players)} players: statistics skipped (smoke)")
        Path(a.out).write_text(json.dumps({**rep, "smoke": True}, indent=1)); return 0
    yp = np.array([lab[pid == p][0] for p in players], float); Ev = E.reindex(players).to_numpy(float)
    out_dir = Path(a.out_dir)
    for arm in ARMS:
        files = sorted(out_dir.glob(f"{arm}_s*.npz"))
        if not files:
            continue
        lps = []
        for f in files:
            z = np.load(f)
            assert np.array_equal(z["keys"], keys), f"{f}: evaluation games out of order"
            lps.append(z["logp"])
        variants = {f"{arm}": np.mean(lps, axis=0), **{f"{arm}_{Path(f).stem.split('_')[-1]}": l for f, l in zip(files, lps)}}
        for name, lp in variants.items():
            s = np.empty(len(players))
            for i, p in enumerate(players):
                tot = lp[pid == p].sum(axis=0); post = np.exp(tot - tot.max()); post /= post.sum()
                s[i] = post @ np.arange(10)
            game_s = np.exp(lp) @ np.arange(10)
            pe = partial(s, yp, [Ev]); ci = boot(lambda idx: partial(s[idx], yp[idx], [Ev[idx]]), len(s))
            r = {"partial_given_E": pe, "ci95": ci, "spearman_y": spearman(s, yp), "game_adjacent_auc": adjacent_auc(game_s, lab), "game_accuracy": float((lp.argmax(1) == lab).mean()),
                 "continue": bool(ci[0] > 0 and pe >= 0.08), "n_seeds": len(files) if name == arm else 1}
            rep["arms"][name] = r
            print(f"{name:10s} partial|E {pe:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  Spearman y {r['spearman_y']:.3f}  game adjacent AUC {r['game_adjacent_auc']:.4f}  "
                  f"game accuracy {r['game_accuracy']:.4f}" + ("  -> CONTINUE" if r["continue"] and name == arm else ""), flush=True)
    rep["E_spearman_y"] = spearman(Ev, yp)
    print(f"reference: E Spearman y {rep['E_spearman_y']:.3f} on these {len(players)} players")
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float)); print("written", a.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("prepare", "train", "stats", "games", "score"))
    ap.add_argument("--data", default="")
    ap.add_argument("--train-folds", type=int, nargs="+", default=list(TRAIN_FOLDS), help="prepare: the training players' folds (M36: 1 2 or 3 4)")
    ap.add_argument("--no-eval", action="store_true", help="prepare: no evaluation split (M36 cross-fit heads)")
    ap.add_argument("--save-head", default=None, help="train: save the trained head here (M36)")
    ap.add_argument("--games", default=None, help="score: a CSV with a game_key column")
    ap.add_argument("--heads", nargs="+", default=[], help="score: saved heads, each game gets logp<j> for head j")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--part-size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=88)
    ap.add_argument("--games-per-player", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/r_variants/e2e"))
    ap.add_argument("--out-dir", default=str(ROOT / "results/r_variants/e2e"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trunk-lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--log-every", type=int, default=100)
    a = ap.parse_args(argv)
    if a.cmd == "train" and not a.arm:
        ap.error("train needs --arm")
    if a.cmd in ("prepare", "train", "stats") and not a.data:
        ap.error(f"{a.cmd} needs --data")
    if a.cmd == "score" and not (a.games and a.heads):
        ap.error("score needs --games and --heads")
    return {"prepare": prepare, "train": train, "stats": stats, "games": games_cmd, "score": score_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
