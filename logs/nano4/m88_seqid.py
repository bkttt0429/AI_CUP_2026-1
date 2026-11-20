"""Section 4M M88 (2026-11-11): single-game identification as the main model -- a sequence-level conditional matcher.
M87's funnel: the opponent is a training player in more than half of the games, the current identifier gets him at top-1 in about a
fifth, and the anchor gain is close to linear in that rate. The M82 / M84 matcher only sees a game side as mean / std (and phase means)
of the trunk vectors. Here a Transformer reads the whole game ply by ply (both players' plies, up to 256): per ply the two 64-dim trunk
vectors (board-pooled, played point), the played point, colour, ply index, the displacement to the previous ply and to the mover's own
previous move, and whether the ply belongs to the SUBJECT side (the side to identify). Output: a unit vector for the subject side.
Training is conditional, as M82: an episode takes P players x M games each; every game is a query against the prototypes (normalised
sums of the players' game vectors, its own left out); cross-entropy over the P players.
Cross-fit as M82: net v never sees fold v's players (validation queries of fold v use it), net -1 sees every player (test, held-out).
  --stage train --net v [--smoke]   trains one network (resumable), keeps the state with the best held-out top-1, writes results/m88/out_{v}.npz
  --stage merge                     results/m84_cache/var_S.npz in m84_variant.py's format (then: m84_variant.py --stage eval --tag S / SP)
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
CACHE, M82, OUT = ROOT / "results/m84_cache", ROOT / "results/m82", ROOT / "results/m88"
DIRS = (ROOT / "results/trunk_vec", ROOT / "results/trunk_vec_x40", ROOT / "results/trunk_vec_x80"); TEST = ROOT / "results/ct_submission/vec"
T_MAX, NETS = 256, (1, 2, 3, 4, -1)
D, LAYERS, HEADS, FFN, DROP, D_OUT = 384, 8, 8, 1536, 0.1, 128
P_EP, M_EP, STEPS, LR, WD, WARM, EVAL_EVERY, MIN_GAMES = 256, 8, 40000, 3e-4, 0.05, 1000, 5000, 8
NONE = 37 * 37


def read_part(csv_path):
    """One trunk-vector part as padded per-game arrays: trunk vectors [G, T_MAX, 128] fp16, point [G, T_MAX] (362 = padding), mover,
    length, the colour of the part's target player (0 Black / 1 White), and the game keys (training) or task numbers (test)."""
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:
        g, l, pos, mover, target, row, t = z["g"], z["l"], z["pos"], z["mover"], z["target"], z["game_row"], z["t"]
    n = len(tab); GL = np.zeros((n, T_MAX, 128), np.float16); POS = np.full((n, T_MAX), 362, np.int16); MOV = np.zeros((n, T_MAX), np.int8); SUBJ = np.zeros(n, np.int8)
    m = t < T_MAX; GL[row[m], t[m], :64] = g[m]; GL[row[m], t[m], 64:] = l[m]; POS[row[m], t[m]] = pos[m]; MOV[row[m], t[m]] = mover[m]; SUBJ[row[target]] = mover[target]
    train = "player_id" in tab.columns
    return (tab["game_key"].to_numpy().astype(str) if train else tab["task"].to_numpy().astype(int)), (tab["player_id"].to_numpy().astype(str) if train else tab["question_id"].to_numpy().astype(str)), GL, POS, MOV, np.minimum(np.bincount(row, minlength=n), T_MAX).astype(np.int16), SUBJ


FULL = False  # --full (M90): also the 595k games extracted by m90_ply_extract.py; outputs to results/m90, variant name S2, 60,000 steps
VAR = "S"
EXT = None  # --ext-ply / --ext-select (M101): external players' per-ply parts (M90 layout) and their game_key -> account table; they train in every network and are never candidates
EXT_SHARE = None  # --ext-share: share of an episode's players that are external (None: no quota)
SEED = 0  # --seed (M90 addendum): further training seeds of the same network, for an average of their standardised scores


def read_m90(path):
    """One part of m90_ply_extract.py as padded arrays (trunk vectors dequantised to fp16)."""
    st = np.load(ROOT / "results/m90/glstats.npz"); z = np.load(path); L = z["LEN"].astype(np.int64); n = len(L); row = np.repeat(np.arange(n), L); t = np.concatenate([np.arange(x) for x in L]) if n else np.zeros(0, np.int64)
    GL = np.zeros((n, T_MAX, 128), np.float16); POS = np.full((n, T_MAX), 362, np.int16); MOV = np.zeros((n, T_MAX), np.int8)
    GL[row, t] = (z["GLQ"].astype(np.float32) / 20.0 * st["sd"] + st["mu"]).astype(np.float16); POS[row, t] = z["POS"]; MOV[row, t] = z["MOV"]
    return z["key"].astype(str), GL, POS, MOV, z["LEN"].astype(np.int16), z["SUBJ"].astype(np.int8)


def load_games_v1(smoke, log):
    files, folds = [], []
    for d in DIRS:
        for f in range(5):
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv")); fs = fs[:12] if smoke else fs; files += fs; folds += [f] * len(fs)
    tf = sorted(TEST.glob("games_s*_p*.csv"))
    with mp.get_context("fork").Pool(12) as pool:
        res = pool.map(read_part, files, chunksize=2); rt = pool.map(read_part, tf, chunksize=2)
    cat = lambda r, j: np.concatenate([x[j] for x in r])  # noqa: E731
    G = dict(key=cat(res, 0), pid=cat(res, 1), GL=cat(res, 2), POS=cat(res, 3), MOV=cat(res, 4), LEN=cat(res, 5), SUBJ=cat(res, 6), fold=np.concatenate([np.full(len(x[0]), f, np.int8) for x, f in zip(res, folds)]))
    if FULL:
        files2 = [f for k in (0, 1, 2) for f in sorted((ROOT / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]; files2 = files2[:20] if smoke else files2
        with mp.get_context("fork").Pool(12) as pool:
            r2 = pool.map(read_m90, files2, chunksize=2)
        tab = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; own = {}
        for f_ in sorted((ROOT / "data/training").glob("train_*.csv")):
            c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str); own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))
        k2 = cat(r2, 0); p2 = np.array([own[k] for k in k2]); f2 = np.array([int(tab.get(p, -9)) for p in p2], np.int8); n_old = len(G["key"])
        G = dict(key=np.concatenate([G["key"], k2]), pid=np.concatenate([G["pid"], p2]), GL=np.concatenate([G["GL"]] + [x[1] for x in r2]), POS=np.concatenate([G["POS"]] + [x[2] for x in r2]), MOV=np.concatenate([G["MOV"]] + [x[3] for x in r2]),
                 LEN=np.concatenate([G["LEN"], cat(r2, 4)]), SUBJ=np.concatenate([G["SUBJ"], cat(r2, 5)]), fold=np.concatenate([G["fold"], f2])); del r2
        log(f"added {len(k2)} extracted games to the {n_old} stored ones")
    o = np.argsort(cat(rt, 0)); Tt = dict(task=cat(rt, 0)[o], GL=cat(rt, 2)[o], POS=cat(rt, 3)[o], MOV=cat(rt, 4)[o], LEN=cat(rt, 5)[o], SUBJ=cat(rt, 6)[o]); assert np.array_equal(Tt["task"], np.arange(len(o)))
    log(f"games loaded: {len(G['key'])} training games ({G['GL'].nbytes / 1e9:.1f} GB), {len(o)} test games; mean plies {float(G['LEN'].mean()):.0f}, truncated at {T_MAX}: {float((G['LEN'] >= T_MAX).mean()):.3f}")
    return G, Tt


def count_part(csv_path):
    return len(pd.read_csv(csv_path, usecols=[0], dtype=str, keep_default_na=False))


def count_m90(path):
    with np.load(path) as z:
        return len(z["LEN"])


def load_games(smoke, log):
    """All games as padded arrays. One worker pool, started before anything large exists, and every part written straight into
    preallocated arrays. The compute nodes run with strict overcommit (vm.overcommit_memory = 2, one commit limit for the whole node):
    the first version (load_games_v1) forked a second pool of twelve workers from a parent already holding ~50 GB and kept every part
    in a list next to the concatenated copy; with five loaders on a node that exhausted the node's commit limit (2026-11-12, 13 of 15
    networks died with MemoryError / ENOSPC). Same arrays in the same order as load_games_v1."""
    files, folds = [], []
    for d in DIRS:
        for f in range(5):
            fs = sorted((d / f"fold{f}").glob("games_s*_p*.csv")); fs = fs[:12] if smoke else fs; files += fs; folds += [f] * len(fs)
    tf = sorted(TEST.glob("games_s*_p*.csv")); files2 = []
    if FULL:
        files2 = [f for k in (0, 1, 2) for f in sorted((ROOT / f"results/m90_ply{k}").glob("ply_*.npz")) if not f.name.endswith(".tmp.npz")]; files2 = files2[:20] if smoke else files2
        if EXT:
            fx = sorted(f for f in Path(EXT[0]).glob("ply_x_*.npz") if not f.name.endswith(".tmp.npz")); files2 += fx  # all of them also in a smoke run: an account's 20 games are spread over the parts
    cat = lambda r, j: np.concatenate([x[j] for x in r])  # noqa: E731
    with mp.get_context("fork").Pool(12) as pool:
        n1 = pool.map(count_part, files, chunksize=8); n2 = pool.map(count_m90, files2, chunksize=8) if files2 else []; n_old = int(sum(n1)); N = n_old + int(sum(n2))
        GL = np.zeros((N, T_MAX, 128), np.float16); POS = np.full((N, T_MAX), 362, np.int16); MOV = np.zeros((N, T_MAX), np.int8); LEN = np.zeros(N, np.int16); SUBJ = np.zeros(N, np.int8); keys, pids, a = [], [], 0
        for x in pool.imap(read_part, files, chunksize=2):
            b = a + len(x[0]); keys.append(x[0]); pids.append(x[1]); GL[a:b] = x[2]; POS[a:b] = x[3]; MOV[a:b] = x[4]; LEN[a:b] = x[5]; SUBJ[a:b] = x[6]; a = b
        assert a == n_old, (a, n_old)
        for x in pool.imap(read_m90, files2, chunksize=2):
            b = a + len(x[0]); keys.append(x[0]); GL[a:b] = x[1]; POS[a:b] = x[2]; MOV[a:b] = x[3]; LEN[a:b] = x[4]; SUBJ[a:b] = x[5]; a = b
        assert a == N, (a, N)
        rt = pool.map(read_part, tf, chunksize=2)
    key = np.concatenate(keys); pid = np.concatenate(pids); fold = np.concatenate([np.full(n, f, np.int8) for n, f in zip(n1, folds)])
    if FULL:
        tab = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]; own = {}
        for f_ in sorted((ROOT / "data/training").glob("train_*.csv")):
            c = pd.read_csv(f_, usecols=["player_id", "game_id", "rank"], dtype=str); own.update(zip(c["rank"] + ":" + c["game_id"], c["player_id"]))
        if EXT:
            xs = pd.read_csv(EXT[1], usecols=["game_key", "account"], dtype=str); own.update(zip(xs["game_key"], "x:" + xs["account"]))
        k2 = key[n_old:]; p2 = np.array([own[k] for k in k2]); f2 = np.array([int(tab.get(p, -9)) for p in p2], np.int8); pid = np.concatenate([pid, p2]); fold = np.concatenate([fold, f2])
        log(f"added {len(k2)} extracted games to the {n_old} stored ones")
    G = dict(key=key, pid=pid, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, fold=fold)
    o = np.argsort(cat(rt, 0)); Tt = dict(task=cat(rt, 0)[o], GL=cat(rt, 2)[o], POS=cat(rt, 3)[o], MOV=cat(rt, 4)[o], LEN=cat(rt, 5)[o], SUBJ=cat(rt, 6)[o]); assert np.array_equal(Tt["task"], np.arange(len(o)))
    log(f"games loaded: {len(G['key'])} training games ({G['GL'].nbytes / 1e9:.1f} GB), {len(o)} test games; mean plies {float(G['LEN'].mean()):.0f}, truncated at {T_MAX}: {float((G['LEN'] >= T_MAX).mean()):.3f}")
    return G, Tt


def build_model():
    import torch
    from torch import nn

    class SeqID(nn.Module):
        def __init__(self):
            super().__init__()
            self.gl = nn.Linear(128, D); self.pos = nn.Embedding(363, D); self.side = nn.Embedding(2, D); self.col = nn.Embedding(2, D); self.tpos = nn.Embedding(T_MAX, D)
            self.dprev = nn.Embedding(NONE + 1, D); self.down = nn.Embedding(NONE + 1, D); self.cls = nn.Parameter(torch.zeros(1, 1, D)); self.norm = nn.LayerNorm(D)
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, HEADS, FFN, DROP, activation="gelu", batch_first=True, norm_first=True), LAYERS, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D_OUT)); self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.register_buffer("mu", torch.zeros(128)); self.register_buffer("sd", torch.ones(128))

        @staticmethod
        def disp(pos, k):
            """Index of the displacement from the ply k steps earlier to this ply ((dx + 18) * 37 + dy + 18), NONE when either is a pass / padding / missing."""
            import torch
            prev = torch.full_like(pos, 362); prev[:, k:] = pos[:, :-k]; ok = (pos < 361) & (prev < 361)
            return torch.where(ok, (pos % 19 - prev % 19 + 18) * 37 + (pos // 19 - prev // 19 + 18), torch.full_like(pos, NONE))

        def forward(self, gl, pos, mov, length, subj):
            import torch
            B, T = pos.shape; x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side((mov == subj[:, None]).long()) + self.col(mov) + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)
    return SeqID()


def stage_train(v, smoke):
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}{'_smoke' if smoke else ''}{f'_s{SEED}' if SEED else ''}"; steps = 300 if smoke else STEPS; eval_every = 100 if smoke else EVAL_EVERY
    log(f"net {v}: GPU {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    G, Tt = load_games(smoke, log); q = np.load(CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(M82 / "held_keys.json")))
    ext_p = sorted(set(p for p in G["pid"] if p.startswith("x:"))); pidx.update({p: C + i for i, p in enumerate(ext_p)}); CT = C + len(ext_p)  # external players get indices after the competition's; prototypes and queries stay competition-only
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}; n_g = len(owner)
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=CT); tp = np.flatnonzero(cnt >= MIN_GAMES); W = int(cnt.max()); tp_ext, tp_own = tp[tp >= C], tp[tp < C]; n_ext = 0 if EXT_SHARE is None or not len(tp_ext) else min(int(round(P_EP * EXT_SHARE)), len(tp_ext))
    if len(ext_p):
        log(f"external players: {len(tp_ext)} with {int((use & (owner >= C)).sum())} games" + ("" if EXT_SHARE is None else f", {n_ext} of the {P_EP} players of every episode"))
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((CT, W), -1, np.int64); tab[o, pos_] = idx
    rng = np.random.default_rng(880 + v + 1000 * SEED); torch.manual_seed(880 + v + 1000 * SEED); net = build_model().to(dev)
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32); lens = G["LEN"][:1]  # noqa: F841
    flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ck = OUT / f"ckpt_{tag}.pt"; start = 0; best = {"top1": -1.0, "step": -1}; best_state = None
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; log(f"resumed at step {start}, best so far {best}")
    tb = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731

    def embed(src, rows, subj):
        """Unit vectors of games `rows` of `src` (G or the test dict) with subject colours `subj`."""
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx]); h_truth = owner[hk]; seen = G["fold"][hk] != v

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        S = embed(G, hk, G["SUBJ"][hk]) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res = {"step": step}
        for name, s in (("seen", seen), ("unseen", ~seen)):
            if s.any():
                res[name] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}
        log(f"  step {step}: held-out own games " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if k != "step"))
        return res, Pm
    hist = []; net.train(); ar = np.arange(P_EP); tl = 0.0; ta = 0.0; t_last = time.time()
    for step in range(start, steps):
        pl = rng.choice(tp, P_EP, replace=False) if not n_ext else np.concatenate([rng.choice(tp_ext, n_ext, replace=False), rng.choice(tp_own, P_EP - n_ext, replace=False)]); u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(tb(G["GL"][rows]), tb(G["POS"][rows].astype(np.int64)), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % (50 if smoke else 500) == 0:
            k_ = 50 if smoke else 500; log(f"  step {step + 1}/{steps}: loss {tl / k_:.4f}, episode top-1 {ta / k_:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / k_:.2f} s/step"); tl = ta = 0.0; t_last = time.time()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            res, _ = evaluate(step + 1) if len(hk) else ({"step": step + 1, "seen": {"n": 0, "top1": float(step + 1), "top5": 0.0, "top20": 0.0}}, None); hist.append(res); key_ = "seen" if "seen" in res else "unseen"
            if res[key_]["top1"] > best["top1"]:
                best = {"top1": res[key_]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state}, ck); t_last = time.time()
    net.load_state_dict(best_state); log(f"net {v}: best held-out top-1 {best['top1']:.3f} at step {best['step']}")
    if not len(hk):  # smoke runs on a subset may hold none of the held-out games: one fake query keeps the code path alive
        hk = np.flatnonzero(~is_held)[:8]; h_truth = owner[hk]; seen = np.ones(len(hk), bool)
    res, Pm = evaluate(best["step"]); Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv); (OUT / f"train_{tag}.json").write_text(json.dumps({"best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum())}, indent=1))
    log(f"net {v}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game)")


def stage_merge(seeds):
    q = np.load(CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, D_OUT), np.float32); qvec = np.zeros((len(seeds), N, D_OUT), np.float16); rep = {}; var = VAR if len(seeds) == 1 else VAR + "e"
    for si, sd_ in enumerate(seeds):
        for j, v in enumerate(NETS):
            t_ = f"{v}{f'_s{sd_}' if sd_ else ''}"; z = np.load(OUT / f"out_{t_}.npz"); protos[si, j] = z["protos"]; qvec[si, z["qidx"]] = z["qvec"]; rep[t_] = json.loads((OUT / f"train_{t_}.json").read_text())["best"]
    np.savez(CACHE / f"var_{var}.npz", protos=protos, qvec=qvec); (CACHE / f"var_{var}.json").write_text(json.dumps(rep, indent=1)); print(f"var_{var} written ({len(seeds)} seed(s)); best held-out top-1 per net: " + ", ".join(f"{k}: {x['top1']:.3f} (step {x['step']})" for k, x in rep.items()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("train", "merge")); ap.add_argument("--net", type=int, default=-1); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", default="0"); ap.add_argument("--ext-ply", default=None); ap.add_argument("--ext-select", default=None); ap.add_argument("--ext-share", type=float, default=None); ap.add_argument("--out", default=None, help="output folder under results/ and variant name, as folder:VAR (M101: m101:S3)"); ap.add_argument("--steps", type=int, default=None, help="training steps = end of the cosine schedule (default 40,000; 60,000 with --full)"); a_ = ap.parse_args(); SEED = a_.seed
    if a_.full:
        FULL, VAR, OUT, STEPS = True, "S2", ROOT / "results/m90", 60000
    if a_.ext_ply:
        assert a_.full and a_.ext_select and a_.out, "--ext-ply needs --full, --ext-select and --out"; EXT, EXT_SHARE = (a_.ext_ply, a_.ext_select), a_.ext_share
    if a_.out:
        OUT, VAR = ROOT / "results" / a_.out.split(":")[0], a_.out.split(":")[1]
    if a_.steps:
        STEPS = a_.steps
    if a_.stage == "train":
        stage_train(a_.net, a_.smoke)
    else:
        stage_merge([int(x) for x in a_.seeds.split(",")])
