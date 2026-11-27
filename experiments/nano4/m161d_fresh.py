#!/usr/bin/env python
"""Section 4M M161d (2026-11-15, pilot): prototypes from games the network has NOT trained on (design doc M161d).

At test time the query is a game the network never saw, the prototype is the mean vector of games it trained on (partly
memorised). Every player's games are split in two halves by a hash of the game key; the network trains on half A only
(m161p_prod's episode loss; variant t1: players drawn by their games, t0: uniform players); the held-out own games are scored
against three sets of prototypes: half A (trained on), half B (never trained on, the same number of games), A + B.
B against A = the effect of freshness alone.

  sbatch -p 8gpus --gres=gpu:2 -c 24 ... run.sbatch python -I logs/nano4/m161d_fresh.py --jobs t1 t0 --steps 12000
"""
import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402
from m112_idaug import sym_table  # noqa: E402

OUT = ROOT / "results/m161"; BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; VARIANTS = {"t1": 1.0, "t0": 0.0}


def train_one(name, G, steps, log):
    import torch
    F = torch.nn.functional; gamma = VARIANTS[name]; dev = "cuda"; v = -1; P_EP, M_EP, D_OUT = base.P_EP, base.M_EP, base.D_OUT; OUT.mkdir(parents=True, exist_ok=True)
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}; half = np.array([zlib.crc32(k.encode()) & 1 for k in G["key"]], bool)
    use = ~is_held & half; rows_a = np.flatnonzero(use); rows_b = np.flatnonzero(~is_held & ~half); cnt = np.bincount(owner[use], minlength=C); cnt_all = np.bincount(owner[~is_held], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max())
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** gamma; p_pl /= p_pl.sum(); log(f"variant {name} (players drawn by games ** {gamma:g}): half A {len(rows_a)} games (trained on), half B {len(rows_b)} games; players trained {len(tp)}; {steps} steps")
    rng = np.random.default_rng(1610 + v); rng2 = np.random.default_rng(161000); torch.manual_seed(1610 + v); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(rows_a, 4000, replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table(); hist = []  # noqa: E731

    def embed(rows):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64); h_truth = owner[hk]
    band = np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi), 0) for n in cnt_all[h_truth]]); share = np.array([cnt_all[(cnt_all >= lo) & (cnt_all < hi)].sum() for lo, hi in BANDS]) / cnt_all.sum()

    def summary(S, valid):
        S = np.where(valid[None], S, -9.0); r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); ok_q = valid[h_truth]
        t1 = np.array([float((r[ok_q & (band == k)] == 0).mean()) if (ok_q & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"top1": float((r[ok_q] == 0).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / share[ok].sum()), "n": int(ok_q.sum())}

    def evaluate(step):
        EA, EB, EH = embed(rows_a), embed(rows_b), embed(hk); SA = np.zeros((C, D_OUT)); np.add.at(SA, owner[rows_a], EA); SB = np.zeros((C, D_OUT)); np.add.at(SB, owner[rows_b], EB); res = {"step": step}
        both = (np.linalg.norm(SA, axis=1) > 0) & (np.linalg.norm(SB, axis=1) > 0)  # candidates and queries: players with games in both halves
        for nm, S in (("A (trained on)", SA), ("B (fresh)", SB), ("A + B", SA + SB)):
            P = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12); x = summary(EH @ P.T, both); res[nm] = x
            log(f"  step {step}: prototypes from half {nm:15s}: held-out top-1 {x['top1']:.3f} ({x['n']} games); by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}")
        # how different are the two halves' vectors of the same player: cosine between the two prototypes (players with >= 30 games)
        big = both & (cnt_all >= 30); ca = SA[big] / np.linalg.norm(SA[big], axis=1, keepdims=True); cb = SB[big] / np.linalg.norm(SB[big], axis=1, keepdims=True); res["proto_cos"] = float((ca * cb).sum(1).mean())
        na = np.linalg.norm(SA[big], axis=1) / np.maximum(cnt[big], 1); nbb = np.linalg.norm(SB[big], axis=1) / np.maximum(np.bincount(owner[rows_b], minlength=C)[big], 1); log(f"  step {step}: cosine between a player's two prototypes {res['proto_cos']:.3f}; norm of the mean vector: half A {float(na.mean()):.3f}, half B {float(nbb.mean()):.3f} (a higher norm = the games of a player lie closer together)")
        return res

    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; t_last = time.time(); ev_steps = {5000, 10000, steps} if steps > 200 else {steps}
    for step in range(steps):
        pl = rng.choice(tp, P_EP, replace=False, p=p_pl); u = rng.random((P_EP, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1)
        pos = SYM[int(rng2.integers(0, 8))][G["POS"][rows].astype(np.int64)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(tb(G["GL"][rows]), tb(pos), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float(); ze = z.reshape(P_EP, M_EP, D_OUT); tot = ze.sum(1); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T
        own = sc * (ze * F.normalize(tot[:, None] - ze, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None]); loss = F.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = 0.0; t_last = time.time()
        if (step + 1) in ev_steps:
            hist.append(evaluate(step + 1)); t_last = time.time()
    (OUT / f"m161d_{name}.json").write_text(json.dumps({"variant": name, "gamma": gamma, "steps": steps, "history": hist, "bands": BANDS, "share_rows": share.tolist()}, indent=1)); log(f"written {OUT / f'm161d_{name}.json'}")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert all(x in VARIANTS for x in a_.jobs); base.FULL, base.EXT, base.VAR = True, None, "S6"
    if a_.smoke:
        OUT = ROOT / "results/m161_smoke"
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs)
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps; G, _ = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); out = open(ROOT / f"logs/nano4/m161_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time(); train_one(name, G, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True))
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = name; log(f"started variant {name} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); bad += status != 0; log(f"variant {kids.pop(pid)} finished with status {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
