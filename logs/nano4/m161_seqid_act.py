#!/usr/bin/env python
"""Section 4M M161 (2026-11-15): the sequence identity network trained for the way it is used (design doc M161).

M160: opponents appear in proportion to their activity (a tenth of the players own half of the rows), the M120 trainer draws
players uniformly (256 players x 8 games per episode: a 500-game player's game is seen ~20 times in 40,000 steps, a 10-game
player's game a thousand times) and compares a game with the 256 players of its episode only; players with 300+ games reach a
held-out top-1 of 0.44. Variants of the all-players network (m120_seqid_sym.train_one otherwise: same model, optimiser, symmetry
per step, evaluation by prototypes = mean vector of a player's non-held games):
  a  players drawn in proportion to their games (8 games each: every game is seen equally often)
  b  in proportion to the square root of their games
  c  a + a cosine classifier over ALL players (every step compares with every class; the two losses are added)
  d  the classifier only, 2,048 games drawn uniformly per step (no episodes), cosine margin 0.2
  e  uniform players + the classifier (separates the sampling from the head)
Reported at every evaluation: held-out top-1 / top-5 overall, by the player's number of training games, and weighted by the rows
(= the expected recognition rate of an opponent drawn by activity); for c / d / e also with the classifier weights as prototypes.
Outputs results/m161/out_-1_<variant>.npz (protos, qidx, qvec as m120; cls_w where there is a head), train_-1_<variant>.json.

  sbatch -p 8gpus --gres=gpu:5 -c 60 ... run.sbatch python -I logs/nano4/m161_seqid_act.py --jobs a b c d e --steps 12000
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402
from m112_idaug import sym_table  # noqa: E402

OUT, CK = ROOT / "results/m161", Path("/work/u2507115/m161_ckpt")
BANDS = [(0, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]
VARIANTS = {"a": dict(gamma=1.0, cls=0.0, only=False, margin=0.0), "b": dict(gamma=0.5, cls=0.0, only=False, margin=0.0), "c": dict(gamma=1.0, cls=1.0, only=False, margin=0.0),
            "d": dict(gamma=1.0, cls=1.0, only=True, margin=0.2), "e": dict(gamma=0.0, cls=1.0, only=False, margin=0.0), "z": dict(gamma=0.0, cls=0.0, only=False, margin=0.0)}  # z = the M120 trainer (check)


def train_one(v, seed, name, G, Tt, steps, log):
    import torch
    F = torch.nn.functional; cfg = VARIANTS[name]; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}_{name}{f'_s{seed}' if seed else ''}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    log(f"net {v} seed {seed} variant {name} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(cfg["gamma"]); p_pl /= p_pl.sum()
    log(f"players trained {len(tp)}, games {int(use.sum())}; an episode's expected share of games from players with >= 300 games: {float(p_pl[cnt[tp] >= 300].sum()):.3f} (uniform {float((cnt[tp] >= 300).mean()):.3f}); the busiest player is in an episode with probability {float(1 - (1 - p_pl.max()) ** P_EP):.3f}")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    groups = [{"params": list(net.parameters()), "weight_decay": base.WD}]; Wc = ls2 = None
    if cfg["cls"] > 0:
        Wc = torch.nn.Parameter(torch.randn(C, D_OUT, device=dev) * 0.01); ls2 = torch.nn.Parameter(torch.tensor(math.log(16.0), device=dev)); groups.append({"params": [Wc, ls2], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=base.LR); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    best = {"top1w": -1.0, "step": -1}; best_state = None; best_w = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731

    def embed(src, rows, subj):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):
        hk = np.flatnonzero(~is_held)[:8]
    h_truth = owner[hk]; band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in cnt[h_truth]]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)

    def summary(S):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[band == k] == 0).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / share[ok].sum())}

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        Eh = embed(G, hk, G["SUBJ"][hk]); res = {"step": step, "proto": summary(Eh @ Pm.T)}
        if Wc is not None:
            res["head"] = summary(Eh @ F.normalize(Wc.detach().float(), dim=-1).cpu().numpy().T)
        for k in ("proto", "head"):
            if k in res:
                x = res[k]; log(f"  step {step}: held-out own games ({len(hk)}), {'prototypes' if k == 'proto' else 'classifier weights'}: top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}")
        return res, Pm

    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}; B = P_EP * M_EP
    for step in range(steps):
        if cfg["only"]:
            rows = rng.choice(all_rows, B, replace=False); cls_t = owner[rows]
        else:
            pl = rng.choice(tp, P_EP, replace=False, p=p_pl); u = rng.random((P_EP, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); cls_t = np.repeat(pl, M_EP)
        pos = G["POS"][rows].astype(np.int64); pos = SYM[int(rng2.integers(0, 8))][pos]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(tb(G["GL"][rows]), tb(pos), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float(); loss = 0.0; acc = None
        if not cfg["only"]:
            ze = z.reshape(P_EP, M_EP, D_OUT); tot = ze.sum(1); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T
            own = sc * (ze * F.normalize(tot[:, None] - ze, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None]); loss = loss + F.cross_entropy(logits, tgt); acc = float((logits.argmax(1) == tgt).float().mean())
        if Wc is not None:
            ct = tb(cls_t); cos = z @ F.normalize(Wc, dim=-1).T; s2 = ls2.exp().clamp(max=64.0)
            if cfg["margin"] > 0:
                cos = cos - cfg["margin"] * F.one_hot(ct, C).to(cos.dtype)
            loss = loss + cfg["cls"] * F.cross_entropy(s2 * cos, ct)
            if acc is None:
                acc = float((cos.argmax(1) == ct).float().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(net.parameters()) + ([Wc] if Wc is not None else []), 1.0); opt.step(); sched.step()
        tl += float(loss.detach()); ta += acc
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, batch top-1 {ta / 500:.3f}, scale {float(net.log_scale.exp()):.1f}" + (f" / head {float(ls2.exp()):.1f}" if ls2 is not None else "") + f", {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = 0.0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1); hist.append(res)
            if res["proto"]["top1w"] > best["top1w"]:
                best = {"top1w": res["proto"]["top1w"], "top1": res["proto"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_w = None if Wc is None else Wc.detach().cpu().clone()
            t_last = time.time()
    net.load_state_dict(best_state); log(f"net {v} seed {seed} variant {name}: best row-weighted held-out top-1 {best['top1w']:.3f} (unweighted {best['top1']:.3f}) at step {best['step']}")
    res, Pm = evaluate(best["step"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    extra = {} if best_w is None else {"cls_w": torch.nn.functional.normalize(best_w.float(), dim=-1).numpy().astype(np.float32)}
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv, **extra)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"variant": name, "config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = CK / f"ckpt_{tag}.pt.tmp"; torch.save({"best": best, "best_state": best_state, "cls_w": best_w, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, CK / f"ckpt_{tag}.pt")
    log(f"net {v} seed {seed} variant {name}: outputs written ({len(mine)} queries); checkpoint = the best state")


def main():
    global OUT, CK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", nargs="+", default=[], help="variant names, one per GPU (a b c d e; z = the M120 trainer)")
    ap.add_argument("--net", default="all", help="all (= the all-players network -1) or 1..4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--smoke", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, "S6"
    v = -1 if a_.net == "all" else int(a_.net); assert a_.jobs and len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs) and v in base.NETS, a_.jobs
    if a_.smoke:  # a smoke run never touches the pilot's outputs
        OUT, CK = ROOT / "results/m161_smoke", Path("/work/u2507115/m161_ckpt_smoke"); base.OUT = OUT; base.EVAL_EVERY = 30
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 60 if a_.smoke else a_.steps
    G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        try:
            pid = os.fork()
        except OSError as e:
            log(f"could not start variant {name}: {e}"); break
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                out = open(ROOT / f"logs/nano4/m161_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time(); train_one(v, a_.seed, name, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True))
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = name; log(f"started variant {name} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); name = kids.pop(pid); bad += status != 0; log(f"variant {name} finished with status {status}")
    for name in a_.jobs:
        f = OUT / f"train_{v}_{name}{f'_s{a_.seed}' if a_.seed else ''}.json"
        if f.exists():
            d = json.load(open(f)); print(f"  variant {name}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: {h['proto']['top1']:.3f} / weighted {h['proto']['top1w']:.3f}" for h in d["history"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
