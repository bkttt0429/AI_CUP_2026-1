#!/usr/bin/env python
"""Section 4M M161b (2026-11-15): the identity network compared with ALL players' prototypes at every step (design doc M161b).

M161's classifier variants (learned centres in AdamW from a random start) collapsed within 500 steps, as in the other session's
runs. Here the comparison with every player uses no adaptively optimised centres:
  after a warm-up on the episode loss alone (3,000 steps) the prototypes of ALL players are computed with the current network
  (mean vector of a player's training games = the prototypes used at test time), refreshed in full every 1,000 steps and, in
  between, moved towards the batch's own vectors for the players of the batch (factor 0.95 per appearance, no gradient);
  bank loss = cross entropy of a game against all prototypes (own-class column: the prototype without the game itself, by
  subtracting the game's current vector), weight ramped from 0 to 1 over 2,000 steps, added to the episode loss.
Variants (all-players network, symmetry per step, otherwise m120_seqid_sym.train_one):
  f  players drawn by their games + the bank loss        g  uniform players + the bank loss
  h  players drawn by their games + LEARNED centres started from the prototypes at the end of the warm-up, plain SGD
     (lr 0.1 x the schedule, momentum 0.9)              i  f with a cosine margin of 0.1 on the own-class column
Evaluation and outputs as m161_seqid_act.py (results/m161/out_-1_<variant>.npz, train_-1_<variant>.json).

  sbatch -p 8gpus --gres=gpu:4 -c 48 ... run.sbatch python -I logs/nano4/m161b_seqid_bank.py --jobs f g h i --steps 12000
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
VARIANTS = {"f": dict(gamma=1.0, bank="proto", margin=0.0), "g": dict(gamma=0.0, bank="proto", margin=0.0), "h": dict(gamma=1.0, bank="param", margin=0.0), "i": dict(gamma=1.0, bank="proto", margin=0.1)}
WARM_B, RAMP_B, REFRESH, EMA, MASK = 3000, 2000, 1000, 0.95, 0.0


def train_one(v, seed, name, G, Tt, steps, log, warm_b=WARM_B, ramp_b=RAMP_B, refresh=REFRESH, mask=MASK, tag=None):
    import torch
    F = torch.nn.functional; cfg = VARIANTS[name]; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = tag or f"{v}_{name}{f'_s{seed}' if seed else ''}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    log(f"net {v} seed {seed} variant {name} {cfg}: {steps} steps, bank from step {warm_b} (ramp {ramp_b}, full refresh every {refresh}), ply mask up to {mask:g}; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(cfg["gamma"]); p_pl /= p_pl.sum()
    log(f"players trained {len(tp)}, games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731
    best = {"top1w": -1.0, "step": -1}; best_state = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731
    cnt_t = tb(cnt.astype(np.float32)); has = tb(cnt > 0); bank = bnorm = Wc = opt_c = None

    def embed(src, rows, subj):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    def train_protos():
        """Mean vector of every player's training games under the current network: (unit prototypes [C, D], norm of the mean [C])."""
        E = embed(G, all_rows, G["SUBJ"][all_rows]); S = np.zeros((C, D_OUT)); np.add.at(S, owner[all_rows], E); S /= np.maximum(cnt, 1)[:, None]; n = np.linalg.norm(S, axis=1)
        return tb((S / np.maximum(n, 1e-12)[:, None]).astype(np.float32)), tb(n.astype(np.float32))

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):
        hk = np.flatnonzero(~is_held)[:8]
    h_truth = owner[hk]; band = np.array([next(k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi) for n in cnt[h_truth]]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)

    def summary(S):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[band == k] == 0).mean()) if (band == k).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        Eh = embed(G, hk, G["SUBJ"][hk]); res = {"step": step, "proto": summary(Eh @ Pm.T)}
        if Wc is not None:
            res["head"] = summary(Eh @ F.normalize(Wc.detach().float(), dim=-1).cpu().numpy().T)
        for k in ("proto", "head"):
            if k in res:
                x = res[k]; log(f"  step {step}: held-out own games ({len(hk)}), {'prototypes' if k == 'proto' else 'learned centres'}: top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}")
        return res, Pm

    net.train(); ar = np.arange(P_EP); tl = ta = tbk = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}
    for step in range(steps):
        if step >= warm_b and (step - warm_b) % refresh == 0 and (cfg["bank"] == "proto" or bank is None):
            t_r = time.time(); bank, bnorm = train_protos()
            if cfg["bank"] == "param" and Wc is None:
                Wc = torch.nn.Parameter(bank.clone()); opt_c = torch.optim.SGD([Wc], lr=0.1, momentum=0.9)
            log(f"  step {step}: prototypes of all players computed ({time.time() - t_r:.0f} s); mean norm of the mean vector {float(bnorm[has].mean()):.3f}"); t_last += time.time() - t_r
        pl = rng.choice(tp, P_EP, replace=False, p=p_pl); u = rng.random((P_EP, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); cls_t = np.repeat(pl, M_EP)
        pos = G["POS"][rows].astype(np.int64); pos = SYM[int(rng2.integers(0, 8))][pos]; gl = tb(G["GL"][rows])
        if mask > 0:
            hit = torch.rand(len(rows), base.T_MAX, device=dev) < torch.rand(len(rows), 1, device=dev) * mask; gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, tb(pos), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float(); ze = z.reshape(P_EP, M_EP, D_OUT); tot = ze.sum(1); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T
        own = sc * (ze * F.normalize(tot[:, None] - ze, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None]); loss = F.cross_entropy(logits, tgt); acc = float((logits.argmax(1) == tgt).float().mean())
        if bank is not None:
            w = min(1.0, (step - warm_b) / max(ramp_b, 1)); ct = tb(cls_t)
            if cfg["bank"] == "param":
                cos = z @ F.normalize(Wc, dim=-1).T; own_c = cos.gather(1, ct[:, None])[:, 0]
            else:
                cos = z @ bank.T; loo = F.normalize((cnt_t[ct] * bnorm[ct])[:, None] * bank[ct] - z, dim=-1); own_c = (z * loo).sum(-1)  # the prototype without this game (its current vector stands in for the stored one)
            cos = cos.scatter(1, ct[:, None], (own_c - cfg["margin"])[:, None]).masked_fill(~has[None], -1e4); lb = F.cross_entropy(sc * cos, ct); loss = loss + w * lb; tbk += float((cos.argmax(1) == ct).float().mean()); nb += 1
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if opt_c is not None:
            for g_ in opt_c.param_groups:
                g_["lr"] = 0.1 * fac(step)
            opt_c.step()
        if bank is not None and cfg["bank"] == "proto":
            with torch.no_grad():
                plt_ = tb(pl); bank[plt_] = F.normalize(EMA * bank[plt_] + (1 - EMA) * F.normalize(ze.detach().mean(1), dim=-1), dim=-1)
        tl += float(loss.detach()); ta += acc
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}" + (f", all-player top-1 {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" if nb else "") + f", scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = tbk = 0.0; nb = 0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1); hist.append(res)
            if res["proto"]["top1w"] > best["top1w"]:
                best = {"top1w": res["proto"]["top1w"], "top1": res["proto"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
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
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"variant": name, "config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "warm_b": warm_b, "ramp_b": ramp_b, "refresh": refresh, "mask": mask, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = CK / f"ckpt_{tag}.pt.tmp"; torch.save({"best": best, "best_state": best_state, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, CK / f"ckpt_{tag}.pt")
    log(f"net {v} seed {seed} variant {name}: outputs written ({len(mine)} queries); checkpoint = the best state")


def main():
    global OUT, CK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", nargs="+", default=[], help="variant names, one per GPU (f g h i)")
    ap.add_argument("--net", default="all"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--smoke", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, "S6"
    v = -1 if a_.net == "all" else int(a_.net); assert a_.jobs and len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs) and v in base.NETS, a_.jobs
    kw = {}
    if a_.smoke:  # a smoke run never touches the pilot's outputs
        OUT, CK = ROOT / "results/m161_smoke", Path("/work/u2507115/m161_ckpt_smoke"); base.OUT = OUT; base.EVAL_EVERY = 40; kw = dict(warm_b=20, ramp_b=20, refresh=20)
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps
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
                t1 = time.time(); train_one(v, a_.seed, name, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
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
