#!/usr/bin/env python
"""Section 4M M161 production trainer (2026-11-15): the cross-fit set of identity networks with the pilot's winning recipe.

One trainer for every recipe of M161 / M161b (m161_seqid_act.py, m161b_seqid_bank.py): players drawn with probability
proportional to games ** gamma, optionally the all-player prototype loss (bank proto: prototypes of all players recomputed every
`refresh` steps after `warm_b` steps, EMA 0.95 in between, own column leave-one-out; bank param: learned centres started from the
prototypes, plain SGD), a cosine margin, the ply mask of M122 / M143. Networks 1-4 never see their fold's players, network -1
sees everyone (m120's cross-fit scheme); outputs in m120's layout and names (out_<net>_s<seed>.npz with protos / qidx / qvec,
train_<net>_s<seed>.json with best = {top1, step, ...}, a slim checkpoint with best_state) in results/m161p and
/work/u2507115/m161p_ckpt, so that m124_tta_queries.py and the merge work on them unchanged (see m161p_tta.py). Checkpoints at
every evaluation; a restarted job resumes. The best state is the one with the highest row-weighted held-out top-1 on the players
the network trained on.

  sbatch -p 8gpus --gres=gpu:5 -c 60 ... run.sbatch python -I logs/nano4/m161p_prod.py --jobs 1 2 3 4 all --seed 20 --steps 40000 --gamma 1 --bank proto --mask 0.3
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

OUT, CK = ROOT / "results/m161p", Path("/work/u2507115/m161p_ckpt")
BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; EMA = 0.95


def train_one(v, seed, G, Tt, steps, log, gamma=1.0, bank_kind="none", margin=0.0, mask=0.0, warm_b=3000, ramp_b=2000, refresh=1000):
    import torch
    F = torch.nn.functional; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}_s{seed}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    cfg = dict(gamma=gamma, bank=bank_kind, margin=margin, mask=mask, warm_b=warm_b, ramp_b=ramp_b, refresh=refresh); log(f"net {v} seed {seed} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(gamma); p_pl /= p_pl.sum()
    log(f"players trained {len(tp)}, games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}; share of an episode's games from players with >= 300 games {float(p_pl[cnt[tp] >= 300].sum()):.3f}")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731
    best = {"top1w": -1.0, "top1": -1.0, "step": -1}; best_state = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731
    cnt_t = tb(cnt.astype(np.float32)); has = tb(cnt > 0); bank = bnorm = Wc = opt_c = None; ck = CK / f"ckpt_{tag}.pt"; start = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if "opt" in st:
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", [])
            if st.get("Wc") is not None:
                Wc = torch.nn.Parameter(st["Wc"].to(dev)); opt_c = torch.optim.SGD([Wc], lr=0.1, momentum=0.9); opt_c.load_state_dict(st["opt_c"])
            log(f"resumed at step {start}, best so far {best}")
        else:  # a finished run's slim checkpoint: only the outputs are rewritten
            start = steps; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")

    def embed(src, rows, subj):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    def train_protos():
        E = embed(G, all_rows, G["SUBJ"][all_rows]); S = np.zeros((C, D_OUT)); np.add.at(S, owner[all_rows], E); S /= np.maximum(cnt, 1)[:, None]; n = np.linalg.norm(S, axis=1)
        return tb((S / np.maximum(n, 1e-12)[:, None]).astype(np.float32)), tb(n.astype(np.float32))

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):
        hk = np.flatnonzero(~is_held)[:8]
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8); band = np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi), 0) for n in cnt[h_truth]]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)

    def summary(S, sel):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[sel & (band == k)] == 0).mean()) if (sel & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"n": int(sel.sum()), "top1": float((r[sel] == 0).mean()), "top5": float((r[sel] < 5).mean()), "top20": float((r[sel] < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        S = embed(G, hk, G["SUBJ"][hk]) @ Pm.T; res = {"step": step}
        for nm, sel in (("seen", seen), ("unseen", ~seen)):
            if sel.any():
                res[nm] = summary(S, sel)
        x = res["seen"]; log(f"  step {step}: held-out own games, players trained on ({x['n']}): top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}" + (f"; unseen players ({res['unseen']['n']}) top-1 {res['unseen']['top1']:.3f}" if "unseen" in res else ""))
        return res, Pm

    net.train(); ar = np.arange(P_EP); tl = ta = tbk = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}
    for step in range(start, steps):
        if bank_kind != "none" and step >= warm_b and (bank is None or (bank_kind == "proto" and (step - warm_b) % refresh == 0)):
            t_r = time.time(); bank, bnorm = train_protos()
            if bank_kind == "param" and Wc is None:
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
            if bank_kind == "param":
                cos = z @ F.normalize(Wc, dim=-1).T; own_c = cos.gather(1, ct[:, None])[:, 0]
            else:
                cos = z @ bank.T; own_c = (z * F.normalize((cnt_t[ct] * bnorm[ct])[:, None] * bank[ct] - z, dim=-1)).sum(-1)
            cos = cos.scatter(1, ct[:, None], (own_c - margin)[:, None]).masked_fill(~has[None], -1e4); loss = loss + w * F.cross_entropy(sc * cos, ct); tbk += float((cos.argmax(1) == ct).float().mean()); nb += 1
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if opt_c is not None:
            for g_ in opt_c.param_groups:
                g_["lr"] = 0.1 * fac(step)
            opt_c.step()
        if bank is not None and bank_kind == "proto":
            with torch.no_grad():
                plt_ = tb(pl); bank[plt_] = F.normalize(EMA * bank[plt_] + (1 - EMA) * F.normalize(ze.detach().mean(1), dim=-1), dim=-1)
        tl += float(loss.detach()); ta += acc
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}" + (f", all-player top-1 {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" if nb else "") + f", scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = tbk = 0.0; nb = 0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1); hist.append(res)
            if res["seen"]["top1w"] > best["top1w"]:
                best = {"top1w": res["seen"]["top1w"], "top1": res["seen"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
            tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state, "hist": hist, "Wc": None if Wc is None else Wc.detach().cpu(), "opt_c": None if opt_c is None else opt_c.state_dict()}, tmp); os.replace(tmp, ck); t_last = time.time()
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best row-weighted held-out top-1 {best['top1w']:.3f} (unweighted {best['top1']:.3f}) at step {best['step']}")
    res, Pm = evaluate(best["step"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, ck)
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game); checkpoint slimmed to the best state")


def main():
    global OUT, CK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", nargs="+", default=[], help="networks, one per GPU: 1 2 3 4 all")
    ap.add_argument("--seed", type=int, required=True, help="a seed number > 0 that results/m161p does not hold yet (it names the outputs: out_<net>_s<seed>.npz)")
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--bank", choices=("none", "proto", "param"), default="none"); ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--mask", type=float, default=0.3); ap.add_argument("--warm-b", type=int, default=3000); ap.add_argument("--ramp-b", type=int, default=2000); ap.add_argument("--refresh", type=int, default=1000); ap.add_argument("--smoke", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, "S6"
    nets = [-1 if x == "all" else int(x) for x in a_.jobs]; assert nets and len(set(nets)) == len(nets) and all(v in base.NETS for v in nets) and a_.seed > 0, a_.jobs
    kw = dict(gamma=a_.gamma, bank_kind=a_.bank, margin=a_.margin, mask=a_.mask, warm_b=a_.warm_b, ramp_b=a_.ramp_b, refresh=a_.refresh)
    if a_.smoke:  # a smoke run never touches the production outputs
        OUT, CK = ROOT / "results/m161p_smoke", Path("/work/u2507115/m161p_ckpt_smoke"); base.OUT = OUT; base.EVAL_EVERY = 40; kw.update(warm_b=20, ramp_b=20, refresh=20)
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(nets), f"{len(nets)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps
    G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, v in enumerate(nets):
        try:
            pid = os.fork()
        except OSError as e:
            log(f"could not start net {v}: {e}"); break
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                out = open(ROOT / f"logs/nano4/m161p_net{v}_s{a_.seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time(); train_one(v, a_.seed, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = v; log(f"started net {v} seed {a_.seed} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); v = kids.pop(pid); bad += status != 0; log(f"net {v} finished with status {status}")
    for v in nets:
        f = OUT / f"train_{v}_s{a_.seed}.json"
        if f.exists():
            d = json.load(open(f)); print(f"  net {v} seed {a_.seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: {h['seen']['top1']:.3f} / weighted {h['seen']['top1w']:.3f}" for h in d["history"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
