#!/usr/bin/env python
"""Section 4M M161f (2026-11-15): prototypes plus LEARNED OFFSETS for the players with enough games (design doc M161f).

M161b's variant h (learned centres started from the prototypes, plain SGD) at 5,000 steps: the centres used as matching vectors
reach 0.555 held-out top-1 for players with 300-999 games (prototypes of the same network 0.472) but only 0.13-0.23 for players
with fewer than 100 games (their centres are rarely updated and go stale). Here every player's matching vector is
normalize(prototype + offset): the prototypes of all players are recomputed from the current network every 1,000 steps (as
M161b's f), the offsets are learned (plain SGD, momentum 0.9, weight decay towards 0) through the all-player loss - a busy
player's offset is updated all the time, a rare player's offset decays to 0 and his vector is his prototype.
Variants (all-players network, players drawn by their games, seed 0):
  v1  offsets, weight decay 0.005        v2  v1 with the wider network (512 / 2,048)        v3  offsets without weight decay
  h2  learned centres as M161b's h, scored also as interpolations lambda * centre + (1 - lambda) * prototype with
      lambda = n / (n + kappa), kappa 30 / 100 / 300 (n = the player's training games)
Every evaluation reports the plain prototypes and the combined vectors; the best state and the stored `protos` are those of
the best scoring. Outputs results/m161/<variant>/out_-1_s0.npz (protos = the matching vectors, plain_protos, extra), train json.

  sbatch -p 8gpus --gres=gpu:4 -c 48 ... run.sbatch python -I logs/nano4/m161f_offset.py --jobs v1 v2 v3 h2 --steps 12000
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
BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]; EMA = 0.95; KAPPAS = (30, 100, 300)
VARIANTS = {"v1": dict(kind="offset", wd_c=0.005), "v2": dict(kind="offset", wd_c=0.005, size=(512, 2048)), "v3": dict(kind="offset", wd_c=0.0), "h2": dict(kind="param", wd_c=0.0)}


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def train_one(v, seed, G, Tt, steps, log, kind="offset", wd_c=0.005, lr_c=0.1, gamma=1.0, mask=0.0, warm_b=3000, ramp_b=2000, refresh=1000, out_dir=None, ck_dir=None, tag=None):
    import torch
    F = torch.nn.functional; out_dir = out_dir or OUT; ck_dir = ck_dir or CK; out_dir.mkdir(parents=True, exist_ok=True); ck_dir.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = tag or f"{v}_s{seed}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    cfg = dict(kind=kind, wd_c=wd_c, lr_c=lr_c, gamma=gamma, mask=mask, warm_b=warm_b, ramp_b=ramp_b, refresh=refresh, width=base.D); log(f"net {v} seed {seed} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(gamma); p_pl /= p_pl.sum(); log(f"players trained {len(tp)}, games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731
    best = {"top1w": -1.0, "top1": -1.0, "step": -1, "scoring": ""}; best_state = None; best_vec = None; best_extra = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731
    cnt_t = tb(cnt.astype(np.float32)); has = tb(cnt > 0); bank = bnorm = Dc = opt_c = None

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

    def summary(S):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[seen & (band == k)] == 0).mean()) if (seen & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"n": int(seen.sum()), "top1": float((r[seen] == 0).mean()), "top5": float((r[seen] < 5).mean()), "top20": float((r[seen] < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm = unit(Pm); Eh = embed(G, hk, G["SUBJ"][hk]); vecs = {"prototypes": Pm}
        if Dc is not None:
            d = Dc.detach().float().cpu().numpy().astype(np.float64)
            if kind == "offset":
                vecs["prototypes + offsets"] = unit(Pm + d)
            else:
                Wn = unit(d); vecs["centres"] = Wn
                for kp in KAPPAS:
                    lam = (cnt / (cnt + float(kp)))[:, None]; vecs[f"interpolated, kappa {kp}"] = unit(lam * Wn + (1 - lam) * Pm)
        res = {"step": step}
        for nm, Vm in vecs.items():
            x = summary(Eh @ Vm.T); res[nm] = x; log(f"  step {step}: held-out own games ({x['n']}), {nm:24s}: top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}")
        if Dc is not None and kind == "offset":
            nrm = np.linalg.norm(Dc.detach().float().cpu().numpy(), axis=1); log(f"  step {step}: norm of the offsets by games: " + "  ".join(f"{lo}+ {float(nrm[(cnt >= lo) & (cnt < hi)].mean()):.3f}" for lo, hi in BANDS if ((cnt >= lo) & (cnt < hi)).any()))
        return res, vecs

    net.train(); ar = np.arange(P_EP); tl = ta = tbk = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}
    for step in range(steps):
        if step >= warm_b and (bank is None or (kind == "offset" and (step - warm_b) % refresh == 0)):
            t_r = time.time(); bank, bnorm = train_protos()
            if Dc is None:
                Dc = torch.nn.Parameter(torch.zeros(C, D_OUT, device=dev) if kind == "offset" else bank.clone()); opt_c = torch.optim.SGD([Dc], lr=lr_c, momentum=0.9, weight_decay=wd_c)
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
            if kind == "offset":
                cos = z @ F.normalize(bank + Dc, dim=-1).T; loo = F.normalize((cnt_t[ct] * bnorm[ct])[:, None] * bank[ct] - z, dim=-1); own_c = (z * F.normalize(loo + Dc[ct], dim=-1)).sum(-1)
            else:
                cos = z @ F.normalize(Dc, dim=-1).T; own_c = cos.gather(1, ct[:, None])[:, 0]
            cos = cos.scatter(1, ct[:, None], own_c[:, None]).masked_fill(~has[None], -1e4); loss = loss + w * F.cross_entropy(sc * cos, ct); tbk += float((cos.argmax(1) == ct).float().mean()); nb += 1
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if opt_c is not None:
            for g_ in opt_c.param_groups:
                g_["lr"] = lr_c * fac(step)
            opt_c.step()
        if bank is not None and kind == "offset":
            with torch.no_grad():
                plt_ = tb(pl); bank[plt_] = F.normalize(EMA * bank[plt_] + (1 - EMA) * F.normalize(ze.detach().mean(1), dim=-1), dim=-1)
        tl += float(loss.detach()); ta += acc
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}" + (f", all-player top-1 {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" if nb else "") + f", scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = tbk = 0.0; nb = 0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, vecs = evaluate(step + 1); hist.append(res); nm = max(vecs, key=lambda k: res[k]["top1w"])
            if res[nm]["top1w"] > best["top1w"]:
                best = {"top1w": res[nm]["top1w"], "top1": res[nm]["top1"], "step": step + 1, "scoring": nm}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_vec = vecs[nm].astype(np.float32); best_extra = {"plain_protos": vecs["prototypes"].astype(np.float32)}
                if Dc is not None:
                    best_extra["offsets" if kind == "offset" else "centres"] = Dc.detach().float().cpu().numpy()
            t_last = time.time()
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best row-weighted held-out top-1 {best['top1w']:.3f} (unweighted {best['top1']:.3f}) at step {best['step']}, scoring = {best['scoring']}")
    j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    np.savez(out_dir / f"out_{tag}.npz", protos=best_vec, qidx=mine, qvec=qv, **best_extra)
    (out_dir / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = ck_dir / f"ckpt_{tag}.pt.tmp"; torch.save({"best": best, "best_state": best_state, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, ck_dir / f"ckpt_{tag}.pt")
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries); checkpoint = the best state")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True); ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    assert len(set(a_.jobs)) == len(a_.jobs) and all(x in VARIANTS for x in a_.jobs), a_.jobs
    base.FULL, base.EXT, base.VAR = True, None, "S6"; root_out, root_ck = (ROOT / "results/m161_smoke", Path("/work/u2507115/m161_ckpt_smoke")) if a_.smoke else (OUT, CK)
    if a_.smoke:
        base.EVAL_EVERY = 40
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(a_.jobs), f"{len(a_.jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps; G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, name in enumerate(a_.jobs):
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k]); spec = dict(VARIANTS[name]); out = open(ROOT / f"logs/nano4/m161_{name}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                if "size" in spec:
                    base.D, base.FFN = spec.pop("size")
                kw = dict(spec, out_dir=root_out / name, ck_dir=root_ck / name)
                if a_.smoke:
                    kw.update(warm_b=20, ramp_b=20, refresh=20)
                t1 = time.time(); train_one(-1, 0, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = name; log(f"started variant {name} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); name = kids.pop(pid); bad += status != 0; log(f"variant {name} finished with status {status}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
