"""Section 4M M112 (2026-11-14): can the identity network be kept from memorising its training games? (design doc M112)
Diagnosis (logs of M90 / M101): on the training games the network reaches 92-98% in-episode accuracy while the held-out games of
the same players stop improving after 5,000-10,000 steps (top-1 0.29-0.30, top-20 even declining) - every training game is shown
40-120 times in exactly the same form, so the network learns "which game is this" instead of "whose play is this".
Here: the all-players network (-1), seed 3, 20,000 steps (the control repeats M101's S3c: 0.270 / 0.293 / 0.294 / 0.294 at
5k / 10k / 15k / 20k), one variant per GPU, all in one job (the games are loaded once, the trainers are forked):
  crop     every presentation is a random window of that many plies of the game (true ply indices kept), evaluation on full games
  pdrop    that share of the window's plies is hidden (padding mask)
  sym      a random one of the eight board symmetries is applied to the move coordinates (trunk vectors unchanged)
  strat    that share of the episodes takes all 256 players from one level (hard negatives: same strength)
  dropout / wd   stronger regularisation
Evaluation as in m88_seqid.py: held-out own games of the training players against the prototypes of all 8,007 (top-1 / 5 / 20).
  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch python -I scripts/m_series/m112_idaug.py [--steps 20000] [--smoke]
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]
import m88_seqid as base  # noqa: E402  (loader, model, constants; torch is imported lazily there)

RV = ROOT / "results/r_variants"
VARIANTS = {
    "v0_control": {},
    "v1_crop128": {"crop": 128},
    "v2_crop128_pdrop": {"crop": 128, "pdrop": 0.2},
    "v3_crop128_pdrop_sym": {"crop": 128, "pdrop": 0.2, "sym": True},
    "v5_v3_strat": {"crop": 128, "pdrop": 0.2, "sym": True, "strat": 0.5},
    "v6_v3_reg": {"crop": 128, "pdrop": 0.2, "sym": True, "dropout": 0.2, "wd": 0.1},
    "v7_crop64_pdrop_sym": {"crop": 64, "pdrop": 0.2, "sym": True},
    # added 05:20 after the first evaluations (the crop variants lose the opening and are far below the control on full games):
    "v8_sym_episode": {"sym": "episode"},            # one random symmetry for the whole episode, full games (the other session's --sym-aug)
    "v9_sym_game": {"sym": True},                    # a random symmetry per game, full games
    "v10_sym_episode_strat": {"sym": "episode", "strat": 0.5},
}


def sym_table():
    """[8, 363]: the eight board symmetries on the point index (x = p % 19, y = p // 19); 361 and 362 (pass / padding) stay."""
    p = np.arange(361); x, y = p % 19, p // 19; T = np.tile(np.arange(363), (8, 1))
    for k, (a, b_) in enumerate(((x, y), (18 - x, y), (x, 18 - y), (18 - x, 18 - y), (y, x), (18 - y, x), (y, 18 - x), (18 - y, 18 - x))):
        T[k, :361] = b_ * 19 + a
    return T


def run_variant(name, spec, G, gpu, steps, jid, smoke):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch
    from aicup.metric import LEVELS
    out = open(ROOT / f"logs/nano4/m112_{name}-{jid}.log", "w", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.DROP = spec.get("dropout", 0.1); WD = spec.get("wd", base.WD); crop = spec.get("crop", base.T_MAX); pdrop = spec.get("pdrop", 0.0); sym = spec.get("sym", False); strat = spec.get("strat", 0.0); v, seed, dev = -1, 3, "cuda"; P_EP, M_EP, D_OUT = base.P_EP, base.M_EP, base.D_OUT
    log(f"variant {name}: {spec}; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}; rk = {lv: i for i, lv in enumerate(LEVELS)}; plab = np.zeros(C, np.int64); plab[owner] = np.array([rk[k.split(":")[0]] for k in G["key"]])
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W = int(cnt.max())
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W), -1, np.int64); tab[o, pos_] = idx
    rng = np.random.default_rng(880 + v + 1000 * seed); rng2 = np.random.default_rng(112000 + sum(map(ord, name))); torch.manual_seed(880 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    tb = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731
    SYM = sym_table(); ar_c = np.arange(crop)

    def forward(gl, pos, mov, length, subj, t_start, hide):
        """As SeqID.forward, with the window's true ply indices and an optional mask of hidden plies."""
        B, T = pos.shape; tix = (t_start[:, None] + torch.arange(T, device=pos.device)[None]).clamp(max=base.T_MAX - 1)
        x = net.gl((gl.float() - net.mu) / net.sd) + net.pos(pos) + net.side((mov == subj[:, None]).long()) + net.col(mov) + net.tpos(tix) + net.dprev(net.disp(pos, 1)) + net.down(net.disp(pos, 2))
        x = torch.cat([net.cls.expand(B, -1, -1), net.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]
        if hide is not None:
            pad[:, 1:] |= hide
        return torch.nn.functional.normalize(net.head(net.enc(x, src_key_padding_mask=pad)[:, 0]).float(), dim=-1)

    def embed(rows):
        net.eval(); res = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; res[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()
        net.train(); return res
    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64); h_truth = owner[hk] if len(hk) else hk; hist = []

    def evaluate(step, train_acc):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(rows); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12); res = {"step": step, "train_episode_top1": train_acc}
        if len(hk):
            S = embed(hk) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res.update(n=int(len(hk)), top1=float((r == 0).mean()), top5=float((r < 5).mean()), top20=float((r < 20).mean()))
            # the same held-out games ranked only among 255 random other players: comparable with the training episodes' accuracy
            sub = np.stack([rng2.choice(C, 255, replace=False) for _ in range(len(hk))]); res["held_top1_of_256"] = float((S[np.arange(len(hk))[:, None], sub] > S[np.arange(len(hk)), h_truth][:, None]).sum(1).__eq__(0).mean())
            log(f"  step {step}: held-out own games ({len(hk)}) top-1 / top-5 / top-20 {res['top1']:.3f} / {res['top5']:.3f} / {res['top20']:.3f}; among 256 random players {res['held_top1_of_256']:.3f} (training games in their episodes: {train_acc:.3f})")
        hist.append(res); (RV / f"m112_{name}{'_smoke' if smoke else ''}.json").write_text(json.dumps({"spec": spec, "history": hist}, indent=1))
    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; last_acc = 0.0; t_last = time.time(); every = 20 if smoke else 500; eval_every = max(steps // 2, 1) if smoke else 5000; ev_steps = set(range(eval_every, steps + 1, eval_every)) | {steps}
    for step in range(steps):
        if strat and rng2.random() < strat:
            c = int(rng2.integers(10)); cand = tp[plab[tp] == c]; cand = cand if len(cand) >= P_EP else tp[np.abs(plab[tp] - c) <= 1]; pl = rng2.choice(cand, P_EP, replace=False) if len(cand) >= P_EP else rng.choice(tp, P_EP, replace=False)
        else:
            pl = rng.choice(tp, P_EP, replace=False)
        u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); ln = G["LEN"][rows].astype(np.int64)
        if crop < base.T_MAX:
            s0 = (rng2.random(len(rows)) * (np.maximum(ln - crop, 0) + 1)).astype(np.int64); ix = s0[:, None] + ar_c[None]; gl, pos, mov = G["GL"][rows[:, None], ix], G["POS"][rows[:, None], ix].astype(np.int64), G["MOV"][rows[:, None], ix].astype(np.int64); lc = np.minimum(ln - s0, crop)
        else:
            s0 = np.zeros(len(rows), np.int64); gl, pos, mov, lc = G["GL"][rows], G["POS"][rows].astype(np.int64), G["MOV"][rows].astype(np.int64), ln
        if sym == "episode":
            pos = SYM[int(rng2.integers(0, 8))][pos]
        elif sym:
            pos = SYM[rng2.integers(0, 8, len(rows))[:, None], pos]
        hide = tb(rng2.random(pos.shape) < pdrop) if pdrop else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = forward(tb(gl), tb(pos), tb(mov), tb(lc), tb(G["SUBJ"][rows].astype(np.int64)), tb(s0), hide)
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % every == 0:
            last_acc = ta / every; log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, episode top-1 {last_acc:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / every:.2f} s/step"); tl = ta = 0.0; t_last = time.time()
        if (step + 1) in ev_steps:
            evaluate(step + 1, last_acc); t_last = time.time()
    log(f"variant {name}: done")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL = True; base.EXT = None; gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; names = a_.variants[:2] if a_.smoke else a_.variants; jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 60 if a_.smoke else a_.steps
    assert len(gpus) >= len(names), f"{len(names)} variants need as many GPUs, got {gpus}"
    G, Tt = base.load_games(a_.smoke, log); del Tt; kids = {}
    for k, name in enumerate(names):
        try:
            pid = os.fork()
        except OSError as e:  # the node's commit limit: go on with the variants already started
            log(f"could not start {name}: {e}"); break
        if pid == 0:
            code = 0
            try:
                run_variant(name, VARIANTS[name], G, gpus[k], steps, jid, a_.smoke)
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = name; log(f"started {name} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); name = kids.pop(pid); bad += status != 0; log(f"{name} finished with status {status}")
    print("\nheld-out own games, top-1 / top-20 at each evaluation (and the training episodes' accuracy):")
    for name in names:
        f = RV / f"m112_{name}{'_smoke' if a_.smoke else ''}.json"
        if f.exists():
            h = json.load(open(f))["history"]; print(f"  {name:22s} " + "  ".join(f"{x['step'] // 1000}k: {x.get('top1', float('nan')):.3f} / {x.get('top20', float('nan')):.3f} (train {x['train_episode_top1']:.2f}, held of 256 {x.get('held_top1_of_256', float('nan')):.2f})" for x in h))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
