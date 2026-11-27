#!/usr/bin/env python
"""Section 4M M120 (2026-11-14): the M88 / M90 sequence identity networks retrained with board-symmetry augmentation.

M112: one random board symmetry of the move coordinates per episode, full games (variant v8) lifts the all-players network's
held-out top-1 from 0.285 to 0.341 at 10,000 steps; random windows hurt. This is m88_seqid.py's stage_train (--full: all
1,000,000 games) with that one change in the training step, the same cross-fit scheme (net v never sees fold v's players,
net -1 sees everyone), the same outputs (out_<tag>.npz: prototypes of the 8,007 players, the query vectors of the network's
queries; train_<tag>.json), and m88's merge. The games are loaded once and one (net, seed) trainer is forked per GPU
(m112_idaug.py's layout). Checkpoints (resume) go to /work/u2507115/m120_ckpt and are slimmed to the best state at the end.

  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch bash scripts/m_series/m120_job.sh 1 2 3 4 all all:1 1:1
  python -I scripts/m_series/m120_seqid_sym.py --merge 0,1,2            (-> results/m84_cache/var_S4e.npz, m84_variant.py's format)
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
from m112_idaug import sym_table  # noqa: E402

OUT, VAR, CK = ROOT / "results/m120", "S4", Path("/work/u2507115/m120_ckpt")


def train_one(v, seed, G, Tt, steps, sym, log, drop_open=0.0, mask=0.0):
    """m88_seqid.stage_train for network v and one seed, with a random board symmetry per episode when sym is set and, with drop_open,
    the opening of that share of an episode's games hidden (first K plies, K uniform in 20..100: trunk vectors -> the mean, played
    points -> the padding point; the other session's --drop-open, its METHOD_NOTES 2.60). With mask, every game draws a share in
    0..mask and that share of its plies get the mean trunk vector (played points kept; the other session's --mask)."""
    import torch
    OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}{f'_s{seed}' if seed else ''}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    log(f"net {v} seed {seed}: {steps} steps, symmetry per episode {bool(sym)}, opening hidden in {drop_open:g} of the games, ply mask up to {mask:g}; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W = int(cnt.max())
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W), -1, np.int64); tab[o, pos_] = idx
    rng = np.random.default_rng(880 + v + 1000 * seed); rng2 = np.random.default_rng(120000 + 10 * (v + 1) + seed); torch.manual_seed(880 + v + 1000 * seed); net = base.build_model().to(dev)
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32)
    flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    ck = CK / f"ckpt_{tag}.pt"; start = 0; best = {"top1": -1.0, "step": -1}; best_state = None; hist = []
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if "opt" in st:
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", []); log(f"resumed at step {start}, best so far {best}")
        else:  # a finished run's slim checkpoint: only the outputs are rewritten
            start = steps; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")
    tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    SYM = sym_table()

    def embed(src, rows, subj):
        """Unit vectors of games `rows` of `src` (G or the test dict) with subject colours `subj` (the games as stored: no symmetry)."""
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):  # a smoke run's subset may hold none of the held-out games: a few fake queries keep the code path alive
        hk = np.flatnonzero(~is_held)[:8]
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8)

    def evaluate(step):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        S = embed(G, hk, G["SUBJ"][hk]) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res = {"step": step}
        for name, s in (("seen", seen), ("unseen", ~seen)):
            if s.any():
                res[name] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}
        log(f"  step {step}: held-out own games " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if k != "step"))
        return res, Pm

    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}
    for step in range(start, steps):
        pl = rng.choice(tp, P_EP, replace=False); u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1)
        pos = G["POS"][rows].astype(np.int64)
        if sym:
            pos = SYM[int(rng2.integers(0, 8))][pos]
        gl, pos_t = tb(G["GL"][rows]), tb(pos)
        if mask > 0:
            hit = torch.rand(len(rows), base.T_MAX, device=dev) < torch.rand(len(rows), 1, device=dev) * mask
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
        if drop_open > 0:
            cut = torch.where(torch.rand(len(rows), device=dev) < drop_open, torch.randint(20, 101, (len(rows),), device=dev), torch.zeros(len(rows), dtype=torch.long, device=dev))
            hide = torch.arange(base.T_MAX, device=dev)[None] < cut[:, None]
            gl = torch.where(hide[..., None], net.mu.to(gl.dtype)[None, None], gl); pos_t = torch.where(hide, torch.full_like(pos_t, 362), pos_t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, pos_t, tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = 0.0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1); hist.append(res); key_ = "seen" if "seen" in res else "unseen"
            if res[key_]["top1"] > best["top1"]:
                best = {"top1": res[key_]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
            tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state, "hist": hist}, tmp); os.replace(tmp, ck); t_last = time.time()
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best held-out top-1 {best['top1']:.3f} at step {best['step']}")
    res, Pm = evaluate(best["step"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "sym": bool(sym), "drop_open": float(drop_open), "mask": float(mask)}, indent=1))
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "hist": hist, "steps": steps, "sym": bool(sym), "drop_open": float(drop_open), "mask": float(mask)}, tmp); os.replace(tmp, ck)
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game); checkpoint slimmed to the best state")


def main():
    global OUT, CK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", nargs="+", default=[], help="network[:seed[:o]] per GPU, e.g. 1 2 3 4 all all:1 1:2:o (all = the all-players network -1: a token that starts with '-' and is not a plain number is taken for an option; third field: o = the opening hidden in 30%% of the games, m = random plies' trunk vectors masked, om = both)")
    ap.add_argument("--parse-only", action="store_true", help="print the parsed jobs and stop")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--no-sym", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", default=None, help="comma-separated seeds: write results/m84_cache/var_S4[e].npz from the finished networks")
    ap.add_argument("--merge-var", default=None, help="with --merge: the variant name to write instead (results/m84_cache/var_<name>.npz), e.g. S4f for seeds 0,1,2,3")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, VAR
    if a_.merge is not None:
        seeds = [int(x) for x in a_.merge.split(",")]
        if a_.merge_var is None:
            base.stage_merge(seeds); return
        q = np.load(base.CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, base.D_OUT), np.float32); qvec = np.zeros((len(seeds), N, base.D_OUT), np.float16); rep = {}
        for si, sd_ in enumerate(seeds):  # m88_seqid.stage_merge with an explicit name
            for j, v in enumerate(base.NETS):
                t_ = f"{v}{f'_s{sd_}' if sd_ else ''}"; z = np.load(OUT / f"out_{t_}.npz"); protos[si, j] = z["protos"]; qvec[si, z["qidx"]] = z["qvec"]; rep[t_] = json.loads((OUT / f"train_{t_}.json").read_text())["best"]
        np.savez(base.CACHE / f"var_{a_.merge_var}.npz", protos=protos, qvec=qvec); (base.CACHE / f"var_{a_.merge_var}.json").write_text(json.dumps(rep, indent=1))
        print(f"var_{a_.merge_var} written ({len(seeds)} seed(s)); best held-out top-1 per net: " + ", ".join(f"{k}: {x['top1']:.3f} (step {x['step']})" for k, x in rep.items()), flush=True); return
    spec = [x.split(":") for x in a_.jobs]; jobs = [(-1 if x[0] == "all" else int(x[0]), int(x[1]) if len(x) > 1 else 0) for x in spec]; opening = {j: (0.3 if len(x) > 2 and "o" in x[2] else 0.0) for j, x in zip(jobs, spec)}; masking = {j: (0.3 if len(x) > 2 and "m" in x[2] else 0.0) for j, x in zip(jobs, spec)}
    assert jobs and len(set(jobs)) == len(jobs) and all(v in base.NETS for v, _ in jobs) and all(len(x) <= 2 or (x[2] and set(x[2]) <= set("om")) for x in spec), a_.jobs
    if a_.parse_only:
        print("jobs: " + ", ".join(f"net {v} seed {s_}{' +opening' if opening[(v, s_)] else ''}{' +mask' if masking[(v, s_)] else ''}" for v, s_ in jobs)); return
    if a_.smoke:  # a smoke run never touches the production outputs
        OUT, CK = ROOT / "results/m120_smoke", Path("/work/u2507115/m120_ckpt_smoke"); base.OUT = OUT
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(jobs), f"{len(jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 60 if a_.smoke else a_.steps
    if a_.smoke:
        base.EVAL_EVERY = 30
    G, Tt = base.load_games(a_.smoke, log); kids = {}
    for k, (v, seed) in enumerate(jobs):
        try:
            pid = os.fork()
        except OSError as e:  # the node's commit limit: go on with the trainers already started
            log(f"could not start net {v} seed {seed}: {e}"); break
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                out = open(ROOT / f"logs/nano4/m120_net{v}_s{seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time()
                train_one(v, seed, G, Tt, steps, not a_.no_sym, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), drop_open=opening[(v, seed)], mask=masking[(v, seed)])
            except BaseException:  # noqa: BLE001
                import traceback
                traceback.print_exc(); code = 1
            finally:
                sys.stdout.flush(); os._exit(code)
        kids[pid] = (v, seed); log(f"started net {v} seed {seed} on GPU {gpus[k]} (pid {pid})")
    bad = 0
    while kids:
        pid, status = os.wait(); v, seed = kids.pop(pid); bad += status != 0; log(f"net {v} seed {seed} finished with status {status}")
    for v, seed in jobs:
        f = OUT / f"train_{v}{f'_s{seed}' if seed else ''}.json"
        if f.exists():
            d = json.load(open(f)); print(f"  net {v} seed {seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: " + " / ".join(f"{k} {h[k]['top1']:.3f}" for k in ("seen", "unseen") if k in h) for h in d["history"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
