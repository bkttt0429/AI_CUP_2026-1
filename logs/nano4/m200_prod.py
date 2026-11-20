#!/usr/bin/env python
"""Section 4M M200 (2026-11-16): the second production recipe (m161q_prod.py) with ONE more input - the "choice residual".

Per ply of the player being identified: r64 = the trunk feature at the played point minus its policy-weighted mean over the
board (the same 64-d projection as the played-point input), and three policy scalars (-log p, entropy, log(1 + rank));
extracted by scripts/choice_residual_extract.py into results/m197_choice64 (int8, standardised x 20; ordinary training games
hold the subject's plies only, validation-question / held-out / test games both sides).

Changes against m161q_prod.py (sampling, losses, centres, symmetries, mask, schedule, outputs and file layout are the same):
  the token sum gets + Linear(67 -> width, no bias, ZERO initialised) of the residual, on the SUBJECT's own plies only
  (0 on the other side's plies and on the padding); for an opponent-side query the subject is flipped, as before, and the
  other side's stored rows are used. The layer is created after every other parameter and the random state is restored, so
  with the same seed the network at step 0, every episode and every mask equal those of the production network of that seed;
  a masked ply loses its residual as well as its engine features;
  the loader attaches G["CR"] / Tt["CR"] (int8 [games, 256, 67], rows at their original ply) and STOPS unless every game
  is covered and length, subject colour, and the point and mover of every stored ply equal the existing input arrays.
Outputs: results/m200 (out_<net>_s<seed>.npz, train_<net>_s<seed>.json), checkpoints /work/u2507115/m200_ckpt.

  sbatch -p 8gpus --gres=gpu:5 -c 60 ... run.sbatch python -I logs/nano4/m200_prod.py --jobs 1:30 2:30 3:30 4:30 all:30 --steps 40000
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402
from m112_idaug import sym_table  # noqa: E402

OUT, CK = ROOT / "results/m200", Path("/work/u2507115/m200_ckpt")
CH = ROOT / "results/m197_choice64"; N_CR, CR_SCALE = 67, 20.0
SELF_CHECK = False
BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]


def read_choice(path):
    with np.load(path, allow_pickle=False) as z:
        return tuple(z[k] for k in ("key", "LEN", "SUBJ", "BOTH", "OFF", "T", "MOV", "POS", "XQ"))


def attach_choice(G, Tt, pool, log):
    """G["CR"], Tt["CR"]: int8 [games, T_MAX, 67], the stored rows at their original ply and zero elsewhere. Raises unless
    every game is covered and the stored plies agree with the existing arrays (length, subject, point and mover per ply)."""
    files = sorted(CH.glob("part_s*of*_p*.npz")); N, NT, T_MAX = len(G["key"]), len(Tt["task"]), base.T_MAX; gi = {k: i for i, k in enumerate(G["key"])}; assert len(gi) == N, "duplicate game keys"
    G["CR"] = np.zeros((N, T_MAX, N_CR), np.int8); Tt["CR"] = np.zeros((NT, T_MAX, N_CR), np.int8); seen = {id(G): np.zeros(N, np.int32), id(Tt): np.zeros(NT, np.int32)}
    bad = dict(length=0, subject=0, point=0, mover=0, coverage=0); plies = both = other = 0; grid = np.arange(T_MAX)[None]
    for key, LEN, SUBJ, BOTH, OFF, T, MOV, POS, XQ in pool.imap(read_choice, files, chunksize=16):
        n = len(key); cnt = np.diff(OFF); row = np.repeat(np.arange(n), cnt); is_t = np.char.startswith(key, "test:"); dst = np.full(n, -1, np.int64)
        for j in range(n):
            dst[j] = int(key[j][5:]) if is_t[j] else gi.get(key[j], -1)
        dst[is_t & (dst >= NT)] = -1; other += int((dst < 0).sum())
        for D_, m in ((G, ~is_t), (Tt, is_t)):
            ok = m & (dst >= 0)
            if not ok.any():
                continue
            pm = ok[row]; r = dst[row[pm]]; t = T[pm].astype(np.int64); g = dst[ok]; D_["CR"][r, t] = XQ[pm]; np.add.at(seen[id(D_)], g, 1); plies += int(pm.sum()); both += int(BOTH[ok].sum())
            bad["point"] += int((D_["POS"][r, t] != POS[pm]).sum()); bad["mover"] += int((D_["MOV"][r, t] != MOV[pm]).sum()); bad["length"] += int((D_["LEN"][g] != LEN[ok]).sum()); bad["subject"] += int((D_["SUBJ"][g] != SUBJ[ok]).sum())
            L = D_["LEN"][g].astype(np.int64); own = ((D_["MOV"][g] == D_["SUBJ"][g][:, None]) & (grid < L[:, None])).sum(1); bad["coverage"] += int((cnt[ok] != np.where(BOTH[ok], L, own)).sum())
    miss = {"training": int((seen[id(G)] == 0).sum()), "test": int((seen[id(Tt)] == 0).sum()), "twice": int((seen[id(G)] > 1).sum() + (seen[id(Tt)] > 1).sum())}
    log(f"choice residual attached: {len(files)} parts, {plies} plies, {both} games with both sides, {other} stored games not loaded here; games without rows {miss}; disagreements with the existing arrays {bad} ({G['CR'].nbytes / 1e9:.1f} GB)")
    if any(miss.values()) or any(bad.values()):
        raise SystemExit(f"choice residual does not match the loaded games: {miss} {bad}")


def build_model():
    """base.build_model() plus the residual input; nothing else of the network or of the random streams changes."""
    import torch
    net = base.build_model(); st = torch.get_rng_state(); net.cr = torch.nn.Linear(N_CR, base.D, bias=False); torch.nn.init.zeros_(net.cr.weight); torch.set_rng_state(st)

    class SeqIDc(type(net)):
        def forward(self, gl, pos, mov, length, subj, cr):
            B, T = pos.shape; own = mov == subj[:, None]
            x = self.gl((gl.float() - self.mu) / self.sd) + self.pos(pos) + self.side(own.long()) + self.col(mov) + self.tpos(torch.arange(T, device=pos.device))[None] + self.dprev(self.disp(pos, 1)) + self.down(self.disp(pos, 2))
            x = x + self.cr(cr.float() * (own[..., None].float() / CR_SCALE))
            x = torch.cat([self.cls.expand(B, -1, -1), self.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # position 0 is [CLS]
            h = self.enc(x, src_key_padding_mask=pad)[:, 0]
            return torch.nn.functional.normalize(self.head(h).float(), dim=-1)
    net.__class__ = SeqIDc
    return net


def train_one(v, seed, G, Tt, steps, log, gamma=1.0, centres=True, mask=0.3, warm_b=3000, ramp_b=2000, lr_c=0.1):
    import torch
    F = torch.nn.functional; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}_s{seed}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    cfg = dict(gamma=gamma, centres=bool(centres), mask=mask, warm_b=warm_b, ramp_b=ramp_b, lr_c=lr_c, episode=[P_EP, M_EP], min_games=base.MIN_GAMES, width=base.D, ffn=base.FFN, layers=base.LAYERS, choice_residual=N_CR)
    log(f"net {v} seed {seed} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(gamma); p_pl /= p_pl.sum(); trained = np.zeros(C, bool); trained[tp] = True
    log(f"players trained {len(tp)} (>= {base.MIN_GAMES} games; they hold {float(cnt[tp].sum() / max(cnt.sum(), 1)):.3f} of the rows), games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}; share of an episode's games from players with >= 300 games {float(p_pl[cnt[tp] >= 300].sum()):.3f}")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731
    best = {"top1w": -1.0, "top1": -1.0, "step": -1}; best_state = None; best_wc = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731
    trained_t = tb(trained); Wc = opt_c = None; ck = CK / f"ckpt_{tag}.pt"; start = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if "opt" in st:
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", [])
            if st.get("best_is_current"):
                best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_wc = None if st.get("Wc") is None else st["Wc"].clone()
            if st.get("Wc") is not None:
                Wc = torch.nn.Parameter(st["Wc"].to(dev)); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9); opt_c.load_state_dict(st["opt_c"])
            log(f"resumed at step {start}, best so far {best}")
        else:  # a finished run's slim checkpoint: only the outputs are rewritten
            start = steps; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")

    def batch(src, r, subj):
        return tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj.astype(np.int64)), tb(src["CR"][r])

    if SELF_CHECK and start == 0:  # with the new layer at zero the network must equal the original one, and the layer must matter once it is not zero
        net.eval(); r = all_rows[:64]; a = batch(G, r, G["SUBJ"][r])
        with torch.no_grad():
            z_new = net(*a); z_old = type(net).__mro__[1].forward(net, *a[:5]); w0 = net.cr.weight.clone(); net.cr.weight.normal_(0, 0.02, generator=torch.Generator(device=dev).manual_seed(1)); z_on = net(*a)
            flip = list(a); flip[4] = 1 - a[4]; z_flip = net(*flip); z_flip0 = type(net).__mro__[1].forward(net, *flip[:5]); net.cr.weight.copy_(w0)
        own_rows = int((a[5][(a[2] == a[4][:, None])] != 0).any(-1).sum()); opp_rows = int((a[5][(a[2] != a[4][:, None])] != 0).any(-1).sum())
        log(f"self-check on 64 training games: zero layer against the original forward, largest difference {float((z_new - z_old).abs().max()):.2e}; random layer, mean change of the vector {float((z_on - z_old).norm(dim=-1).mean()):.4f}; subject flipped (ordinary games store the subject only), change {float((z_flip - z_flip0).norm(dim=-1).mean()):.4f}; plies with a residual: subject {own_rows}, other side {opp_rows}")
        net.train()

    def embed(src, rows, subj):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                out[i:i + 1024] = net(*batch(src, rows[i:i + 1024], subj[i:i + 1024])).float().cpu().numpy()
        net.train(); return out

    def train_protos():
        E = embed(G, all_rows, G["SUBJ"][all_rows]); S = np.zeros((C, D_OUT)); np.add.at(S, owner[all_rows], E); S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        return tb(S.astype(np.float32))

    def unit_centres(W):
        """Unit centre vectors [C, D_OUT] (numpy), zero rows for the players without a centre."""
        Wn = F.normalize(W.detach().float(), dim=-1).cpu().numpy().astype(np.float32); Wn[~trained] = 0.0
        return Wn

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):
        hk = np.flatnonzero(~is_held)[:8]
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8); band = np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi), 0) for n in cnt[h_truth]]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)

    def summary(S, sel):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[sel & (band == k)] == 0).mean()) if (sel & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"n": int(sel.sum()), "top1": float((r[sel] == 0).mean()), "top5": float((r[sel] < 5).mean()), "top20": float((r[sel] < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}

    def line(x):
        return f"top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}"

    def evaluate(step, W):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        Eh = embed(G, hk, G["SUBJ"][hk]); S = Eh @ Pm.T; res = {"step": step}
        for nm, sel in (("seen", seen), ("unseen", ~seen)):
            if sel.any():
                res[nm] = summary(S, sel)
        log(f"  step {step}: held-out own games, players trained on ({res['seen']['n']}), prototypes: " + line(res["seen"]) + (f"; unseen players ({res['unseen']['n']}) top-1 {res['unseen']['top1']:.3f}" if "unseen" in res else ""))
        if W is not None:
            Sc = Eh @ unit_centres(W).T; Sc[:, ~trained] = -2.0; res["centres"] = summary(Sc, seen); log(f"  step {step}: held-out own games, players trained on ({res['centres']['n']}), learned centres: " + line(res["centres"]))
        return res, Pm

    net.train(); ar = np.arange(P_EP); tl = ta = tbk = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}
    for step in range(start, steps):
        if centres and step >= warm_b and Wc is None:
            t_r = time.time(); Wc = torch.nn.Parameter(train_protos().clone()); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9)
            log(f"  step {step}: centres started from the prototypes of {int(trained.sum())} players ({time.time() - t_r:.0f} s)"); t_last += time.time() - t_r
        pl = rng.choice(tp, P_EP, replace=False, p=p_pl); u = rng.random((P_EP, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); cls_t = np.repeat(pl, M_EP)
        pos = G["POS"][rows].astype(np.int64); pos = SYM[int(rng2.integers(0, 8))][pos]; gl = tb(G["GL"][rows]); cr = tb(G["CR"][rows])
        if mask > 0:
            hit = torch.rand(len(rows), base.T_MAX, device=dev) < torch.rand(len(rows), 1, device=dev) * mask; gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl); cr = cr.masked_fill(hit[..., None], 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, tb(pos), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)), cr)
        z = z.float(); ze = z.reshape(P_EP, M_EP, D_OUT); tot = ze.sum(1); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T
        own = sc * (ze * F.normalize(tot[:, None] - ze, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None]); loss = F.cross_entropy(logits, tgt); acc = float((logits.argmax(1) == tgt).float().mean())
        if Wc is not None:
            w = min(1.0, (step - warm_b) / max(ramp_b, 1)); ct = tb(cls_t); cos = (z @ F.normalize(Wc, dim=-1).T).masked_fill(~trained_t[None], -1e4); loss = loss + w * F.cross_entropy(sc * cos, ct); tbk += float((cos.argmax(1) == ct).float().mean()); nb += 1
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if opt_c is not None:
            for g_ in opt_c.param_groups:
                g_["lr"] = lr_c * fac(step)
            opt_c.step()
        tl += float(loss.detach()); ta += acc
        if (step + 1) % 500 == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}" + (f", top-1 against all centres {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" if nb else "") + f", scale {float(sc):.1f}, residual layer norm {float(net.cr.weight.detach().norm()):.2f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = tbk = 0.0; nb = 0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1, Wc); hist.append(res)
            if res["seen"]["top1w"] > best["top1w"]:
                best = {"top1w": res["seen"]["top1w"], "top1": res["seen"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_wc = None if Wc is None else Wc.detach().cpu().clone()
                if "centres" in res:
                    best["centres_top1w"] = res["centres"]["top1w"]; best["centres_top1"] = res["centres"]["top1"]
            same = best["step"] == step + 1  # the best state is the current one: stored once
            tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": None if same else best_state, "best_wc": None if same else best_wc, "best_is_current": same, "hist": hist, "Wc": None if Wc is None else Wc.detach().cpu(), "opt_c": None if opt_c is None else opt_c.state_dict()}, tmp); os.replace(tmp, ck); t_last = time.time()
            (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "steps": steps, "bands": BANDS, "share_rows": share.tolist(), "partial": True}, indent=1))
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best row-weighted held-out top-1 {best['top1w']:.3f} (unweighted {best['top1']:.3f}) at step {best['step']}")
    res, Pm = evaluate(best["step"], None if best_wc is None else best_wc.to(dev)); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])
    loc = {g_: i for i, g_ in enumerate(mine)}
    if len(rows):
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)
    te = mine[q["qtest"][mine] >= 0]
    if len(te):
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)
    extra = {} if best_wc is None else {"centres": unit_centres(best_wc.to(dev)), "has_centre": trained}
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv, **extra)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "best_wc": best_wc, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, ck)
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game; centres {'saved' if extra else 'none'}); checkpoint slimmed to the best state")


def main():
    global OUT, CK, SELF_CHECK
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", nargs="+", required=True, help="one trainer per GPU, NET:SEED with NET in 1 2 3 4 all (e.g. 1:30 all:30)")
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--centres", type=int, default=1); ap.add_argument("--mask", type=float, default=0.3)
    ap.add_argument("--ep", type=int, nargs=2, default=[128, 16]); ap.add_argument("--size", type=int, nargs=2, default=[512, 2048]); ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--warm-b", type=int, default=3000); ap.add_argument("--ramp-b", type=int, default=2000); ap.add_argument("--out", default=None, help="sub-folder of results/m200 and of the checkpoint folder (pilots)"); ap.add_argument("--smoke", action="store_true")
    a_ = ap.parse_args()
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, "S9"; base.P_EP, base.M_EP = a_.ep; base.MIN_GAMES = max(base.MIN_GAMES, a_.ep[1]); base.D, base.FFN = a_.size; base.LAYERS = a_.layers
    jobs = [(-1 if x.split(":")[0] == "all" else int(x.split(":")[0]), int(x.split(":")[1])) for x in a_.jobs]; assert len(set(jobs)) == len(jobs) and all(v in base.NETS and s > 0 for v, s in jobs), a_.jobs
    kw = dict(gamma=a_.gamma, centres=bool(a_.centres), mask=a_.mask, warm_b=a_.warm_b, ramp_b=a_.ramp_b)
    if a_.out:
        OUT, CK = OUT / a_.out, CK / a_.out
    if a_.smoke:  # a smoke run never touches the production outputs
        OUT, CK = ROOT / "results/m200_smoke", Path("/work/u2507115/m200_ckpt_smoke"); base.EVAL_EVERY = 40; kw.update(warm_b=20, ramp_b=20); SELF_CHECK = True
    base.OUT = OUT
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(jobs), f"{len(jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps
    log(f"recipe: episode {base.P_EP} x {base.M_EP}, players with >= {base.MIN_GAMES} games, width {base.D} / feed-forward {base.FFN}, {base.LAYERS} layers, {kw}, choice residual from {CH.resolve()}; outputs {OUT}")
    pool = mp.get_context("fork").Pool(12)  # before anything large exists: the nodes run with strict overcommit (see base.load_games)
    G, Tt = base.load_games(a_.smoke, log); attach_choice(G, Tt, pool, log); pool.close(); pool.join(); kids = {}
    for k, (v, seed) in enumerate(jobs):
        try:
            pid = os.fork()
        except OSError as e:
            log(f"could not start net {v} seed {seed}: {e}"); break
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                out = open(ROOT / f"logs/nano4/m200{'_' + a_.out if a_.out else ''}_net{v}_s{seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time(); train_one(v, seed, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)
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
        f = OUT / f"train_{v}_s{seed}.json"
        if f.exists():
            d = json.load(open(f)); print(f"  net {v} seed {seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: {h['seen']['top1w']:.3f}" + (f" / centres {h['centres']['top1w']:.3f}" if "centres" in h else "") for h in d["history"]))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
