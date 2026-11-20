#!/usr/bin/env python
"""Section 4M M218 (2026-11-17): the production single-game recogniser trained together with external open accounts as strangers.

Design aimed at the diagnosed bottleneck (M192 - M194, M217): for a candidate's unseen games the recogniser is mostly a generic
style metric; the exposure of the candidate's own games adds little. The generic metric is what many more identities improve
(the player-identification session's evidence on never-seen players), and strangers are the hard negatives the candidates'
learned centres never had. Everything else is the second production recipe (m161q_prod.py: by-games sampling, 128 x 16 games,
width 512, 8 layers, mask 0.3, learned centres, 8 symmetries, 40,000 steps, same seeds -> the same initial network).

  --ext-share S   share of an episode's 2,048 games that come from external accounts (8 games per account; the competition
                  players keep 16 games each): S = 0.25 -> 96 competition players + 64 accounts, 0.5 -> 64 + 128
  --oe L          outlier exposure: for the external games, the cross-entropy between the softmax over the competition players'
                  learned centres and the uniform distribution, weight L (same ramp as the centre loss)
External accounts are never candidates and have no centre. Accounts whose hash falls in a fixed 1 / 40 are never trained on: two
games of each are the stranger queries of the open-set reading that every evaluation prints (share of the held-out own games whose
owner is first AND whose margin - first minus the mean of the 2nd to 5th, standardised scores + 0.4 log games - is above the
strangers' 95th percentile).
  --eval-ckpt F --net V --seed S   the same evaluation for a finished checkpoint (the production control)

  python -I logs/nano4/m218_prod.py --jobs 1:30 2:30 3:30 4:30 all:30 --ext-share 0.25 --oe 0.5 --out g25o --ck-dir /dev/shm/m218/g25o
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4")]
import m88_seqid as base  # noqa: E402
import m216_prod as m216  # noqa: E402  (build_model: with base.T_MAX = 256 and no checkpointing it is base.build_model())
from m112_idaug import sym_table  # noqa: E402

OUT, CK = ROOT / "results/m218", Path("/work/u2507115/m218_ckpt"); BANDS = m216.BANDS; M_X, BETA, RES_MOD, NULL_PER = 8, 0.4, 40, 2
XR = Path("/work/u2507115/ext_fox")  # the player-identification session's external (Fox) per-ply parts, read only
EXT_SETS = {"pilot": ([XR / "pilot/ply"], XR / "pilot/select.csv"), "scale1": ([XR / "scale1/ply_a", XR / "scale1/ply_b", XR / "ext_ply_home/scale1_c"], XR / "scale1/select.csv"),
            "scale2_top": ([XR / "ext_ply_home/scale2_top"], XR / "scale2_top/select.csv"), "scale2_5d": ([XR / "ext_ply_home/scale2_5d"], XR / "scale2_5d/select.csv"),
            "scale3b": ([XR / "scale3b/ply", Path.home() / "BK_AICUP_2026-2/data/ext_ply2/scale3b"], XR / "scale3b/select.csv")}
SAVE_OUT = False
KEEP = None  # eval only: the game keys to keep while reading the external parts (set before the pool is created)


def reserved(acc):
    """Never-trained accounts (the strangers of the open-set reading): a fixed 1 / RES_MOD of the hashed accounts."""
    return zlib.crc32(acc.encode()) % RES_MOD == 0


def read_ext(path):
    k, GL, POS, MOV, LEN, SUBJ = base.read_m90(path)
    if KEEP is None:
        return k, GL, POS, MOV, LEN, SUBJ
    m = np.isin(k, KEEP)
    return k[m], GL[m], POS[m], MOV[m], LEN[m], SUBJ[m]


def count_ext(path):
    with np.load(path) as z:
        return len(z["LEN"]) if KEEP is None else int(np.isin(z["key"].astype(str), KEEP).sum())


def ext_tables(sets):
    import pandas as pd
    files, own = [], {}
    for s in sets:
        dirs, sel = EXT_SETS[s]; fs = [f for d in dirs for f in sorted(d.glob("ply_x_*.npz")) if not f.name.endswith(".tmp.npz")]; assert fs, f"no parts for {s}"; files += fs
        t = pd.read_csv(sel, usecols=["game_key", "account", "color"], dtype=str); assert set(t["color"]) <= {"B", "W"}, set(t["color"]); own.update(zip(zip(t["game_key"], t["color"]), t["account"]))  # a game can be listed for both of its players
    return files, own


def load_ext(files, own, pool, log):
    """The external games as padded arrays (as base.load_games), the account index of every game and the accounts' names."""
    n = pool.map(count_ext, files, chunksize=16); N = int(sum(n)); T = base.T_MAX
    GL = np.zeros((N, T, 128), np.float16); POS = np.full((N, T), 362, np.int16); MOV = np.zeros((N, T), np.int8); LEN = np.zeros(N, np.int16); SUBJ = np.zeros(N, np.int8); keys, a = [], 0
    for x in pool.imap(read_ext, files, chunksize=2):
        b = a + len(x[0]); keys.append(x[0]); GL[a:b] = x[1]; POS[a:b] = x[2]; MOV[a:b] = x[3]; LEN[a:b] = x[4]; SUBJ[a:b] = x[5]; a = b
    assert a == N, (a, N); key = np.concatenate(keys) if keys else np.zeros(0, str); pair = list(zip(key.tolist(), np.where(SUBJ == 0, "B", "W").tolist())); seen_, first = set(), np.ones(N, bool)
    for i_, x in enumerate(pair):  # a game side stored in two parts: the first copy is kept
        first[i_] = x not in seen_; seen_.add(x)
    dup = int((~first).sum()); assert dup <= 20, f"{dup} external game sides are stored more than once"; listed = np.array([x in own for x in pair], bool); keep = first & listed
    if dup:
        log(f"external parts: {dup} game side(s) stored more than once, the first copy kept: {[x for x, f_ in zip(pair, first) if not f_][:5]}")
    if not keep.all():  # eval only: the other side of a kept game can belong to an account that is not reserved
        assert KEEP is not None or listed.all(), f"{int((~listed).sum())} external game sides are not in the account tables"
        key, GL, POS, MOV, LEN, SUBJ = key[keep], GL[keep], POS[keep], MOV[keep], LEN[keep], SUBJ[keep]; pair = [x for x, k in zip(pair, keep) if k]; N = len(key)
    names = np.array([own[x] for x in pair]); accs, acc = np.unique(names, return_inverse=True)
    log(f"external games loaded: {N} games of {len(accs)} accounts from {len(files)} parts ({GL.nbytes / 1e9:.1f} GB); mean plies {float(LEN.mean()):.0f}")
    return dict(key=key, GL=GL, POS=POS, MOV=MOV, LEN=LEN, SUBJ=SUBJ, acc=acc.astype(np.int64), accs=accs)


def null_rows(X):
    """The stranger queries: the first NULL_PER games (by key) of every reserved account."""
    res = np.array([reserved(a) for a in X["accs"]]); o = np.lexsort((X["key"], X["acc"])); a_s = X["acc"][o]; rank = np.arange(len(o)) - np.searchsorted(a_s, a_s, side="left")
    return res, np.sort(o[res[a_s] & (rank < NULL_PER)])


def train_one(v, seed, G, Tt, X, steps, log, gamma=1.0, centres=True, mask=0.3, warm_b=3000, ramp_b=2000, lr_c=0.1, share_x=0.25, oe=0.0, eval_ckpt=None, periodic=True):
    import torch
    F = torch.nn.functional; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}_s{seed}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS
    n_x = int(round(share_x * P_EP * M_EP / (2 * M_X))) * 2 * M_X if X is not None and eval_ckpt is None else 0; P_X = n_x // M_X; P_C = (P_EP * M_EP - n_x) // M_EP; n_c = P_C * M_EP; assert n_c + n_x == P_EP * M_EP and P_C >= 8
    cfg = dict(gamma=gamma, centres=bool(centres), mask=mask, warm_b=warm_b, ramp_b=ramp_b, lr_c=lr_c, episode=[P_EP, M_EP], min_games=base.MIN_GAMES, width=base.D, ffn=base.FFN, layers=base.LAYERS, ext_share=share_x, ext_players=P_X, ext_games_each=M_X, comp_players=P_C, oe=oe)
    log(f"net {v} seed {seed} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use); n_all = np.bincount(owner[~is_held], minlength=C)
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx
    p_pl = cnt[tp].astype(np.float64) ** float(gamma); p_pl /= p_pl.sum(); trained = np.zeros(C, bool); trained[tp] = True
    log(f"players trained {len(tp)} (>= {base.MIN_GAMES} games; they hold {float(cnt[tp].sum() / max(cnt.sum(), 1)):.3f} of the rows), games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}")
    nq = np.zeros(0, np.int64)
    if X is not None:
        res_acc, nq = null_rows(X); xo = np.argsort(X["acc"], kind="stable"); xa = X["acc"][xo]; A = len(X["accs"]); xfirst = np.searchsorted(xa, np.arange(A)); xcnt = np.bincount(X["acc"], minlength=A); xtp = np.flatnonzero(~res_acc & (xcnt >= M_X)); WX = int(xcnt.max()) if A else 0
        log(f"external accounts: {A} loaded, {int(res_acc.sum())} reserved as strangers ({len(nq)} stranger queries), {len(xtp)} with >= {M_X} games train ({P_X} per episode, {M_X} games each; {P_C} competition players per episode)")
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); rng3 = np.random.default_rng(218000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = m216.build_model().to(dev)
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731
    best = {"top1w": -1.0, "top1": -1.0, "step": -1}; best_state = None; best_wc = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731
    trained_t = tb(trained); Wc = opt_c = None; ck = CK / f"ckpt_{tag}.pt"; start = 0
    if eval_ckpt is None and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        if "opt" in st:
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", [])
            if st.get("best_is_current"):
                best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_wc = None if st.get("Wc") is None else st["Wc"].clone()
            if st.get("Wc") is not None:
                Wc = torch.nn.Parameter(st["Wc"].to(dev)); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9); opt_c.load_state_dict(st["opt_c"])
            log(f"resumed at step {start}, best so far {best}")
        else:
            start = steps; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")

    def embed(src, rows, subj):
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rows), 1024):
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()
        net.train(); return out

    def train_protos():
        E = embed(G, all_rows, G["SUBJ"][all_rows]); S = np.zeros((C, D_OUT)); np.add.at(S, owner[all_rows], E); S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)
        return tb(S.astype(np.float32))

    def unit_centres(W):
        Wn = F.normalize(W.detach().float(), dim=-1).cpu().numpy().astype(np.float32); Wn[~trained] = 0.0
        return Wn

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)
    if not len(hk):
        hk = np.flatnonzero(~is_held)[:8]
    bandf = lambda n: np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= x < hi), 0) for x in n])  # noqa: E731
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8); band = bandf(cnt[h_truth]); band_all = bandf(n_all[h_truth]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)

    def summary(S, sel, bd):
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[sel & (bd == k)] == 0).mean()) if (sel & (bd == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)
        return {"n": int(sel.sum()), "top1": float((r[sel] == 0).mean()), "top5": float((r[sel] < 5).mean()), "top20": float((r[sel] < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}

    def open_set(S, Sx, valid, ngames):
        """Owner first and margin above the strangers' 95th percentile (standardised scores + BETA log games), held-out own games of the players trained on."""
        def zm(M):
            Z = (M - M[:, valid].mean(1, keepdims=True)) / (M[:, valid].std(1, keepdims=True) + 1e-9) + BETA * np.log(np.maximum(ngames, 1))[None]; Z[:, ~valid] = -1e9; top = -np.sort(-np.partition(Z, -5, axis=1)[:, -5:], axis=1)
            return Z, top[:, 0] - top[:, 1:].mean(1)
        Z, m = zm(S); _, mx = zm(Sx); thr = float(np.quantile(mx, 0.95)); first = Z.argmax(1) == h_truth; clear = first & (m >= thr); wrong = ~first & (m >= thr)
        t = np.array([float(clear[seen & (band == k)].mean()) if (seen & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t)
        return {"threshold": thr, "first": float(first[seen].mean()), "clear": float(clear[seen].mean()), "wrong_above": float(wrong[seen].mean()), "clear_by_band": [None if not np.isfinite(x) else float(x) for x in t], "clear_w": float((share[ok] * t[ok]).sum() / max(share[ok].sum(), 1e-12)), "strangers": int(len(mx))}

    def line(x):
        return f"top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}"

    def oline(x):
        return f"owner first and clear of the strangers {x['clear']:.3f} (weighted by rows {x['clear_w']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["clear_by_band"])) + f"); another player above the threshold {x['wrong_above']:.3f}"

    def evaluate(step, W):
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
        Eh = embed(G, hk, G["SUBJ"][hk]); S = Eh @ Pm.T; res = {"step": step}
        for nm, sel, bd in (("seen", seen, band), ("unseen", ~seen, band_all)):
            if sel.any():
                res[nm] = summary(S, sel, bd)
        log(f"  step {step}: held-out own games, players trained on ({res['seen']['n']}), prototypes: " + line(res["seen"]) + (f"; unseen players ({res['unseen']['n']}) top-1 {res['unseen']['top1']:.3f}, by their games " + " ".join("-" if b is None else format(b, ".3f") for b in res["unseen"]["by_band"]) if "unseen" in res else ""))
        Ex = embed(X, nq, X["SUBJ"][nq]) if len(nq) else None
        if Ex is not None:
            res["open"] = open_set(S, Ex @ Pm.T, n_all > 0, n_all); log(f"  step {step}: prototypes: " + oline(res["open"]))
        if W is not None:
            Wn = unit_centres(W); Sc = Eh @ Wn.T; Sc[:, ~trained] = -2.0; res["centres"] = summary(Sc, seen, band); log(f"  step {step}: held-out own games, players trained on ({res['centres']['n']}), learned centres: " + line(res["centres"]))
            if Ex is not None:
                res["open_centres"] = open_set(Sc, Ex @ Wn.T, trained, cnt); log(f"  step {step}: learned centres: " + oline(res["open_centres"]))
        return res, Pm

    if eval_ckpt is not None:
        st = torch.load(eval_ckpt, map_location=dev, weights_only=False); net.load_state_dict(st["best_state"]); wc = st.get("best_wc"); res, _ = evaluate(int(st["best"]["step"]), None if wc is None else wc.to(dev))
        (OUT / f"eval_{tag}.json").write_text(json.dumps({"checkpoint": str(eval_ckpt), "best": st["best"], "final": res, "bands": BANDS, "share_rows": share.tolist()}, indent=1)); log(f"net {v} seed {seed}: evaluation of {eval_ckpt} written"); return
    net.train(); tl = ta = tbk = toe = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}; K_LOG = 20 if steps <= 200 else 500
    grp = tb(np.concatenate([np.repeat(np.arange(P_C), M_EP), P_C + np.repeat(np.arange(P_X), M_X)])); P_T = P_C + P_X; logC = math.log(max(int(trained.sum()), 1))
    for step in range(start, steps):
        if centres and step >= warm_b and Wc is None:
            t_r = time.time(); Wc = torch.nn.Parameter(train_protos().clone()); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9)
            log(f"  step {step}: centres started from the prototypes of {int(trained.sum())} players ({time.time() - t_r:.0f} s)"); t_last += time.time() - t_r
        pl = rng.choice(tp, P_C, replace=False, p=p_pl); u = rng.random((P_C, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); cls_t = np.repeat(pl, M_EP)
        gl, pos, mov, ln, sj = G["GL"][rows], G["POS"][rows], G["MOV"][rows], G["LEN"][rows], G["SUBJ"][rows]
        if P_X:
            ax = rng3.choice(xtp, P_X, replace=False); ux = rng3.random((P_X, WX)); ux[np.arange(WX)[None] >= xcnt[ax][:, None]] = 2.0; rx = xo[xfirst[ax][:, None] + np.argsort(ux, axis=1)[:, :M_X]].reshape(-1)
            gl = np.concatenate([gl, X["GL"][rx]]); pos = np.concatenate([pos, X["POS"][rx]]); mov = np.concatenate([mov, X["MOV"][rx]]); ln = np.concatenate([ln, X["LEN"][rx]]); sj = np.concatenate([sj, X["SUBJ"][rx]])
        pos = SYM[int(rng2.integers(0, 8))][pos.astype(np.int64)]; gl = tb(gl)
        if mask > 0:
            hit = torch.rand(len(pos), base.T_MAX, device=dev) < torch.rand(len(pos), 1, device=dev) * mask; gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net(gl, tb(pos), tb(mov.astype(np.int64)), tb(ln.astype(np.int64)), tb(sj.astype(np.int64)))
        z = z.float(); tot = torch.zeros(P_T, D_OUT, device=dev).index_add_(0, grp, z); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T
        own = sc * (z * F.normalize(tot[grp] - z, dim=-1)).sum(-1); logits = logits.scatter(1, grp[:, None], own[:, None]); loss = F.cross_entropy(logits, grp); acc = float((logits.argmax(1) == grp).float().mean())
        if Wc is not None:
            w = min(1.0, (step - warm_b) / max(ramp_b, 1)); ct = tb(cls_t); cos = (z @ F.normalize(Wc, dim=-1).T).masked_fill(~trained_t[None], -1e4); loss = loss + w * F.cross_entropy(sc * cos[:n_c], ct); tbk += float((cos[:n_c].argmax(1) == ct).float().mean()); nb += 1
            if oe > 0 and P_X:
                l_oe = -F.log_softmax(sc * cos[n_c:], dim=-1)[:, trained_t].mean() - logC; loss = loss + w * oe * l_oe; toe += float(l_oe.detach())
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if opt_c is not None:
            for g_ in opt_c.param_groups:
                g_["lr"] = lr_c * fac(step)
            opt_c.step()
        tl += float(loss.detach()); ta += acc
        if (step + 1) % K_LOG == 0:
            log(f"  step {step + 1}/{steps}: loss {tl / K_LOG:.4f}, episode top-1 {ta / K_LOG:.3f}" + (f", top-1 against all centres {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" + (f", outlier term above its minimum {toe / max(nb, 1):.3f}" if oe > 0 and P_X else "") if nb else "") + f", scale {float(sc):.1f}, {(time.time() - t_last) / K_LOG:.2f} s/step, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.0f} GB"); tl = ta = tbk = toe = 0.0; nb = 0; t_last = time.time()
        if (step + 1) in ev_steps:
            res, _ = evaluate(step + 1, Wc); hist.append(res)
            if res["seen"]["top1w"] > best["top1w"]:
                best = {"top1w": res["seen"]["top1w"], "top1": res["seen"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_wc = None if Wc is None else Wc.detach().cpu().clone()
                if "centres" in res:
                    best["centres_top1w"] = res["centres"]["top1w"]; best["centres_top1"] = res["centres"]["top1"]
            (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "steps": steps, "bands": BANDS, "share_rows": share.tolist(), "partial": True}, indent=1))
            if periodic:
                same = best["step"] == step + 1
                tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": None if same else best_state, "best_wc": None if same else best_wc, "best_is_current": same, "hist": hist, "Wc": None if Wc is None else Wc.detach().cpu(), "opt_c": None if opt_c is None else opt_c.state_dict()}, tmp); os.replace(tmp, ck)
            t_last = time.time()
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
    if SAVE_OUT:  # off by default (/home is short of space): the direction-averaged vectors are rebuilt from the checkpoint by the TTA stage
        np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv, **extra)
    (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "best_wc": best_wc, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, ck)
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game; centres {'saved' if extra else 'none'}); checkpoint slimmed to the best state")


def main():
    global OUT, CK, KEEP, SAVE_OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--jobs", nargs="+", required=True, help="one trainer per GPU, NET:SEED with NET in 1 2 3 4 all"); ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--ext-share", type=float, default=0.25); ap.add_argument("--oe", type=float, default=0.0); ap.add_argument("--ext-sets", nargs="+", default=list(EXT_SETS)); ap.add_argument("--out", required=True, help="arm name: sub-folder of results/m218 and of the checkpoint folder")
    ap.add_argument("--ck-dir", default=None); ap.add_argument("--no-periodic", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--eval-ckpt", default=None, help="folder with finished checkpoints ckpt_<net>_s<seed>.pt: evaluation only (strangers = the reserved accounts)"); ap.add_argument("--workers", type=int, default=32); ap.add_argument("--save-out", action="store_true", help="also write out_<net>_s<seed>.npz (prototypes, query vectors, centres) - needed by the TTA stage for the centres")
    a_ = ap.parse_args(); t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    SAVE_OUT = bool(a_.save_out)
    base.FULL, base.EXT, base.VAR = True, None, "S11"; base.P_EP, base.M_EP = 128, 16; base.MIN_GAMES = max(base.MIN_GAMES, 16); base.D, base.FFN = 512, 2048; base.LAYERS = 8; assert base.T_MAX == 256 and m216.CKPT_LAYERS == 0
    jobs = [(-1 if x.split(":")[0] == "all" else int(x.split(":")[0]), int(x.split(":")[1])) for x in a_.jobs]; assert len(set(jobs)) == len(jobs) and all(v in base.NETS and s > 0 for v, s in jobs), a_.jobs
    OUT, CK = OUT / a_.out, Path(a_.ck_dir) if a_.ck_dir else CK / a_.out; sets = ["pilot"] if a_.smoke else a_.ext_sets
    if a_.smoke:
        OUT = ROOT / "results/m218_smoke" / a_.out; base.EVAL_EVERY = 40
    base.OUT = OUT; gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(jobs), f"{len(jobs)} trainers need as many GPUs, got {gpus}"
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps; kw = dict(share_x=a_.ext_share, oe=a_.oe, periodic=not a_.no_periodic)
    if a_.smoke:
        kw.update(warm_b=20, ramp_b=20)
    files, own = ext_tables(sets)
    if a_.eval_ckpt:  # only the stranger queries are read from the external parts
        own = {x: acc for x, acc in own.items() if reserved(acc)}; KEEP = np.array(sorted({x[0] for x in own}))
    log(f"arm {a_.out}: external share {a_.ext_share}, outlier weight {a_.oe}, sets {sets} ({len(files)} parts, {len(own)} games listed{'' if KEEP is None else f', {len(KEEP)} of reserved accounts kept'}); outputs {OUT}, checkpoints {CK}")
    pool = mp.get_context("fork").Pool(a_.workers)  # before anything large exists (strict overcommit on the nodes)
    G, Tt = base.load_games(a_.smoke, log); X = load_ext(files, own, pool, log); pool.close(); pool.join(); kids = {}
    for k, (v, seed) in enumerate(jobs):
        try:
            pid = os.fork()
        except OSError as e:
            log(f"could not start net {v} seed {seed}: {e}"); break
        if pid == 0:
            code = 0
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])
                out = open(ROOT / f"logs/nano4/m218_{a_.out}_net{v}_s{seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out
                t1 = time.time(); train_one(v, seed, G, Tt, X, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), eval_ckpt=(Path(a_.eval_ckpt) / f"ckpt_{v}_s{seed}.pt") if a_.eval_ckpt else None, **kw)
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
        f = OUT / (f"eval_{v}_s{seed}.json" if a_.eval_ckpt else f"train_{v}_s{seed}.json")
        if f.exists():
            d = json.load(open(f)); H = [d["final"]] if a_.eval_ckpt else d["history"]
            print(f"  net {v} seed {seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: {h['seen']['top1w']:.3f}" + (f" / centres {h['centres']['top1w']:.3f}" if "centres" in h else "") + (f" / clear {h['open']['clear_w']:.3f}" if "open" in h else "") + (f" / clear, centres {h['open_centres']['clear_w']:.3f}" if "open_centres" in h else "") for h in H))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
