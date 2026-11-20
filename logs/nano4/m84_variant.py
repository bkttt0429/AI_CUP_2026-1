"""Section 4M M84 (2026-11-11): matcher variants with more games per candidate, fused through the cache of the five-component block
score (m84_cache.py), evaluated against the current identifier (M82's five + matcher, "5e") and turned into candidate files without
anyone attending. Network, loss and cross-fit as M82 (aicup_pid.embed.EmbedNet; fold v's queries use the network trained without fold
v's players); per episode one query game and up to 64 support games per player; a candidate's prototype = the sum over ALL his games.
  --stage embed --var ref|A|ALL    ref = M82's stored embeddings (cache check); A = base + x40 + x80 rows (up to 80 games per player);
                                   ALL = every training row (results/m84_rows from m84_rows_extract.py), seeds 0, 1, 2
  --stage eval --tag ref|A|B|B3    A: var A; B: var ALL member 0; B3: var ALL, the three members' standardised scores averaged
  --stage finalize                 upload_next/README.md + candidate files from the eval JSONs (also run at the end of every eval)
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "logs/nano4"), str(Path.home() / "BK_AICUP_2026-2/src")]
import ident_pool as ip  # noqa: E402
import m71_step2 as s2  # noqa: E402
import m81_tactic_id as m81  # noqa: E402
import m82_cond_id as m82  # noqa: E402
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402
from ens_compare import ES, FOLDS, load_fold  # noqa: E402

CACHE, M82, RV, UP = ROOT / "results/m84_cache", ROOT / "results/m82", ROOT / "results/r_variants", ROOT / "upload_next"
NETS = (1, 2, 3, 4, -1)
TAGS = {"ref": ("ref", [0], None), "A": ("A", [0], ("46", "47")), "B": ("ALL", [0], ("48", "49")), "B3": ("ALL", [0, 1, 2], ("50", "51")),
        "P": ("P", [0], ("52", "53")), "P3": ("P", [0, 1, 2], ("54", "55")), "W": ("ref", [0], ("58", "59")), "R": ("R", [0], ("60", "61")), "C": ("PR", [0, 1, 2], ("62", "63")), "S": ("S", [0], ("66", "67")), "SP": ("S", [0], ("68", "69")), "S2": ("S2", [0], ("76", "77")), "SP2": ("S2", [0], ("78", "79")), "S2e": ("S2e", [0, 1, 2], ("80", "81")), "SP2e": ("S2e", [0, 1, 2], ("82", "83")),
        "S3": ("S3", [0], ("86", "87")), "SP3": ("S3", [0], ("88", "89")), "S2x": ("S2x", [0, 1, 2, 3], ("90", "91")), "SP2x": ("S2x", [0, 1, 2, 3], ("92", "93")), "S3c": ("S3c", [0], ("94", "95")), "S2b": ("S2e", [1], None), "S2c": ("S2e", [2], None), "S2y": ("S2y", [0, 1, 2, 3, 4], ("96", "97")), "SP2y": ("S2y", [0, 1, 2, 3, 4], ("98", "99")),  # M101: S3 = the M90 recipe trained with external players (one seed); S2x = S2e's three seeds plus S3
        "S4": ("S4", [0], None), "SP4": ("S4", [0], None), "S4e": ("S4e", [0, 1, 2], None), "SP4e": ("S4e", [0, 1, 2], None), "S2y4": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4": ("S2y", [0, 1, 2, 3, 4], None), "SP2e4": ("S2e", [0, 1, 2], None), "SP4f": ("S4f", [0, 1, 2, 3], None), "SP2y4f": ("S2y", [0, 1, 2, 3, 4], None), "SP4et": ("S4et", [0, 1, 2], None), "SP2y4t1": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4t2": ("S2y", [0, 1, 2, 3, 4], None), "SP2y4et": ("S2y", [0, 1, 2, 3, 4], None), "SP4ft": ("S4ft", [0, 1, 2, 3], None), "S4t": ("S4t", [0], None), "SP4t": ("S4t", [0], None), "SP2y4ft": ("S2y", [0, 1, 2, 3, 4], None), "S4s1": ("S4e", [1], None), "S4s2": ("S4e", [2], None), "S4s3": ("S4f", [3], None),  # M120 / M122 (S4f = four seeds, the fourth trained with the ply mask): the symmetry-augmented networks (no candidate numbers, not in PRIORITY: the formal test is logs/nano4/m121)
        "Pp": ("P", [0], None), "Pp3": ("P", [0, 1, 2], None), "Cp": ("P", [0, 1, 2], None)}  # tag -> (var file, members, ct numbers); Pp / Pp3 = previews of P on the rows extracted so far (no candidate, not in PRIORITY)
PRIORITY = ("SP2y", "S2y", "SP2x", "S2x", "SP2e", "S2e", "SP3", "S3", "S3c", "SP2", "S2", "SP", "S", "C", "P3", "P", "B3", "B", "R", "A", "W")
MULTI = {"S": [("S", [0])], "SP": [("P", [0, 1, 2]), ("S", [0])], "S2": [("S2", [0])], "SP2": [("P", [0, 1, 2]), ("S2", [0])],
         "S2e": [("S2e", [0, 1, 2])], "SP2e": [("P", [0, 1, 2]), ("S2e", [0, 1, 2])],
         "S3": [("S3", [0])], "SP3": [("P", [0, 1, 2]), ("S3", [0])], "S2x": [("S2x", [0, 1, 2, 3])], "SP2x": [("P", [0, 1, 2]), ("S2x", [0, 1, 2, 3])],
         "S3c": [("S3c", [0])], "S2b": [("S2e", [1])], "S2c": [("S2e", [2])],  # M105: seeds 1 and 2 of M90 as single voters (not in PRIORITY: no candidate file)
         "S2y": [("S2y", [0, 1, 2, 3, 4])], "SP2y": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4])],
         "S4": [("S4", [0])], "SP4": [("P", [0, 1, 2]), ("S4", [0])], "S4e": [("S4e", [0, 1, 2])], "SP4e": [("P", [0, 1, 2]), ("S4e", [0, 1, 2])],  # M120: symmetry-augmented networks, seed 0 / three seeds
         "S2y4": [("S2y", [0, 1, 2, 3, 4]), ("S4e", [0, 1, 2])], "SP2y4": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4e", [0, 1, 2])], "SP2e4": [("P", [0, 1, 2]), ("S2e", [0, 1, 2]), ("S4e", [0, 1, 2])],
         "SP4et": [("P", [0, 1, 2]), ("S4et", [0, 1, 2])], "SP2y4t1": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0])], "SP2y4t2": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0, 1])],
         "SP2y4et": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4et", [0, 1, 2])], "SP4ft": [("P", [0, 1, 2]), ("S4ft", [0, 1, 2, 3])],  # S4et / S4ft = three / four new seeds with orientation-averaged queries (M124)
         "S4t": [("S4t", [0])], "SP4t": [("P", [0, 1, 2]), ("S4t", [0])], "SP2y4ft": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4ft", [0, 1, 2, 3])],  # M124: query vectors averaged over the eight board orientations
         "S4s1": [("S4e", [1])], "S4s2": [("S4e", [2])], "S4s3": [("S4f", [3])],  # single seeds of the new networks (report only: plain symmetry / + opening hidden / + ply mask)
         "SP4f": [("P", [0, 1, 2]), ("S4f", [0, 1, 2, 3])], "SP2y4f": [("P", [0, 1, 2]), ("S2y", [0, 1, 2, 3, 4]), ("S4f", [0, 1, 2, 3])]}  # M88: fusion of the cached block with one or two matcher sources, each with its own weight
CGRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
WGRID = (1.0, 1.5, 2.0, None, 3.0, 4.0, 6.0)  # None = M82's weight (2.53), the reference
DESC = {"A": "每位候選用到 80 局", "B": "每位候選用全部對局", "B3": "全部對局＋3 個 seed 平均", "P": "全部對局＋分階段（序盤／中盤／後半）的描述", "P3": "全部對局＋分階段的描述＋3 個 seed 平均", "W": "比對成分的融合權重改以錨點分數（交叉擬合）選定", "R": "比對網路一半的訓練回合只拿同等級的玩家互相區分（至多 80 局）", "C": "合併版：全部對局＋分階段的描述＋同等級回合＋3 個 seed 平均＋權重以錨點分數選定",
        "S": "逐手的認人模型取代原本的比對", "SP": "逐手的認人模型＋分階段描述的比對，兩者並用",
        "S2": "用全部對局訓練的逐手認人模型取代原本的比對", "SP2": "用全部對局訓練的逐手認人模型＋分階段描述的比對",
        "S2e": "全部對局的逐手認人模型（3 個 seed 平均）取代原本的比對", "SP2e": "全部對局的逐手認人模型（3 個 seed 平均）＋分階段描述的比對",
        "S3": "加入外部公開棋譜玩家一起訓練的逐手認人模型（1 個 seed）取代原本的比對", "SP3": "加入外部玩家訓練的逐手認人模型（1 個 seed）＋分階段描述的比對",
        "S2x": "逐手認人模型 4 個成員平均（3 個 seed＋外部玩家版）取代原本的比對", "SP2x": "逐手認人模型 4 個成員平均（3 個 seed＋外部玩家版）＋分階段描述的比對",
        "S3c": "2 萬步排程的逐手認人模型（對照，不加外部玩家，1 個 seed）取代原本的比對", "S2y": "逐手認人模型 5 個成員平均（3 個 seed＋外部玩家版＋對照版）取代原本的比對", "SP2y": "逐手認人模型 5 個成員平均（3 個 seed＋外部玩家版＋對照版）＋分階段描述的比對"}
STEPS, EPISODE, S_EP, W_MAX, MIN_GAMES = 6000, 512, 64, 256, 8
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)


def load_rows(var, log):
    """key, pid, fold, T of every game side with a trunk row, for the support set of `var` (base + x40 [+ x80 [+ the rest]])."""
    B = np.load(M82 / "rows_train.npz"); key, pid, fold, T = [B["key"]], [B["pid"]], [B["fold"]], [B["T"]]; pfold = dict(zip(B["pid"], B["fold"]))
    if var in ("A", "R", "ALL"):
        with mp.get_context("fork").Pool(12) as pool:
            res = pool.map(m82.part_rows, sorted((ROOT / "results/trunk_vec_x80").glob("fold*/games_s*_p*.csv")), chunksize=4)
        k = np.concatenate([t["game_key"].to_numpy() for t, _, _ in res]).astype(str); p = np.concatenate([t["player_id"].to_numpy() for t, _, _ in res]).astype(str)
        key.append(k); pid.append(p); fold.append(np.array([pfold.get(x, -9) for x in p], np.int8)); T.append(np.concatenate([o[0] for _, o, _ in res])); log(f"x80 rows: {len(k)} games")
    if var == "ALL":
        rows = m81.load_rows(); owner = dict(zip(rows.game_key, rows.player_id)); del rows; ks, ts = [], []
        for f in sorted((ROOT / "results/m84_rowsp").glob("rows_*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f); ks.append(z["key"].astype(str)); ts.append(z["T"])
        k = np.concatenate(ks); p = np.array([owner[x] for x in k]); key.append(k); pid.append(p); fold.append(np.array([pfold.get(x, -9) for x in p], np.int8)); T.append(np.concatenate(ts)); log(f"extracted rows: {len(k)} games")
    key, pid, fold, T = np.concatenate(key), np.concatenate(pid), np.concatenate(fold), np.concatenate(T); _, first = np.unique(key, return_index=True); first.sort()
    return key[first], pid[first], fold[first], T[first]


def part_phase(csv_path):
    """(game keys or test task numbers, phase means [2 sides, G, 384]) of one trunk-vector part: the mean 128-dim vector over a side's
    plies with t < 60, 60 <= t < 150, t >= 150 (zeros when a phase has no ply) -- m84_rows_extract.side_rows' definition."""
    import pandas as pd
    csv_path = Path(csv_path); tab = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with np.load(csv_path.with_name(csv_path.name.replace("games_", "moves_").replace(".csv", ".npz"))) as z:
        X = np.hstack([z["g"], z["l"]]).astype(np.float64); target, row, t = z["target"], z["game_row"], z["t"]
    n_g = len(tab); ph = np.zeros((2, n_g, 384), np.float16)
    for side, m in enumerate((target, ~target)):
        for k, pm in enumerate((t < 60, (t >= 60) & (t < 150), t >= 150)):
            mm = m & pm; r = row[mm]; n = np.bincount(r, minlength=n_g); nz = n > 0
            if nz.any():
                order = np.argsort(r, kind="stable"); ph[side][nz, 128 * k:128 * (k + 1)] = np.add.reduceat(X[mm][order], (np.cumsum(n) - n)[nz], axis=0) / n[nz][:, None]
    return (tab["game_key"].to_numpy().astype(str) if "player_id" in tab.columns else tab["task"].to_numpy().astype(int)), ph


def stage_embed(var):
    import torch
    from aicup_pid.embed import EmbedNet
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); N = len(q["qkey"]); side, qnet, qtest = q["side"], q["net"], q["qtest"]
    B = np.load(M82 / "rows_train.npz"); bkey = {k: i for i, k in enumerate(B["key"])}; Rt = np.load(M82 / "rows_test.npz"); held = set(json.load(open(M82 / "held_keys.json")))
    tr = np.flatnonzero(qtest < 0); src = np.array([bkey[k] for k in q["qkey"][tr]]); te = np.flatnonzero(qtest >= 0); QX = np.zeros((N, 256), np.float32)
    QX[tr] = np.where(side[tr][:, None] == 1, B["O"][src], B["T"][src]); QX[te] = np.where(side[te][:, None] == 1, Rt["O"][qtest[te]], Rt["T"][qtest[te]])
    if var == "ref":  # M82's own embeddings, to check that the cache reproduces M82's fusion
        protos = np.zeros((1, 5, C, 128), np.float32); qvec = np.zeros((1, N, 128), np.float16); keep = np.array([k not in held for k in B["key"]]); cand = np.array([pidx[p] for p in B["pid"]])
        for j, v in enumerate(NETS):
            eT = np.load(M82 / f"emb_T_{v}.npy"); Pm = np.zeros((C, 128)); np.add.at(Pm, cand[keep], eT[keep]); protos[0, j] = Pm / np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)
            if v > 0:
                eO = np.load(M82 / f"emb_O_{v}.npy"); s = tr[qnet[tr] == j]; g = np.array([bkey[k] for k in q["qkey"][s]]); qvec[0, s] = np.where(side[s][:, None] == 1, eO[g], eT[g])
            else:
                s = tr[qnet[tr] == j]; qvec[0, s] = eT[[bkey[k] for k in q["qkey"][s]]]; eo, et = np.load(M82 / "emb_test_O.npy"), np.load(M82 / "emb_test_T.npy"); qvec[0, te] = np.where(side[te][:, None] == 1, eo[qtest[te]], et[qtest[te]])
        np.savez(CACHE / "var_ref.npz", protos=protos, qvec=qvec); log("var ref written from M82's embeddings"); return
    key, pid, fold, T = load_rows("ALL" if var in ("P", "PR") else var, log); is_held = np.array([k in held for k in key]); owner = np.array([pidx[p] for p in pid]); seeds = (0,) if var in ("A", "R") else (0, 1, 2)
    if var in ("P", "PR"):  # the phase means join the row: 256 -> 640 dimensions, for the games and for the queries
        kidx = {k: i for i, k in enumerate(key)}; PT = np.zeros((len(key), 384), np.float16); PTB = np.zeros((len(B["key"]), 384), np.float16); POB = np.zeros_like(PTB); PTt = np.zeros((len(Rt["qid"]), 384), np.float16); POt = np.zeros_like(PTt)
        csvs = [f for d_ in ("trunk_vec", "trunk_vec_x40", "trunk_vec_x80") for f in sorted((ROOT / "results" / d_).glob("fold*/games_s*_p*.csv"))]
        with mp.get_context("fork").Pool(12) as pool:
            res = pool.map(part_phase, csvs, chunksize=4); rest = pool.map(part_phase, sorted((ROOT / "results/ct_submission/vec").glob("games_s*_p*.csv")), chunksize=4)
        for ident, ph in res:
            ii = np.array([kidx.get(k, -1) for k in ident]); ok = ii >= 0; PT[ii[ok]] = ph[0][ok]; jb = np.array([bkey.get(k, -1) for k in ident]); ok = jb >= 0; PTB[jb[ok]] = ph[0][ok]; POB[jb[ok]] = ph[1][ok]
        for ident, ph in rest:
            PTt[ident] = ph[0]; POt[ident] = ph[1]
        n_ex = 0
        for f in sorted((ROOT / "results/m84_rowsp").glob("rows_*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            z = np.load(f); ii = np.array([kidx.get(k, -1) for k in z["key"].astype(str)]); ok = ii >= 0; PT[ii[ok]] = z["PT"][ok]; n_ex += int(ok.sum())
        log(f"phase means: {sum(len(i_) for i_, _ in res)} games from stored parts, {n_ex} extracted, {len(PTt)} test; games with no phase mean at all: {int((np.abs(PT).sum(1) == 0).sum())}")
        T = np.hstack([T, PT.astype(np.float32)]); del PT; QP = np.zeros((N, 384), np.float32)
        QP[tr] = np.where(side[tr][:, None] == 1, POB[src], PTB[src]); QP[te] = np.where(side[te][:, None] == 1, POt[qtest[te]], PTt[qtest[te]]); QX = np.hstack([QX, QP])
    D_IN = T.shape[1]
    log(f"var {var}: {len(key)} games of {len(np.unique(owner))} players; games per player: median {int(np.median(np.bincount(owner, minlength=C)))}, max {int(np.bincount(owner, minlength=C).max())}")
    protos = np.zeros((len(seeds), 5, C, 128), np.float32); qvec = np.zeros((len(seeds), N, 128), np.float16); owner_t = torch.as_tensor(owner, device=dev); rep = {}
    for si, seed in enumerate(seeds):
        for j, v in enumerate(NETS):
            torch.manual_seed(1000 * seed + j); rng = np.random.default_rng(1000 * seed + j); use = ~is_held & (fold != v); n_rows = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(n_rows >= MIN_GAMES); W = int(min(n_rows.max(), W_MAX))
            idx = rng.permutation(np.flatnonzero(use)); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos = np.arange(len(o)) - np.searchsorted(o, o, side="left"); k_ = pos < W
            tab = np.full((C, W), -1, np.int64); tab[o[k_], pos[k_]] = idx[k_]; cnt = np.minimum(n_rows, W)
            mu = T[use].mean(0); sd = T[use].std(0) + 1e-6; std = lambda A: torch.as_tensor((A - mu) / sd, dtype=torch.float32, device=dev)  # noqa: E731
            X = std(T); tab_t = torch.as_tensor(tab[tp], device=dev); cnt_t = torch.as_tensor(cnt[tp], device=dev); P = min(EPISODE, len(tp)); S = min(S_EP + 1, W)
            net = EmbedNet(D_IN, 512, 128, 0.1).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=STEPS, pct_start=0.1)
            ar = torch.arange(W, device=dev); tgt_all = torch.arange(P, device=dev); net.train()
            rk_ = torch.as_tensor(q["plab"][tp], device=dev); by_rank = [torch.nonzero(rk_ == r_)[:, 0] for r_ in range(10)] if var in ("R", "PR") else None  # rungs R, C: every other episode inside one level
            for step in range(STEPS):
                if by_rank is not None and step % 2 == 1:
                    pool_ = by_rank[(step // 2) % 10]; b = pool_[torch.randperm(len(pool_), device=dev)[:P]]
                else:
                    b = torch.randperm(len(tp), device=dev)[:P]
                Pn = len(b); tgt = tgt_all[:Pn]; c = cnt_t[b]; u = torch.rand(Pn, W, device=dev).masked_fill(ar[None] >= c[:, None], 2.0)
                rows = tab_t[b].gather(1, u.argsort(dim=1)[:, :S]); zq = net(X[rows[:, 0]]); sm = ar[None, 1:S] < c[:, None]
                zs = net(X[rows[:, 1:].clamp(min=0)].reshape(-1, D_IN)).reshape(Pn, S - 1, -1) * sm[..., None]
                logits = net.log_scale.exp() * zq @ torch.nn.functional.normalize(zs.sum(1), dim=-1).T
                loss = 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) + torch.nn.functional.cross_entropy(logits.T, tgt))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            net.eval()
            with torch.no_grad():
                enc = lambda A: torch.cat([net(std(A[i:i + 65536])) for i in range(0, len(A), 65536)])  # noqa: E731
                eT = enc(T); keep = torch.as_tensor(~is_held, device=dev); Pm = torch.zeros(C, 128, device=dev).index_add_(0, owner_t[keep], eT[keep]); protos[si, j] = torch.nn.functional.normalize(Pm, dim=-1).cpu().numpy()
                s = np.flatnonzero(qnet == j); qvec[si, s] = enc(QX[s]).cpu().numpy()
            log(f"  seed {seed} net {v}: {len(tp)} players, {int(use.sum())} rows, final loss {float(loss.detach()):.4f}, last episode top-1 {float((logits.argmax(1) == tgt).float().mean()):.3f}")
        Q, H = int(q["Q"]), int(q["H"]); hq = np.arange(2 * Q, 2 * Q + H); S_ = qvec[si, hq].astype(np.float32) @ protos[si, 4].T; truth = q["held_truth"]; r = (S_ > S_[np.arange(H), truth][:, None]).sum(1)
        rep[f"seed{seed}"] = {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "top20": float((r < 20).mean())}; log(f"seed {seed}: matcher alone on the held-out own games, top-1 / top-5 / top-20 {rep[f'seed{seed}']['top1']:.3f} / {rep[f'seed{seed}']['top5']:.3f} / {rep[f'seed{seed}']['top20']:.3f} (M82: 0.074 / 0.181 / 0.309)")
    np.savez(CACHE / f"var_{var}.npz", protos=protos, qvec=qvec); (CACHE / f"var_{var}.json").write_text(json.dumps({"games": int(len(key)), "heldout": rep}, indent=1)); log("written")


def anchors(z, seeds):
    """Per fold: scores of the ensemble posterior with the anchor evidence of identifier z (vectorised expected-score decision), and of the base."""
    out = []
    for v in FOLDS:
        y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)
        te, logE, _ = s2.evidence_per_query(z, v); key_row = {k: i for i, k in enumerate(z["key"][te])}; L = np.zeros_like(P)
        for i, gk in enumerate(q["game_keys"]):
            rows = [key_row[k] for k in gk if k in key_row]
            if rows:
                L[i] = logE[rows].sum(axis=0)
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); out.append((SM[(Pf @ SM).argmax(1), y], SM[(P @ SM).argmax(1), y], np.asarray(y), np.asarray(n), np.asarray(pid)))
    return out


def stage_eval(tag):
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    var, members, _ = TAGS[tag]; dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)
    V = np.load(CACHE / f"var_{var}.npz"); protos = torch.as_tensor(V["protos"][members], device=dev); qvec = V["qvec"][members]

    def zm(ids):
        """Matcher score of queries `ids` [n, C]: per member standardised over the valid candidates, averaged over the members."""
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool)
        if (ex >= 0).any():
            r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False
        out = torch.zeros(len(ids), C, device=dev); nj = torch.as_tensor(net[ids], device=dev).long()
        for m in range(len(members)):
            S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)
            for j in range(5):
                s = nj == j
                if s.any():
                    S[s] = qv[s] @ protos[m, j].T
            cnt = valid.sum(1, keepdim=True); mu = (S * valid).sum(1, keepdim=True) / cnt; sd = (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12); out += (S - mu) / sd
        return out / len(members), valid
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy").astype(np.float32); zh = torch.cat([zm(hq[i:i + 4096])[0] for i in range(0, H, 4096)]).cpu().numpy(); truth = q["held_truth"]
    w5e = json.loads((M82 / "report.json").read_text())["weights"]["5e"]["matcher"]; w = np.array([1.0, w5e]) if tag == "ref" else ip.fit_weights([hb, zh], truth)
    r = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1)  # noqa: E731
    rk = r(w[0] * hb + w[1] * zh); acc = {"top1": float((rk == 0).mean()), "top5": float((rk < 5).mean()), "top20": float((rk < 20).mean())}
    log(f"tag {tag}: weights block {w[0]:.2f}, matcher {w[1]:.2f}; held-out own games, fused top-1 / top-5 / top-20 {acc['top1']:.3f} / {acc['top5']:.3f} / {acc['top20']:.3f} (M82 5e: 0.268 / 0.437 / 0.588)")
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]
    assert np.array_equal(idx, np.concatenate([np.arange(2 * Q), np.arange(2 * Q + H, 2 * Q + H + 2 * T)])); b = np.zeros(len(idx), np.int64); m = np.zeros(len(idx))
    for i in range(0, len(idx), 2048):
        ids = idx[i:i + 2048]; z, valid = zm(ids); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev) + w[1] * z; F = F.masked_fill(~valid, -float("inf")); top = F.topk(2, dim=1)
        b[i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()
    del blk; log("all queries fused")
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
    val = {**base, "f_opp_lab": plab[b[o]], "f_opp_m": m[o], "f_opp_best": players[b[o]], "f_imp_lab": plab[b[t_]], "f_imp_m": m[t_]}
    np.savez_compressed(RV / f"m84_{tag}_val.npz", **val); np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[to]], f_opp_m=m[to], f_opp_best=players[b[to]], f_tgt_lab=plab[b[tt]], f_tgt_m=m[tt], f_tgt_best=players[b[tt]])
    same = float((val["f_opp_best"] == d82["f5e_opp_best"]).mean()); same_t = float((players[b[to]] == t82["f5e_opp_best"]).mean()); rep = {"weights": w.tolist(), "heldout_fused": acc, "same_top_candidate_as_5e": {"validation": same, "test": same_t}}
    log(f"same top candidate as M82's 5e identifier: validation opponent side {same:.4f}, test opponent side {same_t:.4f}" + (f" -> cache check {'OK' if min(same, same_t) >= 0.98 else 'FAILED'}" if tag == "ref" else ""))
    if tag == "ref":
        rep["cache_ok"] = bool(min(same, same_t) >= 0.98); (RV / "m84_ref_eval.json").write_text(json.dumps(rep, indent=1)); return
    ZN = {**base, "opp_lab": val["f_opp_lab"], "opp_m": val["f_opp_m"], "imp_m": val["f_imp_m"], "imp_lab": val["f_imp_lab"]}; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        A, B_ = anchors(ZN, seeds), anchors(ZO, seeds); y, n, pid = (np.concatenate([x[k] for x in A]) for k in (2, 3, 4)); res = {}
        for label, a, c in (("new_vs_base", [x[0] for x in A], [x[1] for x in A]), ("5e_vs_base", [x[0] for x in B_], [x[1] for x in B_]), ("new_vs_5e", [x[0] for x in A], [x[0] for x in B_])):
            dd = paired_diff(np.concatenate(a), np.concatenate(c), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(u, x[2], x[3], sizes=SIZES) - balanced_mean_over_n(v_, x[2], x[3], sizes=SIZES) for u, v_, x in zip(a, c, A)]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}
            log(f"  seeds {seeds} {label:12s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["new_vs_5e"]["pass"])
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False
    rep["cache_ok"] = cache_ok; rep["adopt"] = bool(all(ok) and cache_ok); log(f"PRE-REGISTERED: variant {tag} passes against the 5e identifier: {'YES' if rep['adopt'] else 'NO'}" + ("" if cache_ok else " (cache check missing or failed)"))
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float))
    if TAGS[tag][2] is not None:
        finalize()


def stage_eval_multi(tag):
    """M88: as stage_eval, for a fusion of the cached block with several matcher sources (MULTI[tag]); one weight per source, fitted on the
    held-out own games together with the block's."""
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players); src = []
    for var, members in MULTI[tag]:
        V = np.load(CACHE / f"var_{var}.npz"); members = [m_ for m_ in members if m_ < len(V["protos"])]; src.append((torch.as_tensor(V["protos"][members], device=dev), V["qvec"][members]))

    def zs(ids):
        """Per source the matcher score of queries `ids` [n, C] (standardised over the valid candidates, averaged over the members) and the valid mask."""
        ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); cnt = valid.sum(1, keepdim=True); out = []
        for protos, qvec in src:
            z = torch.zeros(len(ids), C, device=dev)
            for m in range(protos.shape[0]):
                S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[m, ids].astype(np.float32), device=dev)
                for j in range(5):
                    s = nj == j
                    if s.any():
                        S[s] = qv[s] @ protos[m, j].T
                mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)
            out.append(z / protos.shape[0])
        return out, valid
    hq = np.arange(2 * Q, 2 * Q + H); hb = np.load(CACHE / "block_held.npy").astype(np.float32); zh = [x.cpu().numpy() for x in zs(hq)[0]]; truth = q["held_truth"]; w = ip.fit_weights([hb, *zh], truth); rnk = lambda F: (F > F[np.arange(H), truth][:, None]).sum(1)  # noqa: E731
    acc = {}
    for name, F in [(f"source{k} ({MULTI[tag][k][0]}) alone", zh[k]) for k in range(len(zh))] + [("fused", w[0] * hb + sum(w[k + 1] * zh[k] for k in range(len(zh))))]:
        r_ = rnk(F); acc[name] = {"top1": float((r_ == 0).mean()), "top5": float((r_ < 5).mean()), "top20": float((r_ < 20).mean())}
    log(f"tag {tag}: weights block {w[0]:.2f}, " + ", ".join(f"{MULTI[tag][k][0]} {w[k + 1]:.2f}" for k in range(len(zh))) + " | held-out own games top-1 / top-5 / top-20: " + "; ".join(f"{k} {x['top1']:.3f}/{x['top5']:.3f}/{x['top20']:.3f}" for k, x in acc.items()) + " (M82 5e fused: 0.268/0.437/0.588)")
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros(len(idx), np.int64); m = np.zeros(len(idx))
    for i in range(0, len(idx), 2048):
        z, valid = zs(idx[i:i + 2048]); F = w[0] * torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)
        for k in range(len(z)):
            F = F + w[k + 1] * z[k]
        top = F.masked_fill(~valid, -float("inf")).topk(2, dim=1); b[i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()
    del blk; o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
    val = {**base, "f_opp_lab": plab[b[o]], "f_opp_m": m[o], "f_opp_best": players[b[o]], "f_imp_lab": plab[b[t_]], "f_imp_m": m[t_]}
    np.savez_compressed(RV / f"m84_{tag}_val.npz", **val); np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[to]], f_opp_m=m[to], f_opp_best=players[b[to]], f_tgt_lab=plab[b[tt]], f_tgt_m=m[tt], f_tgt_best=players[b[tt]])
    rep = {"weights": w.tolist(), "heldout": acc, "same_top_candidate_as_5e": {"validation": float((val["f_opp_best"] == d82["f5e_opp_best"]).mean()), "test": float((players[b[to]] == t82["f5e_opp_best"]).mean())}}
    ZN = {**base, "opp_lab": val["f_opp_lab"], "opp_m": val["f_opp_m"], "imp_m": val["f_imp_m"], "imp_lab": val["f_imp_lab"]}; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; ok = []
    for seeds in ((0, 1, 2), (6, 7, 8)):
        A, B_ = anchors(ZN, seeds), anchors(ZO, seeds); y, n, pid = (np.concatenate([x[k] for x in A]) for k in (2, 3, 4)); res = {}
        for label, a, c in (("new_vs_base", [x[0] for x in A], [x[1] for x in A]), ("5e_vs_base", [x[0] for x in B_], [x[1] for x in B_]), ("new_vs_5e", [x[0] for x in A], [x[0] for x in B_])):
            dd = paired_diff(np.concatenate(a), np.concatenate(c), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(u, x[2], x[3], sizes=SIZES) - balanced_mean_over_n(v_, x[2], x[3], sizes=SIZES) for u, v_, x in zip(a, c, A)]
            good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025); res[label] = {**dd, "per_fold": pf, "pass": bool(good)}
            log(f"  seeds {seeds} {label:12s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m_}: {x:+.4f}" for m_, x in dd["by_n"].items()) + f"  -> {'PASS' if good else 'no'}")
        rep[f"seeds_{seeds[0]}"] = res; ok.append(res["new_vs_5e"]["pass"])
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False
    rep["cache_ok"] = cache_ok; rep["adopt"] = bool(all(ok) and cache_ok); log(f"PRE-REGISTERED: identifier {tag} passes against the 5e identifier: {'YES' if rep['adopt'] else 'NO'}")
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float)); finalize()


def stage_wgrid():
    """Rung W: the matcher's fusion weight chosen by the anchors' score instead of the held-out identification likelihood. One pass fuses
    every weight of WGRID (M82's matcher, cached block); for validation fold v the weight is the one with the best class-balanced score
    (seeds 0-2) on the other three folds; primary = that cross-fitted choice vs M82's weight, M37 on seeds 0-2 and 6-8."""
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)
    V = np.load(CACHE / "var_ref.npz"); protos = torch.as_tensor(V["protos"][0], device=dev); qvec = V["qvec"][0]; w5e = json.loads((M82 / "report.json").read_text())["weights"]["5e"]["matcher"]; grid = [w5e if g is None else g for g in WGRID]
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros((len(grid), len(idx)), np.int64); m = np.zeros((len(grid), len(idx)))
    for i in range(0, len(idx), 2048):
        ids = idx[i:i + 2048]; ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False
        S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[ids].astype(np.float32), device=dev); nj = torch.as_tensor(net[ids], device=dev).long()
        for j in range(5):
            s = nj == j
            if s.any():
                S[s] = qv[s] @ protos[j].T
        cnt = valid.sum(1, keepdim=True); mu = (S * valid).sum(1, keepdim=True) / cnt; z = (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12); B_ = torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)
        for gi, beta in enumerate(grid):
            top = (B_ + beta * z).masked_fill(~valid, -float("inf")).topk(2, dim=1); b[gi, i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[gi, i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()
    del blk; log(f"fused {len(grid)} weights: " + " ".join(f"{x:.2f}" for x in grid))
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
    Z = [{**base, "opp_lab": plab[b[gi][o]], "opp_m": m[gi][o], "imp_m": m[gi][t_], "imp_lab": plab[b[gi][t_]]} for gi in range(len(grid))]; ref = grid.index(w5e); rep = {"grid": grid}; ok = []; pick = None
    for seeds in ((0, 1, 2), (6, 7, 8)):
        A = [anchors(Z[gi], seeds) for gi in range(len(grid))]; fs = np.array([[balanced_mean_over_n(A[gi][f][0], A[gi][f][2], A[gi][f][3], sizes=SIZES) for f in range(len(FOLDS))] for gi in range(len(grid))])  # [weight, fold]
        if pick is None:  # the choice is made once, on seeds 0-2
            pick = [int(np.argmax(fs[:, [u for u in range(len(FOLDS)) if u != f]].mean(1))) for f in range(len(FOLDS))]; final = int(np.argmax(fs.mean(1)))
            log("  score by weight (mean over folds, seeds 0-2): " + " ".join(f"{grid[gi]:.2f}: {fs[gi].mean():.4f}" for gi in range(len(grid))) + f" | cross-fitted picks per fold {[round(grid[x], 2) for x in pick]}; on all folds {grid[final]:.2f}")
        new = [A[pick[f]][f][0] for f in range(len(FOLDS))]; old = [A[ref][f][0] for f in range(len(FOLDS))]; y, n, pid = (np.concatenate([x[k] for x in A[ref]]) for k in (2, 3, 4))
        dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[pick[f], f] - fs[ref, f]) for f in range(len(FOLDS))]; good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
        log(f"  seeds {seeds} cross-fitted weight vs M82's weight {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")
        rep[f"seeds_{seeds[0]}"] = {"new_vs_5e": {**dd, "per_fold": pf, "pass": bool(good)}, "score_by_weight": fs.mean(1).tolist()}; ok.append(good)
    rep.update(picks=[grid[x] for x in pick], final_weight=grid[final], adopt=bool(all(ok) and grid[final] != w5e)); log(f"PRE-REGISTERED: rung W passes: {'YES' if rep['adopt'] else 'NO'}")
    gi = final; np.savez_compressed(RV / "m84_W_val.npz", **base, f_opp_lab=plab[b[gi][o]], f_opp_m=m[gi][o], f_opp_best=players[b[gi][o]], f_imp_lab=plab[b[gi][t_]], f_imp_m=m[gi][t_])
    np.savez_compressed(RV / "m84_W_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[gi][to]], f_opp_m=m[gi][to], f_opp_best=players[b[gi][to]], f_tgt_lab=plab[b[gi][tt]], f_tgt_m=m[gi][tt], f_tgt_best=players[b[gi][tt]])
    (RV / "m84_W_eval.json").write_text(json.dumps(rep, indent=1, default=float)); finalize()


def stage_cgrid(tag="C"):
    """Rung C: the PR matcher (all games, phase rows, same-level episodes, three seeds averaged) with the fusion weight chosen by the
    anchors' score, cross-fitted over folds on seeds 0-2. Primary: that identifier vs M82's 5e, M37 on seeds 0-2 and 6-8."""
    import torch
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731
    var, members, ct_ = TAGS[tag]; dev = "cuda"; q = np.load(CACHE / "queries.npz"); players, plab, excl, net = q["players"], q["plab"], q["excl"], q["net"]; Q, H, T = int(q["Q"]), int(q["H"]), int(q["T"]); C = len(players)
    V = np.load(CACHE / f"var_{var}.npz"); protos = torch.as_tensor(V["protos"][members], device=dev); qvec = V["qvec"][members]; grid = list(CGRID)
    idx = np.concatenate([np.load(CACHE / f"idx_{k}of4.npy") for k in range(4)]); blk = np.concatenate([np.load(CACHE / f"block_{k}of4.npy") for k in range(4)]); order = np.argsort(idx); idx = idx[order]; b = np.zeros((len(grid), len(idx)), np.int64); m = np.zeros((len(grid), len(idx)))
    for i in range(0, len(idx), 2048):
        ids = idx[i:i + 2048]; ex = torch.as_tensor(excl[ids], device=dev); valid = torch.ones(len(ids), C, device=dev, dtype=torch.bool); r = torch.nonzero(ex >= 0)[:, 0]; valid[r, ex[r]] = False; nj = torch.as_tensor(net[ids], device=dev).long(); z = torch.zeros(len(ids), C, device=dev); cnt = valid.sum(1, keepdim=True)
        for mi in range(len(members)):
            S = torch.zeros(len(ids), C, device=dev); qv = torch.as_tensor(qvec[mi, ids].astype(np.float32), device=dev)
            for j in range(5):
                s = nj == j
                if s.any():
                    S[s] = qv[s] @ protos[mi, j].T
            mu = (S * valid).sum(1, keepdim=True) / cnt; z += (S - mu) / (((S - mu) ** 2 * valid).sum(1, keepdim=True) / cnt).sqrt().clamp(min=1e-12)
        z /= len(members); B_ = torch.as_tensor(blk[order[i:i + 2048]].astype(np.float32), device=dev)
        for gi, beta in enumerate(grid):
            top = (B_ + beta * z).masked_fill(~valid, -float("inf")).topk(2, dim=1); b[gi, i:i + 2048] = top.indices[:, 0].cpu().numpy(); m[gi, i:i + 2048] = (top.values[:, 0] - top.values[:, 1]).cpu().numpy()
    del blk; log(f"fused {len(grid)} weights: " + " ".join(f"{x:.2f}" for x in grid))
    o, t_, to, tt = slice(0, Q), slice(Q, 2 * Q), slice(2 * Q, 2 * Q + T), slice(2 * Q + T, 2 * Q + 2 * T); d82 = dict(np.load(RV / "m82_val.npz")); t82 = dict(np.load(RV / "m82_test.npz")); base = {k: d82[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}
    Z = [{**base, "opp_lab": plab[b[gi][o]], "opp_m": m[gi][o], "imp_m": m[gi][t_], "imp_lab": plab[b[gi][t_]]} for gi in range(len(grid))]; ZO = {**base, "opp_lab": d82["f5e_opp_lab"], "opp_m": d82["f5e_opp_m"], "imp_m": d82["f5e_imp_m"], "imp_lab": d82["f5e_imp_lab"]}; rep = {"grid": grid}; ok = []; pick = None
    for seeds in ((0, 1, 2), (6, 7, 8)):
        A = [anchors(Z[gi], seeds) for gi in range(len(grid))]; O = anchors(ZO, seeds); fs = np.array([[balanced_mean_over_n(A[gi][f][0], A[gi][f][2], A[gi][f][3], sizes=SIZES) for f in range(len(FOLDS))] for gi in range(len(grid))])
        fo = np.array([balanced_mean_over_n(O[f][0], O[f][2], O[f][3], sizes=SIZES) for f in range(len(FOLDS))])
        if pick is None:
            pick = [int(np.argmax(fs[:, [u for u in range(len(FOLDS)) if u != f]].mean(1))) for f in range(len(FOLDS))]; final = int(np.argmax(fs.mean(1)))
            log("  score by weight (mean over folds, seeds 0-2): " + " ".join(f"{grid[gi]:.2f}: {fs[gi].mean():.4f}" for gi in range(len(grid))) + f" | 5e {fo.mean():.4f} | cross-fitted picks per fold {[grid[x] for x in pick]}; on all folds {grid[final]:.2f}")
        new = [A[pick[f]][f][0] for f in range(len(FOLDS))]; old = [O[f][0] for f in range(len(FOLDS))]; y, n, pid = (np.concatenate([x[k] for x in O]) for k in (2, 3, 4))
        dd = paired_diff(np.concatenate(new), np.concatenate(old), y, pid, n, sizes=SIZES, n=5000); pf = [float(fs[pick[f], f] - fo[f]) for f in range(len(FOLDS))]; good = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)
        log(f"  seeds {seeds} C (cross-fitted weight) vs 5e {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + f"  -> {'PASS' if good else 'no'}")
        rep[f"seeds_{seeds[0]}"] = {"new_vs_5e": {**dd, "per_fold": pf, "pass": bool(good)}, "score_by_weight": fs.mean(1).tolist()}; ok.append(good)
    ref = RV / "m84_ref_eval.json"; cache_ok = bool(json.loads(ref.read_text())["cache_ok"]) if ref.exists() else False
    rep.update(picks=[grid[x] for x in pick], final_weight=grid[final], cache_ok=cache_ok, adopt=bool(all(ok) and cache_ok)); log(f"PRE-REGISTERED: rung {tag} passes: {'YES' if rep['adopt'] else 'NO'}")
    gi = final; np.savez_compressed(RV / f"m84_{tag}_val.npz", **base, f_opp_lab=plab[b[gi][o]], f_opp_m=m[gi][o], f_opp_best=players[b[gi][o]], f_imp_lab=plab[b[gi][t_]], f_imp_m=m[gi][t_])
    np.savez_compressed(RV / f"m84_{tag}_test.npz", qid=t82["qid"], hd=t82["hd"], target_black=t82["target_black"], f_opp_lab=plab[b[gi][to]], f_opp_m=m[gi][to], f_opp_best=players[b[gi][to]], f_tgt_lab=plab[b[gi][tt]], f_tgt_m=m[gi][tt], f_tgt_best=players[b[gi][tt]])
    (RV / f"m84_{tag}_eval.json").write_text(json.dumps(rep, indent=1, default=float))
    if ct_ is not None:
        finalize()


def finalize():
    """upload_next/: the candidate of the highest-priority variant that passed, built with the unchanged composition, and a README for the user."""
    UP.mkdir(exist_ok=True); st = {t: json.loads((RV / f"m84_{t}_eval.json").read_text()) for t in PRIORITY if (RV / f"m84_{t}_eval.json").exists()}; passing = [t for t in PRIORITY if st.get(t, {}).get("adopt")]
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]  # noqa: E731
    for t in passing:
        a, b = TAGS[t][2]; f = ROOT / f"submission_rank_ct{a}.csv"
        if not f.exists():
            r = subprocess.run([sys.executable, "-I", "logs/nano4/m78_build_candidate.py", "--source", "file", "--val", str(RV / f"m84_{t}_val.npz"), "--test", str(RV / f"m84_{t}_test.npz"), "--prefix", "f_", "--ref", "ct44", "--out", a, b], capture_output=True, text=True, cwd=ROOT)
            (UP / f"build_{t}.log").write_text(r.stdout + "\n" + r.stderr[-3000:]); print(r.stdout[-2500:], flush=True)
    desc = DESC; L = [f"# 上傳指示（自動產生，{time.strftime('%Y-%m-%d %H:%M')}）", "", "**2026-11-14 更新：建議的最終檔改為 ct104b**（`submission_rank_ct104b.csv`，md5 7e00dc32133c，Public 0.8126304，2026-11-14 上傳，歷來最高，已是線上最後一筆；說明見 `ct104_decode.md`）。備援依序為 ct75u（0.8101304）→ ct75s → ct44。以下是 11/13 的說明，保留備查。", "", "**（11/13）建議的最終檔：ct75u**（`submission_rank_ct75u.csv`，md5 9974e151b93e，Public 0.8101304，2026-11-13 上傳，歷來最高）。**注意：11/13 最後兩次上傳是拆解檔（ct75s_out 0.8092107、ct75s_out_up 0.8076304），線上最後一筆目前不是 ct75u——請再傳一次 ct75u，讀數應為 0.8101304。** ct75u＝ct75s＋五個配方全票通過的 16 題改動。依事先寫下的規則（讀數不低於 ct75s 即建議採用）。備援：ct75s（md5 926f0e7c2153，Public 0.8092107）→ ct44（md5 367ab951442f，Public 0.8067107）。其他讀數：ct73 0.8060501、ct75n 0.8044698。之後若再傳別的檔，最後一筆必須傳回選定的最終檔。", ""]
    if (UP / "ct75s_decode.md").exists():  # M102: the shared-changes file and its decode table (written by m102_split.py, kept across rewrites of this README)
        L += ["## 目前不用傳任何檔", "", "- `submission_rank_ct75s.csv` 已上傳，Public 0.8092107：共同的 8 題淨多對 1 題、多 1 題差一級以內（對 ct44 +1.0 分）；由此推得 ct73 獨有的 4 題少對 2 題、ct75n 獨有的 8 題少對 3 題。29 次上傳的讀數聯立仍然相容。對照表與事先寫下的建議見 `ct75s_decode.md`。", "- 第六版打包（加入與外部玩家一起訓練的認人網路、共五個成員）通過，但選到的配方與第五版相同，產生的檔與 ct75／ct75n 逐位元組相同，沒有新的候選檔。", ""]
    if (UP / "ct104_decode.md").exists():  # M129: the candidates of 2026-11-14 (statistic + look-alike bundle); the note is kept across rewrites of this README
        L += ["## 2026-11-14 的新候選檔：ct104、ct104a、ct104b", "", "- 三個檔都通過事先登錄的交叉驗證檢定（對 ct75u 規則 +0.003 到 +0.005），Public 帳本回測沒有否決；說明、md5 與否決線見 `ct104_decode.md`。要不要上傳、最終用哪一個，由使用者決定。", ""]
    if (UP / "ct75u_decode.md").exists():  # M104: the unanimous-vote candidate
        L += ["- `submission_rank_ct75u.csv` 已上傳，Public 0.8101304：對 ct75s 多 1 題差一級以內（+0.37 分），9 題不同。30 次上傳的讀數聯立仍然相容。交叉驗證對 ct75s 的規則 +0.0046／+0.0036（八折全正）。對照表在 `ct75u_decode.md`。", ""]
    plain = lambda s_: s_.replace("id=SP2y", "全部對局的逐手認人模型（5 個成員：3 seed＋外部玩家版＋對照版）＋分階段比對").replace("id=SP2x", "全部對局的逐手認人模型（3 seed＋外部玩家版）＋分階段比對").replace("id=SP2e", "全部對局的逐手認人模型（3 seed）＋分階段比對").replace("id=S2e", "全部對局的逐手認人模型（3 seed）").replace("id=SP2", "全部對局的逐手認人模型＋分階段比對").replace("id=S2", "全部對局的逐手認人模型").replace("id=SP", "逐手認人模型＋分階段比對").replace("id=S", "逐手認人模型").replace("id=P3", "比對改用全部對局＋分階段描述").replace("enc", "加入整局編碼器的證據").replace("cc", "證據表依等級分別估").replace("temp", "20 局的題目證據降溫").replace("pi0", "修正混合權重").replace("+", "；")  # noqa: E731
    bundles = []  # (json, note) in priority order: the first adopted one with a file is the file to upload
    for jn, note in (("m86g_bundle.json", "第六版（再加一組與外部公開棋譜玩家一起訓練的認人網路）"), ("m86f_bundle.json", "第五版（全部對局的逐手認人模型，3 個 seed 平均）"), ("m86e_bundle.json", "第四版（逐手認人模型用全部對局訓練）"), ("m86d_bundle.json", "第三版（含逐手認人模型，全部把握帶）"), ("m86c_bundle.json", "第二版（全部把握帶）"), ("m86b_bundle.json", "第二版（只用高把握帶）"), ("m86_bundle.json", "第一版")):
        if (RV / jn).exists():
            bj_ = json.loads((RV / jn).read_text()); f_ = ROOT / f"submission_rank_{bj_.get('file', 'ct64')}.csv"
            if jn == "m86g_bundle.json" and not f_.exists():  # v6 chose v5's recipe: its file was byte-identical to ct75 and has been removed
                continue
            bundles.append((bj_, note, f_, bool(bj_.get("adopt") and f_.exists())))
    has_bundle = any(x[3] for x in bundles); first = True
    for bj_, note, f_, ok_ in bundles:
        d = [bj_[k]["bundle_vs_off"] for k in ("seeds_0", "seeds_6")]; up_ = {"ct64": 0.8038092, "ct71": 0.8053895, "ct72": 0.8038092}.get(bj_.get("file")); old_ = bj_.get("file") in ("ct64", "ct70", "ct71", "ct72")  # files already uploaded, with their Public readings
        if ok_:
            shutil.copy(f_, UP / f_.name)
            if first and bj_.get("file") not in ("ct64", "ct70", "ct71", "ct72"):  # standing composition since 2026-11-12: no reversal of ct44's changes
                import csv
                rd = lambda n_: {r_["question_id"]: r_["pred_rank"] for r_ in csv.DictReader(open(ROOT / f"submission_rank_{n_}.csv"))}  # noqa: E731
                c28_, c44_, cn_ = rd("ct28"), rd("ct44"), rd(bj_["file"]); fn_ = ROOT / f"submission_rank_{bj_['file']}n.csv"
                with open(fn_, "w", newline="") as fh:
                    w_ = csv.writer(fh, lineterminator="\n"); w_.writerow(["question_id", "pred_rank"]); w_.writerows([k_, c44_[k_] if c44_[k_] != c28_[k_] else cn_[k_]] for k_ in c44_)
                shutil.copy(fn_, UP / fn_.name); nd_ = sum(1 for k_ in c44_ if (c44_[k_] if c44_[k_] != c28_[k_] else cn_[k_]) != c44_[k_])
                same_ = fn_.read_bytes() == (ROOT / "submission_rank_ct73.csv").read_bytes()
                upn_ = {"ct75n": 0.8044698}.get(fn_.stem.replace("submission_rank_", ""))  # no-reversal files already uploaded
                L += [f"## {'不必上傳（內容與現任最終檔 ct73 完全相同）' if same_ else '已上傳過（Public ' + format(upn_, '.7f') + '，高於 0.8027 的否決線）' if upn_ is not None else '要傳的檔'}：`{fn_.name}`（md5 {md5(fn_)}）", "", f"- 內容：`{f_.name}` 的「不改回」版：ct44 已改過的題目維持 ct44 的答案，其餘用新檔的答案（11/12 起的固定組合方式）。對 ct44 改 {nd_} 題。",
                      (f"- 完整版 `{f_.name}`（md5 {md5(f_)}）已上傳（Public {up_:.7f}）；CV 上它比「不改回」版高約 0.002 到 0.004，Public 上因為改回了 ct44 的改動而偏低。最終檔選定後，最後一筆必須再傳那個檔。" if up_ is not None else f"- 完整版 `{f_.name}`（md5 {md5(f_)}）是 CV 最高的版本，兩者期望差約 0.002；有多的上傳次數可以也傳，但最後一筆請留「不改回」版或現任最終檔。"), ""]
            if first:
                L += [f"## {'完整版（已上傳，Public ' + format(up_, '.7f') + '）' if up_ is not None and not old_ else '目前不用傳。最新通過的打包檔（已上傳過，Public ' + format(up_, '.7f') + '）' if up_ is not None else '完整版' if bj_.get('file') not in ('ct64', 'ct70', 'ct71', 'ct72') else '要傳的檔'}：`{f_.name}`（md5 {md5(f_)}）", "", f"- 內容：ct44 加上一包小改良，{note}（{plain(bj_['picks']['all_folds'])}；設計文件 M86–M90）。", f"- 選與驗分開的 CV 估計：{d[0]['diff']:+.4f} [{d[0]['ci95'][0]:+.4f}, {d[0]['ci95'][1]:+.4f}]／{d[1]['diff']:+.4f} [{d[1]['ci95'][0]:+.4f}, {d[1]['ci95'][1]:+.4f}]；對 ct44 改 {bj_['test']['changed_vs_ct44']} 題（Public {bj_['test']['public']}／Private {bj_['test']['private']}）。",
                      "- 已上傳過，不必再傳。讀數高於 0.8027 的否決線，但低於 ct73（0.8060501）與 ct44（0.8067107）。" if up_ is not None and not old_ else "- 已上傳過，不必再傳；它的「不改回」版就是現任最終檔 ct73。" if up_ is not None else "- 傳一次、記下讀數即可；回來後判定（讀數 ≥ 0.8027 才可能取代現任最終檔）。", ""]; first = False
            else:
                L += [f"- 其他已通過的打包檔（不必傳）：`{f_.name}`（md5 {md5(f_)}，{note}，CV {d[0]['diff']:+.4f}／{d[1]['diff']:+.4f}{'，已上傳過，Public ' + format(up_, '.7f') if up_ is not None else ''}）"]
        else:
            L += [f"- 打包檢定{note}：{'沒過' if not bj_.get('passed_cv') else '建檔檢查沒過'}（{d[0]['diff']:+.4f}／{d[1]['diff']:+.4f}）。"]
    L += [""]
    built = [t for t in passing if (ROOT / f"submission_rank_ct{TAGS[t][2][0]}.csv").exists()]
    if built and has_bundle:
        for u in built:
            a2 = TAGS[u][2][0]; f2 = ROOT / f"submission_rank_ct{a2}.csv"; L += [f"- 單項通過 CV 的比對變體（不必傳，已包含在上面的打包檔裡）：`submission_rank_ct{a2}.csv`（md5 {md5(f2)}，{desc[u]}）"]
        L += [""]
    elif built:
        t = built[0]; a = TAGS[t][2][0]; f = ROOT / f"submission_rank_ct{a}.csv"; shutil.copy(f, UP / f.name); d = [st[t][k]["new_vs_5e"]["diff"] for k in ("seeds_0", "seeds_6")]
        L += [f"## 要傳的檔：`submission_rank_ct{a}.csv`（md5 {md5(f)}）", "", f"- 內容：ct44 的做法，比對成分改成「{desc[t]}」。CV 對 ct44 的認人方法 {d[0]:+.4f}／{d[1]:+.4f}，兩組 seed 都通過。", "- 傳一次即可，把 Public 讀數記下來；不必當場判斷。",
              "- 規則（回來後由 Claude 解碼判定）：讀數 ≥ 0.8027（ct44 − 0.004）才取代 ct44。", "- 最後一筆只在 11/24 截止時才算數；有空的話最後再傳一次 ct44，沒空就不用。", ""]
        for u in built[1:]:
            a2 = TAGS[u][2][0]; f2 = ROOT / f"submission_rank_ct{a2}.csv"; shutil.copy(f2, UP / f2.name); L += [f"- 另一個也通過 CV 的檔（次要，有多的上傳次數再傳）：`submission_rank_ct{a2}.csv`（md5 {md5(f2)}，{desc[u]}）"]
    elif not has_bundle:
        L += ["## 目前不用傳", "", "- 還沒有任何新候選通過 CV 規則。ct44 維持最終檔。", ""]
    mj = RV / "m85_mirror_eval.json"; mir = json.loads(mj.read_text()) if mj.exists() else None; f56 = ROOT / "submission_rank_ct56.csv"
    if mir is not None and mir.get("adopt") and f56.exists():
        shutil.copy(f56, UP / f56.name); d = [mir[k]["both_vs_forward"]["diff"] for k in ("seeds_0", "seeds_6")]
        L += ["", f"## 也要傳的檔：`submission_rank_ct56.csv`（md5 {md5(f56)}）", "", f"- 內容：ct44 再加上「鏡像通道」的證據（測試玩家以對手身分出現在訓練棋譜裡）。CV 在 ct44 的做法之上 {d[0]:+.4f}／{d[1]:+.4f}，兩組 seed 都通過；對 ct44 改 {mir['test']['changed_vs_ct44']} 題（Public {mir['test']['public']}／Private {mir['test']['private']}）。",
              "- 一樣傳一次、記下讀數；≥ 0.8027 才可能取代 ct44，回來後判定。", ""]
    elif mir is not None:
        L += ["", f"- 鏡像通道（M85）：已跑完，{'CV 沒過' if not mir.get('passed_cv') else '測試端安全檢查沒過'}，沒有建檔。"]
    else:
        L += ["", "- 鏡像通道（M85）：尚未完成。"]
    L += ["", "## 各變體的狀態", "", "| 變體 | 內容 | 狀態 | 對 ct44 認人方法的錨點差（seed 0–2／6–8） |", "|---|---|---|---|"]
    for t in ("A", "R", "B", "B3", "P", "P3", "W", "C", "S", "SP", "S2", "SP2", "S2e", "SP2e"):
        if t in st:
            d = [st[t][k]["new_vs_5e"] for k in ("seeds_0", "seeds_6")]; L.append(f"| {t} | {desc[t]} | {'通過' if st[t]['adopt'] else '沒過'} | {d[0]['diff']:+.4f} [{d[0]['ci95'][0]:+.4f}, {d[0]['ci95'][1]:+.4f}]／{d[1]['diff']:+.4f} [{d[1]['ci95'][0]:+.4f}, {d[1]['ci95'][1]:+.4f}] |")
        else:
            L.append(f"| {t} | {desc[t]} | 尚未完成 | — |")
    L += ["", "採用順序事先固定為 SP2e > S2e > SP2 > S2 > SP > S > C > P3 > P > B3 > B > R > A > W（設計文件 M84、M88、M90）。下載：`sftp -P 2222 ...:BK_AICUP_2026-1/upload_next/*`（一次 OTP）。"]
    (UP / "README.md").write_text("\n".join(L) + "\n"); shutil.copy(ROOT / "submission_rank_ct44.csv", UP / "submission_rank_ct44.csv"); print("upload_next/README.md written; passing: " + (", ".join(passing) or "none"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True, choices=("embed", "eval", "finalize", "wgrid", "cgrid")); ap.add_argument("--var", default="A"); ap.add_argument("--tag", default="A"); a_ = ap.parse_args()
    if a_.stage == "embed":
        stage_embed(a_.var)
    elif a_.stage == "eval":
        stage_eval_multi(a_.tag) if a_.tag in MULTI else stage_eval(a_.tag)
    elif a_.stage == "wgrid":
        stage_wgrid()
    elif a_.stage == "cgrid":
        stage_cgrid(a_.tag if a_.tag in ("C", "Cp") else "C")
    else:
        finalize()
