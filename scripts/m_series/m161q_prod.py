#!/usr/bin/env python
"""Section 4M M161 production, second recipe (2026-11-15): by-games sampling + 16 games per player + the wider network + learned
class centres (design doc M161, "final recipe").

As m161p_prod.py (players drawn with probability proportional to games ** gamma, m120's cross-fit scheme and file layout), with
  the episode shape and the network size as options (--ep 128 16: only players with >= 16 games are trained; --size 512 2048),
  learned centres for the players the network trains on (--centres 1): after `warm_b` steps the centres start from the
  prototypes of the current network and are trained with plain SGD (lr 0.1 x the schedule, momentum 0.9) through one more loss,
  every game against the centres of all trained players (weight ramped over `ramp_b` steps); pilot M161b-h,
  outputs: out_<net>_s<seed>.npz with protos (prototypes, as before), centres (unit vectors, zero rows for the players without
  one) and has_centre; train_<net>_s<seed>.json; a slim checkpoint with best_state (and best_centres).
The best state is the one with the highest row-weighted held-out top-1 of the PROTOTYPE matching on the players trained on.

  sbatch -p 16gpus --gres=gpu:8 -c 96 ... run.sbatch python -I scripts/m_series/m161q_prod.py --jobs 1:30 2:30 3:30 4:30 all:30 1:31 2:31 3:31 --steps 40000
"""
import argparse  # 解析命令列參數
import json  # 讀寫訓練紀錄
import math  # 餘弦學習率排程
import os  # 分叉子程序、環境變數與檔案替換
import sys  # 設定模組搜尋路徑與結束碼
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m88_seqid as base  # noqa: E402  M88 的資料載入、序列模型與常數
from m112_idaug import sym_table  # noqa: E402  八種盤面對稱的座標對照表

OUT, CK = ROOT / "results/m161q", Path("/work/u2507115/m161q_ckpt")  # 輸出資料夾與檢查點資料夾
BANDS = [(1, 30), (30, 100), (100, 300), (300, 1000), (1000, 10 ** 9)]  # 依玩家局數分帶報告 top-1


def train_one(v, seed, G, Tt, steps, log, gamma=1.0, centres=True, mask=0.3, warm_b=3000, ramp_b=2000, lr_c=0.1):  # 訓練一個網路（含可學中心）並寫出輸出
    import torch  # 在子程序內才匯入 torch
    F = torch.nn.functional; OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}_s{seed}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS  # 建立資料夾；每個 episode 的玩家數與每人局數、輸出維度、網路清單
    cfg = dict(gamma=gamma, centres=bool(centres), mask=mask, warm_b=warm_b, ramp_b=ramp_b, lr_c=lr_c, episode=[P_EP, M_EP], min_games=base.MIN_GAMES, width=base.D, ffn=base.FFN, layers=base.LAYERS)  # 這次訓練的設定（含 episode 形狀與網路大小）
    log(f"net {v} seed {seed} {cfg}: {steps} steps; GPU {torch.cuda.get_device_name(0)}")  # 記錄設定與 GPU
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))  # 查詢表、候選玩家與留出局
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}  # 每局的擁有者、是否留出、局鍵索引
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W_ = int(cnt.max()); all_rows = np.flatnonzero(use)  # 訓練用的局（排除留出局與折 v 的玩家）、每人局數與可訓練的玩家
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W_), -1, np.int64); tab[o, pos_] = idx  # 每位玩家的訓練局列號表，供抽樣
    p_pl = cnt[tp].astype(np.float64) ** float(gamma); p_pl /= p_pl.sum(); trained = np.zeros(C, bool); trained[tp] = True  # 抽人機率；標記有訓練（因而有中心）的玩家
    log(f"players trained {len(tp)} (>= {base.MIN_GAMES} games; they hold {float(cnt[tp].sum() / max(cnt.sum(), 1)):.3f} of the rows), games {int(use.sum())}; classes with a prototype {int((cnt > 0).sum())}; share of an episode's games from players with >= 300 games {float(p_pl[cnt[tp] >= 300].sum()):.3f}")  # 回報訓練規模、所占列數與局數多的玩家比例
    rng = np.random.default_rng(1610 + v + 1000 * seed); rng2 = np.random.default_rng(161000 + 10 * (v + 1) + seed); torch.manual_seed(1610 + v + 1000 * seed); net = base.build_model().to(dev)  # 依網路與種子固定亂數，建立序列模型
    sample = G["GL"][rng.choice(all_rows, min(4000, len(all_rows)), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat  # 以抽樣的局估計輸入的平均與標準差
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); fac = lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)); sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)  # noqa: E731  AdamW 加上暖身與餘弦衰減的學習率
    best = {"top1w": -1.0, "top1": -1.0, "step": -1}; best_state = None; best_wc = None; hist = []; tb = lambda a: torch.as_tensor(a, device=dev); SYM = sym_table()  # noqa: E731  最佳紀錄（含最佳中心）、轉成 GPU 張量的函式與對稱表
    trained_t = tb(trained); Wc = opt_c = None; ck = CK / f"ckpt_{tag}.pt"; start = 0  # 可學中心的狀態與檢查點路徑
    if ck.exists():  # 有檢查點時接續
        st = torch.load(ck, map_location=dev, weights_only=False)  # 讀入檢查點
        if "opt" in st:  # 訓練途中的完整檢查點
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", [])  # 還原網路、最佳化器、排程、最佳中心與紀錄
            if st.get("Wc") is not None:  # 有可學中心時
                Wc = torch.nn.Parameter(st["Wc"].to(dev)); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9); opt_c.load_state_dict(st["opt_c"])  # 還原中心與它的 SGD
            log(f"resumed at step {start}, best so far {best}")  # 回報接續位置
        else:  # 已完成訓練的精簡檢查點：只重寫輸出
            start = steps; best = st["best"]; best_state = st["best_state"]; best_wc = st.get("best_wc"); hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")  # 跳過訓練，直接用最佳狀態與最佳中心

    def embed(src, rows, subj):  # 對指定的局與一方算出身分向量
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)  # 評估模式與輸出陣列
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不算梯度，以 bf16 推論
            for i in range(0, len(rows), 1024):  # 每批 1024 局
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()  # 送入網路取得向量
        net.train(); return out  # 切回訓練模式並回傳

    def train_protos():  # 以目前的網路算出全體訓練玩家的單位原型
        E = embed(G, all_rows, G["SUBJ"][all_rows]); S = np.zeros((C, D_OUT)); np.add.at(S, owner[all_rows], E); S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-12)  # 每位玩家訓練局向量的和，正規化成單位向量
        return tb(S.astype(np.float32))  # 回傳 GPU 上的原型

    def unit_centres(W):  # 中心轉成單位向量
        """Unit centre vectors [C, D_OUT] (numpy), zero rows for the players without a centre."""
        Wn = F.normalize(W.detach().float(), dim=-1).cpu().numpy().astype(np.float32); Wn[~trained] = 0.0  # 正規化後把沒有訓練的玩家清為 0
        return Wn  # 回傳中心

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)  # 留出本人局在訓練局陣列中的列號
    if not len(hk):  # 試跑時可能沒有留出局
        hk = np.flatnonzero(~is_held)[:8]  # 改用前 8 局代替
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8); band = np.array([next((k for k, (lo, hi) in enumerate(BANDS) if lo <= n < hi), 0) for n in cnt[h_truth]]); share = np.array([cnt[(cnt >= lo) & (cnt < hi)].sum() for lo, hi in BANDS]) / max(cnt.sum(), 1)  # 留出局的真實玩家、是否訓練過、局數帶與各帶的列數占比

    def summary(S, sel):  # 彙整一組留出局的認人準度
        r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); t1 = np.array([float((r[sel & (band == k)] == 0).mean()) if (sel & (band == k)).any() else np.nan for k in range(len(BANDS))]); ok = np.isfinite(t1)  # 真實玩家的名次與各局數帶的 top-1
        return {"n": int(sel.sum()), "top1": float((r[sel] == 0).mean()), "top5": float((r[sel] < 5).mean()), "top20": float((r[sel] < 20).mean()), "by_band": [None if not np.isfinite(x) else float(x) for x in t1], "top1w": float((share[ok] * t1[ok]).sum() / max(share[ok].sum(), 1e-12))}  # top-1／5／20、各帶與列數加權的 top-1

    def line(x):  # 把一組評估結果排成一行文字
        return f"top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}; by games " + "  ".join(f"{lo}+ {('-' if b is None else format(b, '.3f'))}" for (lo, hi), b in zip(BANDS, x["by_band"])) + f"; weighted by rows {x['top1w']:.3f}"  # top-1／5／20、各局數帶與列數加權的 top-1

    def evaluate(step, W):  # 評估：留出本人局對原型，有中心時也對中心
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 以非留出局算出每位玩家的單位原型
        Eh = embed(G, hk, G["SUBJ"][hk]); S = Eh @ Pm.T; res = {"step": step}  # 留出局的向量與對原型的相似度
        for nm, sel in (("seen", seen), ("unseen", ~seen)):  # 訓練過與沒訓練過的玩家分開報告
            if sel.any():  # 有這類留出局時
                res[nm] = summary(S, sel)  # 存入彙整
        log(f"  step {step}: held-out own games, players trained on ({res['seen']['n']}), prototypes: " + line(res["seen"]) + (f"; unseen players ({res['unseen']['n']}) top-1 {res['unseen']['top1']:.3f}" if "unseen" in res else ""))  # 回報原型比對的結果
        if W is not None:  # 有中心時
            Sc = Eh @ unit_centres(W).T; Sc[:, ~trained] = -2.0; res["centres"] = summary(Sc, seen); log(f"  step {step}: held-out own games, players trained on ({res['centres']['n']}), learned centres: " + line(res["centres"]))  # 以中心比對（排除沒有中心的玩家）並另外回報
        return res, Pm  # 回傳評估結果與原型

    net.train(); ar = np.arange(P_EP); tl = ta = tbk = 0.0; nb = 0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}  # 進入訓練：統計量歸零並排定評估的步數
    for step in range(start, steps):  # 主訓練迴圈
        if centres and step >= warm_b and Wc is None:  # 暖身結束時建立可學中心
            t_r = time.time(); Wc = torch.nn.Parameter(train_protos().clone()); opt_c = torch.optim.SGD([Wc], lr=lr_c, momentum=0.9)  # 以當下的原型初始化中心，用 SGD 更新
            log(f"  step {step}: centres started from the prototypes of {int(trained.sum())} players ({time.time() - t_r:.0f} s)"); t_last += time.time() - t_r  # 回報耗時，且不計入每步速度
        pl = rng.choice(tp, P_EP, replace=False, p=p_pl); u = rng.random((P_EP, W_)); u[np.arange(W_)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); cls_t = np.repeat(pl, M_EP)  # 依抽人機率選玩家，每位隨機取 M_EP 局
        pos = G["POS"][rows].astype(np.int64); pos = SYM[int(rng2.integers(0, 8))][pos]; gl = tb(G["GL"][rows])  # 落點座標隨機套用八種對稱之一
        if mask > 0:  # 遮手擴增
            hit = torch.rand(len(rows), base.T_MAX, device=dev) < torch.rand(len(rows), 1, device=dev) * mask; gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 每局隨機比例的手換成平均向量
        with torch.autocast("cuda", dtype=torch.bfloat16):  # 以 bf16 前向
            z = net(gl, tb(pos), tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))  # 每局本人方的單位向量
        z = z.float(); ze = z.reshape(P_EP, M_EP, D_OUT); tot = ze.sum(1); sc = net.log_scale.exp(); logits = sc * z @ F.normalize(tot, dim=-1).T  # 每局對 episode 內各玩家原型的相似度
        own = sc * (ze * F.normalize(tot[:, None] - ze, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None]); loss = F.cross_entropy(logits, tgt); acc = float((logits.argmax(1) == tgt).float().mean())  # 自己的原型排除本局（留一）後算 episode 內的交叉熵
        if Wc is not None:  # 中心已啟用
            w = min(1.0, (step - warm_b) / max(ramp_b, 1)); ct = tb(cls_t); cos = (z @ F.normalize(Wc, dim=-1).T).masked_fill(~trained_t[None], -1e4); loss = loss + w * F.cross_entropy(sc * cos, ct); tbk += float((cos.argmax(1) == ct).float().mean()); nb += 1  # 每一局對全部訓練玩家中心的交叉熵（權重逐步升到 1）
        opt.zero_grad(set_to_none=True)  # 清除網路的梯度
        if opt_c is not None:  # 有可學中心時
            opt_c.zero_grad(set_to_none=True)  # 也清除中心的梯度
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()  # 反向傳播、梯度裁剪、更新與學習率排程
        if opt_c is not None:  # 有可學中心時
            for g_ in opt_c.param_groups:  # 中心的學習率
                g_["lr"] = lr_c * fac(step)  # lr_c 乘上與網路相同的排程
            opt_c.step()  # 更新中心
        tl += float(loss.detach()); ta += acc  # 累計損失與批內正確率
        if (step + 1) % 500 == 0:  # 每 500 步回報一次
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f}" + (f", top-1 against all centres {tbk / max(nb, 1):.3f} (weight {min(1.0, (step - warm_b) / max(ramp_b, 1)):.2f})" if nb else "") + f", scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = tbk = 0.0; nb = 0; t_last = time.time()  # 回報平均損失、正確率與速度後歸零
        if (step + 1) in ev_steps:  # 評估步
            res, _ = evaluate(step + 1, Wc); hist.append(res)  # 評估並記錄
            if res["seen"]["top1w"] > best["top1w"]:  # 原型比對的列數加權 top-1 創新高
                best = {"top1w": res["seen"]["top1w"], "top1": res["seen"]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}; best_wc = None if Wc is None else Wc.detach().cpu().clone()  # 更新最佳紀錄並保存權重與中心
                if "centres" in res:  # 有中心的評估時
                    best["centres_top1w"] = res["centres"]["top1w"]; best["centres_top1"] = res["centres"]["top1"]  # 一併記下中心的準度
            tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state, "best_wc": best_wc, "hist": hist, "Wc": None if Wc is None else Wc.detach().cpu(), "opt_c": None if opt_c is None else opt_c.state_dict()}, tmp); os.replace(tmp, ck); t_last = time.time()  # 寫入完整檢查點（先寫暫存檔再替換）
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best row-weighted held-out top-1 {best['top1w']:.3f} (unweighted {best['top1']:.3f}) at step {best['step']}")  # 載回最佳狀態
    res, Pm = evaluate(best["step"], None if best_wc is None else best_wc.to(dev)); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)  # 以最佳狀態重算原型；找出由這個網路負責的查詢
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])  # 訓練資料的查詢：對應的局與要辨識的一方（對手側取另一方）
    loc = {g_: i for i, g_ in enumerate(mine)}  # 查詢列號到輸出位置的對照
    if len(rows):  # 有訓練資料的查詢時
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)  # 算出查詢向量
    te = mine[q["qtest"][mine] >= 0]  # 測試局的查詢
    if len(te):  # 有測試局的查詢時
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)  # 算出測試局的查詢向量
    extra = {} if best_wc is None else {"centres": unit_centres(best_wc.to(dev)), "has_centre": trained}  # 有中心時一併輸出中心與遮罩
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv, **extra)  # 輸出原型、查詢向量與中心（m120 的格式）
    (OUT / f"train_{tag}.json").write_text(json.dumps({"config": cfg, "best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "bands": BANDS, "share_rows": share.tolist()}, indent=1))  # 寫出訓練紀錄
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "best_wc": best_wc, "hist": hist, "steps": steps, "config": cfg}, tmp); os.replace(tmp, ck)  # 檢查點精簡成只含最佳狀態與最佳中心
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game; centres {'saved' if extra else 'none'}); checkpoint slimmed to the best state")  # 回報輸出完成


def main():  # 進入點：每張 GPU 分叉一個訓練子程序
    global OUT, CK  # 先導試驗與試跑時要改輸出位置
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 命令列參數
    ap.add_argument("--jobs", nargs="+", required=True, help="one trainer per GPU, NET:SEED with NET in 1 2 3 4 all (e.g. 1:30 all:30)")  # 每張 GPU 一個訓練：網路:種子
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--centres", type=int, default=1); ap.add_argument("--mask", type=float, default=0.3)  # 步數、抽人指數、是否學中心、遮手比例
    ap.add_argument("--ep", type=int, nargs=2, default=[128, 16]); ap.add_argument("--size", type=int, nargs=2, default=[512, 2048]); ap.add_argument("--layers", type=int, default=8)  # episode 形狀（玩家數 x 每人局數）、寬度與前饋層寬度、層數
    ap.add_argument("--warm-b", type=int, default=3000); ap.add_argument("--ramp-b", type=int, default=2000); ap.add_argument("--out", default=None, help="sub-folder of results/m161q and of the checkpoint folder (pilots)"); ap.add_argument("--smoke", action="store_true")  # 中心損失的暖身步數與權重爬升步數、先導試驗的子資料夾、試跑
    a_ = ap.parse_args()  # 解析參數
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, "S7"; base.P_EP, base.M_EP = a_.ep; base.MIN_GAMES = max(base.MIN_GAMES, a_.ep[1]); base.D, base.FFN = a_.size; base.LAYERS = a_.layers  # 設定 m88 的載入器與模型：含 M90 的局、不加外部帳號、變體名 S7；套用 episode 形狀、最少局數（不少於每人局數）與網路大小
    jobs = [(-1 if x.split(":")[0] == "all" else int(x.split(":")[0]), int(x.split(":")[1])) for x in a_.jobs]; assert len(set(jobs)) == len(jobs) and all(v in base.NETS and s > 0 for v, s in jobs), a_.jobs  # 解析「網路:種子」清單並檢查
    kw = dict(gamma=a_.gamma, centres=bool(a_.centres), mask=a_.mask, warm_b=a_.warm_b, ramp_b=a_.ramp_b)  # 傳給訓練函式的超參數
    if a_.out:  # 先導試驗用的子資料夾
        OUT, CK = OUT / a_.out, CK / a_.out  # 輸出與檢查點都放進子資料夾
    if a_.smoke:  # 試跑絕不碰正式輸出
        OUT, CK = ROOT / "results/m161q_smoke", Path("/work/u2507115/m161q_ckpt_smoke"); base.EVAL_EVERY = 40; kw.update(warm_b=20, ramp_b=20)  # 改用試跑的資料夾並縮短各個間隔
    base.OUT = OUT  # 讓 m88 也使用同一個輸出位置
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(jobs), f"{len(jobs)} trainers need as many GPUs, got {gpus}"  # 每個訓練要一張 GPU
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 80 if a_.smoke else a_.steps  # 工作編號（用於紀錄檔名）與步數
    log(f"recipe: episode {base.P_EP} x {base.M_EP}, players with >= {base.MIN_GAMES} games, width {base.D} / feed-forward {base.FFN}, {base.LAYERS} layers, {kw}; outputs {OUT}")  # 回報這次的配方
    G, Tt = base.load_games(a_.smoke, log); kids = {}  # 分叉前先載入全部訓練局與測試局，子程序共用
    for k, (v, seed) in enumerate(jobs):  # 每個訓練分叉一個子程序
        try:  # 嘗試分叉
            pid = os.fork()  # 分叉
        except OSError as e:  # 分叉失敗
            log(f"could not start net {v} seed {seed}: {e}"); break  # 回報並停止再分叉
        if pid == 0:  # 子程序
            code = 0  # 結束碼
            try:  # 執行訓練
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])  # 只看得到分配給它的 GPU
                out = open(ROOT / f"logs/nano4/m161q{'_' + a_.out if a_.out else ''}_net{v}_s{seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out  # 輸出導向這個訓練自己的紀錄檔
                t1 = time.time(); train_one(v, seed, G, Tt, steps, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), **kw)  # 訓練這個網路
            except BaseException:  # noqa: BLE001  任何錯誤都記錄下來
                import traceback  # 印出錯誤堆疊用
                traceback.print_exc(); code = 1  # 印出錯誤並標記失敗
            finally:  # 不論成敗
                sys.stdout.flush(); os._exit(code)  # 子程序直接結束，不執行父程序的後續程式
        kids[pid] = (v, seed); log(f"started net {v} seed {seed} on GPU {gpus[k]} (pid {pid})")  # 父程序記下子程序
    bad = 0  # 失敗的子程序數
    while kids:  # 等待全部子程序結束
        pid, status = os.wait(); v, seed = kids.pop(pid); bad += status != 0; log(f"net {v} seed {seed} finished with status {status}")  # 收回一個子程序並記錄狀態
    for v, seed in jobs:  # 彙整各訓練的結果
        f = OUT / f"train_{v}_s{seed}.json"  # 訓練紀錄檔
        if f.exists():  # 有寫出時
            d = json.load(open(f)); print(f"  net {v} seed {seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: {h['seen']['top1w']:.3f}" + (f" / centres {h['centres']['top1w']:.3f}" if "centres" in h else "") for h in d["history"]))  # 印出最佳紀錄與各評估步原型／中心的加權 top-1
    sys.exit(1 if bad else 0)  # 有子程序失敗就以 1 結束


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
