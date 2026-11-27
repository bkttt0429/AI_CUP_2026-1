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
import argparse  # 解析命令列參數
import json  # 讀取留出鍵、寫出訓練紀錄
import math  # 餘弦學習率排程
import os  # fork 子行程、設定 GPU 與重導輸出
import sys  # 調整模組搜尋路徑與標準輸出
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m88_seqid as base  # noqa: E402  載入器、模型與常數（torch 在該模組中延後匯入）
from m112_idaug import sym_table  # noqa: E402  M112 的盤面對稱對照表

OUT, VAR, CK = ROOT / "results/m120", "S4", Path("/work/u2507115/m120_ckpt")  # 輸出資料夾、變體名稱、檢查點資料夾


def train_one(v, seed, G, Tt, steps, sym, log, drop_open=0.0, mask=0.0):  # 訓練一個 (網路, 種子) 並寫出輸出
    """m88_seqid.stage_train for network v and one seed, with a random board symmetry per episode when sym is set and, with drop_open,
    the opening of that share of an episode's games hidden (first K plies, K uniform in 20..100: trunk vectors -> the mean, played
    points -> the padding point; the other session's --drop-open, its METHOD_NOTES 2.60). With mask, every game draws a share in
    0..mask and that share of its plies get the mean trunk vector (played points kept; the other session's --mask)."""
    import torch  # 子行程內才匯入 torch
    OUT.mkdir(parents=True, exist_ok=True); CK.mkdir(parents=True, exist_ok=True); dev = "cuda"; tag = f"{v}{f'_s{seed}' if seed else ''}"; P_EP, M_EP, D_OUT, NETS = base.P_EP, base.M_EP, base.D_OUT, base.NETS  # 建立資料夾；輸出標籤 = 網路編號（加種子）
    log(f"net {v} seed {seed}: {steps} steps, symmetry per episode {bool(sym)}, opening hidden in {drop_open:g} of the games, ply mask up to {mask:g}; GPU {torch.cuda.get_device_name(0)}")  # 記錄訓練設定
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))  # 訓練玩家清單與 M82 留出的對局鍵
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}  # 每局的玩家、是否留出與列號
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W = int(cnt.max())  # 交叉擬合：不用折 v 的玩家；夠局數的玩家
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W), -1, np.int64); tab[o, pos_] = idx  # 建表：每位玩家的對局列號
    rng = np.random.default_rng(880 + v + 1000 * seed); rng2 = np.random.default_rng(120000 + 10 * (v + 1) + seed); torch.manual_seed(880 + v + 1000 * seed); net = base.build_model().to(dev)  # 固定亂數並建立模型
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32)  # 抽 4,000 局估計標準化統計量
    flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat  # 略過填充位置後寫入平均與標準差
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=base.WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))  # 最佳化器與暖身後餘弦衰減的學習率
    ck = CK / f"ckpt_{tag}.pt"; start = 0; best = {"top1": -1.0, "step": -1}; best_state = None; hist = []  # 檢查點路徑與訓練狀態
    if ck.exists():  # 有檢查點時續跑
        st = torch.load(ck, map_location=dev, weights_only=False)  # 讀入檢查點
        if "opt" in st:  # 訓練中途的完整檢查點
            net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["step"]; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", []); log(f"resumed at step {start}, best so far {best}")  # 還原模型、最佳化器、排程與紀錄
        else:  # 已完成訓練的精簡檢查點：只重寫輸出
            start = steps; best = st["best"]; best_state = st["best_state"]; hist = st.get("hist", []); log(f"finished checkpoint found, best {best}")  # 跳過訓練，直接用最佳狀態
    tb = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731  陣列轉成 GPU 張量
    SYM = sym_table()  # 八種盤面對稱的對照表

    def embed(src, rows, subj):  # 以評估模式計算一批對局的向量
        """Unit vectors of games `rows` of `src` (G or the test dict) with subject colours `subj` (the games as stored: no symmetry)."""
        net.eval(); out = np.zeros((len(rows), D_OUT), np.float32)  # 切換到評估模式
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不記錄梯度、bf16 計算
            for i in range(0, len(rows), 1024):  # 每批 1024 局
                r = rows[i:i + 1024]; out[i:i + 1024] = net(tb(src["GL"][r]), tb(src["POS"][r].astype(np.int64)), tb(src["MOV"][r].astype(np.int64)), tb(src["LEN"][r].astype(np.int64)), tb(subj[i:i + 1024].astype(np.int64))).float().cpu().numpy()  # 該批對局的向量
        net.train(); return out  # 切回訓練模式並回傳

    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64)  # 留出的本人對局（查詢檔中的 H 段）
    if not len(hk):  # 冒煙測試的子集可能沒有任何留出對局：用幾個假查詢讓程式路徑照常執行
        hk = np.flatnonzero(~is_held)[:8]  # 取前 8 個非留出對局充數
    h_truth = owner[hk]; seen = (G["fold"][hk] != v) | (len(hk) <= 8)  # 正確玩家；本網路訓練過其玩家的留出局

    def evaluate(step):  # 留出對局的認人評估
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(G, rows, G["SUBJ"][rows]); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12)  # 每位玩家的原型 = 非留出對局向量和的單位向量
        S = embed(G, hk, G["SUBJ"][hk]) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res = {"step": step}  # 留出對局對各原型的相似度與正確玩家的名次
        for name, s in (("seen", seen), ("unseen", ~seen)):  # 訓練過與沒訓練過其玩家的兩組
            if s.any():  # 該組有對局時
                res[name] = {"n": int(s.sum()), "top1": float((r[s] == 0).mean()), "top5": float((r[s] < 5).mean()), "top20": float((r[s] < 20).mean())}  # 該組的 top-1/5/20
        log(f"  step {step}: held-out own games " + "; ".join(f"{k} ({x['n']}) top-1 / top-5 / top-20 {x['top1']:.3f} / {x['top5']:.3f} / {x['top20']:.3f}" for k, x in res.items() if k != "step"))  # 記錄評估結果
        return res, Pm  # 回傳結果與原型

    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; t_last = time.time(); ev_steps = set(range(base.EVAL_EVERY, steps + 1, base.EVAL_EVERY)) | {steps}  # 訓練狀態與評估的步數
    for step in range(start, steps):  # 訓練迴圈（續跑時從中途開始）
        pl = rng.choice(tp, P_EP, replace=False); u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1)  # 隨機抽玩家，每人隨機抽 M_EP 局
        pos = G["POS"][rows].astype(np.int64)  # 這批對局的落點
        if sym:  # 對稱擴增
            pos = SYM[int(rng2.integers(0, 8))][pos]  # 整個 episode 套用同一個隨機對稱
        gl, pos_t = tb(G["GL"][rows]), tb(pos)  # 主幹向量與落點張量
        if mask > 0:  # 隨機遮手
            hit = torch.rand(len(rows), base.T_MAX, device=dev) < torch.rand(len(rows), 1, device=dev) * mask  # 每局抽一個比例，選出要遮的手
            gl = torch.where(hit[..., None], net.mu.to(gl.dtype)[None, None], gl)  # 被選中的手改用平均主幹向量
        if drop_open > 0:  # 遮住開局
            cut = torch.where(torch.rand(len(rows), device=dev) < drop_open, torch.randint(20, 101, (len(rows),), device=dev), torch.zeros(len(rows), dtype=torch.long, device=dev))  # 部分對局抽 K（20-100），其餘為 0
            hide = torch.arange(base.T_MAX, device=dev)[None] < cut[:, None]  # 前 K 手的遮罩
            gl = torch.where(hide[..., None], net.mu.to(gl.dtype)[None, None], gl); pos_t = torch.where(hide, torch.full_like(pos_t, 362), pos_t)  # 主幹向量改為平均、落點改為填充點
        with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 混合精度
            z = net(gl, pos_t, tb(G["MOV"][rows].astype(np.int64)), tb(G["LEN"][rows].astype(np.int64)), tb(G["SUBJ"][rows].astype(np.int64)))  # 每局的單位向量
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T  # 每局對各玩家原型的相似度
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])  # 對自己的原型時扣掉這一局本身
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()  # 交叉熵、梯度裁剪與更新
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())  # 累計損失與 episode 內準確率
        if (step + 1) % 500 == 0:  # 每 500 步記錄一次
            log(f"  step {step + 1}/{steps}: loss {tl / 500:.4f}, episode top-1 {ta / 500:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / 500:.2f} s/step"); tl = ta = 0.0; t_last = time.time()  # 印出平均損失、準確率與速度後歸零
        if (step + 1) in ev_steps:  # 到評估步數
            res, _ = evaluate(step + 1); hist.append(res); key_ = "seen" if "seen" in res else "unseen"  # 評估並以「訓練過的玩家」那組為準
            if res[key_]["top1"] > best["top1"]:  # 刷新最佳成績
                best = {"top1": res[key_]["top1"], "step": step + 1}; best_state = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}  # 保存最佳狀態的副本
            tmp = ck.with_name(ck.name + ".tmp"); torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step + 1, "best": best, "best_state": best_state, "hist": hist}, tmp); os.replace(tmp, ck); t_last = time.time()  # 先寫暫存檔再替換，避免中斷時檢查點損毀
    net.load_state_dict(best_state); log(f"net {v} seed {seed}: best held-out top-1 {best['top1']:.3f} at step {best['step']}")  # 改用最佳狀態
    res, Pm = evaluate(best["step"]); j = NETS.index(v); mine = np.flatnonzero(q["net"] == j); qv = np.zeros((len(mine), D_OUT), np.float32)  # 最佳狀態的原型；本網路負責的查詢
    tr = mine[q["qtest"][mine] < 0]; ok = np.array([k in gidx for k in q["qkey"][tr]]); rows = np.array([gidx[k] for k in q["qkey"][tr][ok]], dtype=np.int64); subj = np.where(q["side"][tr][ok] == 1, 1 - G["SUBJ"][rows], G["SUBJ"][rows])  # 非測試查詢（驗證題兩側與留出局）：對局列號與主角顏色（對手側查詢換另一方）
    loc = {g_: i for i, g_ in enumerate(mine)}  # 查詢編號 -> qv 的列
    if len(rows):  # 有非測試查詢時
        qv[[loc[g_] for g_ in tr[ok]]] = embed(G, rows, subj)  # 計算非測試查詢的向量
    te = mine[q["qtest"][mine] >= 0]  # 測試端查詢
    if len(te):  # 有測試端查詢時
        rows = q["qtest"][te].astype(np.int64); subj = np.where(q["side"][te] == 1, 1 - Tt["SUBJ"][rows], Tt["SUBJ"][rows]); qv[[loc[g_] for g_ in te]] = embed(Tt, rows, subj)  # 以測試對局計算查詢向量
    np.savez(OUT / f"out_{tag}.npz", protos=Pm.astype(np.float32), qidx=mine, qvec=qv)  # 寫出原型與查詢向量
    (OUT / f"train_{tag}.json").write_text(json.dumps({"best": best, "history": hist, "final": res, "queries": int(len(mine)), "queries_without_game": int((~ok).sum()), "players_trained": int(len(tp)), "games": int(use.sum()), "steps": steps, "sym": bool(sym), "drop_open": float(drop_open), "mask": float(mask)}, indent=1))  # 寫出訓練紀錄
    tmp = ck.with_name(ck.name + ".tmp"); torch.save({"best": best, "best_state": best_state, "hist": hist, "steps": steps, "sym": bool(sym), "drop_open": float(drop_open), "mask": float(mask)}, tmp); os.replace(tmp, ck)  # 檢查點精簡為只留最佳狀態
    log(f"net {v} seed {seed}: outputs written ({len(mine)} queries, {int((~ok).sum())} without a game); checkpoint slimmed to the best state")  # 記錄完成


def main():  # 命令列進入點
    global OUT, CK  # 冒煙測試時會改寫輸出位置
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 說明取模組說明的第一行
    ap.add_argument("--jobs", nargs="+", default=[], help="network[:seed[:o]] per GPU, e.g. 1 2 3 4 all all:1 1:2:o (all = the all-players network -1: a token that starts with '-' and is not a plain number is taken for an option; third field: o = the opening hidden in 30%% of the games, m = random plies' trunk vectors masked, om = both)")  # 每張 GPU 一個訓練工作
    ap.add_argument("--parse-only", action="store_true", help="print the parsed jobs and stop")  # 只檢查工作清單
    ap.add_argument("--steps", type=int, default=12000)  # 訓練步數
    ap.add_argument("--no-sym", action="store_true")  # 關閉對稱擴增
    ap.add_argument("--smoke", action="store_true")  # 冒煙測試
    ap.add_argument("--merge", default=None, help="comma-separated seeds: write results/m84_cache/var_S4[e].npz from the finished networks")  # 合併已完成的網路
    ap.add_argument("--merge-var", default=None, help="with --merge: the variant name to write instead (results/m84_cache/var_<name>.npz), e.g. S4f for seeds 0,1,2,3")  # 合併時改用的變體名稱
    a_ = ap.parse_args()  # 解析參數
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    base.FULL, base.EXT, base.OUT, base.VAR = True, None, OUT, VAR  # 設定 m88 模組：全部對局、不加外部帳號、本實驗的輸出位置與變體名稱
    if a_.merge is not None:  # 合併模式
        seeds = [int(x) for x in a_.merge.split(",")]  # 要合併的種子
        if a_.merge_var is None:  # 沒指定名稱時
            base.stage_merge(seeds); return  # 沿用 m88 的合併（var_S4 或 var_S4e）
        q = np.load(base.CACHE / "queries.npz"); N, C = len(q["qkey"]), len(q["players"]); protos = np.zeros((len(seeds), 5, C, base.D_OUT), np.float32); qvec = np.zeros((len(seeds), N, base.D_OUT), np.float16); rep = {}  # 原型 [種子, 5 個網路, 玩家, 維度] 與查詢向量
        for si, sd_ in enumerate(seeds):  # 同 m88_seqid.stage_merge，但使用指定的名稱
            for j, v in enumerate(base.NETS):  # 五個交叉擬合網路
                t_ = f"{v}{f'_s{sd_}' if sd_ else ''}"; z = np.load(OUT / f"out_{t_}.npz"); protos[si, j] = z["protos"]; qvec[si, z["qidx"]] = z["qvec"]; rep[t_] = json.loads((OUT / f"train_{t_}.json").read_text())["best"]  # 填入原型、該網路負責的查詢向量與最佳成績
        np.savez(base.CACHE / f"var_{a_.merge_var}.npz", protos=protos, qvec=qvec); (base.CACHE / f"var_{a_.merge_var}.json").write_text(json.dumps(rep, indent=1))  # 寫出配對器快取格式的檔案
        print(f"var_{a_.merge_var} written ({len(seeds)} seed(s)); best held-out top-1 per net: " + ", ".join(f"{k}: {x['top1']:.3f} (step {x['step']})" for k, x in rep.items()), flush=True); return  # 回報後結束
    spec = [x.split(":") for x in a_.jobs]; jobs = [(-1 if x[0] == "all" else int(x[0]), int(x[1]) if len(x) > 1 else 0) for x in spec]; opening = {j: (0.3 if len(x) > 2 and "o" in x[2] else 0.0) for j, x in zip(jobs, spec)}; masking = {j: (0.3 if len(x) > 2 and "m" in x[2] else 0.0) for j, x in zip(jobs, spec)}  # 解析 網路[:種子[:o|m|om]]
    assert jobs and len(set(jobs)) == len(jobs) and all(v in base.NETS for v, _ in jobs) and all(len(x) <= 2 or (x[2] and set(x[2]) <= set("om")) for x in spec), a_.jobs  # 工作不可為空或重複，網路與第三欄須合法
    if a_.parse_only:  # 只檢查工作清單
        print("jobs: " + ", ".join(f"net {v} seed {s_}{' +opening' if opening[(v, s_)] else ''}{' +mask' if masking[(v, s_)] else ''}" for v, s_ in jobs)); return  # 印出後結束
    if a_.smoke:  # 冒煙測試不碰正式輸出
        OUT, CK = ROOT / "results/m120_smoke", Path("/work/u2507115/m120_ckpt_smoke"); base.OUT = OUT  # 改用冒煙測試的資料夾
    gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; assert len(gpus) >= len(jobs), f"{len(jobs)} trainers need as many GPUs, got {gpus}"  # 每個訓練器需要一張 GPU
    jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 60 if a_.smoke else a_.steps  # 工作編號與訓練步數
    if a_.smoke:  # 冒煙測試
        base.EVAL_EVERY = 30  # 每 30 步評估一次
    G, Tt = base.load_games(a_.smoke, log); kids = {}  # 訓練與測試對局只載入一次，子行程共用
    for k, (v, seed) in enumerate(jobs):  # 每個工作 fork 一個子行程
        try:  # 嘗試 fork
            pid = os.fork()  # 子行程繼承已載入的對局
        except OSError as e:  # 節點的記憶體承諾上限：以已啟動的訓練器繼續
            log(f"could not start net {v} seed {seed}: {e}"); break  # 記錄後不再啟動新的訓練器
        if pid == 0:  # 子行程
            code = 0  # 預設成功
            try:  # 執行訓練
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[k])  # 匯入 torch 前先指定 GPU
                out = open(ROOT / f"logs/nano4/m120_net{v}_s{seed}-{jid}.log", "a", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out  # 輸出附加到這個網路的紀錄檔
                t1 = time.time()  # 子行程的計時起點
                train_one(v, seed, G, Tt, steps, not a_.no_sym, lambda s: print(f"{s} ({time.time() - t1:.0f} s)", flush=True), drop_open=opening[(v, seed)], mask=masking[(v, seed)])  # 訓練並寫出輸出
            except BaseException:  # noqa: BLE001  任何例外都記錄後以失敗結束
                import traceback  # 印出錯誤堆疊用
                traceback.print_exc(); code = 1  # 印出錯誤並標記失敗
            finally:  # 不論成敗
                sys.stdout.flush(); os._exit(code)  # 直接結束子行程，不回到父行程的迴圈
        kids[pid] = (v, seed); log(f"started net {v} seed {seed} on GPU {gpus[k]} (pid {pid})")  # 父行程記下子行程
    bad = 0  # 失敗的訓練器數
    while kids:  # 等待所有子行程
        pid, status = os.wait(); v, seed = kids.pop(pid); bad += status != 0; log(f"net {v} seed {seed} finished with status {status}")  # 回收結束的子行程並記錄狀態
    for v, seed in jobs:  # 彙總每個工作的訓練紀錄
        f = OUT / f"train_{v}{f'_s{seed}' if seed else ''}.json"  # 該工作的紀錄檔
        if f.exists():  # 有紀錄才印
            d = json.load(open(f)); print(f"  net {v} seed {seed}: best {d['best']}; " + "  ".join(f"{h['step'] // 1000}k: " + " / ".join(f"{k} {h[k]['top1']:.3f}" for k in ("seen", "unseen") if k in h) for h in d["history"]))  # 最佳成績與每次評估的 top-1
    sys.exit(1 if bad else 0)  # 有訓練器失敗時以 1 結束


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
