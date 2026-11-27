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
import argparse  # 解析命令列參數
import json  # 讀取留出鍵、寫出評估紀錄
import math  # 餘弦學習率排程
import os  # fork 子行程、設定 GPU 與重導輸出
import sys  # 調整模組搜尋路徑與標準輸出
import time  # 計時
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同資料夾的實驗模組可被匯入
import m88_seqid as base  # noqa: E402  載入器、模型與常數（torch 在該模組中延後匯入）

RV = ROOT / "results/r_variants"  # 評估紀錄的輸出資料夾
VARIANTS = {  # 各變體的設定
    "v0_control": {},  # 對照組：原設定
    "v1_crop128": {"crop": 128},  # 隨機 128 手的視窗
    "v2_crop128_pdrop": {"crop": 128, "pdrop": 0.2},  # 再遮住兩成的手
    "v3_crop128_pdrop_sym": {"crop": 128, "pdrop": 0.2, "sym": True},  # 再加上隨機盤面對稱
    "v5_v3_strat": {"crop": 128, "pdrop": 0.2, "sym": True, "strat": 0.5},  # v3 加上一半 episode 同等級抽人
    "v6_v3_reg": {"crop": 128, "pdrop": 0.2, "sym": True, "dropout": 0.2, "wd": 0.1},  # v3 加上更強的正則化
    "v7_crop64_pdrop_sym": {"crop": 64, "pdrop": 0.2, "sym": True},  # 較短的 64 手視窗
    # 05:20 看過第一批評估後加入（視窗變體會丟掉開局，在完整對局上遠低於對照組）：
    "v8_sym_episode": {"sym": "episode"},  # 整個 episode 用同一個隨機對稱、完整對局（另一個工作階段的 --sym-aug）
    "v9_sym_game": {"sym": True},  # 每局各自隨機對稱、完整對局
    "v10_sym_episode_strat": {"sym": "episode", "strat": 0.5},  # episode 對稱加上同等級抽人
}  # 結束變體設定


def sym_table():  # 盤面對稱的點索引對照表（供其他實驗匯入）
    """[8, 363]: the eight board symmetries on the point index (x = p % 19, y = p // 19); 361 and 362 (pass / padding) stay."""
    p = np.arange(361); x, y = p % 19, p // 19; T = np.tile(np.arange(363), (8, 1))  # 每列先設為恆等對應
    for k, (a, b_) in enumerate(((x, y), (18 - x, y), (x, 18 - y), (18 - x, 18 - y), (y, x), (18 - y, x), (y, 18 - x), (18 - y, 18 - x))):  # 八種對稱（翻轉與轉置）
        T[k, :361] = b_ * 19 + a  # 換算回點索引
    return T  # 回傳 [8, 363] 對照表


def run_variant(name, spec, G, gpu, steps, jid, smoke):  # 在子行程中訓練並評估一個變體
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # 匯入 torch 前先指定這個子行程的 GPU
    import torch  # 子行程內才匯入 torch
    from aicup.metric import LEVELS  # 10 個等級名稱
    out = open(ROOT / f"logs/nano4/m112_{name}-{jid}.log", "w", buffering=1); os.dup2(out.fileno(), 1); os.dup2(out.fileno(), 2); sys.stdout = sys.stderr = out  # 每個變體的輸出寫到自己的紀錄檔
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    base.DROP = spec.get("dropout", 0.1); WD = spec.get("wd", base.WD); crop = spec.get("crop", base.T_MAX); pdrop = spec.get("pdrop", 0.0); sym = spec.get("sym", False); strat = spec.get("strat", 0.0); v, seed, dev = -1, 3, "cuda"; P_EP, M_EP, D_OUT = base.P_EP, base.M_EP, base.D_OUT  # 由變體設定取出超參數；固定看全部玩家的網路 -1、種子 3
    log(f"variant {name}: {spec}; GPU {torch.cuda.get_device_name(0)}")  # 記錄變體與 GPU
    q = np.load(base.CACHE / "queries.npz"); players = q["players"]; pidx = {p: i for i, p in enumerate(players)}; C = len(players); held = set(json.load(open(base.M82 / "held_keys.json")))  # 訓練玩家清單與 M82 留出的對局鍵
    owner = np.array([pidx[p] for p in G["pid"]]); is_held = np.array([k in held for k in G["key"]]); gidx = {k: i for i, k in enumerate(G["key"])}; rk = {lv: i for i, lv in enumerate(LEVELS)}; plab = np.zeros(C, np.int64); plab[owner] = np.array([rk[k.split(":")[0]] for k in G["key"]])  # 每局的玩家、是否留出，以及每位玩家的等級
    use = ~is_held & (G["fold"] != v); cnt = np.bincount(owner[use], minlength=C); tp = np.flatnonzero(cnt >= base.MIN_GAMES); W = int(cnt.max())  # 可訓練的對局、每人局數與夠局數的玩家
    idx = np.flatnonzero(use); o = owner[idx]; order = np.argsort(o, kind="stable"); idx, o = idx[order], o[order]; pos_ = np.arange(len(o)) - np.searchsorted(o, o, side="left"); tab = np.full((C, W), -1, np.int64); tab[o, pos_] = idx  # 建表：每位玩家的對局列號
    rng = np.random.default_rng(880 + v + 1000 * seed); rng2 = np.random.default_rng(112000 + sum(map(ord, name))); torch.manual_seed(880 + v + 1000 * seed); net = base.build_model().to(dev)  # 固定亂數並建立模型
    sample = G["GL"][rng.choice(np.flatnonzero(use), min(4000, int(use.sum())), replace=False)].astype(np.float32); flat = sample.reshape(-1, 128); flat = flat[np.abs(flat).sum(1) > 0]; net.mu.copy_(torch.as_tensor(flat.mean(0))); net.sd.copy_(torch.as_tensor(flat.std(0) + 1e-6)); del sample, flat  # 以抽樣對局估計主幹向量的標準化統計量
    opt = torch.optim.AdamW(net.parameters(), lr=base.LR, weight_decay=WD); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / base.WARM) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))  # 最佳化器與暖身後餘弦衰減的學習率
    tb = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731  陣列轉成 GPU 張量
    SYM = sym_table(); ar_c = np.arange(crop)  # 對稱對照表與視窗位移

    def forward(gl, pos, mov, length, subj, t_start, hide):  # 支援視窗與遮罩的前向計算
        """As SeqID.forward, with the window's true ply indices and an optional mask of hidden plies."""
        B, T = pos.shape; tix = (t_start[:, None] + torch.arange(T, device=pos.device)[None]).clamp(max=base.T_MAX - 1)  # 視窗內每手在原局的手數
        x = net.gl((gl.float() - net.mu) / net.sd) + net.pos(pos) + net.side((mov == subj[:, None]).long()) + net.col(mov) + net.tpos(tix) + net.dprev(net.disp(pos, 1)) + net.down(net.disp(pos, 2))  # 主幹向量、落點、是否主角方、顏色、手數與位移的嵌入相加
        x = torch.cat([net.cls.expand(B, -1, -1), net.norm(x)], dim=1); pad = torch.arange(T + 1, device=pos.device)[None] > length[:, None]  # 前面加上 CLS，超出長度的位置遮住
        if hide is not None:  # 有隨機遮手時
            pad[:, 1:] |= hide  # 一併遮住（CLS 不遮）
        return torch.nn.functional.normalize(net.head(net.enc(x, src_key_padding_mask=pad)[:, 0]).float(), dim=-1)  # 取 CLS 輸出並正規化成單位向量

    def embed(rows):  # 以評估模式計算一批對局的向量
        net.eval(); res = np.zeros((len(rows), D_OUT), np.float32)  # 切換到評估模式
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不記錄梯度、bf16 計算
            for i in range(0, len(rows), 1024):  # 每批 1024 局
                r = rows[i:i + 1024]; res[i:i + 1024] = net(tb(G["GL"][r]), tb(G["POS"][r].astype(np.int64)), tb(G["MOV"][r].astype(np.int64)), tb(G["LEN"][r].astype(np.int64)), tb(G["SUBJ"][r].astype(np.int64))).float().cpu().numpy()  # 完整對局的向量
        net.train(); return res  # 切回訓練模式並回傳
    hk = np.array([gidx[k] for k in q["qkey"][2 * int(q["Q"]):2 * int(q["Q"]) + int(q["H"])] if k in gidx], dtype=np.int64); h_truth = owner[hk] if len(hk) else hk; hist = []  # 留出的本人對局（查詢檔中的 H 段）與其正確玩家

    def evaluate(step, train_acc):  # 留出對局的認人評估
        rows = np.flatnonzero(~is_held & (owner < C)); E = embed(rows); Pm = np.zeros((C, D_OUT)); np.add.at(Pm, owner[rows], E); Pm /= np.maximum(np.linalg.norm(Pm, axis=1, keepdims=True), 1e-12); res = {"step": step, "train_episode_top1": train_acc}  # 每位玩家的原型 = 非留出對局向量和的單位向量
        if len(hk):  # 有留出對局時
            S = embed(hk) @ Pm.T; r = (S > S[np.arange(len(hk)), h_truth][:, None]).sum(1); res.update(n=int(len(hk)), top1=float((r == 0).mean()), top5=float((r < 5).mean()), top20=float((r < 20).mean()))  # 正確玩家的名次與 top-1/5/20
            # 同一批留出對局，只在 255 位隨機的其他玩家中排名：可與訓練 episode 的準確率相比
            sub = np.stack([rng2.choice(C, 255, replace=False) for _ in range(len(hk))]); res["held_top1_of_256"] = float((S[np.arange(len(hk))[:, None], sub] > S[np.arange(len(hk)), h_truth][:, None]).sum(1).__eq__(0).mean())  # 256 人中的 top-1
            log(f"  step {step}: held-out own games ({len(hk)}) top-1 / top-5 / top-20 {res['top1']:.3f} / {res['top5']:.3f} / {res['top20']:.3f}; among 256 random players {res['held_top1_of_256']:.3f} (training games in their episodes: {train_acc:.3f})")  # 記錄評估結果
        hist.append(res); (RV / f"m112_{name}{'_smoke' if smoke else ''}.json").write_text(json.dumps({"spec": spec, "history": hist}, indent=1))  # 累積並寫出評估紀錄
    net.train(); ar = np.arange(P_EP); tl = ta = 0.0; last_acc = 0.0; t_last = time.time(); every = 20 if smoke else 500; eval_every = max(steps // 2, 1) if smoke else 5000; ev_steps = set(range(eval_every, steps + 1, eval_every)) | {steps}  # 訓練狀態、紀錄間隔與評估的步數
    for step in range(steps):  # 訓練迴圈
        if strat and rng2.random() < strat:  # 這個 episode 同等級抽人
            c = int(rng2.integers(10)); cand = tp[plab[tp] == c]; cand = cand if len(cand) >= P_EP else tp[np.abs(plab[tp] - c) <= 1]; pl = rng2.choice(cand, P_EP, replace=False) if len(cand) >= P_EP else rng.choice(tp, P_EP, replace=False)  # 隨機等級的玩家，不夠時放寬到相鄰等級，仍不夠則從全部可訓練玩家抽
        else:  # 一般 episode
            pl = rng.choice(tp, P_EP, replace=False)  # 隨機抽玩家
        u = rng.random((P_EP, W)); u[np.arange(W)[None] >= cnt[pl][:, None]] = 2.0; rows = tab[pl[:, None], np.argsort(u, axis=1)[:, :M_EP]].reshape(-1); ln = G["LEN"][rows].astype(np.int64)  # 每人隨機抽 M_EP 局（空位排到最後）
        if crop < base.T_MAX:  # 視窗變體
            s0 = (rng2.random(len(rows)) * (np.maximum(ln - crop, 0) + 1)).astype(np.int64); ix = s0[:, None] + ar_c[None]; gl, pos, mov = G["GL"][rows[:, None], ix], G["POS"][rows[:, None], ix].astype(np.int64), G["MOV"][rows[:, None], ix].astype(np.int64); lc = np.minimum(ln - s0, crop)  # 隨機起點切出視窗
        else:  # 完整對局
            s0 = np.zeros(len(rows), np.int64); gl, pos, mov, lc = G["GL"][rows], G["POS"][rows].astype(np.int64), G["MOV"][rows].astype(np.int64), ln  # 從第 0 手開始
        if sym == "episode":  # 整個 episode 同一種對稱
            pos = SYM[int(rng2.integers(0, 8))][pos]  # 套用同一個對稱
        elif sym:  # 每局各自隨機對稱
            pos = SYM[rng2.integers(0, 8, len(rows))[:, None], pos]  # 每列套用自己的對稱
        hide = tb(rng2.random(pos.shape) < pdrop) if pdrop else None  # 隨機遮手的遮罩
        with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 混合精度
            z = forward(tb(gl), tb(pos), tb(mov), tb(lc), tb(G["SUBJ"][rows].astype(np.int64)), tb(s0), hide)  # 每局的單位向量
        z = z.float().reshape(P_EP, M_EP, D_OUT); tot = z.sum(1); sc = net.log_scale.exp(); logits = sc * z.reshape(-1, D_OUT) @ torch.nn.functional.normalize(tot, dim=-1).T  # 每局對各玩家原型的相似度
        own = sc * (z * torch.nn.functional.normalize(tot[:, None] - z, dim=-1)).sum(-1).reshape(-1); tgt = tb(np.repeat(ar, M_EP)); logits = logits.scatter(1, tgt[:, None], own[:, None])  # 對自己的原型時扣掉這一局本身
        loss = torch.nn.functional.cross_entropy(logits, tgt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()  # 交叉熵、梯度裁剪與更新
        tl += float(loss.detach()); ta += float((logits.argmax(1) == tgt).float().mean())  # 累計損失與 episode 內準確率
        if (step + 1) % every == 0:  # 定期記錄
            last_acc = ta / every; log(f"  step {step + 1}/{steps}: loss {tl / every:.4f}, episode top-1 {last_acc:.3f} of {P_EP}, scale {float(sc):.1f}, {(time.time() - t_last) / every:.2f} s/step"); tl = ta = 0.0; t_last = time.time()  # 印出平均損失、準確率與速度後歸零
        if (step + 1) in ev_steps:  # 到評估步數
            evaluate(step + 1, last_acc); t_last = time.time()  # 評估留出對局
    log(f"variant {name}: done")  # 變體完成


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--smoke", action="store_true"); a_ = ap.parse_args()  # 訓練步數、要跑的變體、冒煙測試
    t0 = time.time(); log = lambda s: print(f"{s} ({time.time() - t0:.0f} s)", flush=True)  # noqa: E731  附上經過秒數的紀錄函式
    base.FULL = True; base.EXT = None; gpus = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]; names = a_.variants[:2] if a_.smoke else a_.variants; jid = os.environ.get("SLURM_JOB_ID", "local"); steps = 60 if a_.smoke else a_.steps  # 使用全部對局、不加外部帳號；冒煙測試只跑兩個變體 60 步
    assert len(gpus) >= len(names), f"{len(names)} variants need as many GPUs, got {gpus}"  # 每個變體需要一張 GPU
    G, Tt = base.load_games(a_.smoke, log); del Tt; kids = {}  # 對局只載入一次，子行程共用；測試對局用不到
    for k, name in enumerate(names):  # 每個變體 fork 一個子行程
        try:  # 嘗試 fork
            pid = os.fork()  # 子行程繼承已載入的對局
        except OSError as e:  # 節點的記憶體承諾上限：以已啟動的變體繼續
            log(f"could not start {name}: {e}"); break  # 記錄後不再啟動新的變體
        if pid == 0:  # 子行程
            code = 0  # 預設成功
            try:  # 執行變體
                run_variant(name, VARIANTS[name], G, gpus[k], steps, jid, a_.smoke)  # 訓練與評估
            except BaseException:  # noqa: BLE001  任何例外都記錄後以失敗結束
                import traceback  # 印出錯誤堆疊用
                traceback.print_exc(); code = 1  # 印出錯誤並標記失敗
            finally:  # 不論成敗
                sys.stdout.flush(); os._exit(code)  # 直接結束子行程，不回到父行程的迴圈
        kids[pid] = name; log(f"started {name} on GPU {gpus[k]} (pid {pid})")  # 父行程記下子行程
    bad = 0  # 失敗的變體數
    while kids:  # 等待所有子行程
        pid, status = os.wait(); name = kids.pop(pid); bad += status != 0; log(f"{name} finished with status {status}")  # 回收結束的子行程並記錄狀態
    print("\nheld-out own games, top-1 / top-20 at each evaluation (and the training episodes' accuracy):")  # 彙總表的標題
    for name in names:  # 逐一變體
        f = RV / f"m112_{name}{'_smoke' if a_.smoke else ''}.json"  # 該變體的評估紀錄
        if f.exists():  # 有紀錄才印
            h = json.load(open(f))["history"]; print(f"  {name:22s} " + "  ".join(f"{x['step'] // 1000}k: {x.get('top1', float('nan')):.3f} / {x.get('top20', float('nan')):.3f} (train {x['train_episode_top1']:.2f}, held of 256 {x.get('held_top1_of_256', float('nan')):.2f})" for x in h))  # 每次評估的 top-1 / top-20
    sys.exit(1 if bad else 0)  # 有變體失敗時以 1 結束


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
