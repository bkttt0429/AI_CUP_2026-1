"""Section 4M M77 (2026-11-10): whole-game bidirectional encoder that recognises WHICH training player one side of a game is.
Input: the first 256 plies as tokens (point, side relative to the subject, absolute colour, ply index, offset to the previous ply,
offset to the subject side's previous ply) + [CLS]; pre-norm Transformer encoder; closed-set classification over the training players.
--exclude-fold v (1-4): every row of fold v's players is left out (cross-fitting: their games are never seen, so recognising the
opponents in them is as on the test set); -1: all rows (the test model). One held-out game per player (>= 8 games) measures the
identification itself. Outputs under results/m77/: val<v>.npz (the validation players' 20-game questions: opponent side and target
side = impostor) or test.npz, heldout_<v>.json, model_<v>.pt.
  python -I scripts/m_series/m77_idbert.py --exclude-fold 1
"""
import argparse  # 解析命令列參數
import json  # 讀取驗證題、寫出留出準度紀錄
import math  # 學習率排程用的 cos
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import torch  # 深度學習框架
from torch import nn  # 神經網路模組

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
OUT = ROOT / "results/m77"  # M77 的輸入與輸出資料夾


class Enc(nn.Module):  # 整局身分編碼器
    def __init__(self, n_classes, d=256, layers=6, heads=8, drop=0.1):  # 建立各層
        super().__init__()  # 初始化 nn.Module
        self.pos, self.side, self.col, self.t = nn.Embedding(363, d), nn.Embedding(3, d), nn.Embedding(3, d), nn.Embedding(256, d)  # 落點（含 pass 與 padding）、本方／對方、顏色、手數的嵌入
        self.o1, self.o2 = nn.Embedding(1370, d), nn.Embedding(1370, d)  # 對上一手、對同方上一手的位移嵌入（37×37 加無效值）
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.ln = nn.LayerNorm(d)  # [CLS] 向量與輸入端的 LayerNorm
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")  # pre-norm 的編碼器層
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)  # 疊成 layers 層
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))  # [CLS] 輸出對應到訓練玩家類別

    @staticmethod  # 不需要實例的工具函式
    def offset(pos, k):  # 與 k 手前落點的相對位移編號
        prev = torch.roll(pos, k, 1); ok = (pos < 361) & (prev < 361); ok[:, :k] = False  # k 手前的落點；兩手都在盤上才有效
        return torch.where(ok, (pos % 19 - prev % 19 + 18) * 37 + (pos // 19 - prev // 19 + 18), torch.full_like(pos, 1369))  # (dx, dy) 編成 0..1368，無效為 1369

    def forward(self, pos, col, subj):  # 前向計算
        pad = pos == 362  # padding 的位置
        side = torch.where(pad, torch.full_like(col, 2), (col != subj[:, None]).long())  # 0 = 主體方、1 = 對方、2 = padding
        h = self.pos(pos) + self.side(side) + self.col(col) + self.t(torch.arange(pos.shape[1], device=pos.device))[None] + self.o1(self.offset(pos, 1)) + self.o2(self.offset(pos, 2))  # 各嵌入相加成 token 表示
        h = torch.cat([self.cls.expand(len(pos), -1, -1), h], 1); mask = torch.cat([torch.zeros(len(pos), 1, dtype=torch.bool, device=pos.device), pad], 1)  # 前置 [CLS] 並建立 padding 遮罩
        return self.out(self.enc(self.ln(h), src_key_padding_mask=mask)[:, 0])  # 取 [CLS] 的輸出做分類


def infer(model, POS, COL, subj, idx, batch=1024):  # 批次推論前 20 名類別
    """top-20 (class index, logit) of every row in idx with the given subject colours."""
    model.eval(); tops, vals = [], []  # 評估模式；收集結果
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # 不算梯度，以 bf16 推論
        for i in range(0, len(idx), batch):  # 逐批次
            j = torch.as_tensor(idx[i:i + batch], device="cuda")  # 本批次的列
            lg = model(POS[j].long(), COL[j].long(), torch.as_tensor(subj[i:i + batch], device="cuda").long()).float()  # 各類別的 logit
            v, k = lg.topk(20, dim=1); tops.append(k.cpu().numpy()); vals.append(v.cpu().numpy())  # 取前 20 名
    return np.concatenate(tops), np.concatenate(vals)  # 回傳類別索引與 logit


def main():  # 訓練並輸出查詢結果
    ap = argparse.ArgumentParser(); ap.add_argument("--exclude-fold", type=int, required=True); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--batch", type=int, default=512)  # 排除的折、epoch 數、批次大小
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--limit", type=int, default=0)  # 學習率、寬度、層數、訓練局數上限
    a = ap.parse_args(); v = a.exclude_fold; t0 = time.time(); torch.manual_seed(20261110 + v); rng = np.random.default_rng(20261110 + v)  # 依排除的折固定種子
    z = np.load(OUT / "data.npz", allow_pickle=False)  # m77_prep.py 準備的資料
    player, fold, hd, subj, plab = z["player"], z["fold"], z["hd"], z["subj"], z["plab"]  # 每列的玩家、折、讓子數與主體顏色（0 黑、1 白）；玩家標籤
    POS, COL = torch.as_tensor(z["POS"].astype(np.int16), device="cuda"), torch.as_tensor(z["COL"], device="cuda")  # 落點與顏色序列放上 GPU
    allowed = (hd < 2) & (fold != v)  # 不用 HD >= 2 的局與折 v 的玩家
    by = {}  # 每位玩家可用的列
    for r in np.flatnonzero(allowed):  # 逐一可用列
        by.setdefault(int(player[r]), []).append(r)  # 依玩家分組
    cls_players = np.array(sorted(by)); cmap = np.full(len(z["players"]), -1); cmap[cls_players] = np.arange(len(cls_players))  # 類別＝有可用列的玩家；玩家索引轉類別
    held = np.array([rng.choice(rs) for rs in by.values() if len(rs) >= 8]); is_held = np.zeros(len(player), bool); is_held[held] = True  # 每位 >= 8 局的玩家留出一局
    train = np.flatnonzero(allowed & ~is_held)  # 訓練列
    if a.limit:  # 有設定局數上限
        train = rng.choice(train, a.limit, replace=False)  # 隨機取子集
    y_all = torch.as_tensor(cmap[player], device="cuda")  # 每列的類別標籤
    model = Enc(len(cls_players), a.d, a.layers).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)  # 模型與 AdamW 最佳化器
    steps = a.epochs * (len(train) // a.batch); warm = 1000  # 總步數與 warmup 步數
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps, 1)))))  # 線性 warmup 後 cosine 衰減
    print(f"exclude fold {v}: {len(train)} training games, {len(cls_players)} classes, {len(held)} held-out games; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)  # 印出訓練規模
    best, best_state, hist = -1.0, None, []  # 最佳 top-1、權重與歷程
    for ep in range(a.epochs):  # 逐 epoch
        model.train(); perm = torch.as_tensor(rng.permutation(train), device="cuda"); tot, nb = 0.0, 0  # 打亂訓練列
        for i in range(0, len(perm) - a.batch + 1, a.batch):  # 逐一完整批次
            j = perm[i:i + a.batch]  # 本批次的列
            with torch.autocast("cuda", dtype=torch.bfloat16):  # 以 bf16 前向計算
                lg = model(POS[j].long(), COL[j].long(), torch.as_tensor(subj, device="cuda")[j].long())  # 以本人為主體的 logit
            loss = nn.functional.cross_entropy(lg.float(), y_all[j], label_smoothing=0.1)  # label smoothing 0.1 的交叉熵
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += float(loss.detach()); nb += 1  # 反向傳播、梯度裁剪與更新
        tk, _ = infer(model, POS, COL, subj[held], held); truth = cmap[player[held]]  # 留出局的前 20 名與真正類別
        acc = {k: float((tk[:, :k] == truth[:, None]).any(1).mean()) for k in (1, 5, 20)}; hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), **{f"top{k}": x for k, x in acc.items()}})  # top-1／5／20 並記入歷程
        print(f"  epoch {ep + 1}: loss {tot / max(nb, 1):.3f}; held-out own games top-1 {acc[1]:.3f} top-5 {acc[5]:.3f} top-20 {acc[20]:.3f} ({time.time() - t0:.0f} s)", flush=True)  # 印出本 epoch 的結果
        if acc[1] > best:  # 留出局 top-1 進步
            best, best_state = acc[1], {k: x.detach().clone() for k, x in model.state_dict().items()}  # 保存最佳權重
    model.load_state_dict(best_state); torch.save({"state": best_state, "classes": cls_players, "args": vars(a)}, OUT / f"model_{v}.pt")  # 載回並存下最佳模型
    json.dump({"exclude_fold": v, "classes": int(len(cls_players)), "held": int(len(held)), "best_top1": best, "history": hist}, open(OUT / f"heldout_{v}.json", "w"), indent=1)  # 寫出留出準度紀錄

    def side_pack(idx, subject):  # 指定主體顏色時的 top-1 標籤、margin 與玩家
        tk, vl = infer(model, POS, COL, subject, idx); best_p = cls_players[tk[:, 0]]  # 第一名類別轉回玩家索引
        return plab[best_p], (vl[:, 0] - vl[:, 1]).astype(np.float32), best_p  # 標籤、與第二名的 logit 差、玩家
    if v > 0:  # 折 v 驗證玩家的 n = 20 題
        key_row = {k: i for i, k in enumerate(z["key"])}; m = json.load(open(ROOT / f"results/trunk_seq/r_val{v}.json"))["val"]; rows, pids, ys = [], [], []  # 棋局鍵對應列；本折的驗證題
        for pid, rk, ng, gk in zip(m["player_id"], m["rank"], m["n_games"], m["game_keys"]):  # 逐題
            if ng == 20:  # 只用 n = 20 的題
                for k in gk:  # 這題的 20 局
                    rows.append(key_row[k]); pids.append(pid); ys.append(int(plab[player[key_row[k]]]))  # 查詢列、玩家與真值
        rows = np.array(rows); ol, om, ob = side_pack(rows, 1 - subj[rows]); il, im, _ = side_pack(rows, subj[rows])  # 對手側與本人側（冒充者）
        np.savez_compressed(OUT / f"val{v}.npz", key=z["key"][rows], pid=np.array(pids), fold=np.full(len(rows), v), y=np.array(ys), hd=hd[rows], target_black=subj[rows] == 0,  # 寫出驗證查詢
                            opp_lab=ol, opp_m=om, opp_best=z["players"][ob], imp_lab=il, imp_m=im)  # 對手側與冒充者的標籤、margin
        print(f"validation fold {v}: {len(rows)} queries written ({time.time() - t0:.0f} s)")  # 印出完成訊息
    else:  # 測試局
        TPOS, TCOL = torch.as_tensor(z["TPOS"].astype(np.int16), device="cuda"), torch.as_tensor(z["TCOL"], device="cuda"); ts = z["tsubj"]; idx = np.arange(len(ts))  # 測試局的序列與本人顏色
        def tp(subject):  # 測試局版本的 side_pack
            tk, vl = infer(model, TPOS, TCOL, subject, idx); bp = cls_players[tk[:, 0]]; return plab[bp], (vl[:, 0] - vl[:, 1]).astype(np.float32), bp  # 標籤、margin 與玩家
        ol, om, ob = tp(1 - ts); tl, tm, tb = tp(ts)  # 對手側與本人側
        np.savez_compressed(OUT / "test.npz", qid=z["tqid"], hd=z["thd"], target_black=ts == 0, opp_lab=ol, opp_m=om, opp_best=z["players"][ob], tgt_lab=tl, tgt_m=tm, tgt_best=z["players"][tb])  # 寫出測試查詢
        print(f"test: {len(idx)} games written ({time.time() - t0:.0f} s)")  # 印出完成訊息


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
