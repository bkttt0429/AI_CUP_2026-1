"""The L1b strength head (issue #14, option b): a scalar strength of each move, trained by pairwise comparison.

StrengthHead is one deep copy of the model's PolicyHead whose channel-0 map is read at the move played:
s(state, move) = head(trunk)[move]. At step 0 that is humanv0's policy logit of the move under the rank-unknown
metadata (l1b.neutral_rows) -- "how natural the move is" -- the same known starting point as RankHeads.

Training (train_strength_head) follows l1btrain.train_heads -- the featcache, the per-player inner split, inner
early stopping on the best weights, a resumable checkpoint, a frozen trunk unless cfg.trunk_lr > 0, torchrun with
one process per GPU -- with the Bradley-Terry objective of the strength estimator (Chen, Shih, Wu 2025) on
positions: within each process's share of a global batch, every ordered pair of positions of different ranks
contributes -log sigmoid(sign(r_i - r_j) (s_i - s_j)), weighted by sqrt(w_i w_j) of the class-balanced position
weights (pairwise_bt); the loss is the weighted mean over the pairs of the global batch. The inner objective is the
same loss over the inner positions in consecutive batches, with the pairwise accuracy and the mean score per rank
reported alongside.

strength_scores has score_positions' shape for one head: [N] float32. save_strength_head / load_strength_head mirror
l1b.save_heads / load_heads (a .pt with a JSON sidecar of kind "strength", sha256 checks, a fine-tuned trunk
restored in place).
"""
import copy  # 深層複製 PolicyHead
import json  # 讀取附屬檔（寫出由 ladder.to_json 負責）
import math  # 向上取整
import time  # 計時
from dataclasses import asdict  # 設定轉成字典
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import torch  # PyTorch 主套件
from katago.train.model_pytorch import PolicyHead  # KataGo 的 policy 頭

from aicup.featcache import read_cache  # 讀取局面快取
from aicup.l1b import atomic_save, neutral_rows, trunk_features, trunk_path  # L1b 的存檔、中性 metadata、主幹特徵與主幹路徑
from aicup.l1btrain import (N_RANKS, HeadsCfg, _all_reduce_, _barrier, _cache_meta, _cfg_key, _dist, _to_tensors, inner_split,  # 沿用 L1b 等級頭的訓練工具
                            position_weights, positions_of, trunk_parameters)  # 局面權重、局面索引與主幹參數
from aicup.ladder import environment, to_json  # 程式來源紀錄與嚴格 JSON
from aicup.metric import LEVELS  # 10 個等級
from aicup.setmodel_io import file_sha256, sidecar_path  # 檔案雜湊與附屬檔路徑
from aicup.settrain import INNER_SEED  # 內層抽樣的固定種子

FORMAT = 1  # 附屬檔的格式版本
KIND = "strength"  # 附屬檔的種類
__all__ = ["HeadsCfg", "StrengthHead", "pairwise_bt", "strength_scores", "evaluate_bt", "train_strength_head",  # 對外公開的名稱
           "save_strength_head", "load_strength_head"]  # 存檔與載入函式


class StrengthHead(torch.nn.Module):  # 每手強度分數的頭
    """One deep copy of a KataGo PolicyHead; forward gives its channel-0 map [B, 362], the per-point strength scores."""

    def __init__(self, policy_head: PolicyHead):  # 由 policy 頭建立
        super().__init__()  # 初始化 nn.Module
        self.head = copy.deepcopy(policy_head)  # 複製一份，不影響原本的模型
        self.requires_grad_(True)  # 頭的參數都要訓練

    def forward(self, trunk, mask, mask_sum_hw, mask_sum):  # 前向計算
        return self.head(trunk, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)[:, 0, :]  # 只取通道 0


def move_scores(model, head, bins, globs, target, color, rows, autocast, grad_trunk=False):  # 每個局面實際著手的強度分數
    """s of each position's move [B] (fp32 head on the trunk's features). grad_trunk: keep the trunk's graph (stage 2)."""
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):  # 主幹只在第二階段求梯度；autocast 開啟時（CUDA）以 bf16 計算
        trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)  # 主幹特徵與遮罩
    out = head(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())  # 頭以 fp32 算出每個點的分數
    return out.gather(1, target[:, None])[:, 0]  # 取出實際著手處的分數


def pairwise_bt(s, rank, w):  # 成對的 Bradley-Terry 損失
    """(weighted loss sum, weight sum, correctly ordered pairs, pairs) over every ordered pair (i, j) of positions of
    different ranks: -log sigmoid(sign(r_i - r_j) (s_i - s_j)) with weight sqrt(w_i w_j). Each unordered pair enters
    twice with the same loss, so the weighted mean is unchanged."""
    diff = s[:, None] - s[None, :]  # 所有配對的分數差
    sign = torch.sign((rank[:, None] - rank[None, :]).to(s.dtype))  # 等級差的正負號
    cross = sign != 0  # 只算不同等級的配對
    pw = torch.sqrt(w[:, None] * w[None, :]) * cross  # 配對權重
    nll = -torch.nn.functional.logsigmoid(sign * diff)  # 每一對的負對數概似
    return (nll * pw).sum(), pw.sum(), ((sign * diff > 0) & cross).sum(), cross.sum()  # 加權損失、權重和、正確數、配對數


def strength_scores(model, head, bins, globs, target, colors, batch=64):  # 評出每一手的強度分數
    """s of each target move: [N] float32 (the trunk under no_grad and, on CUDA, bf16 autocast; the head in fp32)."""
    device = next(head.parameters()).device  # 頭所在的裝置
    n = len(target)  # 手數
    out = np.empty(n, dtype=np.float32)  # 輸出陣列
    rows = neutral_rows()  # 等級未知的 metadata 列
    autocast = device.type == "cuda"  # 只有 CUDA 使用 bf16
    with torch.no_grad():  # 推論不需梯度
        for i in range(0, n, batch):  # 逐批計算
            j = min(n, i + batch)  # 這一批的結尾
            b = torch.as_tensor(np.asarray(bins[i:j]), device=device)  # 棋盤特徵
            g = torch.as_tensor(np.asarray(globs[i:j]), device=device)  # 全域特徵
            c = torch.as_tensor(np.asarray(colors[i:j]), device=device).long()  # 輪到下的顏色
            tgt = torch.as_tensor(np.asarray(target[i:j]), device=device).long()  # 實際著手的位置
            out[i:j] = move_scores(model, head, b, g, tgt, c, rows, autocast).float().cpu().numpy()  # 寫入這一批的分數
    return out  # 回傳每手分數


def evaluate_bt(model, head, arrays, idx, cfg):  # 內層的 Bradley-Terry 評估
    """The Bradley-Terry loss over the inner positions `idx` in consecutive batches of cfg.batch (pairs within a batch),
    with class-balanced weights from the inner set: {"objective", "pair_accuracy", "pairs", "n", "mean_by_rank" [10]
    (None for an absent rank), "ranks_present"}."""
    device, rows = cfg.device, neutral_rows()  # 裝置與中性 metadata 列
    autocast = str(device).startswith("cuda")  # CUDA 上使用 bf16
    idx = np.asarray(idx, dtype=np.int64)  # 內層局面索引
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)  # 每個局面的權重
    weight_of[idx] = position_weights(arrays["rank_idx"][idx])  # 以內層集合計算類別平衡權重
    me, world = _dist()  # 本程序的編號與程序數
    mine = idx[me::world]  # 本程序負責的局面
    acc = torch.zeros(4, dtype=torch.float64, device=device)  # 損失和、權重和、正確配對數、配對數
    sums, counts = np.zeros(N_RANKS), np.zeros(N_RANKS, dtype=np.int64)  # 各等級的分數和與局面數
    was_training = head.training  # 記下原本的模式
    head.eval()  # 評估模式
    with torch.no_grad():  # 不需梯度
        for i in range(0, len(mine), cfg.batch):  # 逐批評估
            sel = np.sort(mine[i:i + cfg.batch])  # 這一批的局面（排序後讀取）
            bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)  # 取出張量
            s = move_scores(model, head, bins, globs, target, color, rows, autocast).double()  # 每手強度分數
            num, den, correct, pairs = pairwise_bt(s, rank, torch.from_numpy(weight_of[sel]).to(device).double())  # 這一批的配對統計
            acc += torch.stack([num, den, correct.double(), pairs.double()])  # 累加四個統計量
            r = rank.cpu().numpy()  # 這一批的等級
            np.add.at(sums, r, s.cpu().numpy())  # 累加各等級的分數
            np.add.at(counts, r, 1)  # 累加各等級的局面數
    if world > 1:  # 多個程序時
        acc = _all_reduce_(acc)  # 加總各程序的統計量
        t = _all_reduce_(torch.as_tensor(np.concatenate([sums, counts.astype(np.float64)]), dtype=torch.float64, device=device)).cpu().numpy()  # 加總各等級的分數與局面數
        sums, counts = t[:N_RANKS], np.rint(t[N_RANKS:]).astype(np.int64)  # 拆回分數和與整數局面數
    head.train(was_training)  # 恢復原本的模式
    num, den, correct, pairs = acc.cpu().numpy().tolist()  # 取出四個統計量
    present = [int(k) for k in np.flatnonzero(counts > 0)]  # 有出現的等級
    return {"objective": float(num / den) if den > 0 else float("nan"), "pair_accuracy": float(correct / pairs) if pairs > 0 else float("nan"),  # 加權平均損失與配對準確率
            "pairs": int(pairs), "n": int(counts.sum()), "mean_by_rank": [float(sums[k] / counts[k]) if counts[k] else None for k in range(N_RANKS)],  # 配對數、局面數與各等級平均分數
            "ranks_present": present}  # 有出現的等級


def save_strength_head(head, path, meta, model_state=None):  # 儲存強度頭與附屬檔
    """state_dict -> path (.pt) and the sidecar path.with_suffix('.json') of kind "strength"; model_state (stage 2) is written
    to l1b.trunk_path(path) and named in the sidecar. Returns the sidecar dict."""
    path = Path(path)  # 轉成 Path 物件
    path.parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    atomic_save(head.state_dict(), path)  # 以不會留下半截檔案的方式存權重
    trunk = {}  # 主幹的附屬檔欄位
    if model_state is not None:  # 第二階段有微調主幹
        tp = trunk_path(path)  # 主幹權重的路徑
        atomic_save(model_state, tp)  # 存下微調後的主幹
        trunk = {"trunk_file": tp.name, "trunk_sha256": file_sha256(tp)}  # 記下檔名與雜湊
    device = next(head.parameters()).device  # 頭所在的裝置
    side = {"format": FORMAT, "kind": KIND, "n_params": int(sum(p.numel() for p in head.parameters())), "state_sha256": file_sha256(path),  # 格式、種類、參數量與權重雜湊
            **trunk, **environment(str(device)), **meta}  # 主幹欄位、程式來源紀錄與訓練紀錄
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")  # 寫出附屬檔
    return json.loads(sidecar_path(path).read_text())  # 讀回寫出的內容並回傳


def load_strength_head(path, model, device="cpu"):  # 載入強度頭並核對
    """(StrengthHead, sidecar, warnings) restored from path onto model.policy_head's architecture; ValueError when the
    sidecar is missing, is not of kind "strength", or a sha256 differs; a stage-2 sidecar's trunk is loaded into `model`."""
    path = Path(path)  # 轉成 Path 物件
    side_path = sidecar_path(path)  # 附屬檔路徑
    if not side_path.exists():  # 沒有附屬檔
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")  # 拒絕載入
    meta = json.loads(side_path.read_text())  # 讀入附屬檔
    if meta.get("format") != FORMAT or meta.get("kind") != KIND:  # 格式或種類不符
        raise ValueError(f"{path}: sidecar format {meta.get('format')!r} / kind {meta.get('kind')!r}; this code reads format {FORMAT} kind {KIND!r}")  # 拒絕載入
    sha = file_sha256(path)  # 權重檔實際的雜湊
    if meta.get("state_sha256") != sha:  # 與記錄的雜湊不符
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(meta.get('state_sha256'))[:12]}")  # 拒絕載入
    if meta.get("trunk_file"):  # 有微調過的主幹
        tp = path.with_name(meta["trunk_file"])  # 主幹權重的路徑
        if not tp.exists():  # 主幹檔不存在
            raise ValueError(f"{path}: the fine-tuned trunk {tp.name} is missing")  # 拒絕載入
        tsha = file_sha256(tp)  # 主幹檔實際的雜湊
        if meta.get("trunk_sha256") != tsha:  # 與記錄的雜湊不符
            raise ValueError(f"{tp}: sha256 {tsha[:12]} differs from the sidecar's {str(meta.get('trunk_sha256'))[:12]}")  # 拒絕載入
        try:  # 嘗試把主幹載入模型
            model.load_state_dict(torch.load(tp, map_location="cpu", weights_only=True))  # 就地還原微調後的主幹
        except RuntimeError as e:  # 權重與模型不相容
            raise ValueError(f"{tp}: state_dict does not fit the model ({e})") from e  # 轉成 ValueError
    head = StrengthHead(model.policy_head)  # 依模型的 policy 頭建立架構
    try:  # 嘗試載入頭的權重
        head.load_state_dict(torch.load(path, map_location=device, weights_only=True))  # 載入強度頭權重
    except RuntimeError as e:  # 權重與架構不相容
        raise ValueError(f"{path}: state_dict does not fit model.policy_head's architecture ({e})") from e  # 轉成 ValueError
    head.to(device).eval()  # 移到裝置並切到評估模式
    warnings = []  # 警告清單
    current = environment("cpu")["code_sha256"]  # 目前程式的雜湊
    if meta.get("code_sha256") != current:  # 訓練時的程式與現在不同
        warnings.append(f"{path}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]} vs {current[:12]})")  # 只記警告
    return head, meta, warnings  # 回傳強度頭、附屬檔與警告


def train_strength_head(model, cache_dir, cfg, out_pt, log=print, resume=True, extra_meta=None, init_head=None):  # 訓練強度頭
    """Train a StrengthHead (and, with cfg.trunk_lr > 0, the trunk) on the cache with the Bradley-Terry objective; returns
    (head with the best inner objective, sidecar meta). init_head: a StrengthHead state_dict to start from (default: a
    copy of model.policy_head). Conventions as l1btrain.train_heads (see the module docstring)."""
    t_start = time.time()  # 開始計時
    me, world = _dist()  # 本程序的編號與程序數
    if me != 0:  # 不是主程序
        log = lambda *a, **k: None  # noqa: E731  非主程序不輸出紀錄
    games, arrays = read_cache(cache_dir)  # 讀取局面快取
    cache_meta = _cache_meta(cache_dir)  # 快取的 meta.json
    train_rows, inner_rows = inner_split(games, cfg.inner_frac)  # 依玩家切出訓練與內層棋局
    train_idx = positions_of(arrays, train_rows)  # 訓練局面
    inner_idx = positions_of(arrays, inner_rows)  # 內層局面
    if len(inner_idx) > cfg.inner_max_positions:  # 內層局面太多
        inner_idx = np.sort(np.random.RandomState(INNER_SEED).choice(inner_idx, cfg.inner_max_positions, replace=False))  # 以固定種子抽樣到上限
    if len(train_idx) == 0:  # 沒有訓練局面
        raise ValueError(f"{cache_dir}: no training positions")  # 無法訓練
    if len(np.unique(arrays["rank_idx"][train_idx])) < 2:  # 等級不到兩種
        raise ValueError(f"{cache_dir}: the Bradley-Terry objective needs positions of at least two ranks")  # 成對目標需要至少兩個等級
    device = cfg.device  # 執行裝置
    torch.manual_seed(cfg.seed)  # 固定亂數
    model.eval()  # 模型保持評估模式
    train_trunk = cfg.trunk_lr > 0  # 是否微調主幹（第二階段）
    if train_trunk:  # 第二階段
        trunk_names = trunk_parameters(model, device)  # 找出通往主幹輸出的參數
    else:  # 第一階段
        model.requires_grad_(False)  # 凍結整個模型
        trunk_names = []  # 沒有要訓練的主幹參數
    trunk_params = [p for n, p in model.named_parameters() if n in set(trunk_names)]  # 要微調的主幹參數
    head = StrengthHead(model.policy_head).to(device)  # 由 policy 頭複製出強度頭
    if init_head is not None:  # 指定了起始權重
        head.load_state_dict(init_head)  # 載入起始權重
    head.train()  # 訓練模式
    groups = [{"params": list(head.parameters()), "lr": cfg.lr}]  # 頭的參數群組
    if train_trunk:  # 第二階段
        groups.append({"params": trunk_params, "lr": cfg.trunk_lr})  # 主幹使用自己的學習率
    opt = torch.optim.Adam(groups)  # Adam 最佳化器
    params = [p for g in groups for p in g["params"]]  # 所有要訓練的參數
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)  # 每個局面的權重
    weight_of[train_idx] = position_weights(arrays["rank_idx"][train_idx])  # 以整個訓練集計算類別平衡權重
    global_batch = cfg.batch * world  # 全域批次大小
    steps_per_epoch = math.ceil(len(train_idx) / global_batch)  # 每個 epoch 的步數
    total_steps = cfg.epochs * steps_per_epoch  # 總步數
    eval_steps = max(1, int(round(cfg.eval_every * steps_per_epoch)))  # 每隔幾步做一次內層評估
    autocast = str(device).startswith("cuda")  # CUDA 上使用 bf16
    rows = neutral_rows()  # 中性 metadata 列
    out_pt = Path(out_pt)  # 輸出權重的路徑
    ckpt_path = out_pt.with_name(out_pt.name + ".ckpt")  # 續跑用的檢查點路徑

    def snapshot():  # 複製目前的權重
        snap = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}  # 頭的權重副本
        if train_trunk:  # 第二階段
            snap["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # 連同主幹的副本
        return snap  # 回傳副本

    use_inner = len(inner_idx) > 0  # 是否有內層評估
    state = {"step": 0, "history": [], "best": None, "since_improvement": 0, "stopped_early": False}  # 訓練狀態（存入檢查點）
    resumed_from = 0  # 續跑的起點
    if resume and ckpt_path.exists():  # 有檢查點就續跑
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # 讀取檢查點
        if ck.get("kind") != KIND or ck["cfg_key"] != _cfg_key(cfg) or ck["steps_per_epoch"] != steps_per_epoch:  # 不是同一次訓練的檢查點
            raise ValueError(f"{ckpt_path}: checkpoint of another run (kind, batch, lr, trunk_lr, seed, world size, inner split or "  # 拒絕續跑
                             "cache differ); delete it or pass --overwrite")  # 錯誤訊息的後半
        head.load_state_dict(ck["head"])  # 還原頭的權重
        if train_trunk:  # 第二階段
            model.load_state_dict(ck["model"])  # 還原主幹
        opt.load_state_dict(ck["opt"])  # 還原最佳化器
        state = ck["state"]  # 還原訓練狀態
        resumed_from = state["step"]  # 記下續跑的步數
        log(f"resumed from {ckpt_path} at step {resumed_from} (epoch {resumed_from / steps_per_epoch:.2f})", flush=True)  # 回報續跑位置

    def save_ckpt():  # 寫出檢查點
        if me == 0:  # 只有主程序寫檔
            ck = {"kind": KIND, "cfg_key": _cfg_key(cfg), "steps_per_epoch": steps_per_epoch, "head": head.state_dict(),  # 種類、設定鍵、步數與頭的權重
                  "opt": opt.state_dict(), "state": state}  # 最佳化器與訓練狀態
            if train_trunk:  # 第二階段
                ck["model"] = model.state_dict()  # 連同主幹
            atomic_save(ck, ckpt_path)  # 以不會留下半截檔案的方式寫入
        _barrier()  # 等所有程序同步

    def evaluate(step):  # 內層評估並判斷是否停止
        res = evaluate_bt(model, head, arrays, inner_idx, cfg)  # 內層的 Bradley-Terry 結果
        epoch = step / steps_per_epoch  # 目前的 epoch 進度
        state["history"].append((epoch, res["objective"], res["pair_accuracy"]))  # 記錄評估結果
        improved = state["best"] is None or (np.isfinite(res["objective"]) and res["objective"] < state["best"]["objective"])  # 是否創新低
        if improved:  # 有進步
            state["best"] = {"objective": res["objective"], "pair_accuracy": res["pair_accuracy"], "step": step,  # 記下最佳的目標值、準確率與步數
                             "mean_by_rank": res["mean_by_rank"], "ranks_present": res["ranks_present"], **snapshot()}  # 各等級平均分數與權重副本
            state["since_improvement"] = 0  # 重設未進步次數
        else:  # 沒有進步
            state["since_improvement"] += 1  # 累計未進步次數
        log(f"epoch {epoch:.2f} inner BT loss {res['objective']:.4f} pair accuracy {res['pair_accuracy']:.4f} "  # 輸出評估進度
            f"(best {state['best']['objective']:.4f} @ {state['best']['step'] / steps_per_epoch:.2f})", flush=True)  # 附上目前的最佳值
        return state["since_improvement"] >= cfg.patience  # 達到 patience 就停止

    if use_inner and state["step"] == 0 and not state["history"]:  # 全新的訓練
        evaluate(0)  # 第 0 步先評估一次
        save_ckpt()  # 寫出初始檢查點
    step = state["step"]  # 目前步數（續跑時不為 0）
    stop = state["stopped_early"]  # 先前是否已提早停止
    t_loop, n_loop = time.time(), 0  # 速度統計
    while step < total_steps and not stop:  # 直到走完全部步數或提早停止
        epoch_i, pos_in_epoch = divmod(step, steps_per_epoch)  # 目前的 epoch 與其中的位置
        order = train_idx[np.random.RandomState(cfg.seed + epoch_i).permutation(len(train_idx))]  # 每個 epoch 以固定種子打亂，續跑時順序相同
        for k in range(pos_in_epoch, steps_per_epoch):  # 從中斷處繼續這個 epoch
            sel = np.sort(order[k * global_batch:(k + 1) * global_batch][me::world])  # 本程序在這個全域批次中的份額
            opt.zero_grad(set_to_none=True)  # 清除梯度
            if len(sel):  # 有分到局面
                bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)  # 取出張量
                w = torch.from_numpy(weight_of[sel]).to(device)  # 這些局面的權重
                s = move_scores(model, head, bins, globs, target, color, rows, autocast, grad_trunk=train_trunk)  # 每手強度分數
                num, den, _, _ = pairwise_bt(s, rank, w)  # 本程序份額內的配對損失
                den_global = _all_reduce_(den.detach().clone())  # 全域批次的權重總和
                loss = num / den_global if float(den_global) > 0 else s.sum() * 0.0  # 以全域權重正規化；沒有配對時損失為 0
                if not torch.isfinite(loss):  # 損失出現非有限值
                    raise FloatingPointError(f"non-finite training loss at step {step}")  # 立即中止
                loss.backward()  # 反向傳播
            else:  # 沒有分到局面
                _all_reduce_(torch.zeros((), device=device))  # 仍參與同一次 all_reduce，以免其他程序卡住
            if world > 1:  # 多個程序時
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]  # 沒有梯度的參數補 0
                flat = _all_reduce_(torch.cat([g.reshape(-1) for g in grads]))  # 一次加總所有梯度
                offset = 0  # 拆回各參數時的位置
                for p, g in zip(params, grads):  # 逐一寫回梯度
                    p.grad = flat[offset:offset + g.numel()].view_as(g).clone()  # 換成加總後的梯度
                    offset += g.numel()  # 移到下一個參數
            opt.step()  # 更新權重
            step += 1  # 累計步數
            n_loop += len(sel) * world  # 估計處理的局面數
            state["step"] = step  # 寫入訓練狀態
            if step % eval_steps == 0 or step == total_steps:  # 到了評估點或最後一步
                if use_inner and evaluate(step):  # 內層評估達到停止條件
                    stop = True  # 停止訓練
                    state["stopped_early"] = True  # 記錄提早停止
                save_ckpt()  # 寫出檢查點
                log(f"  {n_loop / max(time.time() - t_loop, 1e-9):.0f} positions/s since the last report", flush=True)  # 回報速度
                t_loop, n_loop = time.time(), 0  # 重設速度統計
                if stop:  # 已停止
                    break  # 跳出步數迴圈
        else:  # for 迴圈沒有被 break：這個 epoch 已走完
            continue  # 進入下一個 epoch
        break  # for 迴圈被 break（已停止）時也結束 while

    if use_inner:  # 有內層評估時
        best = state["best"]  # 最佳的評估
        head.load_state_dict(best["head"])  # 載回最佳的頭
        if train_trunk:  # 第二階段
            model.load_state_dict(best["model"])  # 載回最佳的主幹
        best_epoch, obj_init, obj_best = best["step"] / steps_per_epoch, state["history"][0][1], best["objective"]  # 最佳的 epoch、初始與最佳的目標值
        acc_init, acc_best = state["history"][0][2], best["pair_accuracy"]  # 初始與最佳的配對準確率
        mean_by_rank, present = best["mean_by_rank"], best["ranks_present"]  # 最佳時各等級的平均分數
    else:  # 沒有內層評估
        best_epoch, obj_init, obj_best, acc_init, acc_best = state["step"] / steps_per_epoch, None, None, None, None  # 使用最後的權重，沒有內層讀數
        mean_by_rank, present = [None] * N_RANKS, []  # 沒有各等級的讀數
    head.eval()  # 評估模式
    per_rank = np.bincount(arrays["rank_idx"][train_idx], minlength=N_RANKS)  # 各等級的訓練局面數
    meta = {"train_cfg": asdict(cfg), "train_folds": cache_meta["folds"], "max_games": cache_meta.get("max_games"),  # 訓練設定、訓練折與每人局數上限
            "n_train_positions": int(len(train_idx)), "n_inner_positions": int(len(inner_idx)),  # 訓練與內層的局面數
            "per_rank_positions": {lv: int(per_rank[k]) for k, lv in enumerate(LEVELS)},  # 各等級的局面數
            "inner_bt_init": obj_init, "inner_bt_best": obj_best, "inner_pair_accuracy_init": acc_init, "inner_pair_accuracy_best": acc_best,  # 初始與最佳的內層指標
            "inner_mean_by_rank_best": mean_by_rank, "inner_ranks_present": present,  # 最佳時各等級的平均分數
            "best_epoch": best_epoch, "stopped_early": bool(state["stopped_early"]),  # 最佳 epoch 與是否提早停止
            "history": [(float(e), float(v), float(a)) for e, v, a in state["history"]],  # 評估紀錄
            "steps_per_epoch": steps_per_epoch, "steps_run": int(state["step"]), "resumed_from_step": int(resumed_from),  # 步數與續跑起點
            "world_size": int(world), "global_batch": int(global_batch), "trunk_trained": bool(train_trunk),  # 程序數、全域批次與是否訓練主幹
            "objective": "Bradley-Terry over the pairs of different ranks within each process's batch share, weights sqrt(w_i w_j)",  # 訓練目標的說明
            "selection": "inner Bradley-Terry loss", "n_trunk_params": int(sum(p.numel() for p in trunk_params)),  # 選模依據與主幹參數量
            "wall_seconds": time.time() - t_start, "cache_dir": str(cache_dir), "cache_meta": cache_meta,  # 耗時與快取資訊
            "head_init": "humanv0_policy_head", **(extra_meta or {})}  # 頭的初始化來源與額外欄位
    if me == 0:  # 只有主程序存檔
        meta = save_strength_head(head, out_pt, meta, model_state=model.state_dict() if train_trunk else None)  # 存下頭（第二階段連同主幹）
    _barrier()  # 等所有程序同步
    return head, meta  # 回傳強度頭與紀錄
