"""Training of the set aggregators (aicup.aggregators) as corrections Δ to cross-fitted base logits.

fit_set_model trains make(d_in) on packed training questions (ladder_data.pack_questions) to
minimise a loss of the per-tick logits base + Δ_t; predict_set_model returns softmax(base + Δ) at
the picked tick. Inputs are standardised with statistics of the training questions only: tokens
over their valid games, qglob over the questions, both with the rankmodel._standardise rule
(ddof=0, std < 1e-12 -> 1); padding is set back to 0 after standardisation. Standardisation is
done batch by batch on the device, so the caller's float32 tokens (several GB for a full fold
set) are never copied to float64 and never modified.

Training: AdamW with a cosine schedule over all steps, a seeded shuffle per epoch, class-balanced
weights (rankmodel.class_weights). Each training player's games recur across his questions with one
label, so a set model can memorise games -> rank (CUDA smoke 2026-10-19: training CE 1.27 -> 0.22 over
20 fixed epochs, validation below the base). With TrainCfg.inner_frac > 0 (the default) fit_set_model
therefore holds out inner validation players (inner_players: 10% of each rank's training players,
INNER_SEED, the same for every rung and seed), trains on the others for at most `epochs` epochs and
returns the weights of the lowest inner CE, evaluation 0 (Δ ≡ 0) included, without refitting on all
players; inner_frac = 0 trains exactly `epochs` epochs on every question. LOSSES and PICKS map
TrainCfg.loss and TrainCfg.pick to their functions: loss "final" (the last tick; the primary loss)
or "ctm_minmax" (the original CTM's min-loss / max-certainty pair of ticks, the r5_ctm_minmax
control); pick "final" (the last tick) or "certain" (each question's most certain tick, label-free,
the tick the original CTM evaluates). cfg.pick chooses the tick of the inner validation CE and the
default tick of predict_set_model, so a model is selected and scored at the same tick; the `pick`
argument of predict_set_model overrides it for a report-only extra.

crossfit_base_logits gives the base logits: for each training fold from a static model fitted on
the other training folds, for the evaluation questions from one fitted on all of them (the static
rung itself).
"""
import math  # ceil、log 等數學函式
from dataclasses import dataclass, field  # 設定與結果的資料類別
from decimal import Decimal  # 以十進位計算內層玩家數
from typing import Callable, Optional  # 型別註記

import numpy as np  # 陣列運算
import torch  # PyTorch 主套件
import torch.nn.functional as F  # 交叉熵等函式
from torch import nn  # 神經網路模組

from aicup.rankmodel import N_CLASSES, _softmax, _standardise, class_weights  # 等級數、softmax、標準化規則、類別權重

BASE_CLIP = 1e-9  # 基礎機率取對數前的下限
STATS_CHUNK = 4096  # 累計 token 統計量時每一塊的題目數
INNER_SEED = 2026  # inner_players 的打亂種子：固定，與 TrainCfg.seed 無關


@dataclass(frozen=True)  # 不可變的訓練設定
class TrainCfg:  # 集合模型的訓練設定
    """epochs: the maximum with inner_frac > 0, the exact number with inner_frac = 0. inner_frac: share of
    each rank's training players held out for early stopping (0: none, data then needs no player_id);
    evals_per_epoch: inner evaluations per epoch (besides evaluation 0); patience: evaluations in a row
    without a lower inner CE before training stops. loss, pick: keys of LOSSES and PICKS; pick is the tick
    of the inner validation CE and the default tick of predict_set_model."""
    epochs: int = 10  # epoch 數：inner_frac > 0 時為上限，inner_frac = 0 時為確切值
    batch: int = 512  # 每批題目數
    lr: float = 1e-3  # 學習率
    weight_decay: float = 1e-4  # AdamW 的權重衰減
    gamma: float = 1e-2  # ‖Δ‖² 懲罰的係數 γ
    seed: int = 0  # 初始化與打亂的種子
    loss: str = "final"  # 損失函式（LOSSES 的鍵）
    pick: str = "final"  # 選用的 tick（PICKS 的鍵）
    device: str = "cuda"  # 執行裝置
    inner_frac: float = 0.1  # 每個等級的訓練玩家中留作內層驗證（提早停止）的比例
    evals_per_epoch: int = 4  # 每個 epoch 的內層評估次數
    patience: int = 8  # 連續未進步幾次就停止


@dataclass  # 可修改的訓練結果
class Fitted:  # 訓練好的集合模型與其紀錄
    """A trained set model with its standardisation statistics (float64) and its training record.

    losses: the mean training loss of each epoch begun -- the last one over the batches done when
    training stopped inside it. With inner early stopping (cfg.inner_frac > 0): inner_ce, the inner CE
    of every evaluation, evaluation 0 (Δ ≡ 0) first; best_eval, the evaluation whose weights the model
    holds; best_epoch, the epochs completed at it (fractional, 0.0 for evaluation 0); stopped_early,
    whether patience ended training before cfg.epochs; n_inner_players / n_inner_questions, the held-out
    players and their questions. With inner_frac = 0: inner_ce is [] and the others are None."""
    model: nn.Module  # 訓練好的模組
    tok_mean: np.ndarray  # token 各欄的平均
    tok_std: np.ndarray  # token 各欄的標準差
    glob_mean: np.ndarray  # 題目層特徵的平均
    glob_std: np.ndarray  # 題目層特徵的標準差
    cfg: TrainCfg  # 使用的訓練設定
    losses: list = field(default_factory=list)  # 每個 epoch 的平均訓練損失
    inner_ce: list = field(default_factory=list)  # 每次評估的內層 CE
    best_eval: Optional[int] = None  # 最佳評估的序號
    best_epoch: Optional[float] = None  # 最佳評估時已完成的 epoch 數
    stopped_early: Optional[bool] = None  # 是否提早停止
    n_inner_players: Optional[int] = None  # 內層玩家數
    n_inner_questions: Optional[int] = None  # 內層題目數


def final_loss(logits_ticks, deltas, y, w, gamma):  # 只看最後一個 tick 的損失
    """CE(base + Δ_T, y) weighted by w (normalised by the batch's sum of w) + γ · mean ‖Δ_T‖².

    logits_ticks, deltas: [B, T, 10]; only the last tick enters."""
    ce = F.cross_entropy(logits_ticks[:, -1], y, reduction="none")  # 最後一個 tick 的逐題交叉熵
    return (w * ce).sum() / w.sum() + gamma * deltas[:, -1].pow(2).sum(dim=-1).mean()  # 加權平均 CE 加上 Δ 懲罰


def certainty(logits):  # 預測的把握程度
    """1 − H(softmax(logits)) / log C over the last axis (C classes): 0 for uniform, 1 for one-hot."""
    logp = logits.log_softmax(dim=-1)  # 對數機率
    return 1.0 + (logp.exp() * logp).sum(dim=-1) / math.log(logits.shape[-1])  # 1 減去正規化的熵


def _last_argmax(x):  # 同分時取較後面的 argmax
    """argmax over dim 1 of x [B, T]; ties go to the later tick."""
    return x.shape[1] - 1 - x.flip(1).argmax(dim=1)  # 反轉後取第一個最大值再換回原索引


def minmax_ticks(logits_ticks, y):  # 找出最低損失與最高把握的 tick
    """Per sample the tick t1 of lowest CE(logits_t, y) and the tick t2 of highest certainty;
    ties go to the later tick. logits_ticks [B, T, 10] -> (t1 [B], t2 [B])."""
    with torch.no_grad():  # 選 tick 不需要梯度
        logp = logits_ticks.log_softmax(dim=-1)  # 各 tick 的對數機率
        ce = -logp.gather(-1, y[:, None, None].expand(-1, logp.shape[1], 1))[..., 0]  # 各 tick 對真實等級的 CE
        return _last_argmax(-ce), _last_argmax(certainty(logits_ticks))  # CE 最低與把握最高的 tick


def ctm_minmax_loss(logits_ticks, deltas, y, w, gamma):  # 原始 CTM 的雙 tick 損失
    """The original CTM loss on a pair of ticks per sample (minmax_ticks: t1 lowest CE, t2 most
    certain): (CE_t1 + CE_t2) / 2 weighted by w (normalised by the batch's sum of w) +
    γ · mean (‖Δ_t1‖² + ‖Δ_t2‖²) / 2 -- weighted as final_loss, so the two differ only in the ticks.

    logits_ticks, deltas: [B, T, 10]."""
    t1, t2 = minmax_ticks(logits_ticks, y)  # 每個樣本的兩個 tick
    rows = torch.arange(len(y), device=logits_ticks.device)  # 樣本索引
    ce = (F.cross_entropy(logits_ticks[rows, t1], y, reduction="none")  # t1 的交叉熵
          + F.cross_entropy(logits_ticks[rows, t2], y, reduction="none")) / 2  # 加上 t2 的交叉熵後取平均
    pen = (deltas[rows, t1].pow(2).sum(dim=-1) + deltas[rows, t2].pow(2).sum(dim=-1)) / 2  # 兩個 tick 的 Δ 懲罰平均
    return (w * ce).sum() / w.sum() + gamma * pen.mean()  # 加權平均 CE 加上 Δ 懲罰


def _pick_torch(logits_ticks, pick):  # 依 pick 取出各題使用的 logits
    """logits_ticks [B, T, C] (torch) -> the logits [B, C] at the tick of `pick`: "final", the last tick, or
    "certain", each question's most certain tick (1 − H / log C; ties -> the later tick), label-free."""
    if pick == "final":  # 取最後一個 tick
        return logits_ticks[:, -1]  # 回傳最後一個 tick 的 logits
    if pick == "certain":  # 取最有把握的 tick
        t = _last_argmax(certainty(logits_ticks))  # 每題把握最高的 tick
        return logits_ticks[torch.arange(len(t), device=t.device), t]  # 取出各題對應的 logits
    raise ValueError(f"unknown pick {pick!r}; known: {list(PICKS)}")  # 未知的 pick


def _pick_final(logits_ticks):  # numpy 版：最後一個 tick
    return logits_ticks[:, -1]  # 回傳最後一個 tick 的 logits


def _pick_certain(logits_ticks):  # numpy 版：最有把握的 tick
    """Each question's most certain tick (1 − H / log 10; ties -> the later tick), label-free (numpy)."""
    return _pick_torch(torch.from_numpy(np.asarray(logits_ticks, dtype=np.float64)), "certain").numpy()  # 轉成張量計算後轉回 numpy


LOSSES = {"final": final_loss, "ctm_minmax": ctm_minmax_loss}  # TrainCfg.loss 對應的損失函式
# logits [Q, T, 10]（float64）-> 每題使用的 logits [Q, 10]
PICKS = {"final": _pick_final, "certain": _pick_certain}  # TrainCfg.pick 對應的取 tick 函式


def _token_stats(tokens, mask, chunk=STATS_CHUNK, rows=None):  # 分塊計算 token 的標準化統計量
    """rankmodel._standardise(tokens[rows][mask[rows]]) -- per-column mean and std (ddof=0) over the
    valid tokens of the questions `rows` (all when None), std < 1e-12 -> 1 -- accumulated in float64
    over chunks of questions (two passes), so no float64 copy of the whole [Q, M, D] array is made
    (and, with rows, no copy of the selected questions beyond one chunk)."""
    def chunks():  # 逐塊產生 token 與遮罩
        for q in range(0, len(tokens) if rows is None else len(rows), chunk):  # 每次一塊題目
            sel = slice(q, q + chunk) if rows is None else rows[q:q + chunk]  # 這一塊的題目
            yield tokens[sel], mask[sel]  # 交出這一塊

    n = int((mask if rows is None else mask[rows]).sum())  # 有效 token 的總數
    total = np.zeros(tokens.shape[2])  # 各欄的總和
    for t, m in chunks():  # 第一趟：累計總和
        total += t[m].sum(axis=0, dtype=np.float64)  # 以 float64 加總有效 token
    mean = total / n  # 各欄平均
    sq = np.zeros(tokens.shape[2])  # 各欄的離差平方和
    for t, m in chunks():  # 第二趟：累計離差平方
        x = t[m].astype(np.float64) - mean  # 這一塊的離差
        sq += np.einsum("ij,ij->j", x, x)  # 逐欄平方和
    std = np.sqrt(sq / n)  # 母體標準差（ddof=0）
    std[std < 1e-12] = 1.0  # 常數欄不縮放
    return mean, std  # 回傳平均與標準差


def _check(data, base):  # 檢查打包資料與基礎 logits
    tokens, mask, qglob = data["tokens"], np.asarray(data["mask"]), data["qglob"]  # 取出三個輸入
    Q = len(tokens)  # 題目數
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != bool or Q == 0:  # 形狀或型別不符
        raise ValueError(f"tokens {tokens.shape} and mask {mask.shape} ({mask.dtype}) are not a non-empty packed set")  # 不是有效的打包資料
    if qglob.ndim != 2 or len(qglob) != Q:  # 題目層特徵與題目數不符
        raise ValueError(f"qglob {qglob.shape} does not match {Q} questions")  # 回報形狀
    if not mask.any(axis=1).all():  # 有題目沒有任何棋局
        raise ValueError(f"{int((~mask.any(axis=1)).sum())} questions have no game")  # 回報空題的數量
    base = np.asarray(base, dtype=np.float64)  # 基礎 logits 轉成雙精度
    if base.shape != (Q, N_CLASSES) or not np.all(np.isfinite(base)):  # 形狀不符或含非有限值
        raise ValueError(f"base must be finite [{Q}, {N_CLASSES}], got {base.shape}")  # 回報錯誤
    return base  # 回傳檢查過的基礎 logits


class _Inputs:  # 放在裝置上的打包題目集合
    """A packed question set on the device; calling it with question indices returns the
    standardised batch (tokens with padding 0, mask, qglob)."""

    def __init__(self, data, fitted, device):  # 把輸入搬到裝置
        if data["tokens"].shape[2] != len(fitted.tok_mean):  # token 欄數與模型不符
            raise ValueError(f"tokens have {data['tokens'].shape[2]} columns, the model has d_in {len(fitted.tok_mean)}")  # 回報欄數
        f32 = dict(dtype=torch.float32, device=device)  # float32 張量的共同參數
        # 在 CPU 上是零複製的視圖，對加速器只傳輸一次
        self.tokens = torch.from_numpy(np.ascontiguousarray(data["tokens"], dtype=np.float32)).to(device)  # 原始 token（尚未標準化）
        self.mask = torch.from_numpy(np.asarray(data["mask"], dtype=bool)).to(device)  # 有效棋局的遮罩
        glob = (np.asarray(data["qglob"], dtype=np.float64) - fitted.glob_mean) / fitted.glob_std  # 以 float64 標準化題目層特徵
        self.qglob = torch.tensor(glob, **f32)  # 轉成 float32 張量
        self.mean, self.std = torch.tensor(fitted.tok_mean, **f32), torch.tensor(fitted.tok_std, **f32)  # token 的標準化參數

    def __call__(self, ix):  # 取出一批並標準化
        mask = self.mask[ix]  # 這一批的遮罩
        tokens = ((self.tokens[ix] - self.mean) / self.std).masked_fill(~mask[..., None], 0.0)  # 標準化後把補齊處設回 0
        return tokens, mask, self.qglob[ix]  # 回傳 token、遮罩與題目層特徵


def _n_inner(n, frac):  # 一個等級的內層玩家數
    """Inner players of a rank with n players: ceil(frac × n) with frac taken as written in decimal
    (0.1 × 30 is 3, not the 4 of float ceil(3.0000000000000004)), at most n − 1, none when n < 2."""
    return 0 if n < 2 else min(n - 1, math.ceil(Decimal(repr(float(frac))) * n))  # 以 Decimal 避免浮點誤差


def inner_players(player_id, y, frac, seed=INNER_SEED):  # 決定內層驗證玩家
    """Per question whether its player is held out as an inner validation player (bool [Q]).

    Per rank, the rank's distinct players are sorted, shuffled with a fresh RandomState(seed) and the
    first _n_inner(n, frac) of them are taken: ceil(frac × n) of the rank's n players, none of a rank
    with one player, never all of them. The choice depends only on the questions' players and ranks --
    not on their order, on TrainCfg.seed or on the model -- so every rung and seed of a split holds out
    the same players. Every player must have one rank; 0 <= frac < 1."""
    pid, y = np.asarray(player_id).astype(str), np.asarray(y, dtype=np.int64)  # 玩家 id 轉字串、等級轉整數
    if pid.ndim != 1 or pid.shape != y.shape:  # 不是每題一個值
        raise ValueError(f"player_id {pid.shape} and y {y.shape} must be one value per question")  # 回報形狀
    if not 0.0 <= frac < 1.0:  # 比例超出範圍
        raise ValueError(f"inner_frac must be in [0, 1), got {frac}")  # 回報錯誤
    players, inv = np.unique(pid, return_inverse=True)  # 已排序的不重複玩家，以及每題對應的玩家
    rank = np.empty(len(players), dtype=np.int64)  # 每位玩家的等級
    rank[inv] = y  # 由題目填入玩家的等級
    if np.any(rank[inv] != y):  # 有玩家的題目等級不一致
        bad = players[np.unique(inv[rank[inv] != y])]  # 找出這些玩家
        raise ValueError(f"{len(bad)} players have questions of more than one rank, e.g. {bad[:3].tolist()}")  # 回報錯誤
    chosen = np.zeros(len(players), dtype=bool)  # 被選為內層的玩家
    for r in np.unique(rank):  # 逐個等級抽選
        members = np.flatnonzero(rank == r)  # 該等級的玩家，依排序後的玩家順序
        chosen[np.random.RandomState(seed).permutation(members)[:_n_inner(len(members), frac)]] = True  # 每個等級用新的同種子亂數抽選
    return chosen[inv]  # 轉回每題的旗標


class _EarlyStop:  # 內層驗證與提早停止
    """Inner validation of fit_set_model on the questions ix: the CE of log_softmax(base + Δ at the
    tick of cfg.pick) for the true rank, weighted by class_weights of their ranks, in float64 and without
    the γ term, computed in eval mode without gradients. Records each CE in fitted.inner_ce, keeps a CPU
    copy of the weights of the lowest one (ties: the earlier) and says stop after cfg.patience
    evaluations in a row without a lower CE."""

    def __init__(self, model, inputs, base, y, ix, cfg, fitted):  # 準備內層驗證的資料
        dev = cfg.device  # 執行裝置
        self.model, self.inputs, self.batch, self.patience, self.fitted = model, inputs, cfg.batch, cfg.patience, fitted  # 記下模型、輸入與設定
        self.pick = cfg.pick  # 評估所用的 tick
        self.ix = torch.from_numpy(ix).to(dev)  # 內層題目的索引
        self.base = torch.tensor(base[ix], dtype=torch.float64, device=dev)  # 內層題目的基礎 logits（float64）
        self.y = torch.tensor(y[ix], device=dev)  # 內層題目的等級
        self.w = torch.tensor(class_weights(y[ix]), dtype=torch.float64, device=dev)  # 內層題目的類別平衡權重
        self.best_state, self.bad = None, 0  # 最佳權重與未進步次數

    def ce(self):  # 計算內層 CE
        self.model.eval()  # 評估模式
        total = torch.zeros((), dtype=torch.float64, device=self.w.device)  # 加權 CE 的總和
        with torch.no_grad():  # 不需梯度
            for q in range(0, len(self.ix), self.batch):  # 逐批計算
                sl = slice(q, q + self.batch)  # 這一批的範圍
                logits = self.base[sl][:, None, :] + self.model(*self.inputs(self.ix[sl])).double()  # 各 tick 的 base + Δ：[B, T, 10]
                logp = _pick_torch(logits, self.pick).log_softmax(dim=-1)  # 選定 tick 的對數機率
                total += (self.w[sl] * -logp.gather(1, self.y[sl, None])[:, 0]).sum()  # 累加加權 CE
        self.model.train()  # 切回訓練模式
        return float(total / self.w.sum())  # 加權平均 CE

    def __call__(self, epochs_done):  # 評估一次並判斷是否停止
        """Evaluate after epochs_done (fractional) epochs; True when training should stop."""
        f, ce = self.fitted, self.ce()  # 訓練紀錄與這次的 CE
        if not math.isfinite(ce):  # CE 不是有限值
            raise FloatingPointError(f"non-finite inner validation CE ({ce}) at inner evaluation {len(f.inner_ce)} "  # 模型已發散，中止訓練
                                     f"(after {epochs_done:.4g} epochs): the model diverged")  # 錯誤訊息的後半
        f.inner_ce.append(ce)  # 記錄這次的 CE
        if f.best_eval is None or ce < f.inner_ce[f.best_eval]:  # 嚴格更低才算進步（同分保留較早者）
            f.best_eval, f.best_epoch, self.bad = len(f.inner_ce) - 1, float(epochs_done), 0  # 更新最佳評估並重設計數
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}  # 保留最佳權重的 CPU 副本
        else:  # 沒有進步
            self.bad += 1  # 累計未進步次數
        return self.bad >= self.patience  # 達到 patience 就停止

    def restore(self):  # 載回最佳權重
        self.model.load_state_dict(self.best_state)  # 換成 CE 最低時的權重


def fit_set_model(make: Callable[[int], nn.Module], data: dict, base, cfg: TrainCfg) -> Fitted:  # 訓練一個集合模型
    """Train make(d_in) on the packed training questions `data` against base logits [Q, 10]
    (cross-fitted, see crossfit_base_logits) with cfg; every question needs a rank (y >= 0) and the
    standardisation statistics must be finite (ValueError otherwise).

    cfg.inner_frac > 0 needs data["player_id"]: inner_players holds those players' questions out, and
    the standardisation statistics, class weights and training steps use the other (training) questions
    only. The cosine schedule spans epochs × n_batches steps (n_batches of the training questions). The
    inner CE (_EarlyStop, at the tick of cfg.pick) is evaluated before the first step (evaluation 0, Δ ≡ 0), then after every
    ceil(n_batches / evals_per_epoch) batches of an epoch and at its end (once per step); training stops
    after `patience` evaluations in a row without a lower CE, and the model returned holds the weights
    of the lowest one -- no refit on all players. A non-finite inner CE raises FloatingPointError.
    cfg.inner_frac = 0: `epochs` epochs over every question, the last weights; player_id is not needed."""
    if cfg.loss not in LOSSES:  # 未知的損失
        raise ValueError(f"unknown loss {cfg.loss!r}; known: {list(LOSSES)}")  # 回報可用的選項
    if cfg.pick not in PICKS:  # 未知的 pick
        raise ValueError(f"unknown pick {cfg.pick!r}; known: {list(PICKS)}")  # 回報可用的選項
    if cfg.epochs < 0 or cfg.batch < 1:  # epoch 數或批次大小不合理
        raise ValueError(f"need epochs >= 0 and batch >= 1, got {cfg.epochs}, {cfg.batch}")  # 回報設定值
    if not 0.0 <= cfg.inner_frac < 1.0 or cfg.evals_per_epoch < 1 or cfg.patience < 1:  # 提早停止的設定不合理
        raise ValueError(f"need 0 <= inner_frac < 1, evals_per_epoch >= 1 and patience >= 1, got {cfg.inner_frac}, "  # 回報設定值
                         f"{cfg.evals_per_epoch}, {cfg.patience}")  # 錯誤訊息的後半
    base = _check(data, base)  # 檢查輸入並取得 float64 基礎 logits
    y = np.asarray(data["y"], dtype=np.int64)  # 每題的等級
    if len(y) != len(base) or y.min() < 0 or y.max() >= N_CLASSES:  # 標籤缺漏或超出範圍
        raise ValueError("every training question needs a rank (0 <= y < 10)")  # 每一題都要有等級
    tokens, mask = data["tokens"], np.asarray(data["mask"], dtype=bool)  # token 與遮罩（不複製 token）
    inner_ix = None  # 內層題目索引，預設沒有
    if cfg.inner_frac > 0:  # 使用內層提早停止
        if "player_id" not in data:  # 沒有玩家 id 就無法依玩家留出
            raise ValueError("inner_frac > 0 holds out players: the data needs player_id (or pass inner_frac=0)")  # 回報需要 player_id
        held = inner_players(data["player_id"], y, cfg.inner_frac)  # 每題是否屬於內層玩家
        if not held.any():  # 沒有任何內層玩家
            raise ValueError("no inner validation players: every rank has fewer than 2 players (pass inner_frac=0)")  # 回報錯誤
        train_ix, inner_ix = np.flatnonzero(~held), np.flatnonzero(held)  # 訓練與內層的題目索引
    else:  # 不留出內層玩家
        train_ix = np.arange(len(y))  # 全部題目都用來訓練
    qglob = np.asarray(data["qglob"], dtype=np.float64)  # 題目層特徵（float64）
    # 有效 token 或 qglob 中的 NaN／inf 會讓該欄的統計量變成非有限值；檢查 O(D) 的統計量而不是檢查
    # 輸入，可以省下再掃一遍數 GB token 的時間；inf - inf 的警告無關緊要，因為出現任何一個這種值
    # 都會以下面的錯誤結束（內層題目中的 NaN／inf 不進統計量，而是進到內層 CE，
    # 同樣會丟出錯誤）
    with np.errstate(invalid="ignore", over="ignore"):  # 暫時關閉無效值與溢位的警告
        tok_mean, tok_std = _token_stats(tokens, mask, rows=None if inner_ix is None else train_ix)  # 只用訓練題目的 token 統計量
        glob_mean, glob_std = _standardise(qglob if inner_ix is None else qglob[train_ix])  # 只用訓練題目的題目層統計量
    for what, mean, std in (("token", tok_mean, tok_std), ("qglob", glob_mean, glob_std)):  # 逐一檢查兩組統計量
        bad = np.flatnonzero(~(np.isfinite(mean) & np.isfinite(std)))  # 非有限值的欄位
        if len(bad):  # 有出問題的欄位
            raise ValueError(f"{what} standardisation statistics are not finite in columns {bad.tolist()}: the training "  # 回報出問題的欄位
                             f"{'tokens of valid games' if what == 'token' else 'qglob'} hold NaN or inf")  # 錯誤訊息的後半
    with torch.random.fork_rng(devices=[]):  # 以種子初始化，不動到全域的亂數狀態
        torch.manual_seed(cfg.seed)  # 固定初始權重
        model = make(tokens.shape[2])  # 依 token 維度建立模組
    fitted = Fitted(model.to(cfg.device), tok_mean, tok_std, glob_mean, glob_std, cfg)  # 建立訓練結果物件
    # 每一題只放上裝置一次（在 CPU 上是零複製的視圖）；以索引陣列挑出訓練題與內層題，
    # 所以切分不會複製 token
    inputs = _Inputs(data, fitted, cfg.device)  # 裝置上的輸入
    base_t = torch.tensor(base, dtype=torch.float32, device=cfg.device)  # 訓練用的 float32 基礎 logits
    y_t = torch.tensor(y, device=cfg.device)  # 等級標籤
    w = np.zeros(len(y))  # 每題的訓練權重（內層題為 0）
    w[train_ix] = class_weights(y[train_ix])  # 訓練題的類別平衡權重
    w_t = torch.tensor(w, dtype=torch.float32, device=cfg.device)  # 權重張量

    Q, loss_fn = len(train_ix), LOSSES[cfg.loss]  # 訓練題數與損失函式
    n_batches = -(-Q // cfg.batch)  # 每個 epoch 的批次數（向上取整）
    every = -(-n_batches // cfg.evals_per_epoch)  # 每隔幾個批次評估一次（向上取整）
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)  # AdamW 最佳化器
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * n_batches))  # 涵蓋全部步數的 cosine 排程
    rng = np.random.RandomState(cfg.seed)  # 每個 epoch 打亂用的亂數
    train_t = torch.from_numpy(train_ix).to(cfg.device)  # 訓練題索引放上裝置
    early, stop, steps = None, False, 0  # 提早停止物件、停止旗標、已走的步數
    model.train()  # 訓練模式
    if inner_ix is not None:  # 有內層驗證時
        fitted.n_inner_players = len(np.unique(np.asarray(data["player_id"])[inner_ix]))  # 記錄內層玩家數
        fitted.n_inner_questions = len(inner_ix)  # 記錄內層題目數
        early = _EarlyStop(model, inputs, base, y, inner_ix, cfg, fitted)  # 建立內層驗證
        early(0.0)  # 評估 0：Δ ≡ 0，即零初始化的輸出層（第一次必定是最佳，不會停止）
    for epoch in range(cfg.epochs):  # 逐個 epoch 訓練
        order = train_t[torch.from_numpy(rng.permutation(Q)).to(cfg.device)]  # 這個 epoch 的題目順序
        total, seen = torch.zeros((), device=cfg.device), 0  # 累計損失與題數
        for b in range(n_batches):  # 逐批訓練
            ix = order[b * cfg.batch:(b + 1) * cfg.batch]  # 這一批的題目
            deltas = model(*inputs(ix))  # 各 tick 的修正量 Δ
            loss = loss_fn(base_t[ix][:, None] + deltas, deltas, y_t[ix], w_t[ix], cfg.gamma)  # 以 base + Δ 計算損失
            opt.zero_grad(set_to_none=True)  # 清除梯度
            loss.backward()  # 反向傳播
            opt.step()  # 更新權重
            sched.step()  # 更新學習率
            total += loss.detach() * len(ix)  # 依題數累計損失
            seen, steps = seen + len(ix), steps + 1  # 累計題數與步數
            if early is not None and ((b + 1) % every == 0 or b + 1 == n_batches):  # 到了評估點或 epoch 結束
                stop = early(epoch + (b + 1) / n_batches)  # 內層評估
                if stop:  # 達到停止條件
                    break  # 跳出批次迴圈
        fitted.losses.append(float(total) / seen)  # 記錄這個 epoch 的平均訓練損失；在 epoch 中途停止時只計已完成的批次
        if stop:  # 已停止
            break  # 結束訓練
    if early is not None:  # 有內層驗證時
        early.restore()  # 載回內層 CE 最低時的權重
        fitted.stopped_early = steps < cfg.epochs * n_batches  # 是否在走完全部步數前停止
    model.eval()  # 切到評估模式
    return fitted  # 回傳訓練結果


def predict_set_model(fitted: Fitted, data: dict, base, ticks=None, pick=None):  # 以集合模型預測
    """softmax(base + Δ) [Q, 10] (float64), Δ taken at the tick chosen by `pick` ("final": the last;
    "certain": each question's most certain tick; None: fitted.cfg.pick, the tick the model was selected
    at); ticks overrides the model's number of ticks."""
    pick = fitted.cfg.pick if pick is None else pick  # 預設用選模時的 tick
    if pick not in PICKS:  # 未知的 pick
        raise ValueError(f"unknown pick {pick!r}; known: {list(PICKS)}")  # 回報可用的選項
    base = _check(data, base)  # 檢查輸入並取得 float64 基礎 logits
    device, model = fitted.cfg.device, fitted.model  # 執行裝置與模型
    inputs = _Inputs(data, fitted, device)  # 裝置上的標準化輸入
    model.eval()  # 評估模式
    out = []  # 各批的 Δ
    with torch.no_grad():  # 不需梯度
        for q in range(0, len(base), fitted.cfg.batch):  # 逐批預測
            ix = torch.arange(q, min(q + fitted.cfg.batch, len(base)), device=device)  # 這一批的題目索引
            out.append(model(*inputs(ix), ticks=ticks).double().cpu().numpy())  # 各 tick 的 Δ（float64）
    logits = base[:, None, :] + np.concatenate(out)  # 各 tick 的 base + Δ
    return _softmax(PICKS[pick](logits))  # 取選定的 tick 後轉成機率


def crossfit_base_logits(make_static, X_by_fold: dict, y_by_fold: dict, X_eval):  # 交叉擬合基礎 logits
    """Cross-fitted base logits log(clip(proba, 1e-9)) of the static model make_static().

    Returns ({f: [n_f, 10]}, [n_eval, 10]): the logits of training fold f from a model fitted on the
    other training folds, and of X_eval from one fitted on all training folds. Training rows are
    always stacked in ascending fold order, so the evaluation logits come from exactly the fit the
    static rung makes on the same folds."""
    folds = sorted(X_by_fold)  # 依折號排序
    if sorted(y_by_fold) != folds:  # 特徵與標籤的折不一致
        raise ValueError(f"X_by_fold folds {folds} differ from y_by_fold folds {sorted(y_by_fold)}")  # 回報錯誤
    if len(folds) < 2:  # 折數不足
        raise ValueError(f"cross-fitting needs at least 2 training folds, got {folds}")  # 交叉擬合至少要兩折
    for f in folds:  # 逐折檢查
        if len(X_by_fold[f]) != len(y_by_fold[f]):  # 列數與標籤數不符
            raise ValueError(f"fold {f}: {len(X_by_fold[f])} rows of X, {len(y_by_fold[f])} labels")  # 回報錯誤

    def fit_on(fs):  # 在指定的折上擬合靜態模型
        return make_static().fit(np.concatenate([X_by_fold[f] for f in fs]), np.concatenate([y_by_fold[f] for f in fs]))  # 依折號順序堆疊後擬合

    def logits(proba):  # 機率轉成 logits
        return np.log(np.clip(proba, BASE_CLIP, None))  # 夾住下限後取對數

    by_fold = {f: logits(fit_on([g for g in folds if g != f]).predict_proba(X_by_fold[f])) for f in folds}  # 每折用其他折擬合的模型預測
    return by_fold, logits(fit_on(folds).predict_proba(X_eval))  # 評估資料用全部折擬合的模型預測
