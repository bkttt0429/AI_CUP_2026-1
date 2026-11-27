"""Sequence encoders over a game's moves, correcting the L1b sums (issue #19).

Every encoder maps a game's step sequence (aicup.seqdata: [T, STEP_DIM], padded with a mask) to a correction Δ [10] and
returns base + Δ, where base is the target's summed L1b log p -- so the "sum" encoder (Δ ≡ 0) is today's features exactly
and every other one starts there (zero-initialised output layer, as the L3 ladder's Δ models). Encoders: sum, conv (two
1-D convolutions, masked mean + max pooling), gru (a GRU, final state + masked mean of the states), ctm_seq (a Continuous
Thought Machine whose ticks are the moves: at step t the synapse reads the latent state and the encoded step, the
neuron-level models turn each neuron's history of pre-activations into its activation, and the output is read from the
synchronisation of neuron pairs accumulated recursively over the steps with learned decays -- the original CTM's
machinery with the data fed one move per tick instead of attended to), transformer (self-attention over the game's steps
with sinusoidal step positions, masked mean + max pooling) and mamba (Mamba blocks: a gated, causal selective state-space
scan over the steps, Gu and Dao 2023, last state + masked mean). Training: class-balanced cross-entropy of the
player's rank per game + gamma ‖Δ‖², Adam, inner early stopping by held-out players (settrain.inner_players), the best
inner CE kept; the step inputs are standardised with statistics of the training steps stored in the model.
"""
import hashlib  # 計算權重的 sha256
import io  # 在記憶體中序列化權重
import math  # 初始化範圍的 sqrt、位置編碼的 log 與批次數的 ceil
import time  # 計算訓練耗時
from dataclasses import asdict, dataclass  # 設定類別與轉成字典

import numpy as np  # 陣列運算
import torch  # PyTorch 主套件
from torch import nn  # 神經網路模組
from torch.nn import functional as F  # 激活與損失函式

from aicup.aggregators import _NLM, zero_linear  # 神經元層級模型與零初始化輸出層
from aicup.metric import LEVELS  # 10 個等級
from aicup.rankmodel import class_weights  # 類別平衡的樣本權重
from aicup.seqdata import STEP_DIM, pad_batch  # 步驟維度與補齊批次
from aicup.settrain import INNER_SEED, inner_players  # 內層留出玩家的抽選

ARCHS = ("sum", "conv", "gru", "ctm_seq", "transformer", "mamba")  # 可用的編碼器結構
N_CLASSES = len(LEVELS)  # 等級數（10）


@dataclass(frozen=True)  # 不可變的設定物件
class SeqCfg:  # 序列編碼器的訓練設定
    arch: str = "gru"  # 編碼器結構
    epochs: int = 10  # 最多訓練幾個 epoch
    batch: int = 256  # 每批棋局數
    lr: float = 1e-3  # 學習率
    weight_decay: float = 1e-4  # 權重衰減
    gamma: float = 1e-3  # ‖Δ‖² 懲罰的係數
    seed: int = 0  # 亂數種子
    device: str = "cuda"  # 執行裝置
    inner_frac: float = 0.1  # 每個等級留作內層驗證的玩家比例
    evals_per_epoch: int = 4  # 每個 epoch 評估幾次
    patience: int = 8  # 連續幾次評估沒進步就停止
    hidden: int = 64  # 隱藏層寬度
    d_latent: int = 128  # CTM 的神經元數
    history: int = 8  # CTM 每個神經元保留的前激活歷史長度
    nlm_hidden: int = 16  # 神經元層級模型的隱藏寬度
    n_sync: int = 256  # CTM 讀出用的神經元對數
    layers: int = 2  # transformer 與 mamba 的層數
    heads: int = 4  # 注意力頭數
    d_state: int = 16  # Mamba 的狀態維度
    d_conv: int = 4  # Mamba 的卷積核寬度
    expand: int = 2  # Mamba 的擴展倍數


class _Encoder(nn.Module):  # 所有編碼器的共同基底
    """Shared: input standardisation buffers and the Δ convention (forward returns base + Δ)."""

    def __init__(self, d_in):  # 建立標準化參數
        super().__init__()  # 初始化 nn.Module
        self.register_buffer("x_mean", torch.zeros(d_in))  # 輸入平均，隨權重存檔
        self.register_buffer("x_std", torch.ones(d_in))  # 輸入標準差，隨權重存檔

    def standardise(self, X):  # 標準化輸入
        return (X - self.x_mean) / self.x_std  # 減去平均再除以標準差

    def forward(self, X, mask, base):  # 前向計算
        return base + self.delta(self.standardise(X), mask).to(base.dtype)  # 回傳 base + Δ；base 是 float64：sum 編碼器會精確回傳它


class SumEncoder(_Encoder):  # Δ 恆為 0 的對照編碼器
    def __init__(self, d_in, cfg):  # 建立對照編碼器
        super().__init__(d_in)  # 建立標準化參數
        self.unused = nn.Parameter(torch.zeros(1))  # 讓最佳化器有一個參數；它從不進入輸出

    def delta(self, X, mask):  # 修正量恆為 0
        return torch.zeros(X.shape[0], N_CLASSES, dtype=X.dtype, device=X.device) + 0.0 * self.unused  # 乘 0 接上參數，使反向傳播可執行


def _masked_mean(h, mask):  # 只對有效步驟取平均
    m = mask.unsqueeze(-1).to(h.dtype)  # 遮罩擴成 [B, T, 1]
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # 有效步驟的總和除以步數（至少 1）


def _masked_max(h, mask):  # 只對有效步驟取最大值
    return h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values.masked_fill(~mask.any(dim=1, keepdim=True), 0.0)  # 補齊處設為 -inf；整局沒有步驟時填 0


class ConvEncoder(_Encoder):  # 一維卷積編碼器
    def __init__(self, d_in, cfg):  # 建立卷積層
        super().__init__(d_in)  # 建立標準化參數
        h = cfg.hidden  # 隱藏寬度
        self.conv1, self.conv2 = nn.Conv1d(d_in, h, 5, padding=2), nn.Conv1d(h, h, 5, padding=2)  # 兩層卷積核大小 5 的一維卷積（padding 2 保持序列長度）
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())  # 池化後的隱藏層
        self.out = zero_linear(h)  # 零初始化的輸出層，起點 Δ = 0

    def delta(self, X, mask):  # 計算修正量
        mt = mask.unsqueeze(1).to(X.dtype)  # [B, 1, T]；每次卷積前把補齊的步驟歸零，如同卷積本身的補零
        h = F.relu(self.conv1(X.transpose(1, 2) * mt)) * mt  # 第一層卷積
        h = (F.relu(self.conv2(h)) * mt).transpose(1, 2)  # 第二層卷積並轉回 [B, T, h]
        return self.out(self.head(torch.cat([_masked_mean(h, mask), _masked_max(h, mask)], dim=-1)))  # 平均＋最大值池化後輸出


class GRUEncoder(_Encoder):  # GRU 編碼器
    def __init__(self, d_in, cfg):  # 建立 GRU
        super().__init__(d_in)  # 建立標準化參數
        h = cfg.hidden  # 隱藏寬度
        self.gru = nn.GRU(d_in, h, batch_first=True)  # 單層 GRU
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())  # 讀出前的隱藏層
        self.out = zero_linear(h)  # 零初始化的輸出層

    def delta(self, X, mask):  # 計算修正量
        lengths = mask.sum(dim=1).clamp_min(1).cpu()  # 每局的有效長度（至少 1）
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)  # 打包以略過補齊的步驟
        out, h_n = self.gru(packed)  # 執行 GRU
        hs, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=X.shape[1])  # 還原成補齊的狀態序列
        return self.out(self.head(torch.cat([h_n[-1], _masked_mean(hs, mask)], dim=-1)))  # 最終狀態＋狀態平均後輸出


class CTMSeqEncoder(_Encoder):  # 每手一個 tick 的 CTM 編碼器
    """CTM with one tick per move. State: the latent z [B, n] and the history A [B, n, history] of pre-activations;
    per tick a = synapse([z, enc(x_t)]) (Linear -> GLU -> LayerNorm, the depth-1 synapse), A <- (A[1:], a), z = NLM(A).
    Output synchronisation of n_sync neuron pairs (i <= j, a seeded random draw) accumulated recursively with learned
    decays d_p = exp(-r_p), r_p in [0, 15]: alpha <- d alpha + z_i z_j, beta <- d beta + 1, S = alpha / sqrt(beta)
    (the closed form of the original's exponentially weighted sum). Padded steps leave the state unchanged, so the
    readout W_out(S) (zero-initialised) is taken after each game's last move."""

    def __init__(self, d_in, cfg):  # 建立 CTM 的各個部件
        super().__init__(d_in)  # 建立標準化參數
        n, hist = cfg.d_latent, cfg.history  # 神經元數與歷史長度
        pairs = torch.triu_indices(n, n).T  # 所有 i <= j 的神經元對
        perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(cfg.seed))  # 以種子固定的隨機順序
        self.register_buffer("idx", pairs[perm[:cfg.n_sync]].clone())  # 取前 n_sync 對，隨權重存檔
        self.r = nn.Parameter(torch.zeros(cfg.n_sync))  # 每一對的衰減參數 r
        b_a, b_z = 1 / math.sqrt(n + hist), 1 / math.sqrt(n)  # 初始狀態的均勻分布範圍
        self.A0 = nn.Parameter(torch.empty(n, hist).uniform_(-b_a, b_a))  # 可學的初始前激活歷史
        self.z0 = nn.Parameter(torch.empty(n).uniform_(-b_z, b_z))  # 可學的初始潛在向量
        self.enc = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU())  # 步驟輸入的編碼層
        self.synapse = nn.Sequential(nn.Linear(n + cfg.hidden, 2 * n), nn.GLU(), nn.LayerNorm(n))  # 深度 1 的 synapse
        self.nlm = _NLM(n, hist, cfg.nlm_hidden)  # 神經元層級模型
        self.out = zero_linear(cfg.n_sync)  # 由同步量讀出的零初始化輸出層

    def delta(self, X, mask):  # 逐手遞迴計算修正量
        self.r.data.clamp_(0.0, 15.0)  # 把 r 限制在 [0, 15]
        decay = torch.exp(-self.r)  # 每一對的衰減係數 d = exp(-r)
        B = X.shape[0]  # 批次大小
        x = self.enc(X)  # 編碼每一步：[B, T, hidden]
        A = self.A0.expand(B, -1, -1)  # 每局從初始歷史出發
        z = self.z0.expand(B, -1)  # 每局從初始潛在向量出發
        i, j = self.idx[:, 0], self.idx[:, 1]  # 神經元對的兩端
        alpha = (z[:, i] * z[:, j])  # 同步量分子的初值
        beta = torch.ones_like(alpha)  # 同步量分母的初值
        for t in range(X.shape[1]):  # 一手一個 tick
            m = mask[:, t].unsqueeze(-1)  # 這一步是否有效
            a = self.synapse(torch.cat([z, x[:, t]], dim=-1))  # 新的前激活
            A_new = torch.cat([A[:, :, 1:], a[:, :, None]], dim=-1)  # 歷史左移並接上新值
            z_new = self.nlm(A_new)  # 由歷史算出新的激活
            alpha_new = decay * alpha + z_new[:, i] * z_new[:, j]  # 遞迴累積同步量
            beta_new = decay * beta + 1.0  # 遞迴累積權重和
            A = torch.where(m.unsqueeze(-1), A_new, A)  # 補齊的步驟不更新歷史
            z = torch.where(m, z_new, z)  # 補齊的步驟不更新激活
            alpha = torch.where(m, alpha_new, alpha)  # 補齊的步驟不更新分子
            beta = torch.where(m, beta_new, beta)  # 補齊的步驟不更新分母
        return self.out(alpha / beta.sqrt())  # 由同步量 S 讀出 Δ


def _step_positions(T, d, like):  # 步驟位置的正弦編碼
    """[T, d] sinusoidal encodings of the step indices (computed per batch, so a game's encoding does not depend on its padding)."""
    pos = torch.arange(T, dtype=like.dtype, device=like.device).unsqueeze(-1)  # 步驟索引 [T, 1]
    freq = torch.exp(torch.arange(0, d, 2, dtype=like.dtype, device=like.device) * (-math.log(10000.0) / d))  # 各維的頻率
    out = torch.zeros(T, d, dtype=like.dtype, device=like.device)  # 輸出陣列
    out[:, 0::2] = torch.sin(pos * freq)  # 偶數維用 sin
    out[:, 1::2] = torch.cos(pos * freq)[:, :d // 2]  # 奇數維用 cos（d 為奇數時截掉多出的一欄）
    return out  # 回傳位置編碼


class TransformerEncoder(_Encoder):  # 自注意力編碼器
    """Pre-norm self-attention over the steps, padded steps masked out; no dropout, like the other encoders."""

    def __init__(self, d_in, cfg):  # 建立 transformer 各層
        super().__init__(d_in)  # 建立標準化參數
        h = cfg.hidden  # 隱藏寬度
        self.proj = nn.Linear(d_in, h)  # 把步驟投影到隱藏寬度
        layer = nn.TransformerEncoderLayer(h, cfg.heads, dim_feedforward=2 * h, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)  # 單層設定：pre-norm、不用 dropout
        self.layers = nn.TransformerEncoder(layer, cfg.layers, norm=nn.LayerNorm(h), enable_nested_tensor=False)  # 堆疊 cfg.layers 層
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())  # 池化後的隱藏層
        self.out = zero_linear(h)  # 零初始化的輸出層

    def delta(self, X, mask):  # 計算修正量
        h = self.proj(X) + _step_positions(X.shape[1], self.proj.out_features, X)  # 投影後加上位置編碼
        h = self.layers(h, src_key_padding_mask=~mask)  # 自注意力，遮掉補齊的步驟
        return self.out(self.head(torch.cat([_masked_mean(h, mask), _masked_max(h, mask)], dim=-1)))  # 平均＋最大值池化後輸出


class _MambaBlock(nn.Module):  # 一個 Mamba 區塊
    """Pre-norm residual block: expand, split into a gate and a path through a causal depthwise convolution and the selective
    state-space scan x_t = exp(delta_t A) x_{t-1} + delta_t B_t u_t, y_t = C_t x_t + D u_t, whose delta, B and C depend on the input."""

    def __init__(self, d, d_state, d_conv, expand):  # 建立區塊的各層
        super().__init__()  # 初始化 nn.Module
        e = expand * d  # 擴展後的寬度
        self.d_state = d_state  # 狀態維度
        self.norm = nn.LayerNorm(d)  # pre-norm 的正規化層
        self.proj_in = nn.Linear(d, 2 * e)  # 投影成路徑與閘門兩份
        self.conv = nn.Conv1d(e, e, d_conv, padding=d_conv - 1, groups=e)  # 逐通道卷積；只取前 T 個輸出即為因果
        self.params = nn.Linear(e, e + 2 * d_state)  # 由輸入產生 delta、B、C
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(e, 1))  # A 的對數，初值為 log 1..N
        self.D = nn.Parameter(torch.ones(e))  # 直通項 D
        self.proj_out = nn.Linear(2 * e, d)  # 投影回原本的寬度

    def forward(self, x):  # 區塊的前向計算
        T = x.shape[1]  # 序列長度
        u, gate = self.proj_in(self.norm(x)).chunk(2, dim=-1)  # 拆成路徑與閘門
        u = F.silu(self.conv(u.transpose(1, 2))[:, :, :T].transpose(1, 2))  # 因果卷積：只留前 T 個輸出
        delta, B, C = self.params(u).split([u.shape[-1], self.d_state, self.d_state], dim=-1)  # 依輸入產生的掃描參數
        delta = F.softplus(delta)  # 讓步長保持為正
        step = torch.exp(-delta.unsqueeze(-1) * torch.exp(self.A_log))  # 離散化的衰減 exp(delta_t A)：[B, T, e, N]
        drive = torch.einsum("bte,btn,bte->bten", delta, B, u)  # 每一步的輸入項 delta_t B_t u_t
        state = torch.zeros(x.shape[0], u.shape[-1], self.d_state, dtype=x.dtype, device=x.device)  # 初始狀態為 0
        ys = []  # 各步的輸出
        for t in range(T):  # 依序掃描每一步
            state = step[:, t] * state + drive[:, t]  # 狀態更新
            ys.append(torch.einsum("ben,bn->be", state, C[:, t]))  # 讀出 C_t x_t
        y = torch.stack(ys, dim=1) + u * self.D  # 加上直通項
        return x + self.proj_out(torch.cat([F.silu(y), F.silu(gate)], dim=-1))  # 路徑與閘門串接投影後做殘差連接


class MambaEncoder(_Encoder):  # Mamba 編碼器
    """Causal, so a game's padded tail cannot reach its steps; the readout is the state after its last move + the masked mean."""

    def __init__(self, d_in, cfg):  # 建立 Mamba 各層
        super().__init__(d_in)  # 建立標準化參數
        h = cfg.hidden  # 隱藏寬度
        self.proj = nn.Linear(d_in, h)  # 把步驟投影到隱藏寬度
        self.blocks = nn.ModuleList(_MambaBlock(h, cfg.d_state, cfg.d_conv, cfg.expand) for _ in range(cfg.layers))  # 堆疊 cfg.layers 個區塊
        self.norm = nn.LayerNorm(h)  # 最後的正規化
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())  # 讀出前的隱藏層
        self.out = zero_linear(h)  # 零初始化的輸出層

    def delta(self, X, mask):  # 計算修正量
        h = self.proj(X)  # 把步驟投影到隱藏寬度
        for block in self.blocks:  # 依序通過各區塊
            h = block(h)  # 通過一個 Mamba 區塊（區塊內含殘差連接）
        h = self.norm(h)  # 最後的 LayerNorm
        last = (mask.sum(dim=1).clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])  # 每局最後一個有效步驟的索引
        return self.out(self.head(torch.cat([h.gather(1, last).squeeze(1), _masked_mean(h, mask)], dim=-1)))  # 最後狀態＋遮罩平均後輸出


ENCODERS = {"sum": SumEncoder, "conv": ConvEncoder, "gru": GRUEncoder, "ctm_seq": CTMSeqEncoder, "transformer": TransformerEncoder, "mamba": MambaEncoder}  # 結構名稱對應編碼器類別


def make_encoder(cfg, d_in=STEP_DIM):  # 依設定建立編碼器
    if cfg.arch not in ENCODERS:  # 未知的結構
        raise ValueError(f"unknown arch {cfg.arch!r}; one of {ARCHS}")  # 回報可用的選項
    torch.manual_seed(cfg.seed)  # 固定初始權重的亂數
    return ENCODERS[cfg.arch](d_in, cfg)  # 建立對應的編碼器


def _batches(seq, idx, batch, rng=None):  # 把棋局索引切成批次
    idx = np.asarray(idx, dtype=np.int64)  # 轉成整數陣列
    if rng is not None:  # 訓練時打亂順序
        idx = idx[rng.permutation(len(idx))]  # 隨機排列
    for i in range(0, len(idx), batch):  # 依批次大小切分
        yield idx[i:i + batch]  # 產生一批索引


def _to_device(seq, idx, device):  # 取出一批並搬到裝置上
    X, mask, base, y = pad_batch(seq, idx)  # 補齊成陣列
    return (torch.from_numpy(X).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(base).to(device, torch.float64),  # 輸入、遮罩與雙精度的 base
            torch.from_numpy(y).to(device))  # 等級標籤


def _balanced_ce(model, seq, idx, cfg, w_of):  # 評估類別平衡交叉熵
    """Class-balanced CE (mean over the games of `idx`, weights w_of per game) and the mean ‖Δ‖²; no grad."""
    model.eval()  # 切到評估模式
    tot, wsum, dsq, n = 0.0, 0.0, 0.0, 0  # 加權損失、權重和、Δ 平方和、局數
    with torch.no_grad():  # 評估時不需梯度
        for b in _batches(seq, idx, cfg.batch):  # 逐批計算
            X, mask, base, y = _to_device(seq, b, cfg.device)  # 搬到裝置
            logits = model(X, mask, base)  # 取得 base + Δ
            w = torch.from_numpy(w_of[b]).to(cfg.device, torch.float64)  # 這批棋局的權重
            tot += float((F.cross_entropy(logits, y, reduction="none") * w).sum())  # 累加加權交叉熵
            wsum += float(w.sum())  # 累加權重
            dsq += float((logits - base).pow(2).sum(dim=-1).sum())  # 累加 ‖Δ‖²
            n += len(b)  # 累加局數
    return tot / max(wsum, 1e-12), dsq / max(n, 1)  # 加權平均 CE 與平均 ‖Δ‖²


def fit_seq_encoder(seq, cfg, log=print):  # 訓練一個序列編碼器
    """Train an encoder on the games of `seq` (a fold_sequences / concat_sequences dict); returns (model, meta).
    Inner validation: inner_frac of each rank's players (settrain.inner_players, INNER_SEED), class-balanced CE evaluated
    at step 0 and evals_per_epoch times per epoch, the best weights kept, stop after `patience` evaluations without a
    lower CE. Weights: every rank the same total weight over the training games."""
    t0 = time.time()  # 開始計時
    device = cfg.device  # 執行裝置
    inner = inner_players(seq["player_id"], seq["rank_idx"], cfg.inner_frac, INNER_SEED) if cfg.inner_frac > 0 else np.zeros(len(seq["rank_idx"]), dtype=bool)  # 標出內層驗證玩家的棋局
    train_idx, inner_idx = np.flatnonzero(~inner), np.flatnonzero(inner)  # 訓練與內層的棋局索引
    if len(train_idx) == 0:  # 沒有訓練資料
        raise ValueError("no training games")  # 無法訓練
    model = make_encoder(cfg).to(device)  # 建立編碼器
    # 以訓練棋局的步驟計算步驟輸入的標準化參數（旗標欄為常數時標準差維持 1）
    steps = np.concatenate([seq["X"][seq["offsets"][g]:seq["offsets"][g + 1]] for g in train_idx]) if len(train_idx) else seq["X"]  # 收集訓練棋局的所有步驟
    mean, std = steps.mean(axis=0), steps.std(axis=0)  # 每欄的平均與標準差
    std[std < 1e-6] = 1.0  # 常數欄不縮放
    model.x_mean.copy_(torch.from_numpy(mean.astype(np.float32)))  # 平均存入模型
    model.x_std.copy_(torch.from_numpy(std.astype(np.float32)))  # 標準差存入模型
    w_all = np.zeros(len(seq["rank_idx"]), dtype=np.float32)  # 訓練用的每局權重
    w_all[train_idx] = class_weights(seq["rank_idx"][train_idx]).astype(np.float32)  # 訓練棋局的類別平衡權重
    w_inner = np.zeros_like(w_all)  # 內層評估用的權重
    if len(inner_idx):  # 有內層棋局時
        w_inner[inner_idx] = class_weights(seq["rank_idx"][inner_idx]).astype(np.float32)  # 內層棋局的類別平衡權重
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)  # Adam 最佳化器
    rng = np.random.RandomState(cfg.seed)  # 打亂批次用的亂數
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch)  # 每個 epoch 的更新步數（批次數）
    eval_every = max(1, steps_per_epoch // max(cfg.evals_per_epoch, 1))  # 每隔幾個更新步做一次內層評估
    history, best, since, step, stopped = [], None, 0, 0, False  # 評估紀錄、最佳狀態、未進步次數、步數、是否提早停止
    losses = []  # 每個 epoch 的平均訓練損失

    def snapshot():  # 複製目前的權重
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # 存成 CPU 上的副本

    def evaluate():  # 內層評估並判斷是否停止
        nonlocal best, since  # 要更新外層的最佳紀錄與未進步次數
        if not len(inner_idx):  # 沒有內層棋局
            return False  # 無法評估，不停止
        ce, dsq = _balanced_ce(model, seq, inner_idx, cfg, w_inner)  # 內層 CE 與 ‖Δ‖²
        model.train()  # 切回訓練模式：評估發生在 epoch 之中，而 cuDNN 的 RNN 在 eval 模式下不允許反向傳播
        epoch = step / steps_per_epoch  # 目前的 epoch 進度
        history.append((epoch, ce, dsq))  # 記錄評估結果
        if best is None or ce < best["ce"]:  # CE 創新低
            best, since = {"ce": ce, "dsq": dsq, "step": step, "epoch": epoch, "state": snapshot()}, 0  # 更新最佳狀態並重設計數
        else:  # 沒有進步
            since += 1  # 累計未進步次數
        log(f"  epoch {epoch:.2f} inner CE {ce:.4f} mean|Δ|² {dsq:.3f} (best {best['ce']:.4f} @ {best['epoch']:.2f})", flush=True)  # 輸出評估進度
        return since >= cfg.patience  # 達到耐心上限就停止

    evaluate()  # 第 0 步先評估一次
    for epoch in range(cfg.epochs):  # 逐個 epoch 訓練
        model.train()  # 訓練模式
        run, nb = 0.0, 0  # 累計損失與批次數
        for b in _batches(seq, train_idx, cfg.batch, rng):  # 打亂後逐批訓練
            X, mask, base, y = _to_device(seq, b, device)  # 搬到裝置
            w = torch.from_numpy(w_all[b]).to(device, torch.float64)  # 這批棋局的權重
            logits = model(X, mask, base)  # 取得 base + Δ
            loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum() + cfg.gamma * (logits - base).pow(2).sum(dim=-1).mean()  # 加權交叉熵 + gamma ‖Δ‖²
            if not torch.isfinite(loss):  # 損失出現非有限值
                raise FloatingPointError(f"non-finite loss at step {step}")  # 立即中止
            opt.zero_grad(set_to_none=True)  # 清除梯度
            loss.backward()  # 反向傳播
            opt.step()  # 更新權重
            run += loss.item()  # 累計損失
            nb += 1  # 累計批次數
            step += 1  # 累計步數
            if step % eval_every == 0 and evaluate():  # 定期評估，達到停止條件時
                stopped = True  # 記下提早停止
                break  # 跳出批次迴圈
        losses.append(run / max(nb, 1))  # 記錄這個 epoch 的平均損失
        if stopped:  # 已提早停止
            break  # 結束訓練
    if best is not None:  # 有內層評估的最佳狀態
        model.load_state_dict(best["state"])  # 載回最佳權重
    model.eval()  # 切到評估模式
    meta = {"cfg": asdict(cfg), "n_train_games": int(len(train_idx)), "n_inner_games": int(len(inner_idx)),  # 設定與棋局數
            "n_inner_players": int(len(np.unique(seq["player_id"][inner_idx]))) if len(inner_idx) else 0,  # 內層玩家數
            "inner_ce_init": history[0][1] if history else None, "inner_ce_best": best["ce"] if best else None,  # 初始與最佳的內層 CE
            "inner_dsq_best": best["dsq"] if best else None, "best_epoch": best["epoch"] if best else float(cfg.epochs),  # 最佳時的 ‖Δ‖² 與 epoch
            "stopped_early": bool(stopped), "history": [(float(e), float(c), float(d)) for e, c, d in history], "losses": losses,  # 是否提早停止、評估紀錄與損失
            "steps_run": int(step), "n_params": int(sum(p.numel() for p in model.parameters())), "wall_seconds": time.time() - t0,  # 步數、參數量與耗時
            "step_names_mean": mean.tolist(), "step_names_std": std.tolist()}  # 步驟輸入的標準化參數
    return model, meta  # 回傳模型與紀錄


def predict_seq(model, seq, cfg):  # 對所有棋局預測
    """base + Δ for every game of `seq`: [G, 10] float64."""
    model.eval()  # 評估模式
    out = np.zeros((len(seq["rank_idx"]), N_CLASSES))  # 輸出陣列
    with torch.no_grad():  # 不需梯度
        for b in _batches(seq, np.arange(len(seq["rank_idx"])), cfg.batch):  # 依原順序逐批預測
            X, mask, base, _ = _to_device(seq, b, cfg.device)  # 搬到裝置
            out[b] = model(X, mask, base).cpu().numpy()  # 寫入這批的結果
    return out  # 回傳每局 10 個等級的 base + Δ


def state_sha256(model):  # 計算權重的雜湊
    buf = io.BytesIO()  # 記憶體緩衝區
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)  # 序列化 CPU 上的權重
    return hashlib.sha256(buf.getvalue()).hexdigest()  # 回傳 sha256


def save_encoder(model, cfg, meta, path):  # 編碼器存檔
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "cfg": asdict(cfg), "meta": meta, "d_in": int(model.x_mean.numel())}, path)  # 存下權重、設定、紀錄與輸入維度


def load_encoder(path, device="cpu"):  # 載入編碼器
    """(model, cfg, meta) from save_encoder's file."""
    d = torch.load(path, map_location=device, weights_only=False)  # 讀取檔案（含設定字典）
    cfg = SeqCfg(**{**d["cfg"], "device": device})  # 還原設定並改用指定裝置
    model = make_encoder(cfg, d["d_in"]).to(device)  # 依設定重建編碼器
    model.load_state_dict(d["state"])  # 載入權重
    model.eval()  # 評估模式
    return model, cfg, d["meta"]  # 回傳模型、設定與紀錄
