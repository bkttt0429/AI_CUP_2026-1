"""Set aggregators of the L3 ladder: a question's game tokens -> corrections Δ to the base logits.

Every model is built as Model(d_in, ...) with d_in the token width (the last dimension of
ladder_data.pack_questions tokens) and maps

    forward(tokens [B, M, d_in], mask [B, M] bool (True = a game), qglob [B, 6], ticks=None) -> deltas [B, T, 10]

on standardised inputs (aicup.settrain). The output layer starts at zero weights and bias, so Δ ≡ 0
before training. The models are permutation invariant over the games and use no positional
encoding; padding rows are zeroed on entry and excluded from every pooling and attention
(key_padding_mask = ~mask), so neither their content nor their number changes the output.
T = 1 models ignore `ticks`; XAttnGRU and CTM run a recurrence of `ticks` steps (default 8) and return the Δ
of every tick. Every question needs at least one game.
"""
import math  # 計算初始化範圍用的平方根

import torch  # 張量運算與參數
from torch import nn  # 神經網路層

N_CLASSES = 10  # 競賽的 10 個等級（D、C、B、A、1D–6D）


def zero_linear(n_in, n_out=N_CLASSES):  # 建立零初始化的輸出層
    """The output layer: Linear with weight and bias at 0, so the first Δ is exactly 0."""
    layer = nn.Linear(n_in, n_out)  # 一般的線性層
    nn.init.zeros_(layer.weight)  # 權重歸零
    nn.init.zeros_(layer.bias)  # 偏置歸零，訓練前輸出恆為 0
    return layer  # 回傳輸出層


def _mlp(n_in, hidden, n_hidden):  # 組出 MLP 頭（T = 1 的四個聚合器使用；XAttnGRU 與 CTM 只用 zero_linear）
    """n_hidden ReLU layers of width `hidden`, then the zero-initialised output layer."""
    layers, d = [], n_in  # 層清單與目前的輸入寬度
    for _ in range(n_hidden):  # 逐層加入隱藏層
        layers += [nn.Linear(d, hidden), nn.ReLU()]  # 線性層接 ReLU
        d = hidden  # 下一層的輸入寬度
    return nn.Sequential(*layers, zero_linear(d))  # 隱藏層後接零初始化的輸出層


def _clean(tokens, mask):  # 把補白列歸零
    """Padding rows set to 0 (masked_fill, so even non-finite padding cannot leak through 0 * x)."""
    return tokens.masked_fill(~mask[..., None], 0.0)  # 非有效局的列填 0


def masked_mean(x, mask):  # 只對有效列取平均
    """Mean of x [B, M, d] over the valid rows -> [B, d]."""
    m = mask[..., None]  # 擴成可廣播的遮罩
    return x.masked_fill(~m, 0.0).sum(dim=1) / m.sum(dim=1).clamp_min(1).to(x.dtype)  # 總和除以有效局數（至少 1）


def masked_max(x, mask):  # 只對有效列取最大值
    """Max of x [B, M, d] over the valid rows -> [B, d]."""
    return x.masked_fill(~mask[..., None], float("-inf")).amax(dim=1)  # 補白設為 -inf 後取最大值


def n_params(model):  # 計算模型的參數量
    """Number of parameters (buffers excluded)."""
    return sum(p.numel() for p in model.parameters())  # 加總每個參數張量的元素數


class GameEncoder(nn.Module):  # 逐局 token 的編碼器
    """Per-game token: Linear(d_in, d_tok) -> ReLU -> Linear(d_tok, d_tok) -> LayerNorm."""

    def __init__(self, d_in, d_tok=128):  # 建立兩層線性層與 LayerNorm
        super().__init__()  # 初始化 nn.Module
        self.net = nn.Sequential(nn.Linear(d_in, d_tok), nn.ReLU(), nn.Linear(d_tok, d_tok), nn.LayerNorm(d_tok))  # 逐局編碼網路

    def forward(self, tokens):  # 前向計算
        return self.net(tokens)  # 每局各自編碼，局與局之間不互動


class PooledDelta(nn.Module):  # r2_pooled_delta：沒有局間交互的對照組
    """r2_pooled_delta, the no-interaction control: mean of the standardised input tokens over the
    valid games, concatenated with qglob, -> MLP(hidden) -> Δ; T = 1. No game encoder."""

    def __init__(self, d_in, d_glob=6, hidden=256):  # 建立單一隱藏層的 MLP 頭
        super().__init__()  # 初始化 nn.Module
        self.head = _mlp(d_in + d_glob, hidden, 1)  # 輸入為平均 token 加上 qglob

    def forward(self, tokens, mask, qglob, ticks=None):  # 前向計算（忽略 ticks）
        return self.head(torch.cat([masked_mean(_clean(tokens, mask), mask), qglob], dim=-1))[:, None]  # 平均池化後算 Δ，補上 T = 1 的維度


class QglobDelta(nn.Module):  # r3_null：只看 qglob 的對照組
    """r3_null: Δ from qglob alone, MLP(hidden); T = 1. Measures what re-calibrating on top of the
    base gains by itself. d_in is accepted for the shared interface and unused."""

    def __init__(self, d_in, d_glob=6, hidden=64):  # d_in 只為共用介面而保留，不使用
        super().__init__()  # 初始化 nn.Module
        self.head = _mlp(d_glob, hidden, 1)  # 只吃 qglob 的 MLP

    def forward(self, tokens, mask, qglob, ticks=None):  # 前向計算，不讀 token
        return self.head(qglob)[:, None]  # 只由題目層全域特徵算 Δ


class DeepSets(nn.Module):  # r3_deepsets：平均與最大值池化
    """GameEncoder -> mean and max over the valid tokens -> concat qglob -> MLP(hidden, hidden) -> Δ; T = 1."""

    def __init__(self, d_in, d_glob=6, d_tok=128, hidden=256):  # 建立編碼器與兩層隱藏層的頭
        super().__init__()  # 初始化 nn.Module
        self.enc = GameEncoder(d_in, d_tok)  # 逐局編碼器
        self.head = _mlp(2 * d_tok + d_glob, hidden, 2)  # 平均、最大值與 qglob 串接後的頭

    def forward(self, tokens, mask, qglob, ticks=None):  # 前向計算（忽略 ticks）
        h = self.enc(_clean(tokens, mask))  # 補白歸零後逐局編碼
        return self.head(torch.cat([masked_mean(h, mask), masked_max(h, mask), qglob], dim=-1))[:, None]  # 兩種池化串接後算 Δ


class _MAB(nn.Module):  # Set Transformer 的多頭注意力區塊
    """Multihead attention block of the Set Transformer (Lee et al. 2019), post-norm:
    H = LN(Q + MHA(Q, K, K)), out = LN(H + rFF(H)), with the padded keys masked."""

    def __init__(self, d, heads):  # 建立注意力、兩個 LayerNorm 與前饋層
        super().__init__()  # 初始化 nn.Module
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)  # 多頭注意力
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)  # 兩次殘差相加後的正規化
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # 逐列前饋層 rFF

    def forward(self, q, kv, key_padding_mask):  # q 讀取 kv，遮掉補白的 key
        h = self.ln1(q + self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)[0])  # 注意力加殘差後正規化
        return self.ln2(h + self.ff(h))  # 前饋加殘差後正規化


class SetTransformer(nn.Module):  # r3_settf；layers=0 時即 r3_pma
    """GameEncoder -> `layers` SABs (self-attention, key_padding_mask = ~mask) -> PMA with one learned
    seed -> concat qglob -> MLP(hidden, hidden) -> Δ; T = 1. The head is the DeepSets head, so the
    two differ only in how the tokens are combined."""

    def __init__(self, d_in, d_glob=6, d_tok=128, heads=4, layers=2, hidden=256):  # 建立編碼器、SAB、PMA 與頭
        super().__init__()  # 初始化 nn.Module
        self.enc = GameEncoder(d_in, d_tok)  # 逐局編碼器
        self.sabs = nn.ModuleList([_MAB(d_tok, heads) for _ in range(layers)])  # 局與局互相注意的 SAB 層
        self.seed = nn.Parameter(torch.randn(1, 1, d_tok) / math.sqrt(d_tok))  # PMA 的學習查詢向量
        self.pma = _MAB(d_tok, heads)  # 注意力池化
        self.head = _mlp(d_tok + d_glob, hidden, 2)  # 池化結果加 qglob 後的頭

    def forward(self, tokens, mask, qglob, ticks=None):  # 前向計算（忽略 ticks）
        kpm = ~mask  # 補白位置為 True 的 key 遮罩
        h = self.enc(_clean(tokens, mask))  # 補白歸零後逐局編碼
        for sab in self.sabs:  # 逐層自注意力
            h = sab(h, h, kpm)  # 局與局交互
        pooled = self.pma(self.seed.expand(len(h), -1, -1), h, kpm)[:, 0]  # 以 seed 查詢，把各局池化成一個向量
        return self.head(torch.cat([pooled, qglob], dim=-1))[:, None]  # 算 Δ 並補上 T = 1 的維度


class XAttnGRU(nn.Module):  # r4_xattn：以交叉注意力反覆讀取各局
    """GameEncoder -> a recurrent state h (learned start h0) that reads the games by cross-attention.
    Each tick: q = W_q h (d_tok), o = MultiheadAttention(q; keys, values = the tokens, key_padding_mask =
    ~mask), h = GRUCell([o, qglob], h), Δ_t = W_out h (zero-initialised). Returns the Δ of all ticks,
    [B, T, 10], T = `ticks` (constructor default 8, overridable in forward)."""

    def __init__(self, d_in, d_glob=6, d_tok=128, d_hidden=256, heads=4, ticks=8):  # 建立編碼器、注意力與 GRU
        super().__init__()  # 初始化 nn.Module
        self.ticks = ticks  # 預設的遞迴步數
        self.enc = GameEncoder(d_in, d_tok)  # 逐局編碼器
        self.h0 = nn.Parameter(torch.zeros(d_hidden))  # 學習的起始狀態
        self.w_q = nn.Linear(d_hidden, d_tok)  # 由狀態產生查詢
        self.attn = nn.MultiheadAttention(d_tok, heads, batch_first=True)  # 讀取各局的交叉注意力
        self.cell = nn.GRUCell(d_tok + d_glob, d_hidden)  # 更新狀態的 GRU 單元
        self.w_out = zero_linear(d_hidden)  # 零初始化的輸出層

    def forward(self, tokens, mask, qglob, ticks=None):  # 執行 ticks 步遞迴
        kpm = ~mask  # 補白的 key 遮罩
        x = self.enc(_clean(tokens, mask))  # 逐局編碼
        h = self.h0.expand(len(x), -1)  # 每題都從同一個起始狀態出發
        deltas = []  # 收集每個 tick 的 Δ
        for _ in range(self.ticks if ticks is None else ticks):  # 未指定時用建構子的步數
            o = self.attn(self.w_q(h)[:, None], x, x, key_padding_mask=kpm, need_weights=False)[0][:, 0]  # 以目前狀態查詢各局
            h = self.cell(torch.cat([o, qglob], dim=-1), h)  # 讀到的內容加 qglob 更新狀態
            deltas.append(self.w_out(h))  # 記下這個 tick 的 Δ
        return torch.stack(deltas, dim=1)  # 所有 tick 的 Δ，[B, T, 10]


class _NLM(nn.Module):  # CTM 的神經元層級模型（seqmodel 也沿用）
    """Neuron-level models: each of the n neurons maps its own history of `history` pre-activations
    through its own MLP history -> hidden -> 1 (ReLU), all neurons at once by einsum.
    A [B, n, history] -> z [B, n]. Initialised as nn.Linear per neuron (U(±1/sqrt(fan_in)))."""

    def __init__(self, n, history, hidden):  # 建立每個神經元各自的兩層權重
        super().__init__()  # 初始化 nn.Module
        b1, b2 = 1 / math.sqrt(history), 1 / math.sqrt(hidden)  # 兩層的均勻分布初始化範圍
        self.w1 = nn.Parameter(torch.empty(n, history, hidden).uniform_(-b1, b1))  # 第一層權重（每個神經元一份）
        self.b1 = nn.Parameter(torch.empty(n, hidden).uniform_(-b1, b1))  # 第一層偏置
        self.w2 = nn.Parameter(torch.empty(n, hidden).uniform_(-b2, b2))  # 第二層權重
        self.b2 = nn.Parameter(torch.empty(n).uniform_(-b2, b2))  # 第二層偏置

    def forward(self, A):  # A：每個神經元的預激活歷史
        h = torch.relu(torch.einsum("bnm,nmh->bnh", A, self.w1) + self.b1)  # 每個神經元各自的隱藏層
        return torch.einsum("bnh,nh->bn", h, self.w2) + self.b2  # 每個神經元的後激活值 z


class _SynapseUNet(nn.Module):  # 原始 CTM 的 U-Net 突觸
    """The synapse of the original CTM at synapse_depth > 1 (SynapseUNET of SakanaAI/continuous-thought-machines,
    its default depth 4): Linear(d_in, w_0) -> LayerNorm -> SiLU, then the depth − 1 down blocks w_i -> w_{i+1}
    (Linear -> LayerNorm -> SiLU) over the widths w = linspace(d, min_width, depth) truncated to int, and the
    matching up blocks w_{i+1} -> w_i in reverse, each output added to the down path's output of width w_i and
    passed through that width's LayerNorm. x [B, d_in] -> [B, w_0 = d]."""

    def __init__(self, d_in, d, depth, min_width=16):  # 建立下行、上行區塊與跳接的正規化
        super().__init__()  # 初始化 nn.Module
        if depth < 2:  # 深度 1 應改用簡單的突觸
            raise ValueError(f"the U-Net synapse needs depth >= 2, got {depth}")  # 拒絕不合法的深度
        w = [int(d + (min_width - d) * i / (depth - 1)) for i in range(depth)]  # 等同 np.linspace(d, min_width, depth) 取整數
        self.widths = w  # 記下各層寬度
        self.first = nn.Sequential(nn.Linear(d_in, w[0]), nn.LayerNorm(w[0]), nn.SiLU())  # 把輸入投影到最寬的一層
        self.down = nn.ModuleList(nn.Sequential(nn.Linear(w[i], w[i + 1]), nn.LayerNorm(w[i + 1]), nn.SiLU())  # 下行區塊：逐步變窄
                                  for i in range(depth - 1))  # 共 depth − 1 個
        self.up = nn.ModuleList(nn.Sequential(nn.Linear(w[i + 1], w[i]), nn.LayerNorm(w[i]), nn.SiLU())  # 上行區塊：逐步變寬
                                for i in range(depth - 1))  # 共 depth − 1 個
        self.skip_ln = nn.ModuleList(nn.LayerNorm(w[i]) for i in range(depth - 1))  # 跳接相加後的正規化

    def forward(self, x):  # U-Net 的前向計算
        outs = [self.first(x)]  # 收集下行路徑各寬度的輸出
        for layer in self.down:  # 逐層下行
            outs.append(layer(outs[-1]))  # 接在前一層的輸出之後
        h = outs[-1]  # 最窄的瓶頸層
        for i in reversed(range(len(self.up))):  # 由窄到寬逐層上行
            h = self.skip_ln[i](self.up[i](h) + outs[i])  # 上行輸出加上同寬度的下行輸出再正規化
        return h  # 輸出 [B, d]


def _sync(Z, idx, r):  # 計算神經元配對的同步量
    """Synchronisation of the neuron pairs idx [P, 2] over the history Z (a list of [B, n], oldest
    first): S_p = Σ_τ w_τ z_i^τ z_j^τ / sqrt(Σ_τ w_τ), w_τ = exp(−r_p (τ_now − τ)) -> [B, P].
    r [P] must already lie in [0, 15] (CTM.forward projects it): a clamp inside the graph would pass
    no gradient at r = 0, where the decays start, and they would never move."""
    Zs = torch.stack(Z, dim=1)  # 疊成 [B, τ, n] 的張量
    age = torch.arange(len(Z) - 1, -1, -1, dtype=Zs.dtype, device=Zs.device)  # 每個時刻的年齡 τ_now − τ
    w = torch.exp(-r[None, :] * age[:, None])  # 每個時刻、每個配對的權重 [τ, P]
    return (w * Zs[..., idx[:, 0]] * Zs[..., idx[:, 1]]).sum(dim=1) / w.sum(dim=0).sqrt()  # 加權乘積和除以權重和的平方根


class CTM(nn.Module):  # r5_ctm：Continuous Thought Machine 聚合器
    """Continuous Thought Machine (Darlow et al. 2025) as a set aggregator: GameEncoder -> a latent
    of d_latent neurons, each with its own history of `history` pre-activations A and its own MLP
    (NLM) giving its post-activation z. The neurons talk to the games and to the output only through
    synchronisation S_p (see _sync) of n_sync_action / n_sync_out neuron pairs i <= j, drawn without
    replacement from the d_latent (d_latent + 1) / 2 pairs by a randperm seeded with pair_seed (the
    first n_sync_action for action, the next n_sync_out for output), each pair with a learned decay
    r_p (starts at 0 = no decay; projected to [0, 15] on every forward, as in the original).

    Each tick t: s_act = sync over [z0, ..., z_{t-1}]; o = MultiheadAttention(W_q s_act; keys,
    values = the tokens, key_padding_mask = ~mask); a = synapse([z, o, qglob]) (synapse_depth 1, the
    default: Linear -> GLU -> LayerNorm(d_latent), the original's depth-1 synapse; synapse_depth > 1: the
    original's U-Net synapse, _SynapseUNet, its default depth 4); A keeps the last `history` a's (starting
    from the learned A0); z_t = NLM(A); Δ_t = W_out(sync over [z0, ..., z_t]) (W_out zero-initialised).
    Both histories start with the learned z0, as in the original, which seeds the action and the output
    accumulators with the start state. Returns the Δ of all ticks, [B, T, 10], T = `ticks` (constructor
    default 8, overridable in forward)."""

    def __init__(self, d_in, d_glob=6, d_tok=128, d_latent=256, history=8, nlm_hidden=16, heads=4, ticks=8,  # 各層寬度、歷史長度、NLM 隱藏寬度、注意力頭數與預設步數
                 n_sync_action=512, n_sync_out=512, pair_seed=0, synapse_depth=1):  # 同步配對數、抽樣種子與突觸深度
        super().__init__()  # 初始化 nn.Module
        if synapse_depth < 1:  # 深度至少為 1
            raise ValueError(f"synapse_depth must be >= 1, got {synapse_depth}")  # 拒絕不合法的深度
        self.ticks = ticks  # 預設的思考步數
        pairs = torch.triu_indices(d_latent, d_latent).T  # 所有 i <= j 的神經元配對 [d_latent (d_latent + 1) / 2, 2]
        if n_sync_action + n_sync_out > len(pairs):  # 要抽的配對數不能超過總數
            raise ValueError(f"{n_sync_action} + {n_sync_out} sync pairs exceed the {len(pairs)} pairs of {d_latent} neurons")  # 配對不足時報錯
        perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(pair_seed))  # 以 pair_seed 固定的不放回抽樣
        self.register_buffer("idx_act", pairs[perm[:n_sync_action]].clone())  # 前 n_sync_action 個配對給動作（查詢）
        self.register_buffer("idx_out", pairs[perm[n_sync_action:n_sync_action + n_sync_out]].clone())  # 接著 n_sync_out 個配對給輸出
        self.r_act = nn.Parameter(torch.zeros(n_sync_action))  # 動作同步的學習衰減，起始為 0
        self.r_out = nn.Parameter(torch.zeros(n_sync_out))  # 輸出同步的學習衰減，起始為 0
        self.enc = GameEncoder(d_in, d_tok)  # 逐局編碼器
        b_a, b_z = 1 / math.sqrt(d_latent + history), 1 / math.sqrt(d_latent)  # 原始實作的起始值範圍
        self.A0 = nn.Parameter(torch.empty(d_latent, history).uniform_(-b_a, b_a))  # 學習的起始預激活歷史
        self.z0 = nn.Parameter(torch.empty(d_latent).uniform_(-b_z, b_z))  # 學習的起始後激活值
        self.w_q = nn.Linear(n_sync_action, d_tok)  # 由動作同步量產生查詢
        self.attn = nn.MultiheadAttention(d_tok, heads, batch_first=True)  # 讀取各局的交叉注意力
        d_syn = d_latent + d_tok + d_glob  # 突觸的輸入寬度：z、注意力輸出與 qglob
        if synapse_depth == 1:  # 深度 1 突觸的模組依此順序建立，所以設定種子後的初始化不變
            self.synapse = nn.Sequential(nn.Linear(d_syn, 2 * d_latent), nn.GLU(), nn.LayerNorm(d_latent))  # 原始實作的深度 1 突觸
        else:  # 深度大於 1
            self.synapse = _SynapseUNet(d_syn, d_latent, synapse_depth)  # 原始實作的 U-Net 突觸
        self.nlm = _NLM(d_latent, history, nlm_hidden)  # 神經元層級模型
        self.w_out = zero_linear(n_sync_out)  # 零初始化的輸出層

    def forward(self, tokens, mask, qglob, ticks=None):  # 執行 ticks 步思考
        # 與原始實作相同，以就地投影把衰減參數夾在 [0, 15]
        # （decay_params.data = clamp(...)）；_sync 之後使用未經 clamp 的參數，梯度才傳得到它們
        self.r_act.data.clamp_(0.0, 15.0)  # 就地投影動作同步的衰減
        self.r_out.data.clamp_(0.0, 15.0)  # 就地投影輸出同步的衰減
        kpm = ~mask  # 補白的 key 遮罩
        x = self.enc(_clean(tokens, mask))  # 逐局編碼
        A = self.A0.expand(len(x), -1, -1)  # 每題都從學習的起始歷史出發
        z = self.z0.expand(len(x), -1)  # 每題都從學習的起始狀態出發
        Z_act, Z_out, deltas = [z], [z], []  # 兩條同步歷史都從 z0 開始
        for _ in range(self.ticks if ticks is None else ticks):  # 未指定時用建構子的步數
            s_act = _sync(Z_act, self.idx_act, self.r_act)  # 到上一步為止的動作同步量
            o = self.attn(self.w_q(s_act)[:, None], x, x, key_padding_mask=kpm, need_weights=False)[0][:, 0]  # 以同步量查詢各局
            a = self.synapse(torch.cat([z, o, qglob], dim=-1))  # 突觸產生新的預激活值
            A = torch.cat([A[:, :, 1:], a[:, :, None]], dim=-1)  # 歷史只保留最近 history 個
            z = self.nlm(A)  # 神經元層級模型給出新的後激活值
            Z_act.append(z)  # 加入動作同步的歷史
            Z_out.append(z)  # 加入輸出同步的歷史
            deltas.append(self.w_out(_sync(Z_out, self.idx_out, self.r_out)))  # 由輸出同步量讀出這個 tick 的 Δ
        return torch.stack(deltas, dim=1)  # 所有 tick 的 Δ，[B, T, 10]
