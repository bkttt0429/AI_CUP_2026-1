"""Set aggregators of the L3 ladder: a question's game tokens -> corrections Δ to the base logits.

Every model is built as Model(d_in, ...) with d_in the token width (the last dimension of
ladder_data.pack_questions tokens) and maps

    forward(tokens [B, M, d_in], mask [B, M] bool (True = a game), qglob [B, 6], ticks=None) -> deltas [B, T, 10]

on standardised inputs (aicup.settrain). The output layer starts at zero weights and bias, so Δ ≡ 0
before training. The models are permutation invariant over the games and use no positional
encoding; padding rows are zeroed on entry and excluded from every pooling and attention
(key_padding_mask = ~mask), so neither their content nor their number changes the output.
T = 1 models ignore `ticks`; XAttnGRU runs a recurrence of `ticks` steps (default 8) and returns the Δ of every
tick. Every question needs at least one game.
"""
import math

import torch
from torch import nn

N_CLASSES = 10


def zero_linear(n_in, n_out=N_CLASSES):
    """The output layer: Linear with weight and bias at 0, so the first Δ is exactly 0."""
    layer = nn.Linear(n_in, n_out)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _mlp(n_in, hidden, n_hidden):
    """n_hidden ReLU layers of width `hidden`, then the zero-initialised output layer."""
    layers, d = [], n_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*layers, zero_linear(d))


def _clean(tokens, mask):
    """Padding rows set to 0 (masked_fill, so even non-finite padding cannot leak through 0 * x)."""
    return tokens.masked_fill(~mask[..., None], 0.0)


def masked_mean(x, mask):
    """Mean of x [B, M, d] over the valid rows -> [B, d]."""
    m = mask[..., None]
    return x.masked_fill(~m, 0.0).sum(dim=1) / m.sum(dim=1).clamp_min(1).to(x.dtype)


def masked_max(x, mask):
    """Max of x [B, M, d] over the valid rows -> [B, d]."""
    return x.masked_fill(~mask[..., None], float("-inf")).amax(dim=1)


def n_params(model):
    """Number of parameters (buffers excluded)."""
    return sum(p.numel() for p in model.parameters())


class GameEncoder(nn.Module):
    """Per-game token: Linear(d_in, d_tok) -> ReLU -> Linear(d_tok, d_tok) -> LayerNorm."""

    def __init__(self, d_in, d_tok=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_tok), nn.ReLU(), nn.Linear(d_tok, d_tok), nn.LayerNorm(d_tok))

    def forward(self, tokens):
        return self.net(tokens)


class PooledDelta(nn.Module):
    """r2_pooled_delta, the no-interaction control: mean of the standardised input tokens over the
    valid games, concatenated with qglob, -> MLP(hidden) -> Δ; T = 1. No game encoder."""

    def __init__(self, d_in, d_glob=6, hidden=256):
        super().__init__()
        self.head = _mlp(d_in + d_glob, hidden, 1)

    def forward(self, tokens, mask, qglob, ticks=None):
        return self.head(torch.cat([masked_mean(_clean(tokens, mask), mask), qglob], dim=-1))[:, None]


class QglobDelta(nn.Module):
    """r3_null: Δ from qglob alone, MLP(hidden); T = 1. Measures what re-calibrating on top of the
    base gains by itself. d_in is accepted for the shared interface and unused."""

    def __init__(self, d_in, d_glob=6, hidden=64):
        super().__init__()
        self.head = _mlp(d_glob, hidden, 1)

    def forward(self, tokens, mask, qglob, ticks=None):
        return self.head(qglob)[:, None]


class DeepSets(nn.Module):
    """GameEncoder -> mean and max over the valid tokens -> concat qglob -> MLP(hidden, hidden) -> Δ; T = 1."""

    def __init__(self, d_in, d_glob=6, d_tok=128, hidden=256):
        super().__init__()
        self.enc = GameEncoder(d_in, d_tok)
        self.head = _mlp(2 * d_tok + d_glob, hidden, 2)

    def forward(self, tokens, mask, qglob, ticks=None):
        h = self.enc(_clean(tokens, mask))
        return self.head(torch.cat([masked_mean(h, mask), masked_max(h, mask), qglob], dim=-1))[:, None]


class _MAB(nn.Module):
    """Multihead attention block of the Set Transformer (Lee et al. 2019), post-norm:
    H = LN(Q + MHA(Q, K, K)), out = LN(H + rFF(H)), with the padded keys masked."""

    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, q, kv, key_padding_mask):
        h = self.ln1(q + self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)[0])
        return self.ln2(h + self.ff(h))


class SetTransformer(nn.Module):
    """GameEncoder -> `layers` SABs (self-attention, key_padding_mask = ~mask) -> PMA with one learned
    seed -> concat qglob -> MLP(hidden, hidden) -> Δ; T = 1. The head is the DeepSets head, so the
    two differ only in how the tokens are combined."""

    def __init__(self, d_in, d_glob=6, d_tok=128, heads=4, layers=2, hidden=256):
        super().__init__()
        self.enc = GameEncoder(d_in, d_tok)
        self.sabs = nn.ModuleList([_MAB(d_tok, heads) for _ in range(layers)])
        self.seed = nn.Parameter(torch.randn(1, 1, d_tok) / math.sqrt(d_tok))
        self.pma = _MAB(d_tok, heads)
        self.head = _mlp(d_tok + d_glob, hidden, 2)

    def forward(self, tokens, mask, qglob, ticks=None):
        kpm = ~mask
        h = self.enc(_clean(tokens, mask))
        for sab in self.sabs:
            h = sab(h, h, kpm)
        pooled = self.pma(self.seed.expand(len(h), -1, -1), h, kpm)[:, 0]
        return self.head(torch.cat([pooled, qglob], dim=-1))[:, None]


class XAttnGRU(nn.Module):
    """GameEncoder -> a recurrent state h (learned start h0) that reads the games by cross-attention.
    Each tick: q = W_q h (d_tok), o = MultiheadAttention(q; keys, values = the tokens, key_padding_mask =
    ~mask), h = GRUCell([o, qglob], h), Δ_t = W_out h (zero-initialised). Returns the Δ of all ticks,
    [B, T, 10], T = `ticks` (constructor default 8, overridable in forward)."""

    def __init__(self, d_in, d_glob=6, d_tok=128, d_hidden=256, heads=4, ticks=8):
        super().__init__()
        self.ticks = ticks
        self.enc = GameEncoder(d_in, d_tok)
        self.h0 = nn.Parameter(torch.zeros(d_hidden))
        self.w_q = nn.Linear(d_hidden, d_tok)
        self.attn = nn.MultiheadAttention(d_tok, heads, batch_first=True)
        self.cell = nn.GRUCell(d_tok + d_glob, d_hidden)
        self.w_out = zero_linear(d_hidden)

    def forward(self, tokens, mask, qglob, ticks=None):
        kpm = ~mask
        x = self.enc(_clean(tokens, mask))
        h = self.h0.expand(len(x), -1)
        deltas = []
        for _ in range(self.ticks if ticks is None else ticks):
            o = self.attn(self.w_q(h)[:, None], x, x, key_padding_mask=kpm, need_weights=False)[0][:, 0]
            h = self.cell(torch.cat([o, qglob], dim=-1), h)
            deltas.append(self.w_out(h))
        return torch.stack(deltas, dim=1)
