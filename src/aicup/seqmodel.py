"""Sequence encoders over a game's moves, correcting the L1b sums (issue #19).

Every encoder maps a game's step sequence (aicup.seqdata: [T, STEP_DIM], padded with a mask) to a correction Δ [10] and
returns base + Δ, where base is the target's summed L1b log p -- so the "sum" encoder (Δ ≡ 0) is today's features exactly
and every other one starts there (zero-initialised output layer, as the L3 ladder's Δ models). Encoders: sum, conv (two
1-D convolutions, masked mean + max pooling), gru (a GRU, final state + masked mean of the states), ctm_seq (a Continuous
Thought Machine whose ticks are the moves: at step t the synapse reads the latent state and the encoded step, the
neuron-level models turn each neuron's history of pre-activations into its activation, and the output is read from the
synchronisation of neuron pairs accumulated recursively over the steps with learned decays -- the original CTM's
machinery with the data fed one move per tick instead of attended to). Training: class-balanced cross-entropy of the
player's rank per game + gamma ‖Δ‖², Adam, inner early stopping by held-out players (settrain.inner_players), the best
inner CE kept; the step inputs are standardised with statistics of the training steps stored in the model.
"""
import hashlib
import io
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aicup.aggregators import _NLM, zero_linear
from aicup.metric import LEVELS
from aicup.rankmodel import class_weights
from aicup.seqdata import STEP_DIM, pad_batch
from aicup.settrain import INNER_SEED, inner_players

ARCHS = ("sum", "conv", "gru", "ctm_seq")
N_CLASSES = len(LEVELS)


@dataclass(frozen=True)
class SeqCfg:
    arch: str = "gru"
    epochs: int = 10
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 1e-3
    seed: int = 0
    device: str = "cuda"
    inner_frac: float = 0.1
    evals_per_epoch: int = 4
    patience: int = 8
    hidden: int = 64
    d_latent: int = 128
    history: int = 8
    nlm_hidden: int = 16
    n_sync: int = 256


class _Encoder(nn.Module):
    """Shared: input standardisation buffers and the Δ convention (forward returns base + Δ)."""

    def __init__(self, d_in):
        super().__init__()
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("x_std", torch.ones(d_in))

    def standardise(self, X):
        return (X - self.x_mean) / self.x_std

    def forward(self, X, mask, base):
        return base + self.delta(self.standardise(X), mask).to(base.dtype)  # base is float64: the sum encoder returns it exactly


class SumEncoder(_Encoder):
    def __init__(self, d_in, cfg):
        super().__init__(d_in)
        self.unused = nn.Parameter(torch.zeros(1))  # so the optimiser has a parameter; it never enters the output

    def delta(self, X, mask):
        return torch.zeros(X.shape[0], N_CLASSES, dtype=X.dtype, device=X.device) + 0.0 * self.unused


def _masked_mean(h, mask):
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def _masked_max(h, mask):
    return h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values.masked_fill(~mask.any(dim=1, keepdim=True), 0.0)


class ConvEncoder(_Encoder):
    def __init__(self, d_in, cfg):
        super().__init__(d_in)
        h = cfg.hidden
        self.conv1, self.conv2 = nn.Conv1d(d_in, h, 5, padding=2), nn.Conv1d(h, h, 5, padding=2)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())
        self.out = zero_linear(h)

    def delta(self, X, mask):
        mt = mask.unsqueeze(1).to(X.dtype)  # [B, 1, T]; padded steps are zeroed before each convolution, like its own zero padding
        h = F.relu(self.conv1(X.transpose(1, 2) * mt)) * mt
        h = (F.relu(self.conv2(h)) * mt).transpose(1, 2)
        return self.out(self.head(torch.cat([_masked_mean(h, mask), _masked_max(h, mask)], dim=-1)))


class GRUEncoder(_Encoder):
    def __init__(self, d_in, cfg):
        super().__init__(d_in)
        h = cfg.hidden
        self.gru = nn.GRU(d_in, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU())
        self.out = zero_linear(h)

    def delta(self, X, mask):
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        out, h_n = self.gru(packed)
        hs, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=X.shape[1])
        return self.out(self.head(torch.cat([h_n[-1], _masked_mean(hs, mask)], dim=-1)))


class CTMSeqEncoder(_Encoder):
    """CTM with one tick per move. State: the latent z [B, n] and the history A [B, n, history] of pre-activations;
    per tick a = synapse([z, enc(x_t)]) (Linear -> GLU -> LayerNorm, the depth-1 synapse), A <- (A[1:], a), z = NLM(A).
    Output synchronisation of n_sync neuron pairs (i <= j, a seeded random draw) accumulated recursively with learned
    decays d_p = exp(-r_p), r_p in [0, 15]: alpha <- d alpha + z_i z_j, beta <- d beta + 1, S = alpha / sqrt(beta)
    (the closed form of the original's exponentially weighted sum). Padded steps leave the state unchanged, so the
    readout W_out(S) (zero-initialised) is taken after each game's last move."""

    def __init__(self, d_in, cfg):
        super().__init__(d_in)
        n, hist = cfg.d_latent, cfg.history
        pairs = torch.triu_indices(n, n).T
        perm = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(cfg.seed))
        self.register_buffer("idx", pairs[perm[:cfg.n_sync]].clone())
        self.r = nn.Parameter(torch.zeros(cfg.n_sync))
        b_a, b_z = 1 / math.sqrt(n + hist), 1 / math.sqrt(n)
        self.A0 = nn.Parameter(torch.empty(n, hist).uniform_(-b_a, b_a))
        self.z0 = nn.Parameter(torch.empty(n).uniform_(-b_z, b_z))
        self.enc = nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU())
        self.synapse = nn.Sequential(nn.Linear(n + cfg.hidden, 2 * n), nn.GLU(), nn.LayerNorm(n))
        self.nlm = _NLM(n, hist, cfg.nlm_hidden)
        self.out = zero_linear(cfg.n_sync)

    def delta(self, X, mask):
        self.r.data.clamp_(0.0, 15.0)
        decay = torch.exp(-self.r)
        B = X.shape[0]
        x = self.enc(X)  # [B, T, hidden]
        A = self.A0.expand(B, -1, -1)
        z = self.z0.expand(B, -1)
        i, j = self.idx[:, 0], self.idx[:, 1]
        alpha = (z[:, i] * z[:, j])
        beta = torch.ones_like(alpha)
        for t in range(X.shape[1]):
            m = mask[:, t].unsqueeze(-1)
            a = self.synapse(torch.cat([z, x[:, t]], dim=-1))
            A_new = torch.cat([A[:, :, 1:], a[:, :, None]], dim=-1)
            z_new = self.nlm(A_new)
            alpha_new = decay * alpha + z_new[:, i] * z_new[:, j]
            beta_new = decay * beta + 1.0
            A = torch.where(m.unsqueeze(-1), A_new, A)
            z = torch.where(m, z_new, z)
            alpha = torch.where(m, alpha_new, alpha)
            beta = torch.where(m, beta_new, beta)
        return self.out(alpha / beta.sqrt())


ENCODERS = {"sum": SumEncoder, "conv": ConvEncoder, "gru": GRUEncoder, "ctm_seq": CTMSeqEncoder}


def make_encoder(cfg, d_in=STEP_DIM):
    if cfg.arch not in ENCODERS:
        raise ValueError(f"unknown arch {cfg.arch!r}; one of {ARCHS}")
    torch.manual_seed(cfg.seed)
    return ENCODERS[cfg.arch](d_in, cfg)


def _batches(seq, idx, batch, rng=None):
    idx = np.asarray(idx, dtype=np.int64)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), batch):
        yield idx[i:i + batch]


def _to_device(seq, idx, device):
    X, mask, base, y = pad_batch(seq, idx)
    return (torch.from_numpy(X).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(base).to(device, torch.float64),
            torch.from_numpy(y).to(device))


def _balanced_ce(model, seq, idx, cfg, w_of):
    """Class-balanced CE (mean over the games of `idx`, weights w_of per game) and the mean ‖Δ‖²; no grad."""
    model.eval()
    tot, wsum, dsq, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for b in _batches(seq, idx, cfg.batch):
            X, mask, base, y = _to_device(seq, b, cfg.device)
            logits = model(X, mask, base)
            w = torch.from_numpy(w_of[b]).to(cfg.device, torch.float64)
            tot += float((F.cross_entropy(logits, y, reduction="none") * w).sum())
            wsum += float(w.sum())
            dsq += float((logits - base).pow(2).sum(dim=-1).sum())
            n += len(b)
    return tot / max(wsum, 1e-12), dsq / max(n, 1)


def fit_seq_encoder(seq, cfg, log=print):
    """Train an encoder on the games of `seq` (a fold_sequences / concat_sequences dict); returns (model, meta).
    Inner validation: inner_frac of each rank's players (settrain.inner_players, INNER_SEED), class-balanced CE evaluated
    at step 0 and evals_per_epoch times per epoch, the best weights kept, stop after `patience` evaluations without a
    lower CE. Weights: every rank the same total weight over the training games."""
    t0 = time.time()
    device = cfg.device
    inner = inner_players(seq["player_id"], seq["rank_idx"], cfg.inner_frac, INNER_SEED) if cfg.inner_frac > 0 else np.zeros(len(seq["rank_idx"]), dtype=bool)
    train_idx, inner_idx = np.flatnonzero(~inner), np.flatnonzero(inner)
    if len(train_idx) == 0:
        raise ValueError("no training games")
    model = make_encoder(cfg).to(device)
    # standardisation of the step inputs from the training games' steps (the flag columns keep std 1 when constant)
    steps = np.concatenate([seq["X"][seq["offsets"][g]:seq["offsets"][g + 1]] for g in train_idx]) if len(train_idx) else seq["X"]
    mean, std = steps.mean(axis=0), steps.std(axis=0)
    std[std < 1e-6] = 1.0
    model.x_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
    model.x_std.copy_(torch.from_numpy(std.astype(np.float32)))
    w_all = np.zeros(len(seq["rank_idx"]), dtype=np.float32)
    w_all[train_idx] = class_weights(seq["rank_idx"][train_idx]).astype(np.float32)
    w_inner = np.zeros_like(w_all)
    if len(inner_idx):
        w_inner[inner_idx] = class_weights(seq["rank_idx"][inner_idx]).astype(np.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.RandomState(cfg.seed)
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch)
    eval_every = max(1, steps_per_epoch // max(cfg.evals_per_epoch, 1))
    history, best, since, step, stopped = [], None, 0, 0, False
    losses = []

    def snapshot():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def evaluate():
        nonlocal best, since
        if not len(inner_idx):
            return False
        ce, dsq = _balanced_ce(model, seq, inner_idx, cfg, w_inner)
        model.train()  # evaluations fall inside the epochs; cuDNN's RNN refuses a backward pass in eval mode
        epoch = step / steps_per_epoch
        history.append((epoch, ce, dsq))
        if best is None or ce < best["ce"]:
            best, since = {"ce": ce, "dsq": dsq, "step": step, "epoch": epoch, "state": snapshot()}, 0
        else:
            since += 1
        log(f"  epoch {epoch:.2f} inner CE {ce:.4f} mean|Δ|² {dsq:.3f} (best {best['ce']:.4f} @ {best['epoch']:.2f})", flush=True)
        return since >= cfg.patience

    evaluate()
    for epoch in range(cfg.epochs):
        model.train()
        run, nb = 0.0, 0
        for b in _batches(seq, train_idx, cfg.batch, rng):
            X, mask, base, y = _to_device(seq, b, device)
            w = torch.from_numpy(w_all[b]).to(device, torch.float64)
            logits = model(X, mask, base)
            loss = (F.cross_entropy(logits, y, reduction="none") * w).sum() / w.sum() + cfg.gamma * (logits - base).pow(2).sum(dim=-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
            step += 1
            if step % eval_every == 0 and evaluate():
                stopped = True
                break
        losses.append(run / max(nb, 1))
        if stopped:
            break
    if best is not None:
        model.load_state_dict(best["state"])
    model.eval()
    meta = {"cfg": asdict(cfg), "n_train_games": int(len(train_idx)), "n_inner_games": int(len(inner_idx)),
            "n_inner_players": int(len(np.unique(seq["player_id"][inner_idx]))) if len(inner_idx) else 0,
            "inner_ce_init": history[0][1] if history else None, "inner_ce_best": best["ce"] if best else None,
            "inner_dsq_best": best["dsq"] if best else None, "best_epoch": best["epoch"] if best else float(cfg.epochs),
            "stopped_early": bool(stopped), "history": [(float(e), float(c), float(d)) for e, c, d in history], "losses": losses,
            "steps_run": int(step), "n_params": int(sum(p.numel() for p in model.parameters())), "wall_seconds": time.time() - t0,
            "step_names_mean": mean.tolist(), "step_names_std": std.tolist()}
    return model, meta


def predict_seq(model, seq, cfg):
    """base + Δ for every game of `seq`: [G, 10] float64."""
    model.eval()
    out = np.zeros((len(seq["rank_idx"]), N_CLASSES))
    with torch.no_grad():
        for b in _batches(seq, np.arange(len(seq["rank_idx"])), cfg.batch):
            X, mask, base, _ = _to_device(seq, b, cfg.device)
            out[b] = model(X, mask, base).cpu().numpy()
    return out


def state_sha256(model):
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_encoder(model, cfg, meta, path):
    torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "cfg": asdict(cfg), "meta": meta, "d_in": int(model.x_mean.numel())}, path)


def load_encoder(path, device="cpu"):
    """(model, cfg, meta) from save_encoder's file."""
    d = torch.load(path, map_location=device, weights_only=False)
    cfg = SeqCfg(**{**d["cfg"], "device": device})
    model = make_encoder(cfg, d["d_in"]).to(device)
    model.load_state_dict(d["state"])
    model.eval()
    return model, cfg, d["meta"]
