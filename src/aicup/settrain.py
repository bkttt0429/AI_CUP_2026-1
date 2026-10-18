"""Training of the set aggregators (aicup.aggregators) as corrections Δ to cross-fitted base logits.

fit_set_model trains make(d_in) on packed training questions (ladder_data.pack_questions) to
minimise a loss of the per-tick logits base + Δ_t; predict_set_model returns softmax(base + Δ) at
the picked tick. Inputs are standardised with statistics of the training questions only: tokens
over their valid games, qglob over the questions, both with the rankmodel._standardise rule
(ddof=0, std < 1e-12 -> 1); padding is set back to 0 after standardisation. Standardisation is
done batch by batch on the device, so the caller's float32 tokens (several GB for a full fold
set) are never copied to float64 and never modified.

Training: AdamW with a cosine schedule over all steps, fixed epochs (no early stopping), a
seeded shuffle per epoch, class-balanced weights (rankmodel.class_weights). LOSSES and PICKS map
TrainCfg.loss and predict_set_model's pick to their functions: loss "final" (the last tick; the
primary loss) or "ctm_minmax" (the original CTM's min-loss / max-certainty pair of ticks, the
r5_ctm_minmax control); pick "final" (the last tick; every primary score) or "certain" (each
question's most certain tick, label-free; report-only).

crossfit_base_logits gives the base logits: for each training fold from a static model fitted on
the other training folds, for the evaluation questions from one fitted on all of them (the static
rung itself).
"""
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aicup.rankmodel import N_CLASSES, _softmax, _standardise, class_weights

BASE_CLIP = 1e-9
STATS_CHUNK = 4096  # questions per chunk when accumulating the token statistics


@dataclass(frozen=True)
class TrainCfg:
    epochs: int = 20
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 1e-2
    seed: int = 0
    loss: str = "final"
    device: str = "cuda"


@dataclass
class Fitted:
    """A trained set model with its standardisation statistics (float64) and per-epoch mean loss."""
    model: nn.Module
    tok_mean: np.ndarray
    tok_std: np.ndarray
    glob_mean: np.ndarray
    glob_std: np.ndarray
    cfg: TrainCfg
    losses: list = field(default_factory=list)


def final_loss(logits_ticks, deltas, y, w, gamma):
    """CE(base + Δ_T, y) weighted by w (normalised by the batch's sum of w) + γ · mean ‖Δ_T‖².

    logits_ticks, deltas: [B, T, 10]; only the last tick enters."""
    ce = F.cross_entropy(logits_ticks[:, -1], y, reduction="none")
    return (w * ce).sum() / w.sum() + gamma * deltas[:, -1].pow(2).sum(dim=-1).mean()


def certainty(logits):
    """1 − H(softmax(logits)) / log C over the last axis (C classes): 0 for uniform, 1 for one-hot."""
    logp = logits.log_softmax(dim=-1)
    return 1.0 + (logp.exp() * logp).sum(dim=-1) / math.log(logits.shape[-1])


def _last_argmax(x):
    """argmax over dim 1 of x [B, T]; ties go to the later tick."""
    return x.shape[1] - 1 - x.flip(1).argmax(dim=1)


def minmax_ticks(logits_ticks, y):
    """Per sample the tick t1 of lowest CE(logits_t, y) and the tick t2 of highest certainty;
    ties go to the later tick. logits_ticks [B, T, 10] -> (t1 [B], t2 [B])."""
    with torch.no_grad():
        logp = logits_ticks.log_softmax(dim=-1)
        ce = -logp.gather(-1, y[:, None, None].expand(-1, logp.shape[1], 1))[..., 0]
        return _last_argmax(-ce), _last_argmax(certainty(logits_ticks))


def ctm_minmax_loss(logits_ticks, deltas, y, w, gamma):
    """The original CTM loss on a pair of ticks per sample (minmax_ticks: t1 lowest CE, t2 most
    certain): (CE_t1 + CE_t2) / 2 weighted by w (normalised by the batch's sum of w) +
    γ · mean (‖Δ_t1‖² + ‖Δ_t2‖²) / 2 -- weighted as final_loss, so the two differ only in the ticks.

    logits_ticks, deltas: [B, T, 10]."""
    t1, t2 = minmax_ticks(logits_ticks, y)
    rows = torch.arange(len(y), device=logits_ticks.device)
    ce = (F.cross_entropy(logits_ticks[rows, t1], y, reduction="none")
          + F.cross_entropy(logits_ticks[rows, t2], y, reduction="none")) / 2
    pen = (deltas[rows, t1].pow(2).sum(dim=-1) + deltas[rows, t2].pow(2).sum(dim=-1)) / 2
    return (w * ce).sum() / w.sum() + gamma * pen.mean()


def _pick_final(logits_ticks):
    return logits_ticks[:, -1]


def _pick_certain(logits_ticks):
    """Each question's most certain tick (1 − H / log 10; ties -> the later tick), label-free."""
    t = _last_argmax(certainty(torch.from_numpy(np.asarray(logits_ticks, dtype=np.float64)))).numpy()
    return logits_ticks[np.arange(len(t)), t]


LOSSES = {"final": final_loss, "ctm_minmax": ctm_minmax_loss}
# logits [Q, T, 10] (float64) -> the logits used per question [Q, 10]; "certain" is report-only
PICKS = {"final": _pick_final, "certain": _pick_certain}


def _token_stats(tokens, mask, chunk=STATS_CHUNK):
    """rankmodel._standardise(tokens[mask]) -- per-column mean and std (ddof=0) over the valid
    tokens, std < 1e-12 -> 1 -- accumulated in float64 over chunks of questions (two passes), so
    no float64 copy of the whole [Q, M, D] array is made."""
    n = int(mask.sum())
    total = np.zeros(tokens.shape[2])
    for q in range(0, len(tokens), chunk):
        total += tokens[q:q + chunk][mask[q:q + chunk]].sum(axis=0, dtype=np.float64)
    mean = total / n
    sq = np.zeros(tokens.shape[2])
    for q in range(0, len(tokens), chunk):
        x = tokens[q:q + chunk][mask[q:q + chunk]].astype(np.float64) - mean
        sq += np.einsum("ij,ij->j", x, x)
    std = np.sqrt(sq / n)
    std[std < 1e-12] = 1.0
    return mean, std


def _check(data, base):
    tokens, mask, qglob = data["tokens"], np.asarray(data["mask"]), data["qglob"]
    Q = len(tokens)
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != bool or Q == 0:
        raise ValueError(f"tokens {tokens.shape} and mask {mask.shape} ({mask.dtype}) are not a non-empty packed set")
    if qglob.ndim != 2 or len(qglob) != Q:
        raise ValueError(f"qglob {qglob.shape} does not match {Q} questions")
    if not mask.any(axis=1).all():
        raise ValueError(f"{int((~mask.any(axis=1)).sum())} questions have no game")
    base = np.asarray(base, dtype=np.float64)
    if base.shape != (Q, N_CLASSES) or not np.all(np.isfinite(base)):
        raise ValueError(f"base must be finite [{Q}, {N_CLASSES}], got {base.shape}")
    return base


class _Inputs:
    """A packed question set on the device; calling it with question indices returns the
    standardised batch (tokens with padding 0, mask, qglob)."""

    def __init__(self, data, fitted, device):
        if data["tokens"].shape[2] != len(fitted.tok_mean):
            raise ValueError(f"tokens have {data['tokens'].shape[2]} columns, the model has d_in {len(fitted.tok_mean)}")
        f32 = dict(dtype=torch.float32, device=device)
        # zero-copy view on the CPU, one transfer to an accelerator
        self.tokens = torch.from_numpy(np.ascontiguousarray(data["tokens"], dtype=np.float32)).to(device)
        self.mask = torch.from_numpy(np.asarray(data["mask"], dtype=bool)).to(device)
        glob = (np.asarray(data["qglob"], dtype=np.float64) - fitted.glob_mean) / fitted.glob_std
        self.qglob = torch.tensor(glob, **f32)
        self.mean, self.std = torch.tensor(fitted.tok_mean, **f32), torch.tensor(fitted.tok_std, **f32)

    def __call__(self, ix):
        mask = self.mask[ix]
        tokens = ((self.tokens[ix] - self.mean) / self.std).masked_fill(~mask[..., None], 0.0)
        return tokens, mask, self.qglob[ix]


def fit_set_model(make: Callable[[int], nn.Module], data: dict, base, cfg: TrainCfg) -> Fitted:
    """Train make(d_in) on the packed training questions `data` against base logits [Q, 10]
    (cross-fitted, see crossfit_base_logits) with cfg; every question needs a rank (y >= 0)."""
    if cfg.loss not in LOSSES:
        raise ValueError(f"unknown loss {cfg.loss!r}; known: {list(LOSSES)}")
    if cfg.epochs < 0 or cfg.batch < 1:
        raise ValueError(f"need epochs >= 0 and batch >= 1, got {cfg.epochs}, {cfg.batch}")
    base = _check(data, base)
    y = np.asarray(data["y"], dtype=np.int64)
    if len(y) != len(base) or y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError("every training question needs a rank (0 <= y < 10)")
    tokens, mask = data["tokens"], np.asarray(data["mask"], dtype=bool)
    tok_mean, tok_std = _token_stats(tokens, mask)
    glob_mean, glob_std = _standardise(np.asarray(data["qglob"], dtype=np.float64))
    with torch.random.fork_rng(devices=[]):  # seeded initialisation without touching the global RNG
        torch.manual_seed(cfg.seed)
        model = make(tokens.shape[2])
    fitted = Fitted(model.to(cfg.device), tok_mean, tok_std, glob_mean, glob_std, cfg)
    inputs = _Inputs(data, fitted, cfg.device)
    base_t = torch.tensor(base, dtype=torch.float32, device=cfg.device)
    y_t = torch.tensor(y, device=cfg.device)
    w_t = torch.tensor(class_weights(y), dtype=torch.float32, device=cfg.device)

    Q, loss_fn = len(y), LOSSES[cfg.loss]
    n_batches = -(-Q // cfg.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * n_batches))
    rng = np.random.RandomState(cfg.seed)
    model.train()
    for _ in range(cfg.epochs):
        order = torch.from_numpy(rng.permutation(Q)).to(cfg.device)
        total = torch.zeros((), device=cfg.device)
        for b in range(n_batches):
            ix = order[b * cfg.batch:(b + 1) * cfg.batch]
            deltas = model(*inputs(ix))
            loss = loss_fn(base_t[ix][:, None] + deltas, deltas, y_t[ix], w_t[ix], cfg.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach() * len(ix)
        fitted.losses.append(float(total) / Q)
    model.eval()
    return fitted


def predict_set_model(fitted: Fitted, data: dict, base, ticks=None, pick="final"):
    """softmax(base + Δ) [Q, 10] (float64), Δ taken at the tick chosen by `pick` ("final": the last;
    "certain": each question's most certain tick, report-only); ticks overrides the model's number of ticks."""
    if pick not in PICKS:
        raise ValueError(f"unknown pick {pick!r}; known: {list(PICKS)}")
    base = _check(data, base)
    device, model = fitted.cfg.device, fitted.model
    inputs = _Inputs(data, fitted, device)
    model.eval()
    out = []
    with torch.no_grad():
        for q in range(0, len(base), fitted.cfg.batch):
            ix = torch.arange(q, min(q + fitted.cfg.batch, len(base)), device=device)
            out.append(model(*inputs(ix), ticks=ticks).double().cpu().numpy())
    logits = base[:, None, :] + np.concatenate(out)
    return _softmax(PICKS[pick](logits))


def crossfit_base_logits(make_static, X_by_fold: dict, y_by_fold: dict, X_eval):
    """Cross-fitted base logits log(clip(proba, 1e-9)) of the static model make_static().

    Returns ({f: [n_f, 10]}, [n_eval, 10]): the logits of training fold f from a model fitted on the
    other training folds, and of X_eval from one fitted on all training folds. Training rows are
    always stacked in ascending fold order, so the evaluation logits come from exactly the fit the
    static rung makes on the same folds."""
    folds = sorted(X_by_fold)
    if sorted(y_by_fold) != folds:
        raise ValueError(f"X_by_fold folds {folds} differ from y_by_fold folds {sorted(y_by_fold)}")
    if len(folds) < 2:
        raise ValueError(f"cross-fitting needs at least 2 training folds, got {folds}")
    for f in folds:
        if len(X_by_fold[f]) != len(y_by_fold[f]):
            raise ValueError(f"fold {f}: {len(X_by_fold[f])} rows of X, {len(y_by_fold[f])} labels")

    def fit_on(fs):
        return make_static().fit(np.concatenate([X_by_fold[f] for f in fs]), np.concatenate([y_by_fold[f] for f in fs]))

    def logits(proba):
        return np.log(np.clip(proba, BASE_CLIP, None))

    by_fold = {f: logits(fit_on([g for g in folds if g != f]).predict_proba(X_by_fold[f])) for f in folds}
    return by_fold, logits(fit_on(folds).predict_proba(X_eval))
