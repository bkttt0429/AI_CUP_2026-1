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
from decimal import Decimal
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aicup.rankmodel import N_CLASSES, _softmax, _standardise, class_weights

BASE_CLIP = 1e-9
STATS_CHUNK = 4096  # questions per chunk when accumulating the token statistics
INNER_SEED = 2026  # inner_players' shuffle: fixed, independent of TrainCfg.seed


@dataclass(frozen=True)
class TrainCfg:
    """epochs: the maximum with inner_frac > 0, the exact number with inner_frac = 0. inner_frac: share of
    each rank's training players held out for early stopping (0: none, data then needs no player_id);
    evals_per_epoch: inner evaluations per epoch (besides evaluation 0); patience: evaluations in a row
    without a lower inner CE before training stops."""
    epochs: int = 10
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 1e-2
    seed: int = 0
    loss: str = "final"
    device: str = "cuda"
    inner_frac: float = 0.1
    evals_per_epoch: int = 4
    patience: int = 8


@dataclass
class Fitted:
    """A trained set model with its standardisation statistics (float64) and its training record.

    losses: the mean training loss of each epoch begun -- the last one over the batches done when
    training stopped inside it. With inner early stopping (cfg.inner_frac > 0): inner_ce, the inner CE
    of every evaluation, evaluation 0 (Δ ≡ 0) first; best_eval, the evaluation whose weights the model
    holds; best_epoch, the epochs completed at it (fractional, 0.0 for evaluation 0); stopped_early,
    whether patience ended training before cfg.epochs; n_inner_players / n_inner_questions, the held-out
    players and their questions. With inner_frac = 0: inner_ce is [] and the others are None."""
    model: nn.Module
    tok_mean: np.ndarray
    tok_std: np.ndarray
    glob_mean: np.ndarray
    glob_std: np.ndarray
    cfg: TrainCfg
    losses: list = field(default_factory=list)
    inner_ce: list = field(default_factory=list)
    best_eval: Optional[int] = None
    best_epoch: Optional[float] = None
    stopped_early: Optional[bool] = None
    n_inner_players: Optional[int] = None
    n_inner_questions: Optional[int] = None


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


def _token_stats(tokens, mask, chunk=STATS_CHUNK, rows=None):
    """rankmodel._standardise(tokens[rows][mask[rows]]) -- per-column mean and std (ddof=0) over the
    valid tokens of the questions `rows` (all when None), std < 1e-12 -> 1 -- accumulated in float64
    over chunks of questions (two passes), so no float64 copy of the whole [Q, M, D] array is made
    (and, with rows, no copy of the selected questions beyond one chunk)."""
    def chunks():
        for q in range(0, len(tokens) if rows is None else len(rows), chunk):
            sel = slice(q, q + chunk) if rows is None else rows[q:q + chunk]
            yield tokens[sel], mask[sel]

    n = int((mask if rows is None else mask[rows]).sum())
    total = np.zeros(tokens.shape[2])
    for t, m in chunks():
        total += t[m].sum(axis=0, dtype=np.float64)
    mean = total / n
    sq = np.zeros(tokens.shape[2])
    for t, m in chunks():
        x = t[m].astype(np.float64) - mean
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


def _n_inner(n, frac):
    """Inner players of a rank with n players: ceil(frac × n) with frac taken as written in decimal
    (0.1 × 30 is 3, not the 4 of float ceil(3.0000000000000004)), at most n − 1, none when n < 2."""
    return 0 if n < 2 else min(n - 1, math.ceil(Decimal(repr(float(frac))) * n))


def inner_players(player_id, y, frac, seed=INNER_SEED):
    """Per question whether its player is held out as an inner validation player (bool [Q]).

    Per rank, the rank's distinct players are sorted, shuffled with a fresh RandomState(seed) and the
    first _n_inner(n, frac) of them are taken: ceil(frac × n) of the rank's n players, none of a rank
    with one player, never all of them. The choice depends only on the questions' players and ranks --
    not on their order, on TrainCfg.seed or on the model -- so every rung and seed of a split holds out
    the same players. Every player must have one rank; 0 <= frac < 1."""
    pid, y = np.asarray(player_id).astype(str), np.asarray(y, dtype=np.int64)
    if pid.ndim != 1 or pid.shape != y.shape:
        raise ValueError(f"player_id {pid.shape} and y {y.shape} must be one value per question")
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"inner_frac must be in [0, 1), got {frac}")
    players, inv = np.unique(pid, return_inverse=True)  # sorted
    rank = np.empty(len(players), dtype=np.int64)
    rank[inv] = y
    if np.any(rank[inv] != y):
        bad = players[np.unique(inv[rank[inv] != y])]
        raise ValueError(f"{len(bad)} players have questions of more than one rank, e.g. {bad[:3].tolist()}")
    chosen = np.zeros(len(players), dtype=bool)
    for r in np.unique(rank):
        members = np.flatnonzero(rank == r)  # in sorted player order
        chosen[np.random.RandomState(seed).permutation(members)[:_n_inner(len(members), frac)]] = True
    return chosen[inv]


class _EarlyStop:
    """Inner validation of fit_set_model on the questions ix: the CE of log_softmax(base + Δ at the
    last tick) for the true rank, weighted by class_weights of their ranks, in float64 and without the
    γ term, computed in eval mode without gradients. Records each CE in fitted.inner_ce, keeps a CPU
    copy of the weights of the lowest one (ties: the earlier) and says stop after cfg.patience
    evaluations in a row without a lower CE."""

    def __init__(self, model, inputs, base, y, ix, cfg, fitted):
        dev = cfg.device
        self.model, self.inputs, self.batch, self.patience, self.fitted = model, inputs, cfg.batch, cfg.patience, fitted
        self.ix = torch.from_numpy(ix).to(dev)
        self.base = torch.tensor(base[ix], dtype=torch.float64, device=dev)
        self.y = torch.tensor(y[ix], device=dev)
        self.w = torch.tensor(class_weights(y[ix]), dtype=torch.float64, device=dev)
        self.best_state, self.bad = None, 0

    def ce(self):
        self.model.eval()
        total = torch.zeros((), dtype=torch.float64, device=self.w.device)
        with torch.no_grad():
            for q in range(0, len(self.ix), self.batch):
                sl = slice(q, q + self.batch)
                logp = (self.base[sl] + self.model(*self.inputs(self.ix[sl]))[:, -1].double()).log_softmax(dim=-1)
                total += (self.w[sl] * -logp.gather(1, self.y[sl, None])[:, 0]).sum()
        self.model.train()
        return float(total / self.w.sum())

    def __call__(self, epochs_done):
        """Evaluate after epochs_done (fractional) epochs; True when training should stop."""
        f, ce = self.fitted, self.ce()
        if not math.isfinite(ce):
            raise FloatingPointError(f"non-finite inner validation CE ({ce}) at inner evaluation {len(f.inner_ce)} "
                                     f"(after {epochs_done:.4g} epochs): the model diverged")
        f.inner_ce.append(ce)
        if f.best_eval is None or ce < f.inner_ce[f.best_eval]:
            f.best_eval, f.best_epoch, self.bad = len(f.inner_ce) - 1, float(epochs_done), 0
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self):
        self.model.load_state_dict(self.best_state)


def fit_set_model(make: Callable[[int], nn.Module], data: dict, base, cfg: TrainCfg) -> Fitted:
    """Train make(d_in) on the packed training questions `data` against base logits [Q, 10]
    (cross-fitted, see crossfit_base_logits) with cfg; every question needs a rank (y >= 0) and the
    standardisation statistics must be finite (ValueError otherwise).

    cfg.inner_frac > 0 needs data["player_id"]: inner_players holds those players' questions out, and
    the standardisation statistics, class weights and training steps use the other (training) questions
    only. The cosine schedule spans epochs × n_batches steps (n_batches of the training questions). The
    inner CE (_EarlyStop) is evaluated before the first step (evaluation 0, Δ ≡ 0), then after every
    ceil(n_batches / evals_per_epoch) batches of an epoch and at its end (once per step); training stops
    after `patience` evaluations in a row without a lower CE, and the model returned holds the weights
    of the lowest one -- no refit on all players. A non-finite inner CE raises FloatingPointError.
    cfg.inner_frac = 0: `epochs` epochs over every question, the last weights; player_id is not needed."""
    if cfg.loss not in LOSSES:
        raise ValueError(f"unknown loss {cfg.loss!r}; known: {list(LOSSES)}")
    if cfg.epochs < 0 or cfg.batch < 1:
        raise ValueError(f"need epochs >= 0 and batch >= 1, got {cfg.epochs}, {cfg.batch}")
    if not 0.0 <= cfg.inner_frac < 1.0 or cfg.evals_per_epoch < 1 or cfg.patience < 1:
        raise ValueError(f"need 0 <= inner_frac < 1, evals_per_epoch >= 1 and patience >= 1, got {cfg.inner_frac}, "
                         f"{cfg.evals_per_epoch}, {cfg.patience}")
    base = _check(data, base)
    y = np.asarray(data["y"], dtype=np.int64)
    if len(y) != len(base) or y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError("every training question needs a rank (0 <= y < 10)")
    tokens, mask = data["tokens"], np.asarray(data["mask"], dtype=bool)
    inner_ix = None
    if cfg.inner_frac > 0:
        if "player_id" not in data:
            raise ValueError("inner_frac > 0 holds out players: the data needs player_id (or pass inner_frac=0)")
        held = inner_players(data["player_id"], y, cfg.inner_frac)
        if not held.any():
            raise ValueError("no inner validation players: every rank has fewer than 2 players (pass inner_frac=0)")
        train_ix, inner_ix = np.flatnonzero(~held), np.flatnonzero(held)
    else:
        train_ix = np.arange(len(y))
    qglob = np.asarray(data["qglob"], dtype=np.float64)
    # a NaN / inf in the valid tokens or in qglob makes its column's statistics non-finite; checking the
    # O(D) statistics instead of the inputs avoids another pass over several GB of tokens, and the
    # warnings of inf - inf are moot because any of them ends in the error below (a NaN / inf among
    # the inner questions reaches the inner CE instead, which raises too)
    with np.errstate(invalid="ignore", over="ignore"):
        tok_mean, tok_std = _token_stats(tokens, mask, rows=None if inner_ix is None else train_ix)
        glob_mean, glob_std = _standardise(qglob if inner_ix is None else qglob[train_ix])
    for what, mean, std in (("token", tok_mean, tok_std), ("qglob", glob_mean, glob_std)):
        bad = np.flatnonzero(~(np.isfinite(mean) & np.isfinite(std)))
        if len(bad):
            raise ValueError(f"{what} standardisation statistics are not finite in columns {bad.tolist()}: the training "
                             f"{'tokens of valid games' if what == 'token' else 'qglob'} hold NaN or inf")
    with torch.random.fork_rng(devices=[]):  # seeded initialisation without touching the global RNG
        torch.manual_seed(cfg.seed)
        model = make(tokens.shape[2])
    fitted = Fitted(model.to(cfg.device), tok_mean, tok_std, glob_mean, glob_std, cfg)
    # every question on the device once (a zero-copy view on the CPU); index arrays pick the training and
    # the inner questions, so the split makes no copy of the tokens
    inputs = _Inputs(data, fitted, cfg.device)
    base_t = torch.tensor(base, dtype=torch.float32, device=cfg.device)
    y_t = torch.tensor(y, device=cfg.device)
    w = np.zeros(len(y))
    w[train_ix] = class_weights(y[train_ix])
    w_t = torch.tensor(w, dtype=torch.float32, device=cfg.device)

    Q, loss_fn = len(train_ix), LOSSES[cfg.loss]
    n_batches = -(-Q // cfg.batch)
    every = -(-n_batches // cfg.evals_per_epoch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs * n_batches))
    rng = np.random.RandomState(cfg.seed)
    train_t = torch.from_numpy(train_ix).to(cfg.device)
    early, stop, steps = None, False, 0
    model.train()
    if inner_ix is not None:
        fitted.n_inner_players = len(np.unique(np.asarray(data["player_id"])[inner_ix]))
        fitted.n_inner_questions = len(inner_ix)
        early = _EarlyStop(model, inputs, base, y, inner_ix, cfg, fitted)
        early(0.0)  # evaluation 0: Δ ≡ 0, the zero-initialised output layer (the first is the best: no stop)
    for epoch in range(cfg.epochs):
        order = train_t[torch.from_numpy(rng.permutation(Q)).to(cfg.device)]
        total, seen = torch.zeros((), device=cfg.device), 0
        for b in range(n_batches):
            ix = order[b * cfg.batch:(b + 1) * cfg.batch]
            deltas = model(*inputs(ix))
            loss = loss_fn(base_t[ix][:, None] + deltas, deltas, y_t[ix], w_t[ix], cfg.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach() * len(ix)
            seen, steps = seen + len(ix), steps + 1
            if early is not None and ((b + 1) % every == 0 or b + 1 == n_batches):
                stop = early(epoch + (b + 1) / n_batches)
                if stop:
                    break
        fitted.losses.append(float(total) / seen)  # a stop inside the epoch: over the batches done
        if stop:
            break
    if early is not None:
        early.restore()
        fitted.stopped_early = steps < cfg.epochs * n_batches
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
