"""Training of the L1b rank heads on a featcache (plan 2026-10-25-l1b-rank-heads, Task 4).

The trunk is frozen (eval, no grad, bf16 autocast on CUDA); only the RankHeads' parameters are
optimised (Adam). Every position's loss is the negative masked log-softmax of the played move under
the head of the player's rank, weighted so that every rank has the same total weight (mean 1,
computed from the full training set). Inner early stopping as the L3 ladder's: a stratified share of
each rank's players is held out by player (settrain.inner_players, INNER_SEED), the class-balanced
inner NLL is evaluated at step 0 and every `eval_every` epoch, the best weights (step 0 included) are
kept, and training stops after `patience` evaluations without improvement. A checkpoint
(<out>.ckpt) after every evaluation makes an interrupted run resume where it stopped; a finished run
keeps its checkpoint so a longer run can continue it (same cfg otherwise).
"""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aicup.featcache import read_cache, unpack_bins, unpack_legal
from aicup.l1b import RankHeads, atomic_save, masked_logp, neutral_rows, save_heads, trunk_features
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.settrain import INNER_SEED, inner_players

N_RANKS = len(LEVELS)


@dataclass(frozen=True)
class HeadsCfg:
    """epochs: maximum epochs; eval_every: inner evaluations every this many epochs (0.25 = four per epoch);
    patience: evaluations in a row without a lower inner NLL before training stops; inner_frac: share of each
    rank's players held out (0: none); inner_max_positions: cap on the inner positions evaluated."""
    epochs: int = 3
    batch: int = 256
    lr: float = 1e-4
    eval_every: float = 0.25
    patience: int = 4
    inner_frac: float = 0.1
    inner_max_positions: int = 400_000
    seed: int = 0
    device: str = "cuda"


def inner_split(games, frac, seed=INNER_SEED):
    """(train game rows, inner game rows) of a cache's games table: per rank, ceil(frac x players) of its players
    are held out with all their games (settrain.inner_players; the same for every seed and run)."""
    if len(games) == 0 or frac <= 0:
        return np.arange(len(games), dtype=np.int64), np.zeros(0, dtype=np.int64)
    y = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64)
    held = inner_players(games["player_id"].to_numpy(dtype=str), y, frac, seed)
    return np.flatnonzero(~held).astype(np.int64), np.flatnonzero(held).astype(np.int64)


def positions_of(arrays, game_rows):
    """Indices of the positions whose game is one of `game_rows`."""
    return np.flatnonzero(np.isin(arrays["game_row"], np.asarray(game_rows, dtype=np.int64)))


def position_weights(rank_idx, n_ranks=N_RANKS):
    """Per-position weights giving every present rank the same total weight, mean 1 (rankmodel.class_weights'
    rule on positions); computed from the whole training set, never from a subsample."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)
    if rank_idx.size == 0:
        return np.zeros(0, dtype=np.float64)
    counts = np.bincount(rank_idx, minlength=n_ranks).astype(np.float64)
    w = 1.0 / counts[rank_idx]
    return w / w.mean()


def batches(arrays, idx, batch, rng=None, device="cpu"):
    """Mini-batches of the positions `idx` (shuffled when rng is given): bins float32 [B, 22, 19, 19], globs
    float32, legal bool [B, 362], target long, rank long, color long -- on `device`."""
    idx = np.asarray(idx, dtype=np.int64)
    if rng is not None:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), batch):
        sel = np.sort(idx[i:i + batch])
        yield _to_tensors(arrays, sel, device)


def _to_tensors(arrays, sel, device):
    bins = torch.from_numpy(unpack_bins(arrays["bins_packed"][sel])).to(device, non_blocking=True).float()
    globs = torch.from_numpy(np.ascontiguousarray(arrays["globs"][sel])).to(device, non_blocking=True)
    legal = torch.from_numpy(unpack_legal(arrays["legal_packed"][sel])).to(device, non_blocking=True)
    target = torch.from_numpy(arrays["target"][sel].astype(np.int64)).to(device, non_blocking=True)
    rank = torch.from_numpy(arrays["rank_idx"][sel].astype(np.int64)).to(device, non_blocking=True)
    color = torch.from_numpy(arrays["color"][sel].astype(np.int64)).to(device, non_blocking=True)
    return bins, globs, legal, target, rank, color


def _true_head_logp(model, heads, bins, globs, legal, target, rank, color, rows, autocast):
    """log p of each position's move under the head of its own rank: [B]."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)
    logits = heads(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())
    logp = masked_logp(logits, legal)
    return logp[torch.arange(len(target), device=logp.device), rank, target]


def evaluate_nll(model, heads, arrays, idx, cfg):
    """Class-balanced per-move NLL of the true-rank head over positions `idx`: {"nll": mean over the ranks present
    of the rank's mean NLL, "nll_by_rank": [10] (None for an absent rank), "ranks_present", "n"}."""
    device, rows = cfg.device, neutral_rows()
    autocast = str(device).startswith("cuda")
    sums, counts = np.zeros(N_RANKS), np.zeros(N_RANKS, dtype=np.int64)
    was_training = heads.training
    heads.eval()
    with torch.no_grad():
        for bins, globs, legal, target, rank, color in batches(arrays, idx, cfg.batch, None, device):
            nll = -_true_head_logp(model, heads, bins, globs, legal, target, rank, color, rows, autocast).double().cpu().numpy()
            r = rank.cpu().numpy()
            np.add.at(sums, r, nll)
            np.add.at(counts, r, 1)
    heads.train(was_training)
    present = [int(k) for k in np.flatnonzero(counts > 0)]
    by_rank = [float(sums[k] / counts[k]) if counts[k] else None for k in range(N_RANKS)]
    nll = float(np.mean([by_rank[k] for k in present])) if present else float("nan")
    return {"nll": nll, "nll_by_rank": by_rank, "ranks_present": present, "n": int(counts.sum())}


def check_init(model, heads, arrays, idx, n=256):
    """max |Δ log p| between every head and the model's own policy under the neutral metadata, fp32 trunk and heads,
    on the first n positions of idx (0 at step 0 up to float error)."""
    device = next(heads.parameters()).device
    rows = neutral_rows()
    worst = 0.0
    with torch.no_grad():
        for bins, globs, legal, target, rank, color in batches(arrays, np.asarray(idx)[:n], 64, None, device):
            trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)
            head_logp = masked_logp(heads(trunk, mask, mask_sum_hw, mask_sum), legal)
            meta = None
            if model.get_has_metadata_encoder():
                meta = torch.as_tensor(rows, device=device)[color]
            own = model.postprocess_output(model(bins, globs, input_meta=meta))[0][0][:, 0, :].float()
            own_logp = masked_logp(own[:, None, :], legal)[:, 0, :]
            diff = (head_logp - own_logp[:, None, :]).abs()
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
            worst = max(worst, float(diff.max()))
    return worst


def _cfg_key(cfg):
    return {k: v for k, v in asdict(cfg).items() if k not in ("epochs", "patience", "device")}


def train_heads(model, cache_dir, cfg, out_pt, log=print, resume=True, extra_meta=None):
    """Train RankHeads on the cache; returns (heads with the best inner NLL, sidecar meta). See the module docstring."""
    t_start = time.time()
    games, arrays = read_cache(cache_dir)
    cache_meta = _cache_meta(cache_dir)
    train_rows, inner_rows = inner_split(games, cfg.inner_frac)
    train_idx = positions_of(arrays, train_rows)
    inner_idx = positions_of(arrays, inner_rows)
    if len(inner_idx) > cfg.inner_max_positions:
        inner_idx = np.sort(np.random.RandomState(INNER_SEED).choice(inner_idx, cfg.inner_max_positions, replace=False))
    if len(train_idx) == 0:
        raise ValueError(f"{cache_dir}: no training positions")
    device = cfg.device
    torch.manual_seed(cfg.seed)
    model.eval()
    model.requires_grad_(False)
    heads = RankHeads(model.policy_head, N_RANKS).to(device)
    heads.train()
    opt = torch.optim.Adam(heads.parameters(), lr=cfg.lr)
    weights = torch.as_tensor(position_weights(arrays["rank_idx"][train_idx]), dtype=torch.float32)
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)
    weight_of[train_idx] = weights.numpy()
    steps_per_epoch = math.ceil(len(train_idx) / cfg.batch)
    total_steps = cfg.epochs * steps_per_epoch
    eval_steps = max(1, int(round(cfg.eval_every * steps_per_epoch)))
    autocast = str(device).startswith("cuda")
    rows = neutral_rows()
    out_pt = Path(out_pt)
    ckpt_path = out_pt.with_name(out_pt.name + ".ckpt")

    use_inner = len(inner_idx) > 0  # inner_frac = 0: no early stopping, the last weights are kept (settrain's rule)
    state = {"step": 0, "history": [], "best": None, "since_improvement": 0, "stopped_early": False}
    resumed_from = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["cfg_key"] != _cfg_key(cfg) or ck["steps_per_epoch"] != steps_per_epoch:
            raise ValueError(f"{ckpt_path}: checkpoint of another run (batch, lr, seed, inner split or cache differ); "
                             "delete it or pass --overwrite")
        heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        resumed_from = state["step"]
        log(f"resumed from {ckpt_path} at step {resumed_from} (epoch {resumed_from / steps_per_epoch:.2f})", flush=True)

    def save_ckpt():
        atomic_save({"cfg_key": _cfg_key(cfg), "steps_per_epoch": steps_per_epoch, "heads": heads.state_dict(),
                     "opt": opt.state_dict(), "state": state}, ckpt_path)

    def evaluate(step):
        """Inner evaluation at `step`; keeps the best weights; True when patience is exhausted."""
        res = evaluate_nll(model, heads, arrays, inner_idx, cfg)
        epoch = step / steps_per_epoch
        state["history"].append((epoch, res["nll"]))
        improved = state["best"] is None or (np.isfinite(res["nll"]) and res["nll"] < state["best"]["nll"])
        if improved:
            state["best"] = {"nll": res["nll"], "step": step, "heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
                             "by_rank": res["nll_by_rank"], "ranks_present": res["ranks_present"]}
            state["since_improvement"] = 0
        else:
            state["since_improvement"] += 1
        log(f"epoch {epoch:.2f} inner NLL {res['nll']:.4f} (best {state['best']['nll']:.4f} @ {state['best']['step'] / steps_per_epoch:.2f})", flush=True)
        return state["since_improvement"] >= cfg.patience

    if use_inner and state["step"] == 0 and not state["history"]:
        evaluate(0)
        save_ckpt()
    step = state["step"]
    stop = state["stopped_early"]
    t_loop, n_loop = time.time(), 0
    while step < total_steps and not stop:
        epoch_i, pos_in_epoch = divmod(step, steps_per_epoch)
        order = train_idx[np.random.RandomState(cfg.seed + epoch_i).permutation(len(train_idx))]
        for k in range(pos_in_epoch, steps_per_epoch):
            sel = np.sort(order[k * cfg.batch:(k + 1) * cfg.batch])
            bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)
            w = torch.from_numpy(weight_of[sel]).to(device)
            logp = _true_head_logp(model, heads, bins, globs, legal, target, rank, color, rows, autocast)
            loss = (-logp * w).sum() / w.sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            n_loop += len(sel)
            state["step"] = step
            if step % eval_steps == 0 or step == total_steps:
                if use_inner and evaluate(step):
                    stop = True
                    state["stopped_early"] = True
                save_ckpt()
                log(f"  {n_loop / max(time.time() - t_loop, 1e-9):.0f} positions/s since the last report", flush=True)
                t_loop, n_loop = time.time(), 0
                if stop:
                    break
        else:
            continue
        break

    if use_inner:
        best = state["best"]
        heads.load_state_dict(best["heads"])
        best_epoch, nll_init, nll_best = best["step"] / steps_per_epoch, state["history"][0][1], best["nll"]
        by_rank, present = best["by_rank"], best["ranks_present"]
    else:
        best_epoch, nll_init, nll_best, by_rank, present = state["step"] / steps_per_epoch, None, None, [None] * N_RANKS, []
    heads.eval()
    per_rank = np.bincount(arrays["rank_idx"][train_idx], minlength=N_RANKS)
    meta = {"train_cfg": asdict(cfg), "train_folds": cache_meta["folds"], "max_games": cache_meta.get("max_games"),
            "n_train_positions": int(len(train_idx)), "n_inner_positions": int(len(inner_idx)),
            "per_rank_positions": {lv: int(per_rank[k]) for k, lv in enumerate(LEVELS)},
            "inner_nll_init": nll_init, "inner_nll_best": nll_best, "inner_nll_best_by_rank": by_rank,
            "best_epoch": best_epoch, "stopped_early": bool(state["stopped_early"]),
            "inner_ranks_present": present, "history": [(float(e), float(v)) for e, v in state["history"]],
            "steps_per_epoch": steps_per_epoch, "steps_run": int(state["step"]), "resumed_from_step": int(resumed_from),
            "wall_seconds": time.time() - t_start, "cache_dir": str(cache_dir), "cache_meta": cache_meta,
            "head_init": "humanv0_policy_head", **(extra_meta or {})}
    meta = save_heads(heads, out_pt, meta)
    return heads, meta


def _cache_meta(cache_dir):
    """The cache's meta.json; ValueError when it is missing or has no folds (the sidecar's train_folds, and so the
    out-of-fold check of l1b_score.py, depend on it -- run l1b_build_cache.py --merge-meta after a sharded build)."""
    p = Path(cache_dir) / "meta.json"
    if not p.exists():
        raise ValueError(f"{p} is missing: a sharded cache needs `l1b_build_cache.py --out {cache_dir} --merge-meta` first")
    meta = json.loads(p.read_text())
    if not meta.get("folds"):
        raise ValueError(f"{p} has no folds; the heads' train_folds would be unknown")
    return meta


def cache_summary(games):
    """Players, games, positions of a cache's games table (for logs)."""
    return {"players": int(games["player_id"].nunique()), "games": int(len(games)), "positions": int(pd.to_numeric(games["n_moves"]).sum())}
