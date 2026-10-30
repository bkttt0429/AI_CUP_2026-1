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

Stage 2 (2026-10-26, issue #6): cfg.trunk_lr > 0 also fine-tunes the trunk -- every model parameter that
reaches `trunkfinal` (found by a probe backward, trunk_parameters), in its own Adam group at trunk_lr; the
model stays in eval mode (humanv0 has no batch statistics: fixscaleonenorm, trunk_normless) and its fine-tuned
state_dict is saved next to the heads (l1b.save_heads, model_state). `init_heads` starts the heads from a saved
state (the stage-1 heads) instead of the policy-head copies. Under torch.distributed (torchrun) every rank
takes rank::world of each global batch of cfg.batch x world positions; the loss is normalised by the global
weight sum and the gradients are summed across ranks, so a step equals one single-process step on the whole
global batch; the inner NLL is evaluated on rank::world of the inner positions and summed; rank 0 alone
writes checkpoints and outputs.

Cross-fit (2026-10-29, issue #16): train_heads takes several caches read as one (featcache.read_caches) and
`exclude_folds`, the folds whose players' games are left out (splits/player_folds.csv), so the five models f_k of
the five-fold cross-fit each train on cache_A + cache_B + cache_0 minus fold k; the sidecar's train_folds is the
caches' folds minus the excluded ones (caches_meta), which l1b_score.py's out-of-fold check reads.
"""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aicup.featcache import read_caches, unpack_bins, unpack_legal
from aicup.l1b import RankHeads, atomic_save, masked_logp, neutral_rows, save_heads, trunk_features
from aicup.ladder import FOLDS_CSV
from aicup.metric import LEVELS, RANK_TO_IDX
from aicup.settrain import INNER_SEED, inner_players

N_RANKS = len(LEVELS)


def cache_dirs_of(cache_dir):
    """[Path, ...] of one directory or a list of them (train_heads takes either)."""
    return [Path(cache_dir)] if isinstance(cache_dir, (str, Path)) else [Path(d) for d in cache_dir]


def excluded_players(exclude_folds, folds_csv=FOLDS_CSV):
    """The player ids of `exclude_folds` in the folds table (ValueError when a fold has no players there)."""
    exclude = sorted({int(f) for f in exclude_folds})
    if not exclude:
        return set()
    folds = pd.read_csv(folds_csv, dtype={"player_id": str})
    missing = [f for f in exclude if not (folds["fold"] == f).any()]
    if missing:
        raise ValueError(f"{folds_csv}: no players in fold(s) {missing}")
    return set(folds.loc[folds["fold"].isin(exclude), "player_id"].astype(str))


def caches_meta(cache_dirs, exclude_folds=()):
    """The meta of the caches read together (issue #16): folds = the union of the caches' folds minus exclude_folds,
    max_games / min_games shared; ValueError when two caches share a fold, an excluded fold is not in the union,
    or the caches disagree on max_games. One cache without exclusions returns its own meta unchanged."""
    dirs = cache_dirs_of(cache_dirs)
    exclude = sorted({int(f) for f in exclude_folds})
    metas = [_cache_meta(d) for d in dirs]
    if len(metas) == 1 and not exclude:
        return metas[0]
    seen, dups = set(), set()
    for m in metas:
        folds = set(int(f) for f in m["folds"])
        dups |= seen & folds
        seen |= folds
    if dups:
        raise ValueError(f"fold(s) {sorted(dups)} are in more than one of the caches {[str(d) for d in dirs]}")
    outside = [f for f in exclude if f not in seen]
    if outside:
        raise ValueError(f"--exclude-folds {outside} are not in the caches' folds {sorted(seen)}")
    if len({json.dumps(m.get("max_games")) for m in metas}) != 1:
        raise ValueError(f"the caches disagree on max_games: {[m.get('max_games') for m in metas]}")
    folds = sorted(seen - set(exclude))
    if not folds:
        raise ValueError("every fold of the caches is excluded")
    return {"folds": folds, "max_games": metas[0].get("max_games"), "min_games": metas[0].get("min_games"),
            "excluded_folds": exclude, "cache_dirs": [str(d) for d in dirs],
            "caches": [{"dir": str(d), **{k: m.get(k) for k in ("folds", "players", "games", "positions", "games_sha256", "model_input_config")}}
                       for d, m in zip(dirs, metas)]}


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
    trunk_lr: float = 0.0  # 0: the trunk is frozen (stage 1); > 0: fine-tuned at this lr (stage 2)
    disc_weight: float = 0.0  # > 0: add disc_weight x the rank cross-entropy of the 10 heads' log p (discriminative)


def _dist():
    """(rank, world size) under an initialised torch.distributed process group, else (0, 1)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_reduce_(t):
    """Sum a tensor across ranks in place (no-op in a single process)."""
    if _dist()[1] > 1:
        import torch.distributed as dist
        dist.all_reduce(t)
    return t


def _barrier():
    if _dist()[1] > 1:
        import torch.distributed as dist
        dist.barrier()


def trunk_parameters(model, device):
    """The names of the model parameters that reach `trunkfinal` (a probe forward/backward on one empty board under
    the neutral metadata); those get requires_grad True, every other parameter (policy / value heads) False."""
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    bins = torch.zeros(1, 22, 19, 19, device=device)
    bins[:, 0] = 1.0
    trunk, _, _, _ = trunk_features(model, bins, torch.zeros(1, 19, device=device), torch.zeros(1, dtype=torch.long, device=device),
                                    neutral_rows())
    trunk.sum().backward()
    names = {n for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    return sorted(names)


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


def _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk=False):
    """log p of each position's move under every head: [B, n_ranks]. grad_trunk: keep the trunk's graph (stage 2)."""
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)
    logits = heads(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())
    logp = masked_logp(logits, legal)
    return logp.gather(2, target[:, None, None].expand(-1, logp.shape[1], 1))[:, :, 0]


def _true_head_logp(model, heads, bins, globs, legal, target, rank, color, rows, autocast, grad_trunk=False):
    """log p of each position's move under the head of its own rank: [B]. grad_trunk: keep the trunk's graph (stage 2)."""
    lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk)
    return lp[torch.arange(len(target), device=lp.device), rank]


def position_losses(lp, rank, disc_weight):
    """Per-position (generative NLL of the true head, rank cross-entropy, total = NLL + disc_weight x CE) from the
    heads' log p [B, n_ranks]: the CE treats the 10 log p of the played move as logits over the ranks (a uniform
    prior), so it rewards heads whose differences point at the player's rank."""
    i = torch.arange(len(rank), device=lp.device)
    gen = -lp[i, rank]
    disc = -torch.log_softmax(lp, dim=1)[i, rank]
    return gen, disc, gen + disc_weight * disc if disc_weight > 0 else gen


def evaluate_nll(model, heads, arrays, idx, cfg):
    """Class-balanced per-move NLL of the true-rank head over positions `idx`: {"nll": mean over the ranks present
    of the rank's mean NLL, "nll_by_rank": [10] (None for an absent rank), "ranks_present", "n"}."""
    device, rows = cfg.device, neutral_rows()
    autocast = str(device).startswith("cuda")
    sums, dsums, counts = np.zeros(N_RANKS), np.zeros(N_RANKS), np.zeros(N_RANKS, dtype=np.int64)
    was_training = heads.training
    heads.eval()
    me, world = _dist()
    with torch.no_grad():
        for bins, globs, legal, target, rank, color in batches(arrays, np.asarray(idx)[me::world], cfg.batch, None, device):
            lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast).double()
            gen, disc, _ = position_losses(lp, rank, 0.0)
            r = rank.cpu().numpy()
            np.add.at(sums, r, gen.cpu().numpy())
            np.add.at(dsums, r, disc.cpu().numpy())
            np.add.at(counts, r, 1)
    if world > 1:
        t = _all_reduce_(torch.as_tensor(np.concatenate([sums, dsums, counts.astype(np.float64)]), dtype=torch.float64, device=device))
        t = t.cpu().numpy()
        sums, dsums, counts = t[:N_RANKS], t[N_RANKS:2 * N_RANKS], np.rint(t[2 * N_RANKS:]).astype(np.int64)
    heads.train(was_training)
    present = [int(k) for k in np.flatnonzero(counts > 0)]
    by_rank = [float(sums[k] / counts[k]) if counts[k] else None for k in range(N_RANKS)]
    nll = float(np.mean([by_rank[k] for k in present])) if present else float("nan")
    disc = float(np.mean([dsums[k] / counts[k] for k in present])) if present else float("nan")
    lam = getattr(cfg, "disc_weight", 0.0)
    return {"nll": nll, "nll_by_rank": by_rank, "ranks_present": present, "n": int(counts.sum()), "disc": disc,
            "objective": nll + lam * disc if lam > 0 else nll}


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


def train_heads(model, cache_dir, cfg, out_pt, log=print, resume=True, extra_meta=None, init_heads=None, exclude_folds=(),
                folds_csv=FOLDS_CSV):
    """Train RankHeads (and, with cfg.trunk_lr > 0, the trunk) on the cache; returns (heads with the best inner NLL,
    sidecar meta). init_heads: a RankHeads state_dict to start from (default: copies of model.policy_head). cache_dir may
    be a list of caches read as one (issue #16), and exclude_folds leaves out the games of those folds' players
    (folds_csv), so the sidecar's train_folds is the caches' union minus them. See the module docstring; under
    torch.distributed only rank 0 logs and writes, and every rank returns the same heads."""
    t_start = time.time()
    me, world = _dist()
    if me != 0:
        log = lambda *a, **k: None  # noqa: E731
    dirs = cache_dirs_of(cache_dir)
    cache_meta = caches_meta(dirs, exclude_folds)
    drop = excluded_players(exclude_folds, folds_csv)
    games, arrays = read_caches(dirs, drop)
    if exclude_folds:
        log(f"caches {[str(d) for d in dirs]}: folds {cache_meta['folds']} after leaving out {sorted(set(map(int, exclude_folds)))} "
            f"({len(drop)} players): {len(games)} games, {len(arrays['target'])} positions", flush=True)
    train_rows, inner_rows = inner_split(games, cfg.inner_frac)
    train_idx = positions_of(arrays, train_rows)
    inner_idx = positions_of(arrays, inner_rows)
    if len(inner_idx) > cfg.inner_max_positions:
        inner_idx = np.sort(np.random.RandomState(INNER_SEED).choice(inner_idx, cfg.inner_max_positions, replace=False))
    if len(train_idx) == 0:
        raise ValueError(f"{[str(d) for d in dirs]}: no training positions")
    device = cfg.device
    torch.manual_seed(cfg.seed)
    model.eval()
    train_trunk = cfg.trunk_lr > 0
    if train_trunk:
        trunk_names = trunk_parameters(model, device)
    else:
        model.requires_grad_(False)
        trunk_names = []
    trunk_params = [p for n, p in model.named_parameters() if n in set(trunk_names)]
    heads = RankHeads(model.policy_head, N_RANKS).to(device)
    if init_heads is not None:
        heads.load_state_dict(init_heads)
    heads.train()
    groups = [{"params": list(heads.parameters()), "lr": cfg.lr}]
    if train_trunk:
        groups.append({"params": trunk_params, "lr": cfg.trunk_lr})
    opt = torch.optim.Adam(groups)
    params = [p for g in groups for p in g["params"]]
    weights = torch.as_tensor(position_weights(arrays["rank_idx"][train_idx]), dtype=torch.float32)
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)
    weight_of[train_idx] = weights.numpy()
    global_batch = cfg.batch * world
    steps_per_epoch = math.ceil(len(train_idx) / global_batch)
    total_steps = cfg.epochs * steps_per_epoch
    eval_steps = max(1, int(round(cfg.eval_every * steps_per_epoch)))
    autocast = str(device).startswith("cuda")
    rows = neutral_rows()
    out_pt = Path(out_pt)
    ckpt_path = out_pt.with_name(out_pt.name + ".ckpt")

    def snapshot():
        snap = {"heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}}
        if train_trunk:
            snap["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return snap

    use_inner = len(inner_idx) > 0  # inner_frac = 0: no early stopping, the last weights are kept (settrain's rule)
    state = {"step": 0, "history": [], "best": None, "since_improvement": 0, "stopped_early": False}
    resumed_from = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["cfg_key"] != _cfg_key(cfg) or ck["steps_per_epoch"] != steps_per_epoch or ck.get("train_folds", cache_meta["folds"]) != cache_meta["folds"]:
            raise ValueError(f"{ckpt_path}: checkpoint of another run (batch, lr, trunk_lr, seed, world size, inner split, cache or "
                             "folds differ); delete it or pass --overwrite")
        heads.load_state_dict(ck["heads"])
        if train_trunk:
            model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        resumed_from = state["step"]
        log(f"resumed from {ckpt_path} at step {resumed_from} (epoch {resumed_from / steps_per_epoch:.2f})", flush=True)

    def save_ckpt():
        if me == 0:
            ck = {"cfg_key": _cfg_key(cfg), "steps_per_epoch": steps_per_epoch, "train_folds": cache_meta["folds"],
                  "heads": heads.state_dict(), "opt": opt.state_dict(), "state": state}
            if train_trunk:
                ck["model"] = model.state_dict()
            atomic_save(ck, ckpt_path)
        _barrier()

    def evaluate(step):
        """Inner evaluation at `step`; keeps the best weights; True when patience is exhausted."""
        res = evaluate_nll(model, heads, arrays, inner_idx, cfg)
        epoch = step / steps_per_epoch
        state["history"].append((epoch, res["objective"]))
        state.setdefault("history_detail", []).append((epoch, res["nll"], res["disc"]))
        improved = state["best"] is None or (np.isfinite(res["objective"]) and res["objective"] < state["best"]["objective"])
        if improved:
            state["best"] = {"nll": res["nll"], "disc": res["disc"], "objective": res["objective"], "step": step, **snapshot(),
                             "by_rank": res["nll_by_rank"], "ranks_present": res["ranks_present"]}
            state["since_improvement"] = 0
        else:
            state["since_improvement"] += 1
        extra = f" rank CE {res['disc']:.4f} objective {res['objective']:.4f}" if cfg.disc_weight > 0 else ""
        log(f"epoch {epoch:.2f} inner NLL {res['nll']:.4f}{extra} (best {state['best']['objective']:.4f} @ "
            f"{state['best']['step'] / steps_per_epoch:.2f})", flush=True)
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
            sel = np.sort(order[k * global_batch:(k + 1) * global_batch][me::world])
            opt.zero_grad(set_to_none=True)
            if len(sel):
                bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)
                w = torch.from_numpy(weight_of[sel]).to(device)
                lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk=train_trunk)
                _, _, per = position_losses(lp, rank, cfg.disc_weight)
                w_sum = _all_reduce_(w.sum().detach().clone())
                loss = (per * w).sum() / w_sum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                loss.backward()
            else:  # the last global batch can leave a rank without positions; it still joins the reductions
                _all_reduce_(torch.zeros((), device=device))
            if world > 1:
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
                flat = _all_reduce_(torch.cat([g.reshape(-1) for g in grads]))
                offset = 0
                for p, g in zip(params, grads):
                    p.grad = flat[offset:offset + g.numel()].view_as(g).clone()
                    offset += g.numel()
            opt.step()
            step += 1
            n_loop += len(sel) * world
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
        if train_trunk:
            model.load_state_dict(best["model"])
        detail = state.get("history_detail") or [(e, v, float("nan")) for e, v in state["history"]]
        best_epoch, nll_init, nll_best = best["step"] / steps_per_epoch, detail[0][1], best["nll"]
        disc_init, disc_best = detail[0][2], best.get("disc")
        obj_init, obj_best = state["history"][0][1], best.get("objective", best["nll"])
        by_rank, present = best["by_rank"], best["ranks_present"]
    else:
        best_epoch, nll_init, nll_best, by_rank, present = state["step"] / steps_per_epoch, None, None, [None] * N_RANKS, []
        disc_init = disc_best = obj_init = obj_best = None
    heads.eval()
    per_rank = np.bincount(arrays["rank_idx"][train_idx], minlength=N_RANKS)
    meta = {"train_cfg": asdict(cfg), "train_folds": cache_meta["folds"], "max_games": cache_meta.get("max_games"),
            "n_train_positions": int(len(train_idx)), "n_inner_positions": int(len(inner_idx)),
            "per_rank_positions": {lv: int(per_rank[k]) for k, lv in enumerate(LEVELS)},
            "inner_nll_init": nll_init, "inner_nll_best": nll_best, "inner_nll_best_by_rank": by_rank,
            "inner_disc_init": disc_init, "inner_disc_best": disc_best, "inner_objective_init": obj_init, "inner_objective_best": obj_best,
            "best_epoch": best_epoch, "stopped_early": bool(state["stopped_early"]),
            "inner_ranks_present": present, "history": [(float(e), float(v)) for e, v in state["history"]],
            "steps_per_epoch": steps_per_epoch, "steps_run": int(state["step"]), "resumed_from_step": int(resumed_from),
            "world_size": int(world), "global_batch": int(global_batch), "trunk_trained": bool(train_trunk),
            "selection": "inner NLL" if cfg.disc_weight <= 0 else f"inner NLL + {cfg.disc_weight:g} x rank CE",
            "history_detail": [(float(e), float(n), float(d)) for e, n, d in state.get("history_detail", [])],
            "n_trunk_params": int(sum(p.numel() for p in trunk_params)),
            "wall_seconds": time.time() - t_start, "cache_dir": str(dirs[0]) if len(dirs) == 1 else [str(d) for d in dirs],
            "cache_dirs": [str(d) for d in dirs], "excluded_folds": sorted({int(f) for f in exclude_folds}), "cache_meta": cache_meta,
            "n_games": int(len(games)), "n_players": int(games["player_id"].nunique()),
            "head_init": "humanv0_policy_head", **(extra_meta or {})}
    if me == 0:
        meta = save_heads(heads, out_pt, meta, model_state=model.state_dict() if train_trunk else None)
    _barrier()
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
