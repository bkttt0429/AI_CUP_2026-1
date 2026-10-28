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
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from katago.train.model_pytorch import PolicyHead

from aicup.featcache import read_cache
from aicup.l1b import atomic_save, neutral_rows, trunk_features, trunk_path
from aicup.l1btrain import (N_RANKS, HeadsCfg, _all_reduce_, _barrier, _cache_meta, _cfg_key, _dist, _to_tensors, inner_split,
                            position_weights, positions_of, trunk_parameters)
from aicup.ladder import environment, to_json
from aicup.metric import LEVELS
from aicup.setmodel_io import file_sha256, sidecar_path
from aicup.settrain import INNER_SEED

FORMAT = 1
KIND = "strength"
__all__ = ["HeadsCfg", "StrengthHead", "pairwise_bt", "strength_scores", "evaluate_bt", "train_strength_head",
           "save_strength_head", "load_strength_head"]


class StrengthHead(torch.nn.Module):
    """One deep copy of a KataGo PolicyHead; forward gives its channel-0 map [B, 362], the per-point strength scores."""

    def __init__(self, policy_head: PolicyHead):
        super().__init__()
        self.head = copy.deepcopy(policy_head)
        self.requires_grad_(True)

    def forward(self, trunk, mask, mask_sum_hw, mask_sum):
        return self.head(trunk, mask=mask, mask_sum_hw=mask_sum_hw, mask_sum=mask_sum, extra_outputs=None)[:, 0, :]


def move_scores(model, head, bins, globs, target, color, rows, autocast, grad_trunk=False):
    """s of each position's move [B] (fp32 head on the trunk's features). grad_trunk: keep the trunk's graph (stage 2)."""
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)
    out = head(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())
    return out.gather(1, target[:, None])[:, 0]


def pairwise_bt(s, rank, w):
    """(weighted loss sum, weight sum, correctly ordered pairs, pairs) over every ordered pair (i, j) of positions of
    different ranks: -log sigmoid(sign(r_i - r_j) (s_i - s_j)) with weight sqrt(w_i w_j). Each unordered pair enters
    twice with the same loss, so the weighted mean is unchanged."""
    diff = s[:, None] - s[None, :]
    sign = torch.sign((rank[:, None] - rank[None, :]).to(s.dtype))
    cross = sign != 0
    pw = torch.sqrt(w[:, None] * w[None, :]) * cross
    nll = -torch.nn.functional.logsigmoid(sign * diff)
    return (nll * pw).sum(), pw.sum(), ((sign * diff > 0) & cross).sum(), cross.sum()


def strength_scores(model, head, bins, globs, target, colors, batch=64):
    """s of each target move: [N] float32 (the trunk under no_grad and, on CUDA, bf16 autocast; the head in fp32)."""
    device = next(head.parameters()).device
    n = len(target)
    out = np.empty(n, dtype=np.float32)
    rows = neutral_rows()
    autocast = device.type == "cuda"
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(n, i + batch)
            b = torch.as_tensor(np.asarray(bins[i:j]), device=device)
            g = torch.as_tensor(np.asarray(globs[i:j]), device=device)
            c = torch.as_tensor(np.asarray(colors[i:j]), device=device).long()
            tgt = torch.as_tensor(np.asarray(target[i:j]), device=device).long()
            out[i:j] = move_scores(model, head, b, g, tgt, c, rows, autocast).float().cpu().numpy()
    return out


def evaluate_bt(model, head, arrays, idx, cfg):
    """The Bradley-Terry loss over the inner positions `idx` in consecutive batches of cfg.batch (pairs within a batch),
    with class-balanced weights from the inner set: {"objective", "pair_accuracy", "pairs", "n", "mean_by_rank" [10]
    (None for an absent rank), "ranks_present"}."""
    device, rows = cfg.device, neutral_rows()
    autocast = str(device).startswith("cuda")
    idx = np.asarray(idx, dtype=np.int64)
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)
    weight_of[idx] = position_weights(arrays["rank_idx"][idx])
    me, world = _dist()
    mine = idx[me::world]
    acc = torch.zeros(4, dtype=torch.float64, device=device)  # loss sum, weight sum, correct pairs, pairs
    sums, counts = np.zeros(N_RANKS), np.zeros(N_RANKS, dtype=np.int64)
    was_training = head.training
    head.eval()
    with torch.no_grad():
        for i in range(0, len(mine), cfg.batch):
            sel = np.sort(mine[i:i + cfg.batch])
            bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)
            s = move_scores(model, head, bins, globs, target, color, rows, autocast).double()
            num, den, correct, pairs = pairwise_bt(s, rank, torch.from_numpy(weight_of[sel]).to(device).double())
            acc += torch.stack([num, den, correct.double(), pairs.double()])
            r = rank.cpu().numpy()
            np.add.at(sums, r, s.cpu().numpy())
            np.add.at(counts, r, 1)
    if world > 1:
        acc = _all_reduce_(acc)
        t = _all_reduce_(torch.as_tensor(np.concatenate([sums, counts.astype(np.float64)]), dtype=torch.float64, device=device)).cpu().numpy()
        sums, counts = t[:N_RANKS], np.rint(t[N_RANKS:]).astype(np.int64)
    head.train(was_training)
    num, den, correct, pairs = acc.cpu().numpy().tolist()
    present = [int(k) for k in np.flatnonzero(counts > 0)]
    return {"objective": float(num / den) if den > 0 else float("nan"), "pair_accuracy": float(correct / pairs) if pairs > 0 else float("nan"),
            "pairs": int(pairs), "n": int(counts.sum()), "mean_by_rank": [float(sums[k] / counts[k]) if counts[k] else None for k in range(N_RANKS)],
            "ranks_present": present}


def save_strength_head(head, path, meta, model_state=None):
    """state_dict -> path (.pt) and the sidecar path.with_suffix('.json') of kind "strength"; model_state (stage 2) is written
    to l1b.trunk_path(path) and named in the sidecar. Returns the sidecar dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(head.state_dict(), path)
    trunk = {}
    if model_state is not None:
        tp = trunk_path(path)
        atomic_save(model_state, tp)
        trunk = {"trunk_file": tp.name, "trunk_sha256": file_sha256(tp)}
    device = next(head.parameters()).device
    side = {"format": FORMAT, "kind": KIND, "n_params": int(sum(p.numel() for p in head.parameters())), "state_sha256": file_sha256(path),
            **trunk, **environment(str(device)), **meta}
    sidecar_path(path).write_text(to_json(side, indent=1) + "\n")
    return json.loads(sidecar_path(path).read_text())


def load_strength_head(path, model, device="cpu"):
    """(StrengthHead, sidecar, warnings) restored from path onto model.policy_head's architecture; ValueError when the
    sidecar is missing, is not of kind "strength", or a sha256 differs; a stage-2 sidecar's trunk is loaded into `model`."""
    path = Path(path)
    side_path = sidecar_path(path)
    if not side_path.exists():
        raise ValueError(f"{path}: sidecar {side_path.name} is missing")
    meta = json.loads(side_path.read_text())
    if meta.get("format") != FORMAT or meta.get("kind") != KIND:
        raise ValueError(f"{path}: sidecar format {meta.get('format')!r} / kind {meta.get('kind')!r}; this code reads format {FORMAT} kind {KIND!r}")
    sha = file_sha256(path)
    if meta.get("state_sha256") != sha:
        raise ValueError(f"{path}: sha256 {sha[:12]} differs from the sidecar's {str(meta.get('state_sha256'))[:12]}")
    if meta.get("trunk_file"):
        tp = path.with_name(meta["trunk_file"])
        if not tp.exists():
            raise ValueError(f"{path}: the fine-tuned trunk {tp.name} is missing")
        tsha = file_sha256(tp)
        if meta.get("trunk_sha256") != tsha:
            raise ValueError(f"{tp}: sha256 {tsha[:12]} differs from the sidecar's {str(meta.get('trunk_sha256'))[:12]}")
        try:
            model.load_state_dict(torch.load(tp, map_location="cpu", weights_only=True))
        except RuntimeError as e:
            raise ValueError(f"{tp}: state_dict does not fit the model ({e})") from e
    head = StrengthHead(model.policy_head)
    try:
        head.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except RuntimeError as e:
        raise ValueError(f"{path}: state_dict does not fit model.policy_head's architecture ({e})") from e
    head.to(device).eval()
    warnings = []
    current = environment("cpu")["code_sha256"]
    if meta.get("code_sha256") != current:
        warnings.append(f"{path}: trained by other training code (code_sha256 {str(meta.get('code_sha256'))[:12]} vs {current[:12]})")
    return head, meta, warnings


def train_strength_head(model, cache_dir, cfg, out_pt, log=print, resume=True, extra_meta=None, init_head=None):
    """Train a StrengthHead (and, with cfg.trunk_lr > 0, the trunk) on the cache with the Bradley-Terry objective; returns
    (head with the best inner objective, sidecar meta). init_head: a StrengthHead state_dict to start from (default: a
    copy of model.policy_head). Conventions as l1btrain.train_heads (see the module docstring)."""
    t_start = time.time()
    me, world = _dist()
    if me != 0:
        log = lambda *a, **k: None  # noqa: E731
    games, arrays = read_cache(cache_dir)
    cache_meta = _cache_meta(cache_dir)
    train_rows, inner_rows = inner_split(games, cfg.inner_frac)
    train_idx = positions_of(arrays, train_rows)
    inner_idx = positions_of(arrays, inner_rows)
    if len(inner_idx) > cfg.inner_max_positions:
        inner_idx = np.sort(np.random.RandomState(INNER_SEED).choice(inner_idx, cfg.inner_max_positions, replace=False))
    if len(train_idx) == 0:
        raise ValueError(f"{cache_dir}: no training positions")
    if len(np.unique(arrays["rank_idx"][train_idx])) < 2:
        raise ValueError(f"{cache_dir}: the Bradley-Terry objective needs positions of at least two ranks")
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
    head = StrengthHead(model.policy_head).to(device)
    if init_head is not None:
        head.load_state_dict(init_head)
    head.train()
    groups = [{"params": list(head.parameters()), "lr": cfg.lr}]
    if train_trunk:
        groups.append({"params": trunk_params, "lr": cfg.trunk_lr})
    opt = torch.optim.Adam(groups)
    params = [p for g in groups for p in g["params"]]
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)
    weight_of[train_idx] = position_weights(arrays["rank_idx"][train_idx])
    global_batch = cfg.batch * world
    steps_per_epoch = math.ceil(len(train_idx) / global_batch)
    total_steps = cfg.epochs * steps_per_epoch
    eval_steps = max(1, int(round(cfg.eval_every * steps_per_epoch)))
    autocast = str(device).startswith("cuda")
    rows = neutral_rows()
    out_pt = Path(out_pt)
    ckpt_path = out_pt.with_name(out_pt.name + ".ckpt")

    def snapshot():
        snap = {"head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        if train_trunk:
            snap["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return snap

    use_inner = len(inner_idx) > 0
    state = {"step": 0, "history": [], "best": None, "since_improvement": 0, "stopped_early": False}
    resumed_from = 0
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("kind") != KIND or ck["cfg_key"] != _cfg_key(cfg) or ck["steps_per_epoch"] != steps_per_epoch:
            raise ValueError(f"{ckpt_path}: checkpoint of another run (kind, batch, lr, trunk_lr, seed, world size, inner split or "
                             "cache differ); delete it or pass --overwrite")
        head.load_state_dict(ck["head"])
        if train_trunk:
            model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        resumed_from = state["step"]
        log(f"resumed from {ckpt_path} at step {resumed_from} (epoch {resumed_from / steps_per_epoch:.2f})", flush=True)

    def save_ckpt():
        if me == 0:
            ck = {"kind": KIND, "cfg_key": _cfg_key(cfg), "steps_per_epoch": steps_per_epoch, "head": head.state_dict(),
                  "opt": opt.state_dict(), "state": state}
            if train_trunk:
                ck["model"] = model.state_dict()
            atomic_save(ck, ckpt_path)
        _barrier()

    def evaluate(step):
        res = evaluate_bt(model, head, arrays, inner_idx, cfg)
        epoch = step / steps_per_epoch
        state["history"].append((epoch, res["objective"], res["pair_accuracy"]))
        improved = state["best"] is None or (np.isfinite(res["objective"]) and res["objective"] < state["best"]["objective"])
        if improved:
            state["best"] = {"objective": res["objective"], "pair_accuracy": res["pair_accuracy"], "step": step,
                             "mean_by_rank": res["mean_by_rank"], "ranks_present": res["ranks_present"], **snapshot()}
            state["since_improvement"] = 0
        else:
            state["since_improvement"] += 1
        log(f"epoch {epoch:.2f} inner BT loss {res['objective']:.4f} pair accuracy {res['pair_accuracy']:.4f} "
            f"(best {state['best']['objective']:.4f} @ {state['best']['step'] / steps_per_epoch:.2f})", flush=True)
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
                s = move_scores(model, head, bins, globs, target, color, rows, autocast, grad_trunk=train_trunk)
                num, den, _, _ = pairwise_bt(s, rank, w)
                den_global = _all_reduce_(den.detach().clone())
                loss = num / den_global if float(den_global) > 0 else s.sum() * 0.0
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                loss.backward()
            else:
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
        head.load_state_dict(best["head"])
        if train_trunk:
            model.load_state_dict(best["model"])
        best_epoch, obj_init, obj_best = best["step"] / steps_per_epoch, state["history"][0][1], best["objective"]
        acc_init, acc_best = state["history"][0][2], best["pair_accuracy"]
        mean_by_rank, present = best["mean_by_rank"], best["ranks_present"]
    else:
        best_epoch, obj_init, obj_best, acc_init, acc_best = state["step"] / steps_per_epoch, None, None, None, None
        mean_by_rank, present = [None] * N_RANKS, []
    head.eval()
    per_rank = np.bincount(arrays["rank_idx"][train_idx], minlength=N_RANKS)
    meta = {"train_cfg": asdict(cfg), "train_folds": cache_meta["folds"], "max_games": cache_meta.get("max_games"),
            "n_train_positions": int(len(train_idx)), "n_inner_positions": int(len(inner_idx)),
            "per_rank_positions": {lv: int(per_rank[k]) for k, lv in enumerate(LEVELS)},
            "inner_bt_init": obj_init, "inner_bt_best": obj_best, "inner_pair_accuracy_init": acc_init, "inner_pair_accuracy_best": acc_best,
            "inner_mean_by_rank_best": mean_by_rank, "inner_ranks_present": present,
            "best_epoch": best_epoch, "stopped_early": bool(state["stopped_early"]),
            "history": [(float(e), float(v), float(a)) for e, v, a in state["history"]],
            "steps_per_epoch": steps_per_epoch, "steps_run": int(state["step"]), "resumed_from_step": int(resumed_from),
            "world_size": int(world), "global_batch": int(global_batch), "trunk_trained": bool(train_trunk),
            "objective": "Bradley-Terry over the pairs of different ranks within each process's batch share, weights sqrt(w_i w_j)",
            "selection": "inner Bradley-Terry loss", "n_trunk_params": int(sum(p.numel() for p in trunk_params)),
            "wall_seconds": time.time() - t_start, "cache_dir": str(cache_dir), "cache_meta": cache_meta,
            "head_init": "humanv0_policy_head", **(extra_meta or {})}
    if me == 0:
        meta = save_strength_head(head, out_pt, meta, model_state=model.state_dict() if train_trunk else None)
    _barrier()
    return head, meta
