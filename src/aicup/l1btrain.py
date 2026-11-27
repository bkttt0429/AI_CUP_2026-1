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
import json  # 讀取快取的 meta 與比對 max_games
import math  # 計算每個 epoch 的步數
import time  # 計時
from dataclasses import asdict, dataclass  # 訓練設定的資料類別
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列
import pandas as pd  # 讀取分折表
import torch  # 網路訓練

from aicup.featcache import read_caches, unpack_bins, unpack_legal  # 讀取並解壓局面快取
from aicup.l1b import RankHeads, atomic_save, masked_logp, neutral_rows, save_heads, trunk_features  # 等級頭與相關工具
from aicup.ladder import FOLDS_CSV  # 分折表的路徑
from aicup.metric import LEVELS, RANK_TO_IDX  # 10 個等級與其索引
from aicup.settrain import INNER_SEED, inner_players  # 與 L3 梯子共用的內層留出規則

N_RANKS = len(LEVELS)  # 等級頭的個數（10）


def cache_dirs_of(cache_dir):  # 統一成快取資料夾的清單
    """[Path, ...] of one directory or a list of them (train_heads takes either)."""
    return [Path(cache_dir)] if isinstance(cache_dir, (str, Path)) else [Path(d) for d in cache_dir]  # 單一路徑包成清單


def excluded_players(exclude_folds, folds_csv=FOLDS_CSV):  # 被排除之折的玩家
    """The player ids of `exclude_folds` in the folds table (ValueError when a fold has no players there)."""
    exclude = sorted({int(f) for f in exclude_folds})  # 去除重複並排序
    if not exclude:  # 沒有要排除的折
        return set()  # 回傳空集合
    folds = pd.read_csv(folds_csv, dtype={"player_id": str})  # 讀入分折表
    missing = [f for f in exclude if not (folds["fold"] == f).any()]  # 表中沒有玩家的折
    if missing:  # 指定了不存在的折
        raise ValueError(f"{folds_csv}: no players in fold(s) {missing}")  # 拒絕
    return set(folds.loc[folds["fold"].isin(exclude), "player_id"].astype(str))  # 這些折的玩家


def caches_meta(cache_dirs, exclude_folds=()):  # 合併並核對多個快取的 meta
    """The meta of the caches read together (issue #16): folds = the union of the caches' folds minus exclude_folds,
    max_games / min_games shared; ValueError when two caches share a fold, an excluded fold is not in the union,
    or the caches disagree on max_games. One cache without exclusions returns its own meta unchanged."""
    dirs = cache_dirs_of(cache_dirs)  # 快取資料夾清單
    exclude = sorted({int(f) for f in exclude_folds})  # 要排除的折
    metas = [_cache_meta(d) for d in dirs]  # 各快取的 meta.json
    if len(metas) == 1 and not exclude:  # 單一快取且不排除任何折
        return metas[0]  # 原樣回傳
    seen, dups = set(), set()  # 已見過的折與重複的折
    for m in metas:  # 逐一快取
        folds = set(int(f) for f in m["folds"])  # 這個快取的折
        dups |= seen & folds  # 與先前快取重疊的折
        seen |= folds  # 加入聯集
    if dups:  # 有折重複出現
        raise ValueError(f"fold(s) {sorted(dups)} are in more than one of the caches {[str(d) for d in dirs]}")  # 拒絕
    outside = [f for f in exclude if f not in seen]  # 不在快取中的排除折
    if outside:  # 排除了快取沒有的折
        raise ValueError(f"--exclude-folds {outside} are not in the caches' folds {sorted(seen)}")  # 拒絕
    if len({json.dumps(m.get("max_games")) for m in metas}) != 1:  # 各快取的 max_games 不一致（以 JSON 字串比較）
        raise ValueError(f"the caches disagree on max_games: {[m.get('max_games') for m in metas]}")  # 拒絕
    if len({cache_side(m) for m in metas}) != 1:  # 目標方／對手方的局面不一致
        raise ValueError(f"the caches disagree on side (target / opponent positions): {[cache_side(m) for m in metas]}")  # 拒絕
    folds = sorted(seen - set(exclude))  # 實際用於訓練的折
    if not folds:  # 全部的折都被排除
        raise ValueError("every fold of the caches is excluded")  # 拒絕
    return {"folds": folds, "max_games": metas[0].get("max_games"), "min_games": metas[0].get("min_games"), "side": cache_side(metas[0]),  # 合併後的 meta
            "excluded_folds": exclude, "cache_dirs": [str(d) for d in dirs],  # 排除的折與快取來源
            "caches": [{"dir": str(d), **{k: m.get(k) for k in ("folds", "players", "games", "positions", "games_sha256", "model_input_config", "side")}}  # 每個快取的摘要
                       for d, m in zip(dirs, metas)]}  # 逐一快取


def cache_side(meta):  # 快取存的是哪一方的局面
    """"target" (the labelled player's positions, the default of caches built before issue #33) or "opponent"."""
    return meta.get("side") or "target"  # 舊快取沒有 side 欄時視為 target


@dataclass(frozen=True)  # 不可變的資料類別
class HeadsCfg:  # 等級頭的訓練設定
    """epochs: maximum epochs; eval_every: inner evaluations every this many epochs (0.25 = four per epoch);
    patience: evaluations in a row without a lower inner NLL before training stops; inner_frac: share of each
    rank's players held out (0: none); inner_max_positions: cap on the inner positions evaluated."""
    epochs: int = 3  # 最多 3 個 epoch
    batch: int = 256  # 每個行程的批次大小
    lr: float = 1e-4  # 等級頭的學習率
    eval_every: float = 0.25  # 每 1/4 個 epoch 做一次內層評估
    patience: int = 4  # 連續 4 次沒有改善就停止
    inner_frac: float = 0.1  # 每級留出 10% 的玩家作為內層
    inner_max_positions: int = 400_000  # 內層最多評估 40 萬個局面
    seed: int = 0  # 訓練的亂數種子
    device: str = "cuda"  # 訓練用的裝置
    trunk_lr: float = 0.0  # 0：主幹凍結（第一階段）；> 0：以此學習率微調主幹（第二階段）
    disc_weight: float = 0.0  # > 0：另加 disc_weight x 10 個頭之 log p 的等級交叉熵（鑑別式）


def _dist():  # 取得分散式訓練的 rank 與 world size
    """(rank, world size) under an initialised torch.distributed process group, else (0, 1)."""
    import torch.distributed as dist  # 用到時才匯入
    if dist.is_available() and dist.is_initialized():  # 在 torchrun 之下
        return dist.get_rank(), dist.get_world_size()  # 目前行程的編號與總行程數
    return 0, 1  # 單一行程


def _all_reduce_(t):  # 跨行程加總
    """Sum a tensor across ranks in place (no-op in a single process)."""
    if _dist()[1] > 1:  # 多個行程才需要
        import torch.distributed as dist  # 用到時才匯入
        dist.all_reduce(t)  # 預設的運算為加總
    return t  # 回傳同一個張量


def _barrier():  # 跨行程同步
    if _dist()[1] > 1:  # 多個行程才需要
        import torch.distributed as dist  # 用到時才匯入
        dist.barrier()  # 等所有行程到齊


def trunk_parameters(model, device):  # 找出會影響主幹輸出的參數
    """The names of the model parameters that reach `trunkfinal` (a probe forward/backward on one empty board under
    the neutral metadata); those get requires_grad True, every other parameter (policy / value heads) False."""
    model.requires_grad_(True)  # 先讓所有參數都能收到梯度
    model.zero_grad(set_to_none=True)  # 清掉舊的梯度
    bins = torch.zeros(1, 22, 19, 19, device=device)  # 一個空棋盤
    bins[:, 0] = 1.0  # 平面 0：所有點都在盤上
    trunk, _, _, _ = trunk_features(model, bins, torch.zeros(1, 19, device=device), torch.zeros(1, dtype=torch.long, device=device),  # 探測用的前向計算
                                    neutral_rows())  # 使用中性的 metadata
    trunk.sum().backward()  # 反向傳播到所有相關的參數
    names = {n for n, p in model.named_parameters() if p.grad is not None}  # 收到梯度者即為主幹參數
    model.zero_grad(set_to_none=True)  # 清掉探測產生的梯度
    for n, p in model.named_parameters():  # 逐一參數
        p.requires_grad_(n in names)  # 只有主幹參數可以訓練
    return sorted(names)  # 回傳排序後的名稱


def inner_split(games, frac, seed=INNER_SEED):  # 依玩家切出內層驗證集
    """(train game rows, inner game rows) of a cache's games table: per rank, ceil(frac x players) of its players
    are held out with all their games (settrain.inner_players; the same for every seed and run)."""
    if len(games) == 0 or frac <= 0:  # 沒有局或不留出
        return np.arange(len(games), dtype=np.int64), np.zeros(0, dtype=np.int64)  # 全部當作訓練
    y = games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64)  # 每局的等級索引
    held = inner_players(games["player_id"].to_numpy(dtype=str), y, frac, seed)  # 被留出的局
    return np.flatnonzero(~held).astype(np.int64), np.flatnonzero(held).astype(np.int64)  # 訓練局與內層局


def positions_of(arrays, game_rows):  # 取出指定局的局面
    """Indices of the positions whose game is one of `game_rows`."""
    return np.flatnonzero(np.isin(arrays["game_row"], np.asarray(game_rows, dtype=np.int64)))  # 依局列號篩選


def position_weights(rank_idx, n_ranks=N_RANKS):  # 類別平衡的局面權重
    """Per-position weights giving every present rank the same total weight, mean 1 (rankmodel.class_weights'
    rule on positions); computed from the whole training set, never from a subsample."""
    rank_idx = np.asarray(rank_idx, dtype=np.int64)  # 每個局面的等級
    if rank_idx.size == 0:  # 沒有局面
        return np.zeros(0, dtype=np.float64)  # 回傳空陣列
    counts = np.bincount(rank_idx, minlength=n_ranks).astype(np.float64)  # 每級的局面數
    w = 1.0 / counts[rank_idx]  # 讓每級的總權重相同
    return w / w.mean()  # 正規化成平均為 1


def batches(arrays, idx, batch, rng=None, device="cpu"):  # 產生小批次
    """Mini-batches of the positions `idx` (shuffled when rng is given): bins float32 [B, 22, 19, 19], globs
    float32, legal bool [B, 362], target long, rank long, color long -- on `device`."""
    idx = np.asarray(idx, dtype=np.int64)  # 局面索引
    if rng is not None:  # 需要打亂
        idx = idx[rng.permutation(len(idx))]  # 依 rng 打亂順序
    for i in range(0, len(idx), batch):  # 逐批次
        sel = np.sort(idx[i:i + batch])  # 取出這一批並把索引排序
        yield _to_tensors(arrays, sel, device)  # 解壓並轉成張量


def _to_tensors(arrays, sel, device):  # 把選中的局面解壓成張量
    bins = torch.from_numpy(unpack_bins(arrays["bins_packed"][sel])).to(device, non_blocking=True).float()  # 輸入平面
    globs = torch.from_numpy(np.ascontiguousarray(arrays["globs"][sel])).to(device, non_blocking=True)  # 全域特徵
    legal = torch.from_numpy(unpack_legal(arrays["legal_packed"][sel])).to(device, non_blocking=True)  # 合法手遮罩
    target = torch.from_numpy(arrays["target"][sel].astype(np.int64)).to(device, non_blocking=True)  # 實際落點
    rank = torch.from_numpy(arrays["rank_idx"][sel].astype(np.int64)).to(device, non_blocking=True)  # 玩家的等級
    color = torch.from_numpy(arrays["color"][sel].astype(np.int64)).to(device, non_blocking=True)  # 輪到的顏色
    return bins, globs, legal, target, rank, color  # 回傳六個張量


def _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk=False):  # 實際落子在各頭下的 log p
    """log p of each position's move under every head: [B, n_ranks]. grad_trunk: keep the trunk's graph (stage 2)."""
    with torch.set_grad_enabled(grad_trunk and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):  # 主幹只在第二階段保留梯度，CUDA 上用 bf16
        trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)  # 主幹特徵
    logits = heads(trunk.float(), mask.float(), mask_sum_hw.float(), mask_sum.float())  # 等級頭以 fp32 計算
    logp = masked_logp(logits, legal)  # 遮罩非法點
    return logp.gather(2, target[:, None, None].expand(-1, logp.shape[1], 1))[:, :, 0]  # 取實際落點在每個頭下的 log p


def _true_head_logp(model, heads, bins, globs, legal, target, rank, color, rows, autocast, grad_trunk=False):  # 自身等級之頭下的 log p
    """log p of each position's move under the head of its own rank: [B]. grad_trunk: keep the trunk's graph (stage 2)."""
    lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk)  # 所有頭的 log p
    return lp[torch.arange(len(target), device=lp.device), rank]  # 取各自等級的那一欄


def position_losses(lp, rank, disc_weight):  # 逐局面的損失
    """Per-position (generative NLL of the true head, rank cross-entropy, total = NLL + disc_weight x CE) from the
    heads' log p [B, n_ranks]: the CE treats the 10 log p of the played move as logits over the ranks (a uniform
    prior), so it rewards heads whose differences point at the player's rank."""
    i = torch.arange(len(rank), device=lp.device)  # 批次內的索引
    gen = -lp[i, rank]  # 真實等級之頭的 NLL
    disc = -torch.log_softmax(lp, dim=1)[i, rank]  # 把 10 個 log p 當作 logits 的等級交叉熵
    return gen, disc, gen + disc_weight * disc if disc_weight > 0 else gen  # 未啟用鑑別項時總損失即為 NLL


def evaluate_nll(model, heads, arrays, idx, cfg):  # 內層評估
    """Class-balanced per-move NLL of the true-rank head over positions `idx`: {"nll": mean over the ranks present
    of the rank's mean NLL, "nll_by_rank": [10] (None for an absent rank), "ranks_present", "n"}."""
    device, rows = cfg.device, neutral_rows()  # 裝置與中性的 metadata
    autocast = str(device).startswith("cuda")  # CUDA 上用 bf16 autocast
    sums, dsums, counts = np.zeros(N_RANKS), np.zeros(N_RANKS), np.zeros(N_RANKS, dtype=np.int64)  # 每級的 NLL 和、CE 和與局面數
    was_training = heads.training  # 記下原本的模式
    heads.eval()  # 切成評估模式
    me, world = _dist()  # 本行程的編號與總行程數
    with torch.no_grad():  # 只做推論
        for bins, globs, legal, target, rank, color in batches(arrays, np.asarray(idx)[me::world], cfg.batch, None, device):  # 每個行程評估自己分到的局面
            lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast).double()  # 以 float64 累加
            gen, disc, _ = position_losses(lp, rank, 0.0)  # NLL 與 CE
            r = rank.cpu().numpy()  # 每個局面的等級
            np.add.at(sums, r, gen.cpu().numpy())  # 依等級累加 NLL
            np.add.at(dsums, r, disc.cpu().numpy())  # 依等級累加 CE
            np.add.at(counts, r, 1)  # 依等級計數
    if world > 1:  # 多個行程時合併結果
        t = _all_reduce_(torch.as_tensor(np.concatenate([sums, dsums, counts.astype(np.float64)]), dtype=torch.float64, device=device))  # 三組數字一起加總
        t = t.cpu().numpy()  # 搬回 CPU
        sums, dsums, counts = t[:N_RANKS], t[N_RANKS:2 * N_RANKS], np.rint(t[2 * N_RANKS:]).astype(np.int64)  # 拆回三組
    heads.train(was_training)  # 恢復原本的模式
    present = [int(k) for k in np.flatnonzero(counts > 0)]  # 有局面的等級
    by_rank = [float(sums[k] / counts[k]) if counts[k] else None for k in range(N_RANKS)]  # 每級的平均 NLL
    nll = float(np.mean([by_rank[k] for k in present])) if present else float("nan")  # 類別平衡的 NLL
    disc = float(np.mean([dsums[k] / counts[k] for k in present])) if present else float("nan")  # 類別平衡的 CE
    lam = getattr(cfg, "disc_weight", 0.0)  # 設定物件沒有 disc_weight 時視為 0
    return {"nll": nll, "nll_by_rank": by_rank, "ranks_present": present, "n": int(counts.sum()), "disc": disc,  # 評估結果
            "objective": nll + lam * disc if lam > 0 else nll}  # 選模用的目標值


def check_init(model, heads, arrays, idx, n=256):  # 檢查第 0 步的等級頭等於原本的 policy
    """max |Δ log p| between every head and the model's own policy under the neutral metadata, fp32 trunk and heads,
    on the first n positions of idx (0 at step 0 up to float error)."""
    device = next(heads.parameters()).device  # 等級頭所在的裝置
    rows = neutral_rows()  # 中性的 metadata
    worst = 0.0  # 目前的最大差
    with torch.no_grad():  # 只做推論
        for bins, globs, legal, target, rank, color in batches(arrays, np.asarray(idx)[:n], 64, None, device):  # 前 n 個局面
            trunk, mask, mask_sum_hw, mask_sum = trunk_features(model, bins, globs, color, rows)  # fp32 的主幹特徵
            head_logp = masked_logp(heads(trunk, mask, mask_sum_hw, mask_sum), legal)  # 各等級頭的 log p
            meta = None  # 預設不給 metadata
            if model.get_has_metadata_encoder():  # 模型有 metadata 編碼器
                meta = torch.as_tensor(rows, device=device)[color]  # 依輪到的顏色取中性列
            own = model.postprocess_output(model(bins, globs, input_meta=meta))[0][0][:, 0, :].float()  # 模型自身 policy 的第 0 通道
            own_logp = masked_logp(own[:, None, :], legal)[:, 0, :]  # 以相同遮罩算 log p
            diff = (head_logp - own_logp[:, None, :]).abs()  # 每個頭與原 policy 的差
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))  # 非法點（-inf 相減）不計
            worst = max(worst, float(diff.max()))  # 更新最大差
    return worst  # 回傳最大差


def _cfg_key(cfg):  # 判斷檢查點是否屬於同一設定的鍵
    return {k: v for k, v in asdict(cfg).items() if k not in ("epochs", "patience", "device")}  # 可以延長或更換的欄位不列入比對


def train_heads(model, cache_dir, cfg, out_pt, log=print, resume=True, extra_meta=None, init_heads=None, exclude_folds=(),  # 訓練等級頭的主函式
                folds_csv=FOLDS_CSV):  # 分折表路徑
    """Train RankHeads (and, with cfg.trunk_lr > 0, the trunk) on the cache; returns (heads with the best inner NLL,
    sidecar meta). init_heads: a RankHeads state_dict to start from (default: copies of model.policy_head). cache_dir may
    be a list of caches read as one (issue #16), and exclude_folds leaves out the games of those folds' players
    (folds_csv), so the sidecar's train_folds is the caches' union minus them. See the module docstring; under
    torch.distributed only rank 0 logs and writes, and every rank returns the same heads."""
    t_start = time.time()  # 計時起點
    me, world = _dist()  # 本行程的編號與總行程數
    if me != 0:  # 不是 rank 0
        log = lambda *a, **k: None  # noqa: E731  只有 rank 0 輸出 log
    dirs = cache_dirs_of(cache_dir)  # 快取資料夾清單
    cache_meta = caches_meta(dirs, exclude_folds)  # 合併並核對快取的 meta
    drop = excluded_players(exclude_folds, folds_csv)  # 要排除的玩家
    games, arrays = read_caches(dirs, drop)  # 讀入所有局面
    if exclude_folds:  # 有排除折時記下實際的資料量
        log(f"caches {[str(d) for d in dirs]}: folds {cache_meta['folds']} after leaving out {sorted(set(map(int, exclude_folds)))} "  # 快取、使用的折與排除的折
            f"({len(drop)} players): {len(games)} games, {len(arrays['target'])} positions", flush=True)  # 排除的玩家數、局數與局面數
    train_rows, inner_rows = inner_split(games, cfg.inner_frac)  # 依玩家切出內層
    train_idx = positions_of(arrays, train_rows)  # 訓練局面
    inner_idx = positions_of(arrays, inner_rows)  # 內層局面
    if len(inner_idx) > cfg.inner_max_positions:  # 內層局面太多
        inner_idx = np.sort(np.random.RandomState(INNER_SEED).choice(inner_idx, cfg.inner_max_positions, replace=False))  # 以固定種子抽出上限數量
    if len(train_idx) == 0:  # 沒有訓練局面
        raise ValueError(f"{[str(d) for d in dirs]}: no training positions")  # 拒絕
    device = cfg.device  # 訓練用的裝置
    torch.manual_seed(cfg.seed)  # 固定亂數
    model.eval()  # 模型維持 eval 模式（humanv0 沒有批次統計量，第二階段微調時也不切換）
    train_trunk = cfg.trunk_lr > 0  # 第二階段：微調主幹
    if train_trunk:  # 要微調主幹
        trunk_names = trunk_parameters(model, device)  # 找出主幹參數並開啟梯度
    else:  # 第一階段
        model.requires_grad_(False)  # 主幹完全凍結
        trunk_names = []  # 沒有主幹參數要訓練
    trunk_params = [p for n, p in model.named_parameters() if n in set(trunk_names)]  # 主幹參數物件
    heads = RankHeads(model.policy_head, N_RANKS).to(device)  # 從 policy 頭複製 10 個等級頭
    if init_heads is not None:  # 指定了起始權重
        heads.load_state_dict(init_heads)  # 例如第一階段訓練好的等級頭
    heads.train()  # 等級頭進入訓練模式
    groups = [{"params": list(heads.parameters()), "lr": cfg.lr}]  # 等級頭的參數組
    if train_trunk:  # 第二階段
        groups.append({"params": trunk_params, "lr": cfg.trunk_lr})  # 主幹自己的參數組與學習率
    opt = torch.optim.Adam(groups)  # Adam 最佳化器
    params = [p for g in groups for p in g["params"]]  # 所有要更新的參數（跨行程加總梯度時使用）
    weights = torch.as_tensor(position_weights(arrays["rank_idx"][train_idx]), dtype=torch.float32)  # 由完整訓練集算出的類別平衡權重
    weight_of = np.zeros(len(arrays["target"]), dtype=np.float32)  # 每個局面的權重（非訓練局面為 0）
    weight_of[train_idx] = weights.numpy()  # 填入訓練局面的權重
    global_batch = cfg.batch * world  # 全域批次大小
    steps_per_epoch = math.ceil(len(train_idx) / global_batch)  # 每個 epoch 的步數
    total_steps = cfg.epochs * steps_per_epoch  # 總步數上限
    eval_steps = max(1, int(round(cfg.eval_every * steps_per_epoch)))  # 每隔幾步評估一次
    autocast = str(device).startswith("cuda")  # CUDA 上用 bf16 autocast
    rows = neutral_rows()  # 中性的 metadata
    out_pt = Path(out_pt)  # 輸出路徑
    ckpt_path = out_pt.with_name(out_pt.name + ".ckpt")  # 續跑用的檢查點路徑

    def snapshot():  # 複製目前的權重
        snap = {"heads": {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}}  # 等級頭權重的 CPU 副本
        if train_trunk:  # 第二階段
            snap["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}  # 連同主幹一起保存
        return snap  # 回傳快照

    use_inner = len(inner_idx) > 0  # inner_frac = 0：不提早停止，保留最後的權重（settrain 的規則）
    state = {"step": 0, "history": [], "best": None, "since_improvement": 0, "stopped_early": False}  # 訓練狀態（會存進檢查點）
    resumed_from = 0  # 從第幾步續跑
    if resume and ckpt_path.exists():  # 有檢查點就續跑
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)  # 讀入檢查點（含一般的 Python 物件）
        if ck["cfg_key"] != _cfg_key(cfg) or ck["steps_per_epoch"] != steps_per_epoch or ck.get("train_folds", cache_meta["folds"]) != cache_meta["folds"]:  # 設定、每 epoch 步數或訓練折不同
            raise ValueError(f"{ckpt_path}: checkpoint of another run (batch, lr, trunk_lr, seed, world size, inner split, cache or "  # 拒絕其他執行的檢查點
                             "folds differ); delete it or pass --overwrite")  # 提示處理方式
        heads.load_state_dict(ck["heads"])  # 還原等級頭
        if train_trunk:  # 第二階段
            model.load_state_dict(ck["model"])  # 還原主幹
        opt.load_state_dict(ck["opt"])  # 還原最佳化器的狀態
        state = ck["state"]  # 還原訓練狀態
        resumed_from = state["step"]  # 記下續跑的步數
        log(f"resumed from {ckpt_path} at step {resumed_from} (epoch {resumed_from / steps_per_epoch:.2f})", flush=True)  # 記錄續跑

    def save_ckpt():  # 寫入檢查點
        if me == 0:  # 只有 rank 0 寫檔
            ck = {"cfg_key": _cfg_key(cfg), "steps_per_epoch": steps_per_epoch, "train_folds": cache_meta["folds"],  # 續跑時比對用的設定
                  "heads": heads.state_dict(), "opt": opt.state_dict(), "state": state}  # 權重、最佳化器與訓練狀態
            if train_trunk:  # 第二階段
                ck["model"] = model.state_dict()  # 連同主幹
            atomic_save(ck, ckpt_path)  # 原子寫入
        _barrier()  # 等所有行程

    def evaluate(step):  # 內層評估並更新最佳權重
        """Inner evaluation at `step`; keeps the best weights; True when patience is exhausted."""
        res = evaluate_nll(model, heads, arrays, inner_idx, cfg)  # 內層的 NLL 與 CE
        epoch = step / steps_per_epoch  # 目前的 epoch（可為小數）
        state["history"].append((epoch, res["objective"]))  # 記錄目標值
        state.setdefault("history_detail", []).append((epoch, res["nll"], res["disc"]))  # 記錄 NLL 與 CE 的細節
        improved = state["best"] is None or (np.isfinite(res["objective"]) and res["objective"] < state["best"]["objective"])  # 是否比目前最佳更好
        if improved:  # 有改善
            state["best"] = {"nll": res["nll"], "disc": res["disc"], "objective": res["objective"], "step": step, **snapshot(),  # 保存最佳的權重與分數
                             "by_rank": res["nll_by_rank"], "ranks_present": res["ranks_present"]}  # 每級的 NLL 與有局面的等級
            state["since_improvement"] = 0  # 重設計數
        else:  # 沒有改善
            state["since_improvement"] += 1  # 累計未改善的次數
        extra = f" rank CE {res['disc']:.4f} objective {res['objective']:.4f}" if cfg.disc_weight > 0 else ""  # 有鑑別項時多印兩個值
        log(f"epoch {epoch:.2f} inner NLL {res['nll']:.4f}{extra} (best {state['best']['objective']:.4f} @ "  # 記錄評估結果
            f"{state['best']['step'] / steps_per_epoch:.2f})", flush=True)  # 最佳值所在的 epoch
        return state["since_improvement"] >= cfg.patience  # patience 用完就停止

    if use_inner and state["step"] == 0 and not state["history"]:  # 全新的執行
        evaluate(0)  # 先評估第 0 步（起點也可能是最佳）
        save_ckpt()  # 寫入第一個檢查點
    step = state["step"]  # 目前的步數（續跑時不為 0）
    stop = state["stopped_early"]  # 是否已提早停止
    t_loop, n_loop = time.time(), 0  # 計算吞吐量用
    while step < total_steps and not stop:  # 逐 epoch 訓練
        epoch_i, pos_in_epoch = divmod(step, steps_per_epoch)  # 目前的 epoch 與 epoch 內的位置
        order = train_idx[np.random.RandomState(cfg.seed + epoch_i).permutation(len(train_idx))]  # 每個 epoch 以固定種子打亂，續跑時順序相同
        for k in range(pos_in_epoch, steps_per_epoch):  # 從中斷處繼續這個 epoch
            sel = np.sort(order[k * global_batch:(k + 1) * global_batch][me::world])  # 本行程在全域批次中分到的局面
            opt.zero_grad(set_to_none=True)  # 清除梯度
            if len(sel):  # 這個行程有局面
                bins, globs, legal, target, rank, color = _to_tensors(arrays, sel, device)  # 解壓成張量
                w = torch.from_numpy(weight_of[sel]).to(device)  # 這批局面的權重
                lp = _head_logps(model, heads, bins, globs, legal, target, color, rows, autocast, grad_trunk=train_trunk)  # 各頭的 log p
                _, _, per = position_losses(lp, rank, cfg.disc_weight)  # 逐局面的總損失
                w_sum = _all_reduce_(w.sum().detach().clone())  # 全域的權重總和
                loss = (per * w).sum() / w_sum  # 以全域權重總和正規化
                if not torch.isfinite(loss):  # 損失不是有限值
                    raise FloatingPointError(f"non-finite training loss at step {step}")  # 立刻停止
                loss.backward()  # 反向傳播
            else:  # 最後一個全域批次可能讓某個行程（rank）分不到局面；它仍要參與加總
                _all_reduce_(torch.zeros((), device=device))  # 對應其他行程的權重總和加總
            if world > 1:  # 多個行程時加總梯度
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]  # 沒有梯度的參數補 0
                flat = _all_reduce_(torch.cat([g.reshape(-1) for g in grads]))  # 攤平後一次加總
                offset = 0  # 拆回各參數時的位置
                for p, g in zip(params, grads):  # 逐一參數
                    p.grad = flat[offset:offset + g.numel()].view_as(g).clone()  # 寫回加總後的梯度
                    offset += g.numel()  # 推進位置
            opt.step()  # 更新參數
            step += 1  # 完成一次參數更新
            n_loop += len(sel) * world  # 累計處理的局面數（以本行程估計全域）
            state["step"] = step  # 記進訓練狀態，存成檢查點後續跑會從這一步接著跑
            if step % eval_steps == 0 or step == total_steps:  # 到了評估時間或最後一步
                if use_inner and evaluate(step):  # 評估後發現 patience 已用完
                    stop = True  # 停止訓練
                    state["stopped_early"] = True  # 記錄提早停止
                save_ckpt()  # 寫入檢查點
                log(f"  {n_loop / max(time.time() - t_loop, 1e-9):.0f} positions/s since the last report", flush=True)  # 記錄吞吐量
                t_loop, n_loop = time.time(), 0  # 重設計時
                if stop:  # 要停止
                    break  # 跳出這個 epoch
        else:  # 這個 epoch 正常跑完（沒有 break）
            continue  # 進入下一個 epoch
        break  # 內層迴圈 break 時結束訓練

    if use_inner:  # 有內層評估
        best = state["best"]  # 最佳的紀錄
        heads.load_state_dict(best["heads"])  # 還原最佳的等級頭
        if train_trunk:  # 第二階段
            model.load_state_dict(best["model"])  # 還原最佳的主幹
        detail = state.get("history_detail") or [(e, v, float("nan")) for e, v in state["history"]]  # 沒有細節紀錄時以目標值代替
        best_epoch, nll_init, nll_best = best["step"] / steps_per_epoch, detail[0][1], best["nll"]  # 最佳 epoch、起點與最佳的 NLL
        disc_init, disc_best = detail[0][2], best.get("disc")  # 起點與最佳的 CE
        obj_init, obj_best = state["history"][0][1], best.get("objective", best["nll"])  # 起點與最佳的目標值
        by_rank, present = best["by_rank"], best["ranks_present"]  # 最佳時的每級 NLL
    else:  # 沒有內層：保留最後的權重
        best_epoch, nll_init, nll_best, by_rank, present = state["step"] / steps_per_epoch, None, None, [None] * N_RANKS, []  # 沒有內層分數
        disc_init = disc_best = obj_init = obj_best = None  # 其餘欄位也留空
    heads.eval()  # 切成推論模式
    per_rank = np.bincount(arrays["rank_idx"][train_idx], minlength=N_RANKS)  # 每級的訓練局面數
    meta = {"train_cfg": asdict(cfg), "train_folds": cache_meta["folds"], "max_games": cache_meta.get("max_games"), "side": cache_side(cache_meta),  # 設定與資料來源
            "n_train_positions": int(len(train_idx)), "n_inner_positions": int(len(inner_idx)),  # 訓練與內層的局面數
            "per_rank_positions": {lv: int(per_rank[k]) for k, lv in enumerate(LEVELS)},  # 每級的局面數
            "inner_nll_init": nll_init, "inner_nll_best": nll_best, "inner_nll_best_by_rank": by_rank,  # 內層 NLL 的起點與最佳
            "inner_disc_init": disc_init, "inner_disc_best": disc_best, "inner_objective_init": obj_init, "inner_objective_best": obj_best,  # CE 與目標值的起點與最佳
            "best_epoch": best_epoch, "stopped_early": bool(state["stopped_early"]),  # 最佳 epoch 與是否提早停止
            "inner_ranks_present": present, "history": [(float(e), float(v)) for e, v in state["history"]],  # 內層出現的等級與評估歷史
            "steps_per_epoch": steps_per_epoch, "steps_run": int(state["step"]), "resumed_from_step": int(resumed_from),  # 步數資訊
            "world_size": int(world), "global_batch": int(global_batch), "trunk_trained": bool(train_trunk),  # 分散式設定與是否微調主幹
            "selection": "inner NLL" if cfg.disc_weight <= 0 else f"inner NLL + {cfg.disc_weight:g} x rank CE",  # 選模的依據
            "history_detail": [(float(e), float(n), float(d)) for e, n, d in state.get("history_detail", [])],  # 評估歷史的細節
            "n_trunk_params": int(sum(p.numel() for p in trunk_params)),  # 微調的主幹參數量
            "wall_seconds": time.time() - t_start, "cache_dir": str(dirs[0]) if len(dirs) == 1 else [str(d) for d in dirs],  # 耗時與快取來源
            "cache_dirs": [str(d) for d in dirs], "excluded_folds": sorted({int(f) for f in exclude_folds}), "cache_meta": cache_meta,  # 快取清單、排除的折與合併後的 meta
            "n_games": int(len(games)), "n_players": int(games["player_id"].nunique()),  # 局數與玩家數
            "head_init": "humanv0_policy_head", **(extra_meta or {})}  # 等級頭起點的標記與呼叫端附加的資訊
    if me == 0:  # 只有 rank 0 寫檔
        meta = save_heads(heads, out_pt, meta, model_state=model.state_dict() if train_trunk else None)  # 儲存等級頭與側檔；第二階段另存微調後的主幹
    _barrier()  # 等所有行程
    return heads, meta  # 回傳等級頭與 meta


def _cache_meta(cache_dir):  # 讀取快取的 meta.json
    """The cache's meta.json; ValueError when it is missing or has no folds (the sidecar's train_folds, and so the
    out-of-fold check of l1b_score.py, depend on it -- run l1b_build_cache.py --merge-meta after a sharded build)."""
    p = Path(cache_dir) / "meta.json"  # meta 檔的路徑
    if not p.exists():  # 沒有 meta.json（分片建立的快取要先以 --merge-meta 合併）
        raise ValueError(f"{p} is missing: a sharded cache needs `l1b_build_cache.py --out {cache_dir} --merge-meta` first")  # 提示先合併 meta
    meta = json.loads(p.read_text())  # 讀入 meta
    if not meta.get("folds"):  # 沒有記錄折
        raise ValueError(f"{p} has no folds; the heads' train_folds would be unknown")  # 拒絕
    return meta  # 回傳 meta


def cache_summary(games):  # 快取的規模摘要
    """Players, games, positions of a cache's games table (for logs)."""
    return {"players": int(games["player_id"].nunique()), "games": int(len(games)), "positions": int(pd.to_numeric(games["n_moves"]).sum())}  # 玩家數、局數與局面數
