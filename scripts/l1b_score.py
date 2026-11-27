#!/usr/bin/env python
"""Score the evaluation games of folds under the L1b rank heads, in scripts/score_folds.py's layout.

Plan 2026-10-25-l1b-rank-heads, Task 5. The games are scoring.fold_players' (<= --max-games per player
with >= --min-games, the defaults 20 / 5 being results/scores' selection), so every fold scored here
has the same games as results/scores/fold{F}/ and the two can be compared move by move. Output per
fold: <out>/fold{F}/games_s{shard}_p{k:03d}.csv (game_key, player_id, rank, color, n_moves, sum_<D..6D>)
and moves_s{shard}_p{k:03d}.npz (logp [M, 10] float32, game_row, t), written atomically and resumable
per game (scoring.resume_plan); plus <out>/fold{F}/meta_s{shard}.json with the heads' sidecar summary,
the environment and the counts.

The heads must not have seen the fold: a fold in the sidecar's train_folds is refused (exit 2) unless
--allow-seen, and then only into an --out whose name contains "_seen" (diagnostics only). Heads trained on the
opponent's positions (sidecar side "opponent", l1b_build_cache.py --opponent; issue #33) are refused without
--opponent: they read the labelled player's rank off the opponent's play.

  python -I scripts/l1b_score.py --heads results/l1b/heads_A.pt --folds 3 4 0 --out results/l1b/scores --workers 10
  bash scripts/nano4/shards.sh 8 python -I scripts/l1b_score.py --heads results/l1b/heads_B.pt --folds 1 2 0 --out results/l1b/scores_B
"""
import argparse  # 解析命令列參數
import multiprocessing as mp  # CPU 特徵抽取的行程池
import os  # 以改名完成原子寫入
import sys  # 設定匯入路徑與結束碼
import time  # 計算耗時
from datetime import datetime  # 記錄完成時間
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列與 npz 輸出
import pandas as pd  # 表格處理

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.l1b import load_heads, score_positions  # noqa: E402  載入等級頭並逐手評分
from aicup.ladder import FOLDS_CSV, environment, to_json  # noqa: E402  分折表、執行環境紀錄與 JSON 輸出
from aicup.metric import LEVELS  # noqa: E402  10 個等級（也是等級頭的順序）
from aicup.scoring import (fold_players, init_worker, load_games, part_paths, player_games, read_game_keys, restrict_players,  # noqa: E402  與 Human SL 評分共用的選局、讀檔與分段工具
                           resume_plan, score_stream)  # 續跑計畫與串流評分

CKPT = ROOT / "models" / "katago" / "b18c384nbt-humanv0.ckpt"  # Human SL 網路的檢查點，當主幹（第二階段的等級頭會換上微調主幹）
HEADS_SUMMARY = ("train_folds", "max_games", "best_epoch", "inner_nll_init", "inner_nll_best", "stopped_early", "n_params",  # 從側檔抄進評分紀錄的欄位
                 "state_sha256", "trunk_file", "trunk_sha256", "code_sha256", "git_head", "git_dirty", "humanv0_sha256", "model_source", "side")  # 含權重雜湊與來源，供日後核對


def check_side(meta, opponent):  # 以對手局面訓練的等級頭只能評對手落子
    """ValueError when heads trained on the opponent's positions (sidecar side "opponent", issue #33) would score the target's
    moves: they read the labelled player's rank off the opponent's play and are only meant for --opponent. Target-side heads
    (the default, and every sidecar before issue #33) score either side, as fused_opp does."""
    side = meta.get("side") or "target"  # 舊側檔沒有 side 時視為目標端
    if side == "opponent" and not opponent:  # 對手端的等級頭卻要評目標玩家
        raise ValueError("the heads were trained on the opponent's positions (sidecar side \"opponent\"); score with --opponent")  # 拒絕並提示改用 --opponent


def load_trunk(ckpt, model_config, device):  # 載入主幹模型
    import torch  # 只在需要時才載入 PyTorch
    if model_config:  # 測試用：隨機初始化的 KataGo 設定
        from katago.train import modelconfigs  # KataGo 的模型設定表
        from katago.train.model_pytorch import Model  # KataGo 的 PyTorch 模型
        torch.manual_seed(0)  # 固定初始化，結果可重現
        model = Model(modelconfigs.config_of_name[model_config], 19)  # 建立 19 路的模型
        model.initialize()  # 隨機初始化權重
        return model.to(device).eval()  # 移到裝置並切到推論模式
    from aicup.humansl import load_humansl  # 載入 Human SL 權重的工具
    return load_humansl(str(ckpt), device)[0]  # 回傳 Human SL 模型本體


def make_pool(workers, config):  # 建立特徵抽取的行程池
    """The CPU feature pool; fork it before anything touches CUDA (PyTorch does not support forking a CUDA process)."""
    return mp.get_context("fork").Pool(workers, initializer=init_worker, initargs=(config,))  # 每個子行程各自建立 KataGo 特徵物件


def check_unseen(meta, folds, allow_seen, out):  # 確認等級頭沒看過要評分的折
    """ValueError when the heads were trained on one of `folds` (unless allow_seen into a *_seen directory), or when the
    sidecar does not record train_folds at all (a cache without meta.json): out-of-fold cannot be shown."""
    if meta.get("train_folds") is None:  # 側檔沒有記錄訓練折
        raise ValueError("the heads' sidecar has no train_folds (the cache had no meta.json?); cannot show the features are out-of-fold")  # 無法證明折外，拒絕
    seen = sorted(set(meta.get("train_folds")) & set(folds))  # 等級頭訓練時看過的折
    if seen and not allow_seen:  # 有看過的折且未允許
        raise ValueError(f"the heads' train_folds {meta.get('train_folds')} include {seen}; their features would not be out-of-fold "  # 拒絕：特徵不是折外
                         "(--allow-seen for diagnostics only)")  # 提示只有診斷時可放行
    if allow_seen and "_seen" not in Path(out).name:  # 放行時輸出名稱必須標明 _seen
        raise ValueError(f"--allow-seen writes only into a directory whose name contains '_seen', not {out}")  # 避免與正式的折外評分混淆


def score_fold(fold, players, out, shard, nshards, part_size, pool, scorer, meta_heads, log=print, opponent=False):  # 評分一折
    """Score one fold's pending parts into out/fold{F}; returns (games in shard, positions scored, seconds). opponent: score
    the moves of the target player's opponent instead (the other colour; the table's color column is the side scored)."""
    games = [(p.rank, p.player_id, k) for p in players.itertuples() for k in p.game_keys][shard::nshards]  # 本分片負責的棋局（每 nshards 局取一局）
    info = {g[2]: g for g in games}  # 棋局鍵 → (等級, 玩家, 鍵)
    fold_dir = Path(out) / f"fold{fold}"  # 這一折的輸出資料夾
    fold_dir.mkdir(parents=True, exist_ok=True)  # 確保資料夾存在
    plan = resume_plan([g[2] for g in games], fold_dir, shard, part_size)  # 略過已完成的棋局，排出待評的分段
    pending = sum(len(keys) for _, keys in plan)  # 待評分的局數
    log(f"fold {fold}: {len(players)} players, {len(games)} games in this shard, {len(games) - pending} already done, "  # 回報這一折的進度
        f"{pending} to score in {len(plan)} parts", flush=True)  # 以及待評的局數與分段數
    t0, n_pos = time.time(), 0  # 計時與局面計數
    if plan:  # 還有待評分的分段
        data = load_games([k for _, keys in plan for k in keys])  # 讀入待評棋局的 SGF 與顏色
        for j, (k, keys) in enumerate(plan):  # 逐一處理每個分段
            side = {key: (("W" if data[key][1] == "B" else "B") if opponent else data[key][1]) for key in keys}  # 要評分的顏色：對手模式取另一色
            tasks = [(i, data[key][0], side[key]) for i, key in enumerate(keys)]  # 每局的 (序號, SGF, 顏色)
            per_task, arrays, n, wall = score_stream(pool, None, tasks, None, log_every=10**9, scorer=scorer)  # CPU 抽特徵、GPU 以等級頭評分
            logp, task_idx, ts = arrays if arrays is not None else (np.zeros((0, len(LEVELS)), np.float32), np.zeros(0, np.int32), np.zeros(0, np.int16))  # 沒有任何局面時用空陣列
            recs = [{"game_key": key, "player_id": info[key][1], "rank": info[key][0], "color": side[key], "n_moves": per_task[i][0],  # 每局一列：鍵、玩家、等級、顏色、手數
                     **{f"sum_{lv}": float(s) for lv, s in zip(LEVELS, per_task[i][1])}} for i, key in enumerate(keys)]  # 以及 10 個等級頭的 log p 總和
            csv_path, npz_path = part_paths(fold_dir, shard, k)  # 這個分段的正式檔名
            tmp_csv, tmp_npz = csv_path.with_suffix(".tmp.csv"), npz_path.with_name(npz_path.stem + ".tmp.npz")  # 先寫到暫存檔名
            np.savez_compressed(tmp_npz, logp=logp.astype(np.float32), game_row=task_idx, t=ts)  # 寫出逐手 log p
            pd.DataFrame(recs).to_csv(tmp_csv, index=False)  # 寫出逐局總和表
            os.replace(tmp_npz, npz_path)  # 先改名 npz
            os.replace(tmp_csv, csv_path)  # 再改名 csv，兩個檔都在才算完成
            n_pos += n  # 累計局面數
            log(f"  fold {fold} part {j + 1}/{len(plan)} (index {k}): {len(keys)} games, {n} positions, {n / max(wall, 1e-9):.0f} positions/s", flush=True)  # 回報這個分段的速度
    secs = time.time() - t0  # 評分耗時
    meta = {"fold": fold, "shard": shard, "nshards": nshards, "part_size": part_size, "games": len(games), "positions": int(n_pos),  # 本分片的設定與計數
            "scoring_wall_s": secs, "levels": list(LEVELS), "heads": meta_heads, "side": "opponent" if opponent else "target", "finished": datetime.now().isoformat(timespec="seconds"),  # 耗時、等級頭摘要與評分端
            **environment("cpu")}  # 執行環境
    (fold_dir / f"meta_s{shard}.json").write_text(to_json(meta, indent=1) + "\n")  # 寫出本分片的紀錄
    return len(games), n_pos, secs  # 回傳局數、局面數與秒數


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--heads", required=True)  # 等級頭權重（.pt，旁邊要有側檔）
    ap.add_argument("--folds", type=int, nargs="+", required=True)  # 要評分的折
    ap.add_argument("--out", required=True)  # 輸出資料夾
    ap.add_argument("--max-games", type=int, default=20)  # 每位玩家最多評幾局
    ap.add_argument("--min-games", type=int, default=5)  # 局數不足的玩家略過
    ap.add_argument("--workers", type=int, default=10)  # CPU 特徵行程數
    ap.add_argument("--shard", type=int, default=0)  # 本行程的分片編號
    ap.add_argument("--nshards", type=int, default=1)  # 分片總數
    ap.add_argument("--part-size", type=int, default=100)  # 每個分段的局數
    ap.add_argument("--ckpt", default=str(CKPT))  # 主幹的檢查點（Human SL）
    ap.add_argument("--model-config", default=None, help="random KataGo config instead of --ckpt (tests)")  # 測試用的隨機模型設定
    ap.add_argument("--device", default="cuda")  # 評分用的裝置
    ap.add_argument("--folds-csv", default=str(FOLDS_CSV))  # 分折表
    ap.add_argument("--allow-seen", action="store_true")  # 允許評等級頭看過的折（僅供診斷）
    ap.add_argument("--opponent", action="store_true", help="score the opponent's moves (the other colour) instead of the target player's")  # 改評對手的落子
    ap.add_argument("--game-keys", default=None, help="score only these games of each fold's selection (a CSV with a game_key column or one key "  # 只重評清單中的棋局
                    "per line), e.g. the HD >= 2 games again into a patch directory that scripts/patch_scored.py merges (issue #42)")  # 例如 HD >= 2 的讓子局
    a = ap.parse_args(argv)  # 解析參數
    model = load_trunk(a.ckpt, a.model_config, "cpu")  # 先放在 CPU，等行程池 fork 之後才搬到 GPU
    try:  # 等級頭或側檔有問題時改為回報錯誤
        heads, meta, warnings = load_heads(a.heads, model, "cpu")  # 載入等級頭（第二階段的檔也載入微調的主幹）
        check_unseen(meta, a.folds, a.allow_seen, a.out)  # 確認是折外評分
        check_side(meta, a.opponent)  # 確認評分端相符
    except ValueError as e:  # 檢查不通過
        print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示拒絕執行
    for w in warnings:  # 例如訓練程式碼版本不同
        print(f"WARNING: {w}", file=sys.stderr, flush=True)  # 印出警告
    meta_heads = {"path": str(a.heads), **{k: meta.get(k) for k in HEADS_SUMMARY}}  # 要寫進評分紀錄的等級頭摘要
    all_games, folds_df = player_games(), pd.read_csv(a.folds_csv)  # 訓練集所有棋局與分折表
    pool = make_pool(a.workers, model.config)  # 在碰 CUDA 之前先 fork 行程池
    model = model.to(a.device)  # 主幹搬到 GPU
    heads = heads.to(a.device)  # 等級頭搬到 GPU

    def scorer(bins, globs, legal, target, colors, rows):  # 給 score_stream 的評分函式
        return score_positions(model, heads, bins, globs, legal, target, colors)  # 每手在 10 個等級頭下的 log p

    try:  # 不論成功與否都要關閉行程池
        only = read_game_keys(a.game_keys) if a.game_keys else None  # 只重評的棋局清單
        for fold in a.folds:  # 逐折評分
            players = fold_players(all_games, folds_df, fold, max_games=a.max_games, min_games=a.min_games)  # 預設參數下與 results/scores 相同的選局
            if only is not None:  # 有指定清單時
                players = restrict_players(players, only)  # 只留清單中的棋局
            n_games, n_pos, secs = score_fold(fold, players, a.out, a.shard, a.nshards, a.part_size, pool, scorer, meta_heads, opponent=a.opponent)  # 評分這一折
            print(f"fold {fold} done: {n_games} games, {n_pos} positions scored, {secs / 60:.1f} min", flush=True)  # 回報這一折完成
    finally:  # 收尾
        pool.close()  # 不再接受新工作
        pool.join()  # 等待子行程結束
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
