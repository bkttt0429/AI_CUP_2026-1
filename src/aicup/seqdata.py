"""Per-game move sequences from scored folds (issue #19: is there signal in the order of the moves?).

A scored fold holds, per part, a games table and per-move arrays (logp [M, S], game_row, t). fold_sequences merges, per
game, the target player's moves (18 Human SL log p from the results/scores fold + 10 L1b log p from an l1b_score.py fold)
and the opponent's moves (10 L1b log p from an l1b_score.py --opponent fold) in move order into one step sequence:
STEP_DIM = 18 + 10 + 3 columns -- hsl (zero on the opponent's steps), l1b, is_target, is_opponent, t / T_SCALE. The target's
summed L1b log p (`base`, the l1b_score csv sums) is what the submission's features use today; a sequence encoder adds a
correction to it (aicup.seqmodel).
"""
from pathlib import Path  # 路徑處理

import numpy as np  # 陣列運算
import pandas as pd  # 讀取棋局表

from aicup.humansl import PROFILES  # Human SL 的 18 個剖面
from aicup.metric import LEVELS, RANK_TO_IDX  # 10 個等級與等級索引
from aicup.scoring import completed_parts, part_paths  # 分段檔的列舉與路徑

N_HSL, N_L1B = len(PROFILES), len(LEVELS)  # Human SL 與 L1b 的欄數（18、10）
STEP_DIM = N_HSL + N_L1B + 3  # 每一步的特徵維度
COL_TARGET, COL_OPP, COL_T = N_HSL + N_L1B, N_HSL + N_L1B + 1, N_HSL + N_L1B + 2  # 三個附加欄的位置
T_SCALE = 300.0  # 手數索引的縮放常數
STEP_NAMES = [f"hsl_{p}" for p in PROFILES] + [f"l1b_{lv}" for lv in LEVELS] + ["is_target", "is_opponent", "t_norm"]  # 每一欄的名稱
SUM_TOL = 0.05  # 逐手加總必須重現 csv 的總和（bf16 批次造成的雜訊約為每局 1e-3）


def read_moves(fold_dir):  # 讀取一折的逐手結果
    """(games table, {game_key: (t [n] int64, logp [n, S] float32)}) of every finished part of one scored fold."""
    frames, out = [], {}  # 各分段的棋局表與逐手結果
    parts = completed_parts(fold_dir)  # 已完成的分段
    if not parts:  # 沒有任何分段
        raise FileNotFoundError(f"no finished parts in {fold_dir}")  # 回報錯誤
    for shard, k in parts:  # 逐一處理分段
        csv, npz = part_paths(fold_dir, shard, k)  # 分段的兩個檔案
        games = pd.read_csv(csv, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False)  # 讀棋局表，id 維持字串
        with np.load(npz, allow_pickle=False) as z:  # 讀逐手陣列
            logp, row, t = z["logp"].astype(np.float32), z["game_row"].astype(np.int64), z["t"].astype(np.int64)  # 統一型別
        if not np.isfinite(logp).all():  # 出現非有限值
            raise ValueError(f"{npz}: non-finite log p")  # 視為資料錯誤
        order = np.argsort(row, kind="stable")  # 依所屬列穩定排序，保留局內順序
        logp, row, t = logp[order], row[order], t[order]  # 套用排序
        bounds = np.searchsorted(row, np.arange(len(games) + 1))  # 每一局在陣列中的起訖位置
        for g, key in enumerate(games["game_key"]):  # 逐局切出逐手資料
            sl = slice(bounds[g], bounds[g + 1])  # 這一局的範圍
            out[key] = (t[sl], logp[sl])  # 記下手數索引與對數機率
        frames.append(games)  # 收集棋局表
    games = pd.concat(frames, ignore_index=True)  # 合併各分段的棋局表
    if not games["game_key"].is_unique:  # 有重複的棋局
        raise ValueError(f"{fold_dir}: games scored more than once")  # 視為錯誤
    return games, out  # 回傳棋局表與逐手結果


def fold_sequences(hsl_dir, l1b_dir, opp_dir, fold):  # 組出一折各局的步驟序列
    """The step sequences of one fold's games.

    Returns {"game_key" [G], "player_id" [G], "rank" [G] (str), "rank_idx" [G] int64, "color" [G], "n_moves" [G] (the
    target's), "n_steps" [G], "offsets" [G + 1], "X" [sum n_steps, STEP_DIM] float32, "base" [G, 10] float64 (the target's
    summed L1b log p, checked against the csv sums), "games" (the L1b games table)}. The three folds must hold the same
    games; the Human SL and L1b arrays must describe the same target moves. opp_dir None: target steps only."""
    hsl_games, hsl = read_moves(Path(hsl_dir) / f"fold{fold}")  # Human SL 的目標玩家逐手結果
    l1b_games, l1b = read_moves(Path(l1b_dir) / f"fold{fold}")  # L1b 的目標玩家逐手結果
    opp = read_moves(Path(opp_dir) / f"fold{fold}")[1] if opp_dir else None  # L1b 的對手逐手結果（可省略）
    keys = l1b_games["game_key"].tolist()  # 以 L1b 棋局表的順序為準
    if set(keys) != set(hsl_games["game_key"]):  # 兩份評分的棋局不同
        raise ValueError(f"fold {fold}: {hsl_dir} and {l1b_dir} score different games")  # 拒絕合併
    if opp is not None and set(keys) != set(opp):  # 對手評分涵蓋的棋局不同
        raise ValueError(f"fold {fold}: {opp_dir} covers other games than {l1b_dir}")  # 拒絕合併
    csv_sums = l1b_games[[f"sum_{lv}" for lv in LEVELS]].to_numpy(dtype=float)  # csv 記錄的每局 L1b 總和
    blocks, n_steps, base = [], np.zeros(len(keys), dtype=np.int64), np.zeros((len(keys), N_L1B))  # 各局序列、步數與目標玩家的 L1b 對數機率加總（base）
    for g, key in enumerate(keys):  # 逐局組合序列
        th, lh = hsl[key]  # Human SL 的手數索引與對數機率
        tl, ll = l1b[key]  # L1b 的手數索引與對數機率
        if len(th) != len(tl) or not np.array_equal(th, tl):  # 兩邊評的手不同
            raise ValueError(f"fold {fold}: game {key}: Human SL and L1b arrays describe different moves")  # 拒絕合併
        base[g] = ll.astype(np.float64).sum(axis=0)  # 以雙精度加總目標玩家的 L1b 對數機率
        if not np.all(np.abs(base[g] - csv_sums[g]) <= SUM_TOL):  # 與 csv 的總和差太多
            raise ValueError(f"fold {fold}: game {key}: per-move L1b sums differ from the csv sums by up to {np.abs(base[g] - csv_sums[g]).max():.4f}")  # 回報最大差距
        tgt = np.zeros((len(tl), STEP_DIM), dtype=np.float32)  # 目標玩家的步驟
        tgt[:, :N_HSL], tgt[:, N_HSL:N_HSL + N_L1B], tgt[:, COL_TARGET], tgt[:, COL_T] = lh, ll, 1.0, tl / T_SCALE  # 填入 hsl、l1b、目標旗標與縮放後的手數索引 t / T_SCALE
        if opp is not None:  # 有對手的手
            to, lo = opp[key]  # 對手的手數索引與 L1b 對數機率
            op = np.zeros((len(to), STEP_DIM), dtype=np.float32)  # 對手的步驟（hsl 欄為 0）
            op[:, N_HSL:N_HSL + N_L1B], op[:, COL_OPP], op[:, COL_T] = lo, 1.0, to / T_SCALE  # 填入 l1b、對手旗標與縮放後的手數索引（hsl 欄維持 0）
            steps = np.concatenate([tgt, op])  # 合併雙方的步驟
            steps = steps[np.argsort(np.concatenate([tl, to]), kind="stable")]  # 依著手順序排列
        else:  # 只用目標玩家的手
            steps = tgt[np.argsort(tl, kind="stable")]  # 依著手順序排列
        blocks.append(steps)  # 收集這一局的序列
        n_steps[g] = len(steps)  # 記下步數
    offsets = np.concatenate([[0], np.cumsum(n_steps)])  # 每一局在 X 中的起點
    X = np.concatenate(blocks) if blocks else np.zeros((0, STEP_DIM), dtype=np.float32)  # 所有步驟接成一個陣列
    return {"game_key": np.asarray(keys, dtype=str), "player_id": l1b_games["player_id"].to_numpy(dtype=str),  # 棋局鍵與玩家
            "rank": l1b_games["rank"].to_numpy(dtype=str), "rank_idx": l1b_games["rank"].map(RANK_TO_IDX).to_numpy(dtype=np.int64),  # 等級字串與索引
            "color": l1b_games["color"].to_numpy(dtype=str), "n_moves": l1b_games["n_moves"].to_numpy(dtype=np.int64),  # 顏色與目標玩家的手數
            "n_steps": n_steps, "offsets": offsets, "X": X, "base": base, "games": l1b_games}  # 序列資料與棋局表


def concat_sequences(parts):  # 合併多折的序列
    """Several fold_sequences results as one (offsets re-based)."""
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("game_key", "player_id", "rank", "rank_idx", "color", "n_moves", "n_steps", "X", "base")}  # 逐欄串接
    out["offsets"] = np.concatenate([[0], np.cumsum(out["n_steps"])])  # 依合併後的步數重算起點
    out["games"] = pd.concat([p["games"] for p in parts], ignore_index=True)  # 合併棋局表
    return out  # 回傳合併結果


def pad_batch(seq, idx):  # 取出一批棋局並補齊長度
    """Games `idx` of a sequences dict as padded arrays: X [B, T, D] float32, mask [B, T] bool, base [B, 10], y [B]."""
    idx = np.asarray(idx, dtype=np.int64)  # 棋局索引
    lengths = seq["n_steps"][idx]  # 各局的步數
    T = int(lengths.max()) if len(idx) else 1  # 補齊到最長的一局
    X = np.zeros((len(idx), T, STEP_DIM), dtype=np.float32)  # 補 0 的輸入陣列
    mask = np.zeros((len(idx), T), dtype=bool)  # 有效步驟的遮罩
    for b, g in enumerate(idx):  # 逐局填入
        n = seq["n_steps"][g]  # 這一局的步數
        X[b, :n] = seq["X"][seq["offsets"][g]:seq["offsets"][g] + n]  # 複製這一局的步驟
        mask[b, :n] = True  # 標記有效步驟
    return X, mask, seq["base"][idx], seq["rank_idx"][idx]  # 回傳輸入、遮罩、基準與標籤
