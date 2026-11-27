"""Per-game staged evidence vectors: the input of the L3 aggregator ladder.

A game's per-move log-likelihoods ll [moves, 18] (one column per Human SL profile) are
reduced to a fixed-length vector per game stage. Within a stage, every move is made relative
to its profile mean, d = ll - mean_r ll, so the vector keeps the *shape* of the evidence over
profiles and drops the move's overall level l = mean_r ll; the level survives only as the
stage's mean surprisal. There is no unit-length normalisation.

Stages are defined on the game's move number t + 1 (t is the 0-based index in the game, the
target colour's moves only). Each stage contributes SEG_DIM = 75 fields:
  d_mean 18 | d_std 18 (ddof=0) | d_q10 18 | d_q90 18 (linear quantiles) | n_eff | mean_surprisal | present
n_eff is the number of target moves in the stage, mean_surprisal = -mean(l), present = 1 when
the stage has any move. A stage without moves is all zeros with present = 0.
"""
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列
import pandas as pd  # 讀取局表 CSV

from aicup.humansl import PROFILES  # 18 個 Human SL 剖面
from aicup.qfeatures import SUM_COLS  # CSV 中各剖面總和的欄名
from aicup.scoring import completed_parts, part_paths  # 評分分段檔的列舉與路徑

SEGMENTS = ((1, 40), (41, 100), (101, 180), (181, None))  # 含端點的手數範圍；None 表示沒有上限
WHOLE = ((1, None),)  # 整局當作一個階段
SEG_DIM = 4 * len(PROFILES) + 3  # 每個階段 75 欄：四種統計量各 18 欄，再加 3 欄
CSV_TOL = 1e-2  # 每局每剖面「逐手 ll 總和」與 CSV 總和的最大容許差（CSV 的總和是 float32 加總）


def _segment_block(ll):  # 一個階段的 75 個欄位
    """The SEG_DIM fields of one stage from its per-move ll [n >= 1, 18] (float64)."""
    level = ll.mean(axis=1)  # 每一手對 18 個剖面取平均的水準 l
    d = ll - level[:, None]  # 去掉水準後的形狀
    q10, q90 = np.quantile(d, [0.1, 0.9], axis=0, method="linear")  # 各剖面的 10% 與 90% 分位數
    return np.concatenate([d.mean(axis=0), d.std(axis=0, ddof=0), q10, q90, [len(ll), -level.mean(), 1.0]])  # 四種統計量加上 n_eff、平均驚奇度與 present


def game_vector(logp, t, segments=SEGMENTS):  # 一局的分段證據向量
    """logp [n, 18] and 0-based move index t [n] -> float32 vector of SEG_DIM * len(segments) fields."""
    ll = np.asarray(logp, dtype=np.float64)  # 以 float64 計算
    number = np.asarray(t, dtype=np.int64) + 1  # 轉成從 1 起算的手數
    if ll.ndim != 2 or ll.shape[1] != len(PROFILES) or number.shape != (len(ll),):  # 形狀須為 n 手 x 18 剖面
        raise ValueError(f"logp {ll.shape} and t {number.shape} do not describe n moves x {len(PROFILES)} profiles")  # 拒絕
    out = np.zeros((len(segments), SEG_DIM))  # 沒有手的階段保持全 0
    for i, (lo, hi) in enumerate(segments):  # 逐階段處理
        inside = (number >= lo) if hi is None else (number >= lo) & (number <= hi)  # 落在這個階段的手
        if inside.any():  # 這個階段有手
            out[i] = _segment_block(ll[inside])  # 計算該階段的欄位
    return out.reshape(-1).astype(np.float32)  # 攤平成一維的 float32


def vec_names(segments=SEGMENTS):  # 向量各欄的名稱
    """Field names aligned with game_vector: s1_mean_12k ... s4_present (prefix w_ for WHOLE)."""
    names = []  # 收集欄名
    for i in range(len(segments)):  # 逐階段
        p = "w" if tuple(segments) == WHOLE else f"s{i + 1}"  # 整局用 w，分段依序用 s1、s2…（預設 SEGMENTS 為 s1–s4）
        names += [f"{p}_{stat}_{prof}" for stat in ("mean", "std", "q10", "q90") for prof in PROFILES]  # 四種統計量 x 18 個剖面
        names += [f"{p}_n_eff", f"{p}_mean_surprisal", f"{p}_present"]  # 最後三欄
    return names  # 回傳欄名清單


def _part_vectors(games, moves, label):  # 一個分段的向量（先做一致性檢查）
    """Vectors of one finished part after checking its per-move scores against the CSV."""
    n = len(games)  # 局數
    logp, row, t = moves["logp"], moves["game_row"].astype(np.int64), moves["t"]  # 逐手分數、所屬的局與手數索引
    if not np.isfinite(logp).all():  # 有非有限值
        raise ValueError(f"{label}: non-finite log p in the per-move scores")  # 拒絕
    if len(row) and (row.min() < 0 or row.max() >= n):  # 局列號超出 CSV 的範圍
        raise ValueError(f"{label}: game_row outside the {n} games of the CSV")  # 拒絕
    ll = logp.astype(np.float64)  # 以 float64 計算
    count = np.bincount(row, minlength=n)  # 每局實際評分的手數
    if not np.array_equal(count, games["n_moves"].to_numpy()):  # 須與 CSV 的 n_moves 相同
        bad = int(np.flatnonzero(count != games["n_moves"].to_numpy())[0])  # 第一個不符的局
        raise ValueError(f"{label}: game {games['game_key'][bad]} has {count[bad]} scored moves, "  # 拒絕並指出是哪一局
                         f"n_moves says {games['n_moves'][bad]}")  # 錯誤訊息的後半
    csv_sums = games[SUM_COLS].to_numpy(dtype=float)  # CSV 記錄的各剖面總和
    sums = np.zeros((n, len(PROFILES)))  # 由逐手分數重算的總和
    np.add.at(sums, row, ll)  # 依局累加
    off = ~(np.abs(sums - csv_sums) <= CSV_TOL)  # 差超過容許值者（也會抓到 CSV 中的 NaN）
    if off.any():  # 有不一致
        g, p = (int(i[0]) for i in np.nonzero(off))  # 第一個不一致的局與剖面
        raise ValueError(f"{label}: {int(off.any(axis=1).sum())} games differ from the CSV sums, e.g. game "  # 拒絕並報告局數
                         f"{games['game_key'][g]} {SUM_COLS[p]}: per-move sum {sums[g, p]:.4f} vs CSV {csv_sums[g, p]:.4f}")  # 舉出一個例子
    order = np.argsort(row, kind="stable")  # 依局排序，同一局內維持原本的順序
    ll, t = ll[order], t[order]  # 重排逐手資料
    bounds = np.searchsorted(row[order], np.arange(n + 1))  # 每局在排序後的起訖位置
    staged = np.zeros((n, SEG_DIM * len(SEGMENTS)), dtype=np.float32)  # 分段向量 [n, 300]
    whole = np.zeros((n, SEG_DIM * len(WHOLE)), dtype=np.float32)  # 整局向量 [n, 75]
    for g in range(n):  # 逐局計算
        sl = slice(bounds[g], bounds[g + 1])  # 這一局的手
        staged[g] = game_vector(ll[sl], t[sl], SEGMENTS)  # 四個階段的向量
        whole[g] = game_vector(ll[sl], t[sl], WHOLE)  # 整局的向量
    return staged, whole, csv_sums  # 回傳兩種向量與 CSV 總和


def fold_vectors(fold_dir):  # 一個已評分折的所有局向量
    """Vectors of every game in the finished parts of one scored fold, in completed-part order.

    Returns game_key (fixed-width unicode), staged [G, 300], whole [G, 75] (float32), sums
    [G, 18] (float64, the CSV values, so question features equal qfeatures.build_matrix bit for
    bit) and n_moves [G]. The per-move scores must reproduce the CSV sums and move counts.
    """
    fold_dir = Path(fold_dir)  # 統一成 Path
    parts = completed_parts(fold_dir)  # 已完成的分段
    if not parts:  # 沒有任何已完成的分段
        raise FileNotFoundError(f"no finished parts in {fold_dir}")  # 停止並說明
    keys, staged, whole, sums, n_moves = [], [], [], [], []  # 各輸出的收集清單
    for shard, k in parts:  # 逐段處理
        csv_path, npz_path = part_paths(fold_dir, shard, k)  # 兩個檔的路徑
        games = pd.read_csv(csv_path)  # 局表
        moves = dict(np.load(npz_path))  # 每個陣列只讀一次
        s, w, csv_sums = _part_vectors(games, moves, csv_path.name)  # 核對並計算向量
        keys += games["game_key"].tolist()  # 收集局鍵
        staged.append(s)  # 收集分段向量
        whole.append(w)  # 收集整局向量
        sums.append(csv_sums)  # 收集 CSV 總和
        n_moves.append(games["n_moves"].to_numpy(dtype=np.int64))  # 收集手數
    if len(set(keys)) != len(keys):  # 同一局被評分不只一次（例如出現在多個分段）
        raise ValueError(f"{fold_dir}: {len(keys) - len(set(keys))} games scored more than once")  # 拒絕重複評分
    return {"game_key": np.asarray(keys, dtype=str), "staged": np.concatenate(staged), "whole": np.concatenate(whole),  # 局鍵與兩種向量
            "sums": np.concatenate(sums), "n_moves": np.concatenate(n_moves)}  # 總和與手數
