"""Question features from a strength head's per-game statistics (issue #14, option b).

experiments/se_score.py stores, per game, the moves judged and statistics of the strength scores s of the target's (or,
with --opponent, the opponent's) moves: mean, standard deviation, the 10 / 50 / 90 % quantiles and the mean by
stage (moves with t < 40, 40 <= t < 100, t >= 100; a stage without moves takes the game's mean). se_features
turns a question's games into SE_FEATURE_NAMES: the move-weighted mean of the per-game means, the move-weighted
standard deviation of the per-game means (between games), the move-weighted mean of the per-game standard deviations
(within games), and the move-weighted means of the quantiles and stage means. build_se_matrix has
qfeatures.build_matrix's shape for a scored-games table with SE_STATS columns.
"""
import numpy as np  # 數值運算

from aicup.metric import RANK_TO_IDX  # 等級名稱轉索引

SE_STATS = ("se_mean", "se_std", "se_q10", "se_q50", "se_q90", "se_s1", "se_s2", "se_s3")  # 每局存下的 8 個強度統計量
SE_FEATURE_NAMES = ["se_mean", "se_between_std", "se_within_std", "se_q10", "se_q50", "se_q90", "se_s1", "se_s2", "se_s3"]  # 題目層的 9 個特徵名稱
STAGES = ((0, 40), (40, 100), (100, 10 ** 9))  # 三個階段的手數區間（左閉右開）


def game_stats(s, t):  # 計算一局的強度統計量
    """SE_STATS of one game from its per-move scores s [n] and move indices t [n] (zeros for a game without moves)."""
    s, t = np.asarray(s, dtype=np.float64), np.asarray(t)  # 轉成陣列，分數用雙精度
    if len(s) == 0:  # 這一局沒有評到的手
        return dict.fromkeys(SE_STATS, 0.0)  # 全部填 0
    mean = float(s.mean())  # 整局平均
    q10, q50, q90 = (float(v) for v in np.quantile(s, [0.1, 0.5, 0.9]))  # 三個分位數
    stages = []  # 各階段的平均
    for lo, hi in STAGES:  # 逐一處理三個階段
        m = (t >= lo) & (t < hi)  # 落在這個階段的手
        stages.append(float(s[m].mean()) if m.any() else mean)  # 沒有手時改用整局平均
    return {"se_mean": mean, "se_std": float(s.std()), "se_q10": q10, "se_q50": q50, "se_q90": q90,  # 整局的平均、標準差與分位數
            "se_s1": stages[0], "se_s2": stages[1], "se_s3": stages[2]}  # 三個階段的平均


def se_features(stats, n_moves):  # 把一題各局的統計量合成題目特徵
    """[9] SE_FEATURE_NAMES from a question's games: stats [m, 8] in SE_STATS order and n_moves [m]."""
    stats = np.asarray(stats, dtype=np.float64).reshape(-1, len(SE_STATS))  # 整理成 [局數, 8]
    n = np.asarray(n_moves, dtype=np.float64)  # 每局的手數
    w = n / max(n.sum(), 1.0)  # 以手數為權重，避免除以 0
    mean_w = float((stats[:, 0] * w).sum())  # 各局平均的加權平均
    between = float(np.sqrt(max((w * (stats[:, 0] - mean_w) ** 2).sum(), 0.0)))  # 局間的加權標準差
    within = float((stats[:, 1] * w).sum())  # 局內標準差的加權平均
    rest = [float((stats[:, k] * w).sum()) for k in range(2, len(SE_STATS))]  # 分位數與階段平均的加權平均
    return np.array([mean_w, between, within, *rest])  # 組成 9 維特徵


def build_se_matrix(games, questions):  # 建立所有題目的特徵矩陣
    """X [Q, 9], y [Q] (-1 when the rank is unknown), games per question -- for a scored-games table with SE_STATS."""
    g = games.set_index("game_key")  # 以棋局鍵為索引
    stats = g[list(SE_STATS)].to_numpy(dtype=float)  # 每局的統計量矩陣
    n = g["n_moves"].to_numpy(dtype=float)  # 每局的手數
    row = {k: i for i, k in enumerate(g.index)}  # 棋局鍵對應列號
    X = np.stack([se_features(stats[idx], n[idx]) for idx in ([row[k] for k in keys] for keys in questions["game_keys"])])  # 每題取出自己的棋局算特徵
    if "rank" in questions:  # 題目表帶有等級欄位
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(int).to_numpy()  # 轉成等級索引，未知為 -1
    else:  # 沒有等級欄位時
        y = np.full(len(questions), -1)  # 全部標為未知
    return X, y, questions["game_keys"].map(len).to_numpy()  # 回傳特徵、標籤與每題局數
