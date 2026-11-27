"""Question-level features (L2/L3 inputs) from per-game Human SL log-likelihood sums.

A question is a set of games by one player; each game contributes, per profile, the sum of
log p over the target player's moves and the number of those moves.
"""
import numpy as np  # 特徵的數值運算
import pandas as pd  # 輸出題目資料表

from aicup.evidence import posterior_mean_index  # 後驗平均剖面索引
from aicup.humansl import PROFILES  # 18 個 Human SL 剖面
from aicup.metric import RANK_TO_IDX  # 等級名稱轉索引

def sum_cols(profiles):  # 評分表的總和欄名
    """The per-game sum columns of a scored games table: sum_<profile> for each profile (or L1b head)."""
    return [f"sum_{p}" for p in profiles]  # 依剖面順序產生欄名


def feature_names(profiles):  # 題目特徵的欄名
    """question_features' layout for `profiles`: rel_<p> per profile, then the six curve and evidence features."""
    return [f"rel_{p}" for p in profiles] + ["best_logp", "pmi_sum", "peak_interp", "log_moves", "log_games", "game_argmax_std"]  # 每個剖面一欄，再加六欄


SUM_COLS = sum_cols(PROFILES)  # 18 個 Human SL 剖面的總和欄
FEATURE_NAMES = feature_names(PROFILES)  # 24 維題目特徵（q24）的欄名


def _peak_interp(curve):  # 以拋物線內插求峰值位置
    """Argmax refined by a parabola through it and its two neighbours (profile-index units)."""
    i = int(np.argmax(curve))  # 曲線最高點的剖面索引
    if 0 < i < len(curve) - 1:  # 兩側都有鄰點才能內插
        a, b, c = curve[i - 1], curve[i], curve[i + 1]  # 峰值與左右鄰點的值
        den = a - 2 * b + c  # 二階差分（曲率）
        if den < 0:  # 開口向下才是真正的峰
            return i + 0.5 * (a - c) / den  # 拋物線頂點的位置
    return float(i)  # 在端點或無法內插時用原始的最大值索引


def question_features(sums, n_moves):  # 一個題目的特徵（剖面數加 6 欄；18 個剖面時為 24 維）
    """sums [games, 18] and n_moves [games] -> feature vector aligned with FEATURE_NAMES.

    rel_*: per-move mean log p under each profile minus its maximum (curve shape);
    best_logp: that maximum; pmi_sum: posterior-mean profile index of the summed log p;
    peak_interp: parabola-refined peak of the per-move curve; log_moves, log_games:
    evidence amount; game_argmax_std: spread of the best profile across games.
    """
    sums = np.asarray(sums, dtype=float)  # 每局各剖面的 log p 總和
    n = np.asarray(n_moves, dtype=float)  # 每局目標玩家的手數
    total = sums.sum(axis=0)  # 全題各剖面的 log p 總和
    moves = max(n.sum(), 1.0)  # 全題總手數，至少當 1
    mean = total / moves  # 逐手平均 log p 曲線
    best = mean.max()  # 曲線的最高值（best_logp）
    per_game = sums / np.maximum(n, 1.0)[:, None]  # 每局各自的逐手平均曲線
    return np.concatenate([mean - best, [best, posterior_mean_index(total), _peak_interp(mean), np.log(moves),  # 依序為 rel_*、best_logp、pmi_sum、peak_interp、log_moves
                                         np.log(len(sums)), per_game.argmax(axis=1).std()]])  # log_games 與各局最佳剖面的標準差


def build_matrix(games, questions, profiles=PROFILES):  # 整批題目的特徵矩陣
    """Feature matrix X [Q, F], labels y [Q] (-1 when the rank is unknown) and games per question; the games table
    holds sum_cols(profiles) (the 18 Human SL profiles by default, the 10 L1b heads with profiles=LEVELS)."""
    g = games.set_index("game_key")  # 棋局表以棋局鍵為索引
    sums = g[sum_cols(profiles)].to_numpy(dtype=float)  # 每局各剖面（或等級頭）的總和
    n = g["n_moves"].to_numpy(dtype=float)  # 每局手數
    row = {k: i for i, k in enumerate(g.index)}  # 棋局鍵轉列號
    X = np.stack([question_features(sums[idx], n[idx]) for idx in ([row[k] for k in keys] for keys in questions["game_keys"])])  # 逐題計算特徵
    if "rank" in questions:  # 有等級欄的題目（訓練或驗證）
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(int).to_numpy()  # 等級轉索引，未知為 -1
    else:  # 測試題沒有等級欄
        y = np.full(len(questions), -1)  # 全部標為未知
    return X, y, questions["game_keys"].map(len).to_numpy()  # 特徵、標籤與每題局數


def sample_questions(players, per_player, rng, sizes=range(5, 21)):  # 抽樣產生訓練題
    """Random training questions: per player, `per_player` subsets of its scored games."""
    rows = []  # 收集抽出的題目
    for p in players.itertuples():  # 逐位玩家
        keys = list(p.game_keys)  # 該玩家已評分的棋局
        allowed = [s for s in sizes if s <= len(keys)]  # 不超過該玩家局數的題目大小
        if not allowed:  # 局數太少
            continue  # 略過這位玩家
        for j in range(per_player):  # 每位玩家抽 per_player 題
            size = int(rng.choice(allowed))  # 隨機選題目大小
            pick = rng.choice(len(keys), size=size, replace=False)  # 不重複地抽棋局
            rows.append({"question_id": f"s_{p.player_id[:8]}_{j}", "rank": p.rank, "player_id": p.player_id,  # 題號、等級與玩家
                         "n_games": size, "game_keys": [keys[i] for i in pick]})  # 局數與抽到的棋局鍵
    return pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])  # 與驗證題欄位相同的資料表
