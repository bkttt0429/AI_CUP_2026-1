"""Official Task-2 scoring: 1 for the exact rank, e^-1 for an adjacent rank, 0 otherwise."""
import math  # 計算相鄰等級的分數 e^-1

import numpy as np  # 期望得分的向量運算

LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]  # 10 個等級，由弱到強對應索引 0–9
RANK_TO_IDX = {r: i for i, r in enumerate(LEVELS)}  # 等級名稱轉成類別索引
ADJACENT = math.exp(-1)  # 差一級時的得分 e^-1


def score(pred_idx, true_idx):  # 單一題目的官方分數
    d = abs(int(pred_idx) - int(true_idx))  # 預測與真實等級相差幾級
    return 1.0 if d == 0 else ADJACENT if d == 1 else 0.0  # 完全正確 1、差一級 e^-1、其餘 0


def mean_score(preds, trues):  # 多個題目的平均官方分數
    preds, trues = list(preds), list(trues)  # 轉成清單以便檢查長度
    if len(preds) != len(trues) or not preds:  # 空輸入或長度不一致
        raise ValueError("preds and trues must be non-empty and the same length")  # 拒絕計算
    return sum(score(p, t) for p, t in zip(preds, trues)) / len(preds)  # 逐題計分後取平均


def expected_scores(probs):  # 預測每個等級的期望得分
    """Expected official score of predicting each class k: p_k + e^-1 (p_{k-1} + p_{k+1})."""
    p = np.asarray(probs, dtype=float)  # 一題的 10 級機率
    padded = np.concatenate([[0.0], p, [0.0]])  # 兩端補 0，讓越界的鄰級項為 0
    return p + ADJACENT * (padded[:-2] + padded[2:])  # 本級機率加上左右鄰級機率乘 e^-1


def expected_score_decision(probs):  # 期望得分決策（決策層使用）
    return int(np.argmax(expected_scores(probs)))  # 選期望得分最高的等級索引
