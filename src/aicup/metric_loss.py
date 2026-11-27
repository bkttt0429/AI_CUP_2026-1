"""Losses aligned with the official score (architecture v2 section 4F).

S(k, y) is the official score of predicting class k when the truth is y: 1, e^-1 for a neighbour, 0 otherwise
(aicup.metric). Two losses are built on it and shared by the trees, the MLP and the C / T arms:

  kernel targets   t(k | y) = S(k, y) / sum_j S(j, y); the loss is the cross-entropy against this soft label. A proper
                   loss: the minimiser is the class posterior blurred by the kernel, so the matching decision is the plain
                   argmax (the expected-score decision would blur twice).
  expected score   l(z, y) = -sum_k softmax(z)_k S(k, y), the negative expected official score. A linear scoring rule
                   whose minimiser is a one-hot on the expected-score decision, so it serves only as a second stage that
                   starts from a cross-entropy solution and stops on held-out players.

The numpy functions give XGBoost custom objectives their gradient and hessian (hessian 2 p (1 - p) as in XGBoost's own
softmax objective, floored, times the sample weight); the torch functions go into the training loops.
"""
import numpy as np  # XGBoost 自訂目標的梯度與 Hessian 運算
import torch  # 訓練迴圈用的張量版損失

from aicup.metric import ADJACENT  # 相鄰等級的得分 e^-1

N_CLASSES = 10  # 等級數
HESS_FLOOR = 1e-6  # Hessian 的下限，避免趨近 0


def score_matrix(n_classes=N_CLASSES):  # 建立官方得分矩陣
    """S[k, y]: the official score of predicting k when the truth is y."""
    d = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])  # 每對 (k, y) 的等級差
    return np.where(d == 0, 1.0, np.where(d == 1, ADJACENT, 0.0))  # 等級差 0、1、其他分別得 1、e^-1、0


def kernel_targets(y, n_classes=N_CLASSES):  # 核目標軟標籤
    """[N, n_classes] soft targets t(k | y) = S(k, y) / sum_j S(j, y): interior classes 0.576 / 0.212 / 0.212, the two
    edge classes 0.731 / 0.269."""
    T = score_matrix(n_classes)[:, np.asarray(y, dtype=int)].T  # 每個樣本取其真實等級那一欄的得分
    return T / T.sum(axis=1, keepdims=True)  # 每列正規化為和 1


def softmax(z):  # numpy 版 softmax
    z = np.asarray(z, dtype=float)  # 轉成浮點陣列
    e = np.exp(z - z.max(axis=1, keepdims=True))  # 先減去每列最大值，避免溢位
    return e / e.sum(axis=1, keepdims=True)  # 每列正規化為機率


def _weighted(g, h, weight):  # 梯度與 Hessian 乘上樣本權重
    if weight is None:  # 沒有樣本權重
        return g, h  # 原樣回傳
    w = np.asarray(weight, dtype=float)[:, None]  # 權重轉成欄向量以便廣播
    return g * w, h * w  # 逐樣本加權


def kernel_grad_hess(margin, y, weight=None):  # 核目標的 XGBoost 自訂目標
    """Gradient p - t and hessian 2 p (1 - p) of the kernel-target cross-entropy at raw margins [N, K]."""
    p = softmax(margin)  # 原始 margin 轉成機率
    return _weighted(p - kernel_targets(y, p.shape[1]), np.maximum(2.0 * p * (1.0 - p), HESS_FLOOR), weight)  # 梯度 p - t，Hessian 設下限後一起加權


def expected_score_np(proba, y):  # numpy 版逐樣本期望分數
    """Per-sample expected official score sum_k p_k S(k, y)."""
    p = np.asarray(proba, dtype=float)  # 機率轉成浮點陣列
    return (p * score_matrix(p.shape[1])[:, np.asarray(y, dtype=int)].T).sum(axis=1)  # 機率乘上真實等級的得分後加總


def expected_score_grad_hess(margin, y, weight=None):  # 期望分的 XGBoost 自訂目標
    """Gradient -p_j (S(j, y) - sum_k p_k S(k, y)) of the negative expected score at raw margins [N, K]; the hessian is the
    softmax surrogate 2 p (1 - p) (the exact second derivative is indefinite)."""
    p = softmax(margin)  # 原始 margin 轉成機率
    S = score_matrix(p.shape[1])[:, np.asarray(y, dtype=int)].T  # 每個樣本的 S(j, y)
    g = -p * (S - (p * S).sum(axis=1, keepdims=True))  # 經 softmax 連鎖律得到的梯度
    return _weighted(g, np.maximum(2.0 * p * (1.0 - p), HESS_FLOOR), weight)  # Hessian 取替代值並設下限，再加權


def kernel_ce_torch(logits, targets):  # 訓練迴圈用的核目標交叉熵
    """Per-sample cross-entropy against soft targets [N, K] (kernel_targets as a tensor)."""
    return -(targets * torch.log_softmax(logits, dim=1)).sum(dim=1)  # 以軟目標加權的負對數機率


def expected_score_loss_torch(logits, y, S=None):  # 訓練迴圈用的負期望分損失
    """Per-sample negative expected official score; S is score_matrix as a tensor on logits' device (built when None)."""
    if S is None:  # 呼叫端沒有傳入得分矩陣
        S = torch.as_tensor(score_matrix(logits.shape[1]), dtype=logits.dtype, device=logits.device)  # 在 logits 的裝置上建立
    return -(torch.softmax(logits, dim=1) * S[:, y].T).sum(dim=1)  # 負的期望官方分數
