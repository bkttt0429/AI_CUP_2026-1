"""Question-level evaluation and paired model comparison on pooled evaluation questions.

A question is scored by the official metric of the expected-score decision taken from its
class probabilities (see aicup.metric). All headline numbers are class-balanced: the mean of
per-class means, per question size, then averaged over the sizes. Classes are the competition's
10 levels (index 0 = D ... 9 = 6D, see aicup.metric.LEVELS).
"""
import numpy as np  # 評估的數值運算

from aicup.evidence import bootstrap_ci_mean_over_sizes  # 玩家層級 bootstrap 信賴區間
from aicup.metric import LEVELS, expected_score_decision, score  # 等級、期望得分決策與官方計分

SIZES = (5, 10, 20)  # 驗證題的三種題目大小（局數）
N_CLASSES = len(LEVELS)  # 10 個等級
P_FLOOR = 1e-12  # 計算 NLL 取對數前，機率先截在這個下限


def balanced(values, y, mask=None):  # 類別平衡平均
    """Mean over classes present (inside `mask`) of the per-class mean of `values`."""
    mask = np.ones(len(y), dtype=bool) if mask is None else mask  # 沒給遮罩就納入全部題目
    return float(np.mean([values[mask & (y == k)].mean() for k in range(N_CLASSES) if (mask & (y == k)).any()]))  # 各類別平均再平均，略過沒有題目的類別


def _balanced_by_n(values, y, n_games, sizes):  # 每種題目大小的類別平衡平均
    return {int(m): balanced(values, y, n_games == m) for m in sizes}  # 以題目大小為鍵


def balanced_mean_over_n(scores, y, n_games, sizes=SIZES):  # 標題分數的統計量
    """Class-balanced mean of `scores` per question size, averaged over `sizes`."""
    scores, y, n_games = np.asarray(scores, dtype=float), np.asarray(y, dtype=int), np.asarray(n_games)  # 轉成陣列
    return float(np.mean(list(_balanced_by_n(scores, y, n_games, sizes).values())))  # 各大小的類別平衡平均再平均


def evaluate(proba, y, qs, sizes=SIZES):  # 題目層的完整評估
    """Official scores and diagnostics of class probabilities `proba` [Q, 10] for true classes `y` [Q].

    qs: DataFrame with an n_games column (question size). Returns (metrics, scores [Q]).
    `nll_balanced`: per true class, the mean of -log max(p(true class), 1e-12), then the mean over the
    classes present. The clip keeps one-hot models finite (a miss costs -log 1e-12 = 27.63).
    `brier`: per true class, the mean of the multi-class Brier score sum_k (p_k - 1[k = y])^2 over the
    10 classes (0 for a perfect, 2 for a one-hot miss; not divided by 10), then the mean over the classes present.
    """
    proba = np.asarray(proba, dtype=float)  # 類別機率 [Q, 10]
    y = np.asarray(y, dtype=int)  # 真實類別
    pred = np.array([expected_score_decision(p) for p in proba])  # 每題以期望得分決策
    s = np.array([score(p, t) for p, t in zip(pred, y)])  # 每題的官方分數
    n = qs["n_games"].to_numpy()  # 每題局數

    out = {"balanced_by_n": _balanced_by_n(s, y, n, sizes)}  # 各題目大小的類別平衡分數
    out["balanced_mean_over_n"] = float(np.mean(list(out["balanced_by_n"].values())))  # 標題分數
    out["question_mean"] = float(s.mean())  # 不加權的逐題平均
    out["per_class"] = {LEVELS[k]: float(s[y == k].mean()) for k in range(N_CLASSES)}  # 各等級的平均分數
    out["exact"] = float((pred == y).mean())  # 完全猜中的比例
    out["within1"] = float((np.abs(pred - y) <= 1).mean())  # 差一級以內的比例
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # 混淆矩陣
    np.add.at(conf, (y, pred), 1)  # 列為真實、欄為預測逐題累加
    out["confusion_rows_truth_cols_pred"] = conf.tolist()  # 轉成可寫入 JSON 的清單
    nll, brier = nll_brier(proba, y)  # 逐題 NLL 與 Brier
    out["nll_balanced"] = balanced(nll, y)  # 類別平衡 NLL
    out["brier"] = balanced(brier, y)  # 類別平衡 Brier
    return out, s  # 指標與逐題分數


def nll_brier(proba, y):  # 逐題 NLL 與 Brier 分數
    """Per question: the NLL -log max(p(true class), P_FLOOR) and the multi-class Brier score
    sum_k (p_k - 1[k = y])^2 (not divided by 10) of probabilities `proba` [Q, 10] -> (nll [Q], brier [Q]).
    evaluate's nll_balanced and brier are their class-balanced means (see balanced)."""
    proba, y = np.asarray(proba, dtype=float), np.asarray(y, dtype=int)  # 轉成陣列
    return -np.log(np.maximum(proba[np.arange(len(y)), y], P_FLOOR)), ((proba - np.eye(N_CLASSES)[y]) ** 2).sum(axis=1)  # 截斷後的負對數機率，以及與 one-hot 的平方差和


def paired_diff(s_a, s_b, y, player_id, n_games, sizes=SIZES, n=5000, seed=42):  # 兩個模型的成對比較
    """Paired difference A - B of the class-balanced mean over question sizes, with a player bootstrap CI.

    s_a, s_b: per-question scores of the two models on the same questions. The statistic is linear
    in the scores, so the bootstrap interval of s_a - s_b is the paired interval of the difference
    (players are resampled within each class, once per replicate and shared by every size).
    Only questions whose size is in `sizes` enter, the difference and the interval alike.
    Returns {"diff", "ci95": [lo, hi], "by_n": {size: difference at that size}}.
    """
    s_a, s_b = np.asarray(s_a, dtype=float), np.asarray(s_b, dtype=float)  # 兩個模型的逐題分數
    y, player_id, n_games = np.asarray(y, dtype=int), np.asarray(player_id), np.asarray(n_games)  # 標籤、玩家與局數
    keep = np.isin(n_games, list(sizes))  # 只保留指定大小的題目
    s_a, s_b, y, player_id, n_games = s_a[keep], s_b[keep], y[keep], player_id[keep], n_games[keep]  # 套用篩選
    by_a, by_b = _balanced_by_n(s_a, y, n_games, sizes), _balanced_by_n(s_b, y, n_games, sizes)  # 兩者在各大小的分數
    lo, hi = bootstrap_ci_mean_over_sizes(s_a - s_b, player_id, y, n_games, n=n, seed=seed)  # 對逐題差值做玩家 bootstrap
    return {"diff": balanced_mean_over_n(s_a, y, n_games, sizes) - balanced_mean_over_n(s_b, y, n_games, sizes),  # 標題分數的差
            "ci95": [lo, hi],  # 95% 信賴區間
            "by_n": {m: by_a[m] - by_b[m] for m in by_a}}  # 各題目大小的差
