"""Turn per-profile log-likelihoods into class decisions and separability statistics.

Profiles are the 18 Human SL ranks (index 0 = 12k ... 17 = 6d, see aicup.humansl.PROFILES);
classes are the competition's 10 levels (index 0 = D ... 9 = 6D, see aicup.metric.LEVELS).
"""
import numpy as np  # 數值陣列

from aicup.metric import score  # 官方的單題計分

# 競賽類別 -> 剖面索引，採用名義上的級位範圍（D = 10-12k ... A = 1-3k）。
HARD_BINS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12], [13], [14], [15], [16], [17]]  # D、C、B、A 各三個級位，段位一對一


def class_scores_hard_bins(L):  # 硬分箱的類別分數
    """[..., 18] log-likelihoods -> [..., 10]: each class scored by its best profile."""
    L = np.asarray(L, dtype=float)  # 轉成浮點陣列
    return np.stack([L[..., b].max(axis=-1) for b in HARD_BINS], axis=-1)  # 每個類別取所屬剖面的最大值


def posterior_mean_index(L):  # 後驗平均剖面索引（pmi）
    """Mean profile index under softmax(L) along the last axis (uniform prior over profiles)."""
    L = np.asarray(L, dtype=float)  # 轉成浮點陣列
    p = np.exp(L - L.max(axis=-1, keepdims=True))  # 減去最大值以免溢位
    p /= p.sum(axis=-1, keepdims=True)  # 正規化成後驗機率
    return (p * np.arange(L.shape[-1])).sum(axis=-1)  # 索引的期望值


def auc(pos, neg):  # 兩組分數的 AUC
    """P(pos > neg) + 0.5 P(pos == neg) over all pairs (Mann-Whitney)."""
    pos, neg = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]  # 擴成所有配對
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())  # 平手算一半


def fit_thresholds(s, y, n_classes=10, weights=None):  # 以動態規劃擬合單調門檻
    """Monotone cut points on statistic s maximising the (weighted) mean official score (exact DP).

    Equal s values always land in the same class. Returns cuts [n_classes - 1] with
    class = number of cuts strictly below s (see apply_thresholds); unused classes get
    collapsed cuts (+inf above the highest used class, -inf below the lowest).
    """
    s, y = np.asarray(s, dtype=float), np.asarray(y, dtype=int)  # 統計量與真實類別
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)  # 未給權重時每題權重為 1
    values, inverse = np.unique(s, return_inverse=True)  # 不重複的 s 值與每題對應的位置
    gain = np.zeros((len(values), n_classes))  # 每個 s 值判成每一類時的總得分
    for c in range(n_classes):  # 逐類計算
        gain[:, c] = np.bincount(inverse, weights=[score(c, t) * wi for t, wi in zip(y, w)], minlength=len(values))  # 同一 s 值之題目的加權得分加總
    best = np.empty_like(gain)  # best[b, c]：第 b 個值判成 c 時，前 b + 1 個值的最佳總分
    back = np.zeros(gain.shape, dtype=int)  # 回溯用：前一個值的類別
    best[0] = gain[0]  # 第一個值可判成任一類
    for b in range(1, len(values)):  # 依 s 由小到大推進
        run_arg = 0  # 前一個值在類別 <= c 中的最佳類別
        for c in range(n_classes):  # 逐類更新
            if best[b - 1, c] > best[b - 1, run_arg]:  # 類別 c 更好
                run_arg = c  # 更新最佳類別
            back[b, c] = run_arg  # 記下回溯來源
            best[b, c] = best[b - 1, run_arg] + gain[b, c]  # 類別不遞減限制下的最佳總分
    cls = np.empty(len(values), dtype=int)  # 每個 s 值的類別
    cls[-1] = int(np.argmax(best[-1]))  # 最後一個值取最佳類別
    for b in range(len(values) - 1, 0, -1):  # 由後往前回溯
        cls[b - 1] = back[b, cls[b]]  # 前一個值的類別
    cuts = np.empty(n_classes - 1)  # 切點
    for j in range(1, n_classes):  # 第 j 個切點分開類別 < j 與 >= j
        below, above = values[cls < j], values[cls >= j]  # 切點兩側的 s 值
        if len(above) == 0:  # 沒有值屬於 >= j 的類別
            cuts[j - 1] = np.inf  # 沒有值判成 >= j 的類別：切點放在 +inf，任何 s 都不會越過
        elif len(below) == 0:  # 沒有值屬於 < j 的類別
            cuts[j - 1] = -np.inf  # 沒有值判成 < j 的類別：切點放在 -inf，任何有限的 s 都會越過
        else:  # 兩側都有值
            cuts[j - 1] = (below.max() + above.min()) / 2  # 取兩側相鄰值的中點
    return cuts  # 回傳切點


def apply_thresholds(s, cuts):  # 依切點把統計量轉成類別
    return (np.asarray(s, dtype=float)[:, None] > np.asarray(cuts)[None, :]).sum(axis=1)  # 類別 = 嚴格小於 s 的切點個數


def bootstrap_ci(scores, players, ranks, n=1000, seed=42, alpha=0.05, balanced=False):  # 以玩家重抽樣的信賴區間
    """95% CI of the mean question score, resampling players with replacement within each rank.

    balanced=True reports the mean of per-rank means (every rank weighs the same) instead of
    the mean over questions.
    """
    scores, players, ranks = np.asarray(scores, dtype=float), np.asarray(players), np.asarray(ranks)  # 轉成陣列
    rng = np.random.RandomState(seed)  # 固定種子的亂數產生器
    by_rank = {}  # 每個等級：每位玩家的題目分數
    for r in np.unique(ranks):  # 逐級處理
        ps = np.unique(players[ranks == r])  # 這一級的玩家
        by_rank[r] = [scores[(players == p) & (ranks == r)] for p in ps]  # 每位玩家的分數組
    means = np.empty(n)  # 每次重抽的統計量
    for i in range(n):  # 重抽 n 次
        per_rank = [np.concatenate([groups[j] for j in rng.randint(len(groups), size=len(groups))])  # 在級內有放回地抽玩家
                    for groups in by_rank.values()]  # 每一級各抽一次
        means[i] = np.mean([g.mean() for g in per_rank]) if balanced else np.concatenate(per_rank).mean()  # 各級平均或題目平均
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])  # 取兩端的分位數
    return float(lo), float(hi)  # 回傳區間


def bootstrap_ci_mean_over_sizes(scores, players, ranks, sizes, n=1000, seed=42, alpha=0.05):  # 主要指標的信賴區間
    """95% CI of the mean over question sizes of the class-balanced score.

    Players are resampled with replacement within each rank, once per replicate and shared by
    every size, so the interval belongs to exactly the statistic reported as the headline.
    """
    scores, players, ranks, sizes = (np.asarray(a) for a in (scores, players, ranks, sizes))  # 轉成陣列
    scores = scores.astype(float)  # 分數轉成浮點
    rng = np.random.RandomState(seed)  # 固定種子的亂數產生器
    size_values = np.unique(sizes)  # 出現的題長
    rank_values = np.unique(ranks)  # 出現的等級
    groups = {r: [np.flatnonzero((players == p) & (ranks == r)) for p in np.unique(players[ranks == r])] for r in rank_values}  # 每級每位玩家的題目索引
    means = np.empty(n)  # 每次重抽的統計量
    for i in range(n):  # 重抽 n 次
        idx = {r: np.concatenate([g[j] for j in rng.randint(len(g), size=len(g))]) for r, g in groups.items()}  # 每級抽一次玩家，各題長共用
        per_size = []  # 各題長的類別平衡分數
        for s in size_values:  # 逐一題長
            per_rank = [scores[ix[sizes[ix] == s]].mean() for ix in idx.values() if (sizes[ix] == s).any()]  # 這個題長下各級的平均
            per_size.append(np.mean(per_rank))  # 各級平均再平均
        means[i] = np.mean(per_size)  # 對題長取平均
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])  # 取兩端的分位數
    return float(lo), float(hi)  # 回傳區間


def question_matrix(games, questions, profiles):  # 由每局總和組出每題的矩陣
    """Per question: summed log p [Q, P], per-game-normalised sums [Q, P], target-move counts [Q].

    games: DataFrame with game_key, n_moves and sum_<profile> columns.
    questions: DataFrame with a game_keys column holding lists of game keys.
    """
    g = games.set_index("game_key")  # 以局鍵查表
    sums = g[[f"sum_{p}" for p in profiles]].to_numpy(dtype=float)  # 每局各剖面的 log p 總和
    n = g["n_moves"].to_numpy(dtype=float)  # 每局的目標手數
    row = {k: i for i, k in enumerate(g.index)}  # 局鍵 -> 列號
    L_sum = np.zeros((len(questions), len(profiles)))  # 每題直接相加
    L_norm = np.zeros_like(L_sum)  # 每題先除以各局手數再相加
    n_moves = np.zeros(len(questions), dtype=int)  # 每題的總手數
    for qi, keys in enumerate(questions["game_keys"]):  # 逐題處理
        idx = [row[k] for k in keys]  # 題內各局的列號
        L_sum[qi] = sums[idx].sum(axis=0)  # 加總 log p
        L_norm[qi] = (sums[idx] / n[idx, None]).sum(axis=0)  # 每局正規化後再加總
        n_moves[qi] = int(n[idx].sum())  # 加總手數
    return L_sum, L_norm, n_moves  # 回傳三者
