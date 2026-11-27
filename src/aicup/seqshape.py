"""Per-game "shape" features of the per-move L1b log p (issue #19 step 0; issue #21 fused_opp_shape).

One game's per-move L1b log p [n, 10] gives two scalar sequences: pmi_t, the expected rank index under the softmax of move
t's ten log p, and s_t = -max_k log p_t,k, the surprisal of the move's best rank. FEATURES per game: mean_pmi, log_moves,
std_pmi, ac1_pmi, ac2_pmi (lag-1 / lag-2 autocorrelation of pmi about the game mean), half_diff_pmi (second half minus
first half), slope_pmi (OLS slope over the normalised move index), mean_s, ac1_s. SHAPE is FEATURES without log_moves
(the question features already carry n_moves). The sums behind today's features see none of these except through the
mean, so a question-level mean of SHAPE over the question's games is an independent channel for within-game order.

Issue #29 (the "distribution range" of the per-move score): game_pmi_hist is the game's histogram of pmi_t over N_BINS equal
bins of [0, 9] (fractions of the game's moves), the whole per-move distribution rather than its moments; HIST names the bins.
"""
import numpy as np  # 數值運算
import pandas as pd  # 建立每局特徵表

from aicup.metric import LEVELS  # 10 個等級
from aicup.seqdata import read_moves  # 讀取一折的逐手結果

FEATURES = ["mean_pmi", "log_moves", "std_pmi", "ac1_pmi", "ac2_pmi", "half_diff_pmi", "slope_pmi", "mean_s", "ac1_s"]  # 每局的 9 個形狀特徵
SHAPE = [f for f in FEATURES if f != "log_moves"]  # 題目層使用的特徵（不含 log_moves）
N_BINS = 10  # 直方圖的區間數
HIST_EDGES = np.linspace(0.0, float(len(LEVELS) - 1), N_BINS + 1)  # [0, 9] 的等寬區間邊界
HIST = [f"hist_{i}" for i in range(N_BINS)]  # 各區間的欄名


def _ac(x, lag):  # 序列的自相關
    x = x - x.mean()  # 減去該局平均
    if len(x) <= lag + 1 or np.var(x) < 1e-12:  # 太短或幾乎沒有變化
        return 0.0  # 自相關記為 0
    return float(np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / np.var(x))  # 落後 lag 的自共變異除以變異數


def move_pmi(logp):  # 每手的期望等級索引
    """pmi_t [n]: the expected rank index under the softmax of each move's log p [n, 10] (float64)."""
    lp = np.asarray(logp, dtype=np.float64)  # 轉成雙精度
    z = lp - lp.max(axis=1, keepdims=True)  # 減去最大值以穩定數值
    p = np.exp(z)  # 未正規化的機率
    p /= p.sum(axis=1, keepdims=True)  # 正規化成每手的等級分布
    return p @ np.arange(lp.shape[1])  # 期望等級索引


def game_shape(logp):  # 一局的形狀特徵
    """The FEATURES of one game from its per-move L1b log p [n, 10]."""
    lp = np.asarray(logp, dtype=np.float64)  # 轉成雙精度
    n = len(lp)  # 手數
    if n == 0:  # 沒有手
        return np.zeros(len(FEATURES))  # 全部為 0
    pmi = move_pmi(lp)  # 每手的期望等級
    s = -lp.max(axis=1)  # 每手最佳等級的驚奇度
    t = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)  # 正規化到 [0, 1] 的手數索引
    half = n // 2  # 前半的手數
    half_diff = float(pmi[half:].mean() - pmi[:half].mean()) if half else 0.0  # 後半平均減前半平均
    slope = float(np.polyfit(t, pmi, 1)[0]) if n > 2 else 0.0  # pmi 對正規化手數索引的最小平方法斜率（至少 3 手才算）
    return np.array([pmi.mean(), np.log(n), pmi.std(), _ac(pmi, 1), _ac(pmi, 2), half_diff, slope, s.mean(), _ac(s, 1)])  # 依 FEATURES 的順序組成


def game_pmi_hist(logp):  # 一局的 pmi 直方圖
    """[N_BINS]: the fraction of the game's moves whose pmi_t falls in each bin of HIST_EDGES (zeros for a game without moves)."""
    lp = np.asarray(logp, dtype=np.float64)  # 轉成雙精度
    if len(lp) == 0:  # 沒有手
        return np.zeros(N_BINS)  # 全部為 0
    counts, _ = np.histogram(move_pmi(lp), bins=HIST_EDGES)  # 各區間的手數
    return counts / len(lp)  # 換算成比例


def fold_table(fold_dir, fn=game_shape, names=FEATURES):  # 一折每局的特徵表
    """game_key, player_id, rank, n_moves and fn(per-move log p) -> `names` of every game of one scored fold (an l1b_score.py fold directory)."""
    games, moves = read_moves(fold_dir)  # 讀取棋局表與逐手結果
    rows = np.stack([fn(moves[k][1]) for k in games["game_key"]]) if len(games) else np.zeros((0, len(names)))  # 每局套用特徵函式
    out = pd.DataFrame(rows, columns=list(names))  # 特徵欄
    for c in ("n_moves", "rank", "player_id", "game_key"):  # 依序插到最前面
        out.insert(0, c, games[c].to_numpy())  # 結果的前四欄為 game_key、player_id、rank、n_moves
    return out  # 回傳特徵表


def fold_shape_table(fold_dir):  # 一折的形狀特徵表
    """fold_table with the FEATURES."""
    return fold_table(fold_dir, game_shape, FEATURES)  # 套用 game_shape


def fold_hist_table(fold_dir):  # 一折的直方圖特徵表
    """fold_table with the HIST bins of game_pmi_hist."""
    return fold_table(fold_dir, game_pmi_hist, HIST)  # 套用 game_pmi_hist


def question_shape(table, questions, cols=SHAPE, with_std=False):  # 題目層的形狀特徵
    """X [Q, k] (or [Q, 2k] with the standard deviation over the games appended): the mean of `cols` over each question's games."""
    g = table.set_index("game_key")  # 以棋局鍵為索引
    vals = g[list(cols)].to_numpy(dtype=float)  # 每局的特徵值
    row = {k: i for i, k in enumerate(g.index)}  # 棋局鍵對應列號
    k = len(cols)  # 特徵數
    out = np.zeros((len(questions), 2 * k if with_std else k))  # 輸出矩陣
    for q, keys in enumerate(questions["game_keys"]):  # 逐題計算
        v = vals[[row[key] for key in keys]]  # 這一題各局的特徵
        out[q, :k] = v.mean(axis=0)  # 各局平均
        if with_std:  # 需要標準差時
            out[q, k:] = v.std(axis=0)  # 附上各局標準差
    return out  # 回傳題目特徵
