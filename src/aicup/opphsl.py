"""Section 4M M14: the opponent's moves scored under the 18 Human SL profiles as question features.

The training side scores them with scripts/score_folds.py --opponent (results/scores_opp_hd, results/scores_opp_hd_x40); the
test side with predict_submission.py's opponent pass (the shards' hsl_opp_sums / hsl_opp_move_logp). Both hold, per game, the
opponent's move count and its 18 log p sums (one per profile, own = opp = profile metadata, as the target's own HSL scores).

question_features gives the 39 columns of one question: the mean and std over its games of the per-game per-move mean log p
curve centred by its max (18 + 18), the mean and std over games of the opponent's HSL posterior-mean index (2), and the
target's own HSL posterior-mean index mean minus the opponent's (1). opp_hsl_features is the table form (scored-game tables
as the training side reads them). The R-level ablation (experiments/nano4/r_variants.py opphsl) was +0.0200 [+0.0156, +0.0245].
"""
import numpy as np  # 特徵的數值運算

from aicup.evidence import posterior_mean_index  # 後驗平均剖面索引（pmi）
from aicup.humansl import PROFILES  # 18 個 Human SL 剖面名稱

SUM_COLS = [f"sum_{p}" for p in PROFILES]  # 評分表中各剖面的 log p 總和欄
OPP_HSL_FEATURE_NAMES = ([f"opp_hsl_mean_{p}" for p in PROFILES] + [f"opp_hsl_std_{p}" for p in PROFILES]  # 曲線的題內平均與標準差（18 + 18）
                         + ["opp_hsl_pmi_mean", "opp_hsl_pmi_std", "own_minus_opp_hsl_pmi"])  # 對手 pmi 平均、標準差與本人減對手（3 欄）
N_OPP_HSL = len(OPP_HSL_FEATURE_NAMES)  # 特徵欄數（39）


def per_game_curve(opp_sums, opp_n):  # 每局對手的相對似然曲線
    """[G, 18]: the per-move mean log p of the opponent's moves under each profile, centred by the per-game maximum."""
    oc = np.asarray(opp_sums, dtype=float) / np.maximum(np.asarray(opp_n, dtype=float), 1.0)[:, None]  # 總和除以手數得逐手平均，手數至少當 1
    return oc - oc.max(axis=1, keepdims=True)  # 以每局最大值置中，只留剖面間的相對高低


def question_features(own_sums, opp_sums, opp_n):  # 一個題目的對手 HSL 特徵
    """[N_OPP_HSL] of one question: own_sums [G, 18] (the target's HSL sums), opp_sums [G, 18] and opp_n [G] (the opponent's)."""
    own_sums, opp_sums = np.asarray(own_sums, dtype=float), np.asarray(opp_sums, dtype=float)  # 轉成浮點陣列
    if own_sums.shape != opp_sums.shape or own_sums.ndim != 2 or own_sums.shape[1] != len(PROFILES) or len(opp_n) != len(opp_sums):  # 形狀不一致
        raise ValueError(f"own sums {own_sums.shape}, opponent sums {opp_sums.shape}, {len(opp_n)} move counts: expected [G, {len(PROFILES)}] each")  # 拒絕錯位的輸入
    C = per_game_curve(opp_sums, opp_n)  # 每局對手的置中曲線
    pm, hm = posterior_mean_index(opp_sums), posterior_mean_index(own_sums)  # 每局對手與本人的 pmi
    return np.concatenate([C.mean(axis=0), C.std(axis=0), [pm.mean(), pm.std(), hm.mean() - pm.mean()]])  # 依 OPP_HSL_FEATURE_NAMES 的順序串接


def opp_hsl_features(hsl, opp_hsl, questions):  # 整批題目的對手 HSL 特徵表
    """[Q, N_OPP_HSL] from the scored-game tables (game_key, n_moves, sum_<profile>) of the target (hsl) and of the opponent
    (opp_hsl, whose n_moves is the opponent's); questions: a frame with game_keys lists. ValueError for a game missing from a table."""
    o, h = opp_hsl.set_index("game_key"), hsl.set_index("game_key")  # 兩張評分表以棋局鍵為索引
    orow, hrow = {k: i for i, k in enumerate(o.index)}, {k: i for i, k in enumerate(h.index)}  # 棋局鍵轉列號
    O, On, H = o[SUM_COLS].to_numpy(dtype=float), o["n_moves"].to_numpy(dtype=float), h[SUM_COLS].to_numpy(dtype=float)  # 對手總和、對手手數、本人總和
    out = np.zeros((len(questions), N_OPP_HSL))  # 每個題目一列的輸出
    for qi, keys in enumerate(questions["game_keys"]):  # 逐題處理
        keys = list(keys)  # 該題的棋局鍵
        missing = [k for k in keys if k not in orow or k not in hrow]  # 不同時在兩張表中的棋局
        if missing:  # 有缺局
            raise ValueError(f"question {qi}: {len(missing)} games are not in both HSL tables, e.g. {missing[:3]}")  # 停止，避免靜默算錯
        oi, hi = [orow[k] for k in keys], [hrow[k] for k in keys]  # 該題各局在兩張表中的列號
        out[qi] = question_features(H[hi], O[oi], On[oi])  # 算出該題的 39 欄
    return out  # [Q, N_OPP_HSL] 特徵表
