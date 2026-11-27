"""Question packing and the static rungs (0)(1)(2) of the L3 aggregator ladder.

pack_questions turns a question DataFrame into a padded set of per-game tokens over the vectors
of aicup.gamevec (`vec`: game_key, staged, whole, sums, n_moves). static_features turns packed
questions into the fixed-length question features of the static rungs, in float64; every kind is
a symmetric function of the question's games, so the order of game_keys does not matter.

Notation (per question): m games, sums_g [18] and n_g target moves of game g, L = sum_g sums_g,
n = sum_g n_g, c_g = sums_g / n_g (per-move curve), r_g = c_g - mean_r c_g (its shape over profiles).
log m and log n are the evidence amount; counts=False is the game-length ablation (METHOD_NOTES
§2.6): it drops the explicit count inputs (token n_eff, present and log1p(n_g); qglob beyond log m;
log n; q24's log_moves), as listed in pack_questions and static_features.
"""
import numpy as np  # 數值運算

from aicup.gamevec import SEGMENTS, WHOLE, vec_names  # 分段定義與每局向量的欄名
from aicup.metric import RANK_TO_IDX  # 等級轉索引
from aicup.qfeatures import FEATURE_NAMES, question_features  # 24 維題目特徵（q24）
from aicup.rankmodel import MLPModel, SoftmaxModel  # 靜態梯級使用的模型

QGLOB_NAMES = ["log_games", "log_target_moves", "log1p_seg1", "log1p_seg2", "log1p_seg3", "log1p_seg4"]  # 題目層全域輸入（qglob）的欄名
REPS = {"staged": SEGMENTS, "whole": WHOLE}  # 兩種每局向量表示：分四段或整局
CLIP_QUANTILE = 0.95  # 截斷寬度取的分位數


def _stat_cols(names, stat):  # 找出某個逐段欄位的欄索引
    """Columns of one per-stage field (e.g. "n_eff"), in stage order: names are {prefix}_{stat}."""
    return [i for i, name in enumerate(names) if name.split("_", 1)[1] == stat]  # 去掉段前綴後比對


_STAGED = vec_names(SEGMENTS)  # 分段向量的欄名
SEG_N_EFF, SEG_PRESENT = _stat_cols(_STAGED, "n_eff"), _stat_cols(_STAGED, "present")  # 各段的手數欄與有無欄


def count_cols(rep):  # 局長消融要歸零的 token 欄
    """Token columns zeroed by counts=False: each stage's n_eff and present, and log1p(n_moves_g) (the last)."""
    names = vec_names(REPS[rep])  # 該表示的欄名
    return _stat_cols(names, "n_eff") + _stat_cols(names, "present") + [len(names)]  # 各段 n_eff、present 與最後的 log1p(手數) 欄


def pack_questions(questions, vec, rep="staged", counts=True, max_games=20):  # 把題目打包成集合模型的輸入
    """Questions -> padded game sets over the vectors in `vec` (see the module docstring).

    Returns tokens float32 [Q, max_games, D + 1] (the game's `rep` vector, D = 300 staged or 75
    whole, then log1p(n_moves_g); padding rows are 0), mask bool [Q, max_games] (True = a game),
    qglob float32 [Q, 6] aligned with QGLOB_NAMES (log m, log n and log1p of each stage's total
    target moves, always from the staged n_eff), y int64 [Q] (-1 without a rank), idx (per question
    the vec rows of its games, in game_keys order), player_id, n_games and question_id arrays.
    counts=False zeroes the count_cols(rep) token columns and qglob[:, 1:].
    """
    if rep not in REPS:  # 不認得的表示
        raise ValueError(f"rep must be one of {list(REPS)}, got {rep!r}")  # 拒絕
    feats = vec[rep]  # 每局的向量
    D = len(vec_names(REPS[rep]))  # 向量維度
    if feats.shape[1] != D:  # 向量檔與欄名定義不一致
        raise ValueError(f"vec[{rep!r}] has {feats.shape[1]} columns, expected {D}")  # 拒絕
    keys = vec["game_key"].tolist()  # 向量檔中各局的鍵
    row = {k: i for i, k in enumerate(keys)}  # 棋局鍵轉列號
    if len(row) != len(keys):  # 有重複的鍵
        raise ValueError(f"vec has {len(keys) - len(row)} duplicate game keys")  # 拒絕，避免對錯局
    idx = []  # 每題各局的列號
    for qid, gk in zip(questions["question_id"], questions["game_keys"]):  # 逐題處理
        if not 1 <= len(gk) <= max_games:  # 局數超出範圍
            raise ValueError(f"question {qid}: {len(gk)} games, need 1..{max_games}")  # 拒絕
        missing = [k for k in gk if k not in row]  # 不在向量檔中的局
        if missing:  # 有缺局
            raise KeyError(f"question {qid}: game {missing[0]} is not in the vectors")  # 拒絕
        idx.append(np.array([row[k] for k in gk], dtype=np.int64))  # 依 game_keys 的順序記下列號
    n_games = np.array([len(ix) for ix in idx], dtype=np.int64)  # 每題局數
    if "n_games" in questions and not np.array_equal(questions["n_games"].to_numpy(dtype=np.int64), n_games):  # 題目表的局數欄與實際不符
        raise ValueError("n_games column differs from the number of game_keys")  # 拒絕

    n_moves, seg = vec["n_moves"], vec["staged"][:, SEG_N_EFF].astype(np.float64)  # 每局目標手數與各段手數
    tokens = np.zeros((len(idx), max_games, D + 1), dtype=np.float32)  # 補齊到 max_games 的 token
    mask = np.zeros((len(idx), max_games), dtype=bool)  # 哪些位置是真的棋局
    qglob = np.zeros((len(idx), len(QGLOB_NAMES)))  # 題目層全域輸入
    for q, ix in enumerate(idx):  # 逐題填入
        m = len(ix)  # 該題局數
        tokens[q, :m, :D] = feats[ix]  # 前 D 欄放每局向量
        tokens[q, :m, D] = np.log1p(n_moves[ix])  # 最後一欄放 log1p(手數)
        mask[q, :m] = True  # 標記有效的局
        qglob[q] = [np.log(m), np.log(max(n_moves[ix].sum(), 1)), *np.log1p(seg[ix].sum(axis=0))]  # log m、log n 與各段總手數的 log1p
    if not counts:  # 局長消融
        tokens[:, :, count_cols(rep)] = 0.0  # 歸零 token 的計數欄
        qglob[:, 1:] = 0.0  # 只留 log m
    if "rank" in questions:  # 有等級欄
        y = questions["rank"].map(RANK_TO_IDX).fillna(-1).astype(np.int64).to_numpy()  # 等級轉索引，未知為 -1
    else:  # 測試題沒有等級
        y = np.full(len(idx), -1, dtype=np.int64)  # 全部標為未知
    return {"tokens": tokens, "mask": mask, "qglob": qglob.astype(np.float32), "y": y, "idx": idx,  # 模型輸入與標籤
            "player_id": questions["player_id"].to_numpy(dtype=str), "n_games": n_games,  # 玩家與局數
            "question_id": questions["question_id"].to_numpy(dtype=str)}  # 題號


def _profile_shape(vec, ix):  # 每局曲線在剖面間的形狀
    """r_g = c_g - mean_r c_g [m, 18] with c_g = sums_g / n_g (float64)."""
    c = vec["sums"][ix].astype(np.float64) / np.maximum(vec["n_moves"][ix].astype(np.float64), 1.0)[:, None]  # 逐手平均曲線 c_g
    return c - c.mean(axis=1, keepdims=True)  # 減去剖面平均，只留形狀


def _evidence(vec, ix, counts):  # 證據量欄
    """[log m, log n], or [log m] when counts=False."""
    m, n = len(ix), max(float(vec["n_moves"][ix].sum()), 1.0)  # 局數與總手數（至少 1）
    return [np.log(m), np.log(n)] if counts else [np.log(m)]  # 局長消融時拿掉 log n


def _robust(vec, ix, kind, clip_k):  # 梯級 (1) 的穩健平均
    """18 robust per-profile means of r_g: clipped around the median ("clip") or 20% trimmed ("trim")."""
    r = _profile_shape(vec, ix)  # 每局的形狀 r_g
    if kind == "clip":  # 以中位數為中心截斷
        med = np.median(r, axis=0)  # 各剖面的中位數
        return med + np.clip(r - med, -clip_k, clip_k).mean(axis=0)  # 偏差截在 ±k 後平均，再加回中位數
    cut = len(r) // 5  # 每側去掉 floor(0.2 m) 局
    return np.sort(r, axis=0)[cut:len(r) - cut].mean(axis=0)  # 各剖面排序後取截尾平均


def _sum(vec, ix):  # 梯級 (0) 的加總特徵
    """The calibrated-sum form of §2.3, linearised: (L - mean_r L) / sqrt(n) and / n."""
    L = vec["sums"][ix].astype(np.float64).sum(axis=0)  # 全題各剖面的 log p 總和 L
    n = max(float(vec["n_moves"][ix].sum()), 1.0)  # 總手數（至少 1）
    d = L - L.mean()  # 減去剖面平均
    return np.concatenate([d / np.sqrt(n), d / n])  # 兩種縮放並列


def _q24(packed, vec, counts):  # 24 維題目特徵
    X = np.stack([question_features(vec["sums"][ix], vec["n_moves"][ix]) for ix in packed["idx"]])  # 逐題計算 q24
    return X if counts else np.delete(X, FEATURE_NAMES.index("log_moves"), axis=1)  # 局長消融時刪掉 log_moves


def _pooled(packed, vec, counts):  # 梯級 (2) 的平均 token 特徵
    """Mean token over the question's games, qglob and q24; counts=False strips the count columns."""
    tok, mask = packed["tokens"], packed["mask"]  # 打包好的 token 與遮罩
    reps = {len(vec_names(s)) + 1: r for r, s in REPS.items()}  # 由 token 欄數反推表示
    if tok.shape[2] not in reps:  # 欄數不屬於任何表示
        raise ValueError(f"tokens have {tok.shape[2]} columns, not a packed rep")  # 拒絕
    # 逐題計算，避免把整個 [Q, 20, D + 1] 的 token 一次複製成 float64
    mean = np.stack([tok[q, mask[q]].astype(np.float64).mean(axis=0) for q in range(len(tok))])  # 每題有效各局 token 的平均
    qglob = packed["qglob"].astype(np.float64)  # 題目層全域輸入
    if counts:  # 保留計數欄
        if np.any(mean[:, -1] == 0):  # log1p(手數) 的平均為 0，表示打包時已歸零
            raise ValueError("pooled with counts=True needs questions packed with counts=True")  # 打包與特徵設定不一致
    else:  # 局長消融
        mean[:, count_cols(reps[tok.shape[2]])] = 0.0  # 歸零計數欄
        qglob[:, 1:] = 0.0  # 只留 log m
    return np.hstack([mean, qglob, _q24(packed, vec, counts)])  # 平均 token、qglob 與 q24 並列


def _length(vec, ix):  # 只有局長的對照特徵
    """Game-length statistics only (§2.6): log m, log n, mean and std (ddof=0) of n_g, per-stage
    fraction of games with the stage present and mean n_eff."""
    n_g = vec["n_moves"][ix].astype(np.float64)  # 每局目標手數
    st = vec["staged"][ix].astype(np.float64)  # 每局的分段向量
    return np.concatenate([_evidence(vec, ix, True), [n_g.mean(), n_g.std()], st[:, SEG_PRESENT].mean(axis=0),  # log m、log n、手數平均與標準差、各段出現比例
                           st[:, SEG_N_EFF].mean(axis=0)])  # 各段平均手數


def static_features(packed, vec, kind, counts=True, clip_k=None):  # 靜態梯級的題目特徵
    """Question features X float64 [Q, F] of one static kind (see the module docstring for notation).

    sum: [(L - mean_r L)/sqrt(n) (18), (L - mean_r L)/n (18), log m, log n]
    clip: [med + mean_g clip(r_g - med, -k, k) (18), log m, log n], med = per-profile median_g r_g,
          k = clip_k (from fit_clip_k on training questions only)
    trim: [per-profile 20% trimmed mean of r_g, floor(0.2 m) games off each side (18), log m, log n]
    q24: qfeatures.question_features of the question's games, identical to qfeatures.build_matrix
    pooled: [mean token over the valid games (D + 1), qglob (6), q24 (24)]
    length: [log m, log n, mean and std of n_g, per-stage present fraction (4), per-stage mean n_eff (4)]
    counts=False: sum/clip/trim drop log n, q24 drops log_moves, pooled zeroes the token count columns
    and qglob[:, 1:] and uses the reduced q24; length is refused.
    """
    idx = packed["idx"]  # 每題各局在 vec 中的列
    if kind == "q24":  # 24 維題目特徵
        return _q24(packed, vec, counts)  # 與 build_matrix 相同的特徵
    if kind == "pooled":  # 平均 token
        return _pooled(packed, vec, counts)  # 平均 token、qglob 與 q24
    if kind == "length":  # 只有局長
        if not counts:  # 局長特徵本身就是計數
            raise ValueError("the length-only kind is made of counts; it has no counts=False form")  # 沒有消融版本
        return np.stack([_length(vec, ix) for ix in idx])  # 逐題局長統計
    if kind == "sum":  # 梯級 (0)
        return np.stack([np.concatenate([_sum(vec, ix), _evidence(vec, ix, counts)]) for ix in idx])  # 加總特徵加證據量
    if kind in ("clip", "trim"):  # 梯級 (1)
        if kind == "clip" and clip_k is None:  # 截斷版需要訓練題的寬度
            raise ValueError("kind 'clip' needs clip_k (fit_clip_k on the training questions)")  # 拒絕
        return np.stack([np.concatenate([_robust(vec, ix, kind, clip_k), _evidence(vec, ix, counts)]) for ix in idx])  # 穩健平均加證據量
    raise ValueError(f"unknown static feature kind {kind!r}")  # 不認得的種類


def fit_clip_k(packed, vec):  # 由訓練題求截斷寬度
    """Clip width k: the 95th percentile of |r_g - med| over every game and profile of the (training) questions."""
    dev = []  # 收集每局每剖面的偏差
    for ix in packed["idx"]:  # 逐題
        r = _profile_shape(vec, ix)  # 該題各局的形狀
        dev.append(np.abs(r - np.median(r, axis=0)).ravel())  # 與題內中位數的絕對偏差
    return float(np.quantile(np.concatenate(dev), CLIP_QUANTILE))  # 取第 95 百分位數


# 梯級 -> (靜態特徵種類, 向量表示, 無參數的模型工廠函式)
STATIC_RUNGS = {  # 靜態梯級的定義
    "r0_sum": ("sum", "staged", lambda: SoftmaxModel(l2=1e-3)),  # (0) 校準後似然累加 → softmax 迴歸
    "r1_clip": ("clip", "staged", lambda: SoftmaxModel(l2=1e-3)),  # (1) 以中位數為中心截斷後平均
    "r1_trim": ("trim", "staged", lambda: SoftmaxModel(l2=1e-3)),  # (1) 20% 截尾平均
    "r2_mlp24": ("q24", "staged", lambda: MLPModel(hidden=32)),  # 當時的現行提交模型，也是集合梯級的基準
    "r2_staged": ("pooled", "staged", lambda: MLPModel(hidden=64)),  # (2) 分段 token 的題內平均 → MLP h64
    "r2_whole": ("pooled", "whole", lambda: MLPModel(hidden=64)),  # (2) 整局 token 的題內平均 → MLP h64
    "r2_length_only": ("length", "staged", lambda: MLPModel(hidden=32)),  # 只有局長統計的對照
}  # 結束梯級定義
