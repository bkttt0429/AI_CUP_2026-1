"""TSFEL features of the per-move score series of a game (issue #36: the user's "let TSFEL generate the features").

Each game gives two scalar series over its moves (aicup.seqshape): pmi_t, the posterior-mean rank index of the move's ten
L1b log p, and s_t, the surprisal of the move's best rank. TSFEL (Time Series Feature Extraction Library, 0.2.0) turns a
series into its statistical, temporal, spectral and fractal descriptors; game_tsfel runs it on both series and prefixes the
names with pmi_ / s_. The four wavelet features are left out (they break under pandas 3: duplicate column labels); games
shorter than MIN_MOVES, and any feature TSFEL cannot compute for a series, give NaN. Some TSFEL features have a length-
dependent number of outputs, so a fold's table takes the union of the names and NaN where a game has none; the CV side
standardises with the training questions' statistics and imputes NaN by the training mean (standardise_impute), then
reduces the block with a PCA fitted on the training questions only (pca_fit / pca_apply). fold_tsfel_table caches nothing
itself; experiments/tsfel_check.py writes the per-fold tables to disk because extraction costs about 11 ms per series.
"""
import multiprocessing as mp  # 多程序平行抽取特徵
import warnings  # 隱藏 TSFEL 與 numpy 的警告

import numpy as np  # 數值運算
import pandas as pd  # 特徵表

from aicup.seqdata import read_moves  # 讀取一折的逐手結果
from aicup.seqshape import move_pmi  # 每手的期望等級索引

MIN_MOVES = 12  # TSFEL 的 LPCC／MFCC 至少需要這麼多個樣本；較短的棋局特徵為 NaN
WAVELET_BROKEN = ("Wavelet absolute mean", "Wavelet energy", "Wavelet standard deviation", "Wavelet variance")  # 在 pandas 3 下會出錯而排除的四個小波特徵
SERIES = ("pmi", "s")  # 兩條序列的名稱前綴
_CFG = None  # 快取的 TSFEL 設定


def tsfel_config():  # 取得 TSFEL 設定
    """TSFEL's full configuration minus WAVELET_BROKEN (cached)."""
    global _CFG  # 使用模組層級的快取
    if _CFG is None:  # 第一次呼叫時才建立
        import tsfel  # 延後匯入選用套件 TSFEL（見 requirements-tsfel.txt）
        cfg = tsfel.get_features_by_domain()  # 所有領域的特徵設定
        for name in WAVELET_BROKEN:  # 逐一移除會出錯的小波特徵
            cfg.get("spectral", {}).pop(name, None)  # 從頻譜領域刪除
        _CFG = cfg  # 存入快取
    return _CFG  # 回傳設定


def game_series(logp):  # 一局的兩條序列
    """(pmi_t, s_t) of one game from its per-move L1b log p [n, 10]."""
    lp = np.asarray(logp, dtype=np.float64)  # 轉成雙精度
    if len(lp) == 0:  # 沒有手
        return np.zeros(0), np.zeros(0)  # 兩條空序列
    return move_pmi(lp), -lp.max(axis=1)  # 期望等級與驚奇度


def series_features(x, prefix):  # 一條序列的 TSFEL 特徵
    """{prefix_<TSFEL name>: value} of one series; {} when it is too short or TSFEL fails on it."""
    x = np.asarray(x, dtype=np.float64)  # 轉成雙精度
    if len(x) < MIN_MOVES:  # 序列太短
        return {}  # 不產生特徵
    import tsfel  # 延後匯入選用套件 TSFEL（序列夠長才需要）
    try:  # TSFEL 可能對某些序列出錯
        with warnings.catch_warnings():  # 暫時改變警告設定
            warnings.simplefilter("ignore")  # 隱藏 TSFEL 的警告
            df = tsfel.time_series_features_extractor(tsfel_config(), pd.DataFrame({prefix: x}), fs=1, verbose=0)  # 抽取特徵，取樣頻率設為每手 1 個樣本
    except Exception:  # noqa: BLE001  這條序列算不出的特徵記為 NaN，而不是讓程式中止
        return {}  # 視為沒有特徵
    row = df.iloc[0]  # 唯一的一列結果
    return {str(k): float(v) for k, v in row.items() if np.isfinite(v)}  # 只保留有限值


def game_tsfel(logp):  # 一局的全部 TSFEL 特徵
    """The TSFEL features of both series of one game (names prefixed pmi_ and s_)."""
    pmi, s = game_series(logp)  # 取得兩條序列
    out = series_features(pmi, "pmi")  # pmi 序列的特徵
    out.update(series_features(s, "s"))  # 加上驚奇度序列的特徵
    return out  # 回傳特徵字典


def _one(args):  # 平行處理用的單局包裝
    key, logp = args  # 棋局鍵與逐手 log p
    return key, game_tsfel(logp)  # 回傳棋局鍵與特徵


def fold_tsfel_table(fold_dir, workers=1):  # 一折的 TSFEL 特徵表
    """game_key, player_id, rank, n_moves and the union of the games' TSFEL features (NaN where a game has none) of one
    scored fold (an l1b_score.py fold directory)."""
    games, moves = read_moves(fold_dir)  # 讀取棋局表與逐手結果
    keys = games["game_key"].tolist()  # 棋局順序
    tasks = [(k, moves[k][1]) for k in keys]  # 每局一個任務
    if workers and workers > 1:  # 要平行處理時
        with mp.get_context("fork").Pool(workers) as pool:  # 以 fork 建立程序池
            rows = dict(pool.imap_unordered(_one, tasks, chunksize=64))  # 不依順序收集結果
    else:  # 單一程序時
        rows = dict(map(_one, tasks))  # 依序計算
    feats = pd.DataFrame([rows[k] for k in keys])  # 依原棋局順序組成表，缺少的特徵為 NaN
    feats = feats.reindex(sorted(feats.columns), axis=1)  # 欄位依名稱排序
    out = pd.concat([games[["game_key", "player_id", "rank", "n_moves"]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)  # 前面接上四個棋局欄位
    return out  # 回傳特徵表


def save_tsfel_table(path, table):  # 把一折的特徵表存成 npz
    """One fold's table as an npz (keys, player_id, rank, n_moves, cols, X); npz files under results/seq are not committed."""
    cols = feature_columns(table)  # TSFEL 特徵欄
    np.savez_compressed(path, keys=table["game_key"].to_numpy(dtype=str), player_id=table["player_id"].to_numpy(dtype=str),  # 棋局鍵與玩家
                        rank=table["rank"].to_numpy(dtype=str), n_moves=table["n_moves"].to_numpy(dtype=np.int64),  # 等級與手數
                        cols=np.asarray(cols, dtype=str), X=table[cols].to_numpy(dtype=np.float32))  # 欄名與 float32 特徵矩陣


def load_tsfel_table(path):  # 讀回存檔的特徵表
    with np.load(path, allow_pickle=False) as z:  # 開啟 npz
        cols = [str(c) for c in z["cols"]]  # 特徵欄名
        out = pd.DataFrame({"game_key": z["keys"].astype(str), "player_id": z["player_id"].astype(str), "rank": z["rank"].astype(str), "n_moves": z["n_moves"]})  # 四個棋局欄位
        feats = pd.DataFrame(z["X"].astype(np.float64), columns=cols)  # 特徵轉回雙精度
    return pd.concat([out, feats], axis=1)  # 合成與 fold_tsfel_table 相同格式的表


def feature_columns(table):  # 列出 TSFEL 特徵欄
    """The TSFEL columns of a fold_tsfel_table (everything after the four game columns)."""
    return [c for c in table.columns if c not in ("game_key", "player_id", "rank", "n_moves")]  # 排除四個棋局欄位


def question_tsfel(table, questions, cols):  # 題目層的 TSFEL 特徵
    """X [Q, k]: the nanmean of `cols` over each question's games (NaN when no game of the question has the feature)."""
    g = table.set_index("game_key")  # 以棋局鍵為索引
    vals = g.reindex(columns=list(cols)).to_numpy(dtype=float)  # 依指定欄位取值，缺少的欄位為 NaN
    row = {k: i for i, k in enumerate(g.index)}  # 棋局鍵對應列號
    out = np.full((len(questions), len(cols)), np.nan)  # 預設為 NaN
    with warnings.catch_warnings():  # 暫時改變警告設定
        warnings.simplefilter("ignore", category=RuntimeWarning)  # 全為 NaN 時的平均警告不顯示
        for q, keys in enumerate(questions["game_keys"]):  # 逐題計算
            out[q] = np.nanmean(vals[[row[key] for key in keys]], axis=0)  # 忽略 NaN 取各局平均
    return out  # 回傳題目特徵


def standardise_impute(X_tr, X_va):  # 標準化並填補缺值
    """Standardise both with the training nan-mean / nan-std, impute NaN by 0 (the training mean) and drop the columns that
    are constant or all-NaN in training; returns (Z_tr, Z_va, kept column indices)."""
    mu = np.nanmean(X_tr, axis=0)  # 訓練端各欄的平均
    sd = np.nanstd(X_tr, axis=0)  # 訓練端各欄的標準差
    keep = np.where(np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12))[0]  # 保留有值且有變化的欄位
    mu, sd = mu[keep], sd[keep]  # 只留保留欄位的統計量

    def z(X):  # 套用標準化與填補
        Z = (X[:, keep] - mu) / sd  # 以訓練統計量標準化
        return np.where(np.isfinite(Z), Z, 0.0)  # NaN 填 0（即訓練平均）

    return z(X_tr), z(X_va), keep  # 回傳兩端的結果與保留欄位


def pca_fit(Z, k):  # 在訓練資料上擬合 PCA
    """Principal directions [d, k'] of the standardised training block (k' = min(k, rank)) by SVD of the centred data."""
    Zc = Z - Z.mean(axis=0)  # 中心化
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)  # 奇異值分解
    r = int((s > 1e-9 * max(s[0], 1e-300)).sum()) if len(s) else 0  # 數值上的秩
    k = max(0, min(int(k), r))  # 主成分數不超過秩
    return {"mean": Z.mean(axis=0), "components": vt[:k].T, "explained": (s[:k] ** 2 / max((s ** 2).sum(), 1e-300)).tolist()}  # 平均、主方向與解釋變異比例


def pca_apply(pca, Z):  # 套用 PCA 投影
    return (Z - pca["mean"]) @ pca["components"]  # 中心化後投影到主方向
