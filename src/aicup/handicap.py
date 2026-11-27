"""The handicap information of dataset v1.1 (HD[n], issue #42) as question-level features (proposal (a)).

The server assigns the handicap from the two players' ranks: HD[1] (讓先, about 16% of games) when they are about one
rank apart, HD[n] stones (3.6%) when n ranks apart, HD[0] (80%) when equal. It is label-side information, like the
opponent channel of fused_opp_diff but more direct. The `color` column is the ACTUAL colour (issue #42's census), so the
labelled player is the upper hand of a handicap game when their colour is W.

Per game: hd_even (HD = 0), hd_sente (HD = 1), hd_stones (HD >= 2), hd_gap_up (the gap when the labelled player is the
upper hand, else 0), hd_gap_down (the gap when they are the lower hand, else 0); the signed gap is up - down. A record
without HD (dataset v1.0) gives zeros in every column. Per question: the mean over its games.
"""
from pathlib import Path  # 路徑處理

import numpy as np  # 數值陣列
import pandas as pd  # 讀寫表格

from aicup.sgf import handicap  # 從 SGF 讀出 HD[n]

HD_FEATURE_NAMES = ["hd_even", "hd_sente", "hd_stones", "hd_gap_up", "hd_gap_down"]  # 5 個讓子特徵欄
HD_COLUMNS = ["game_key", "player_id", "rank", "color", "hd"]  # hd_table 的欄位


def hd_table(train_dir):  # 由訓練 CSV 建立每局的讓子表
    """game_key, player_id, rank, color, hd (HD[n] of the record as float; NaN when absent) of every game of the training
    CSVs in `train_dir` (reads the sgf_content column once; a few CPU minutes over a million rows)."""
    frames = []  # 每個 CSV 一張表
    for p in sorted(Path(train_dir).glob("train_*.csv")):  # 逐一讀取訓練 CSV
        df = pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str, keep_default_na=False)  # 只讀需要的欄位
        frames.append(pd.DataFrame({"game_key": df["rank"] + ":" + df["game_id"], "player_id": df["player_id"], "rank": df["rank"], "color": df["color"],  # 局鍵為「等級:局號」
                                    "hd": df["sgf_content"].map(handicap).astype(float)}))  # 解析每局的 HD
    if not frames:  # 沒有任何訓練 CSV
        raise FileNotFoundError(f"no train_*.csv in {train_dir}")  # 停止並說明
    return pd.concat(frames, ignore_index=True)  # 合併成一張表


def read_hd_table(path):  # 讀回 hd_table
    """An hd_table written with to_csv (hd NaN when absent)."""
    t = pd.read_csv(path, dtype={"game_key": str, "player_id": str, "rank": str, "color": str}, keep_default_na=False, na_values={"hd": [""]})  # 只有 hd 欄的空值視為 NaN
    t["hd"] = pd.to_numeric(t["hd"], errors="coerce").astype(float)  # hd 轉成浮點
    return t  # 回傳表格


def signed_gap(hd, color):  # 有號的等級差距
    """The server's rank gap seen from the labelled player: +HD when they are the upper hand (actual colour W), -HD when
    the lower hand (B); 0 for an even game or an absent HD. Arrays in, array out."""
    h = np.nan_to_num(np.asarray(hd, dtype=float), nan=0.0)  # 沒有 HD 視為 0
    sign = np.where(np.asarray(color, dtype=str) == "W", 1.0, -1.0)  # 執白為上手
    return h * sign  # 有號差距


def game_hd_features(hd, color):  # 每局的 5 個讓子特徵
    """[m, 5]: HD_FEATURE_NAMES of every game from its HD (NaN = absent) and the labelled player's actual colour."""
    h = np.asarray(hd, dtype=float)  # HD 陣列
    known = np.isfinite(h)  # 有 HD 的局
    gap = signed_gap(h, color)  # 有號差距
    return np.column_stack([(known & (h == 0)).astype(float), (known & (h == 1)).astype(float), (known & (h >= 2)).astype(float),  # 互先、讓先、讓子三個指示欄
                            np.maximum(gap, 0.0), np.maximum(-gap, 0.0)])  # 上手與下手時的差距


def question_hd_features(table, questions):  # 每題的讓子特徵（訓練端）
    """X [Q, 5]: the mean of game_hd_features over each question's game_keys, looked up in an hd_table (KeyError for a game
    the table lacks)."""
    g = table.drop_duplicates("game_key").set_index("game_key")  # 每局一列
    F = game_hd_features(g["hd"].to_numpy(), g["color"].to_numpy())  # 每局的特徵
    row = {k: i for i, k in enumerate(g.index)}  # 局鍵 -> 列號
    rows = [F[[row[k] for k in ks]].mean(axis=0) for ks in questions["game_keys"]]  # 題內各局取平均
    return np.stack(rows) if rows else np.zeros((0, len(HD_FEATURE_NAMES)))  # 沒有題目時回傳空矩陣


def question_hd_features_from_games(games):  # 每題的讓子特徵（提交端）
    """[5]: the mean of game_hd_features over (sgf, colour) pairs, for the submission's questions (the test files' HD[])."""
    if not games:  # 沒有局
        return np.zeros(len(HD_FEATURE_NAMES))  # 回傳全 0
    hd = [np.nan if (h := handicap(sgf)) is None else float(h) for sgf, _ in games]  # 解析每局的 HD，沒有時為 NaN
    return game_hd_features(hd, [c for _, c in games]).mean(axis=0)  # 各局特徵取平均


def hd_summary(table):  # 讓子分布的摘要
    """Counts of HD values (\"absent\" for NaN) and the shares of games with HD >= 1 and HD >= 2."""
    h = table["hd"].to_numpy(dtype=float)  # HD 陣列
    known = np.isfinite(h)  # 有 HD 的局
    counts = {"absent": int((~known).sum())} if (~known).any() else {}  # 沒有 HD 的局數（有才記）
    for v in sorted(set(h[known].astype(int).tolist())):  # 逐一 HD 值
        counts[str(v)] = int((h == v).sum())  # 該值的局數
    return {"games": int(len(h)), "hd": counts, "share_hd1": float((known & (h >= 1)).mean()) if len(h) else 0.0,  # 總局數、各值計數與 HD >= 1 的比例
            "share_hd2": float((known & (h >= 2)).mean()) if len(h) else 0.0}  # HD >= 2 的比例
