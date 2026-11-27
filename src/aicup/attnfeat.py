"""Section 4M M36: per-game log-softmax of the cross-fitted spatial-attention heads (experiments/nano4/e2e_pilot.py score, heads A then B)
as question-level R columns. Head A trained on folds 1-2, head B on folds 3-4: a game of fold 1 or 2 takes B, of fold 3 or 4 takes A,
of fold 0 (and the test set) their mean, so no game is scored by a head that saw its player."""
import glob  # 找出評分分段檔
from pathlib import Path  # 組合路徑

import numpy as np  # 數值陣列
import pandas as pd  # 讀取局表 CSV（每局所屬的折）

N_COLS = 20  # 題內各局 10 個 log-softmax 值的平均與標準差


def load_scores(score_dir, games_csv):  # 讀入每局經交叉擬合選定的分數
    """{game_key: chosen [10] float32} from the score parts and the games CSV (game_key, fold)."""
    files = sorted(glob.glob(str(Path(score_dir) / "part_*.npz")))  # 依名稱排序的分段檔
    if not files:  # 沒有任何分段檔
        raise FileNotFoundError(f"{score_dir}: no score parts")  # 停止並說明
    fold = pd.read_csv(games_csv, dtype={"game_key": str}).set_index("game_key")["fold"].to_dict()  # 每局所屬的折
    out = {}  # 收集結果
    for f in files:  # 逐一讀入分段檔
        z = np.load(f)  # 讀入 npz
        a, b = z["logp0"], z["logp1"]  # A 頭與 B 頭的 log-softmax
        for k, la, lb in zip(z["keys"], a, b):  # 逐局處理
            fo = fold[str(k)]  # 這一局的折
            out[str(k)] = (lb if fo in (1, 2) else la if fo in (3, 4) else 0.5 * (la + lb)).astype(np.float32)  # 折 1、2 取 B 頭，折 3、4 取 A 頭（都沒看過該玩家）；其他（折 0、測試集）取兩頭平均
    return out  # 回傳每局的分數


def question_features(game_keys, scores):  # 組出每題的 20 欄
    """[Q, 20]: per question the mean and the std over its games of the chosen log-softmax; KeyError for an unscored game."""
    rows = []  # 每題一列
    for ks in game_keys:  # 逐題處理
        m = np.stack([scores[str(k)] for k in ks])  # 題內各局的分數 [m, 10]
        rows.append(np.r_[m.mean(axis=0), m.std(axis=0)])  # 平均與標準差串成 20 欄
    return np.asarray(rows, dtype=np.float32)  # 回傳 [Q, 20]
