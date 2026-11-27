"""Data for the rank-evidence replay wall (experiments/viz/rank_replay.html).

Each board replays one game; under it, the posterior-mean Human SL profile of the moves the
target player has made so far (uniform prior over the 18 profiles) shows how the rank
evidence builds up move by move, against the band of best-fitting profiles that the pilot
measured for the player's true class.

Page capacity comes from the lab01 GPU measurements (input-to-paint P95, one headless
Chrome page): 50 boards 28 ms, 100 boards 54 ms, 200 boards 105 ms; 20 / 40 separate
windows 24 / 52 ms. So a page holds at most MAX_BOARDS boards unless forced.
"""
import re  # 比對評分分段的檔名
from datetime import datetime  # 記錄資料的產生時間
from pathlib import Path  # 路徑處理

import numpy as np  # 數值運算
import pandas as pd  # 讀取棋局清單

from aicup.evidence import posterior_mean_index  # 後驗平均剖面索引
from aicup.humansl import PROFILES  # 18 個 Human SL 剖面
from aicup.metric import LEVELS  # 10 個等級
from aicup.sgf import handicap, setup_and_moves  # 讓子數與著手序列

MAX_BOARDS = 100  # 一頁棋盤數的實測舒適上限
DEFAULT_BOARDS = 50  # 預設的棋盤數
MAX_WINDOWS = 40  # 獨立視窗數的實測上限


def check_board_count(n, force=False):  # 檢查一頁的棋盤數
    if n > MAX_BOARDS and not force:  # 超過上限且未強制
        raise ValueError(f"{n} boards on one page; the measured comfortable limit is {MAX_BOARDS} "  # 拒絕並說明實測依據
                         f"(P95 54 ms at 100, 105 ms at 200). Use fewer boards, separate windows, or force.")  # 建議的替代做法


def evidence_series(logp):  # 逐手累積的等級證據
    """Per target move: posterior-mean profile index and best profile of the cumulative log p."""
    cum = np.cumsum(np.asarray(logp, dtype=float), axis=0)  # 各剖面 log p 的逐手累積
    return {"pmi": [round(float(v), 3) for v in posterior_mean_index(cum)], "best": [int(i) for i in cum.argmax(axis=1)]}  # 每手的 pmi（取三位小數）與最佳剖面


def class_bands(best_profile_counts, lo=0.1, hi=0.9):  # 各類別的最佳剖面區帶
    """{class: [low, high]} profile indices covering the middle of each class's best-profile counts."""
    out = {}  # 類別到區帶的字典
    for cls, counts in best_profile_counts.items():  # 逐類別處理
        cum = np.cumsum(counts) / max(sum(counts), 1)  # 累積比例
        out[cls] = [int(np.searchsorted(cum, lo - 1e-12)), int(np.searchsorted(cum, hi - 1e-12))]  # 累積比例到達 lo 與 hi 的剖面索引
    return out  # {類別: [low, high]}


def select_games(games, per_rank, seed=42, ranks=LEVELS):  # 挑選要重播的棋局
    """per_rank players per rank, one random game each, in weak-to-strong order."""
    rng = np.random.RandomState(seed)  # 固定種子的抽樣器
    picked = []  # 收集選出的棋局
    for rank in ranks:  # 由弱到強逐等級
        sub = games[games["rank"] == rank]  # 該等級的棋局
        players = sorted(sub["player_id"].unique())  # 該等級的玩家，排序以求可重現
        for pid in [players[i] for i in sorted(rng.choice(len(players), size=min(per_rank, len(players)), replace=False))]:  # 不重複地抽出至多 per_rank 位玩家
            keys = sorted(sub.loc[sub["player_id"] == pid, "game_key"])  # 該玩家的棋局鍵
            picked.append({"game_key": keys[rng.randint(len(keys))], "player_id": pid, "rank": rank})  # 隨機選一局
    return pd.DataFrame(picked, columns=["game_key", "player_id", "rank"])  # 選出的棋局表


def _pairs(directory, fold, layout):  # 列出評分輸出的 CSV/npz 成對檔案
    d = Path(directory)  # 評分輸出的資料夾
    if layout == "pilot":  # experiments/pilot_score.py 的輸出格式（分片檔直接在資料夾內）
        for csv in sorted(d.glob(f"games_fold{fold}_shard*.csv")):  # 逐個分片的棋局表
            yield csv, d / csv.name.replace("games_", "moves_").replace(".csv", ".npz")  # 配對同名的逐手檔
    elif layout == "scores":  # scripts/score_folds.py 的輸出格式（每折一個子資料夾）
        d = d / f"fold{fold}"  # 該折的子資料夾
        for csv in sorted(d.glob("games_s*_p*.csv")):  # 逐個分段的棋局表
            if re.fullmatch(r"games_s\d+_p\d+\.csv", csv.name):  # 只接受完全符合命名規則的檔案
                yield csv, d / csv.name.replace("games_", "moves_").replace(".csv", ".npz")  # 配對同名的逐手檔
    else:  # 不認得的格式
        raise ValueError(f"unknown layout {layout!r}")  # 拒絕


def load_move_logp(directory, fold, layout, keys=None):  # 讀取逐手 log p
    """{game_key: {"logp": [n, 18] float32, "t": [n]}} from CSV/npz pairs; finished pairs only."""
    want = None if keys is None else set(keys)  # 只要指定棋局時的集合
    out = {}  # 棋局鍵到逐手資料
    for csv, npz in _pairs(directory, fold, layout):  # 逐個成對檔案
        if not npz.exists():  # 逐手檔還沒寫出（該段未完成）
            continue  # 略過
        names = pd.read_csv(csv, usecols=["game_key"])["game_key"].tolist()  # 這一段的棋局鍵，順序即列號
        if want is not None and not want.intersection(names):  # 這一段沒有要的棋局
            continue  # 不讀 npz
        z = np.load(npz)  # 逐手陣列
        rows, logp, ts = z["game_row"], z["logp"], z["t"]  # 每手所屬的棋局列、各剖面的 log p 與手序
        order = np.argsort(rows, kind="stable")  # 依棋局列穩定排序
        bounds = np.searchsorted(rows[order], np.arange(len(names) + 1))  # 每局在排序後陣列中的起訖
        for r, key in enumerate(names):  # 逐局取出
            if want is None or key in want:  # 是要的棋局
                idx = order[bounds[r]:bounds[r + 1]]  # 該局各手的位置
                out[key] = {"logp": logp[idx].astype(np.float32), "t": ts[idx]}  # 保存該局的逐手 log p 與手序
    return out  # 以棋局鍵為索引的逐手資料


def build_payload(rows, sgf_map, logp_map, bands, title, source):  # 組成頁面讀取的資料
    games = []  # 每個棋盤的資料
    for r in rows.itertuples():  # 逐局處理
        sgf, color = sgf_map[r.game_key]  # 棋譜與目標玩家的顏色
        ev = logp_map.get(r.game_key)  # 該局的逐手 log p（可能沒有）
        stones, moves = setup_and_moves(sgf)  # 讓子石與著手序列
        games.append({  # 一個棋盤的資料
            "key": r.game_key, "player": r.player_id[:8], "rank": r.rank, "color": color,  # 棋局、玩家 id 前 8 碼、等級與顏色
            "moves": [None if x is None else [x, y] for _, x, y in moves],  # 著手座標，虛手為 None
            "stones": [[x, y] for x, y in stones], "first": moves[0][0] if moves else "B", "handicap": handicap(sgf),  # 讓子石、第一手的顏色與讓子數
            "band": bands.get(r.rank),  # 該等級的最佳剖面區帶
            "evidence": None if ev is None else {"t": [int(t) for t in ev["t"]], **evidence_series(ev["logp"])},  # 逐手證據曲線
        })  # 結束這個棋盤
    return {"title": title, "source": source, "created": datetime.now().isoformat(timespec="seconds"),  # 頁面標題、資料來源與產生時間
            "profiles": PROFILES, "levels": LEVELS, "games": games}  # 剖面、等級與棋盤清單
