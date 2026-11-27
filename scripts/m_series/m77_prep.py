"""Section 4M M77 data (2026-11-10): every training game and every test game as move arrays for the whole-game identity encoder.
POS [N, 256] uint16 (0-360 board point y * 19 + x, 361 pass, 362 padding), COL [N, 256] uint8 (0 B, 1 W, 2 padding), the row's
player index, fold, actual colour, HD; the same for the test games (question id, target colour). Output results/m77/data.npz.
  python -I scripts/m_series/m77_prep.py
"""
import multiprocessing as mp  # 平行解析棋譜
import re  # 從 SGF 取出讓子數 HD
import sys  # 調整模組搜尋路徑
import time  # 計時
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取訓練與測試資料

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src")]  # 讓 aicup 套件可匯入
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序
from aicup.sgf import parse_moves  # noqa: E402  解析 SGF 的著手

T = 256  # 每局只取前 256 手
RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引


def arr(sgf):  # 一局棋譜轉成落點與顏色陣列
    m = parse_moves(sgf)[:T]  # 只取前 T 手
    pos = np.full(T, 362, np.uint16); col = np.full(T, 2, np.uint8)  # 預設全為 padding
    for t, (c, x, y) in enumerate(m):  # 逐手填入
        pos[t] = 361 if x is None else y * 19 + x; col[t] = 0 if c == "B" else 1  # pass 記為 361；黑 0、白 1
    h = re.search(r"HD\[(\d+)\]", sgf)  # 讓子數標記
    return pos, col, len(m), int(h.group(1)) if h else 0  # 落點、顏色、手數、HD


def pack(sgfs):  # 平行轉換多局並堆疊
    with mp.get_context("fork").Pool(12) as pool:  # 12 個工作行程
        res = pool.map(arr, sgfs, chunksize=1024)  # 逐局轉換
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res]), np.array([r[2] for r in res], np.int16), np.array([r[3] for r in res], np.int8)  # 回傳 POS、COL、LEN、HD 四個陣列


def main():  # 主流程
    t0 = time.time()  # 開始時間
    rows = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank", "color", "sgf_content"], dtype=str) for p in sorted((ROOT / "data/training").glob("train_*.csv"))], ignore_index=True)  # 讀入全部訓練列
    players = np.array(sorted(rows.player_id.unique())); pidx = {p: i for i, p in enumerate(players)}  # 玩家清單與索引
    plab = rows.drop_duplicates("player_id").set_index("player_id")["rank"].reindex(players).map(RANK).to_numpy()  # 每位玩家的等級標籤
    folds = pd.read_csv(ROOT / "splits/player_folds.csv", dtype={"player_id": str}).set_index("player_id")["fold"]  # 玩家所屬的折
    POS, COL, LEN, HD = pack(rows.sgf_content.tolist()); print(f"training games packed: {POS.shape} ({time.time() - t0:.0f} s)", flush=True)  # 轉換全部訓練局
    recs = []  # 測試局的（題號, 棋譜, 執色）
    for h in ("public", "private"):  # Public 與 Private 測試檔
        df = pd.read_csv(ROOT / f"data/tests/rank_prediction_test_{h}.csv", dtype=str, keep_default_na=False)  # 讀入測試題
        for d in df.to_dict("records"):  # 逐題
            for k in range(1, int(d["num_games"]) + 1):  # 這題的每一局
                s, c = d.get(f"sgf_{k}", ""), d.get(f"color_{k}", "")  # 棋譜與本人執色
                if s and c in ("B", "W"):  # 只收有效的局
                    recs.append((d["question_id"], s, c))  # 加入一筆
    TPOS, TCOL, TLEN, THD = pack([r[1] for r in recs])  # 轉換全部測試局
    np.savez(ROOT / "results/m77/data.npz", POS=POS, COL=COL, LEN=LEN, hd=HD, subj=(rows.color.to_numpy() == "W").astype(np.uint8), player=rows.player_id.map(pidx).to_numpy(np.int32),  # 寫出訓練局陣列、本人顏色與玩家
             fold=rows.player_id.map(folds).to_numpy(np.int8), key=(rows["rank"] + ":" + rows["game_id"]).to_numpy(str), players=players, plab=plab,  # 折、棋局鍵、玩家清單與標籤
             TPOS=TPOS, TCOL=TCOL, TLEN=TLEN, thd=THD, tsubj=np.array([r[2] == "W" for r in recs], np.uint8), tqid=np.array([r[0] for r in recs]))  # 測試局的陣列、本人顏色與題號
    print(f"written: {len(rows)} training games ({int((HD >= 2).sum())} with HD >= 2, median length {int(np.median(LEN))}), {len(recs)} test games ({time.time() - t0:.0f} s)")  # 印出摘要


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
