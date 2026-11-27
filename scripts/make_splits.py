#!/usr/bin/env python
"""Player-grouped 5-fold split and 5/10/20-game pseudo-questions (METHOD_NOTES §2.1).

Within each rank, players are sorted by game count (ties shuffled with the seed) and dealt
round-robin into folds so every fold gets a similar share of heavy players. For a held-out
player, games are shuffled once and carved into questions round-robin over the sizes
5 → 10 → 20, at most `max_q` per size, never reusing a game.
"""
import argparse  # 解析命令列參數
import glob  # 找出各等級的訓練 CSV
import sys  # 設定匯入路徑
from pathlib import Path  # 處理路徑

import numpy as np  # 亂數與排列
import pandas as pd  # 讀寫表格

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.metric import LEVELS  # noqa: E402  10 個等級的固定順序

SEED = 42  # 分折與出題共用的種子
SIZES = (5, 10, 20)  # 假題的局數，依序輪流切


def assign_folds(players, n_folds=5, seed=SEED):  # 把玩家分到各折
    """players: DataFrame[rank, player_id, n_games] -> same + fold (0..n_folds-1)."""
    rng = np.random.RandomState(seed)  # 固定種子，分折可重現
    out = players.copy()  # 不改動呼叫端的表
    out["_tie"] = rng.rand(len(out))  # 局數相同時用來打亂的隨機鍵
    out = out.sort_values(["rank", "n_games", "_tie"], ascending=[True, False, True])  # 每級內依局數由多到少排序
    out["fold"] = out.groupby("rank").cumcount() % n_folds  # 每級內輪流發到各折
    return out.drop(columns="_tie").sort_values(["rank", "fold", "player_id"]).reset_index(drop=True)  # 去掉輔助欄並整理順序


def make_questions(keys, fold, rank, player_id, sizes=SIZES, max_q=2):  # 把一位玩家的棋局切成題目
    """Carve a player's (already shuffled) game keys into disjoint questions."""
    qs, pos, counts = [], 0, {n: 0 for n in sizes}  # 題目清單、目前位置、各大小已切的題數
    progressed = True  # 上一輪是否還切得出題目
    while progressed:  # 直到每種大小都切不出為止
        progressed = False  # 本輪先假設切不出
        for n in sizes:  # 依 5 → 10 → 20 輪流
            if counts[n] < max_q and pos + n <= len(keys):  # 該大小未滿額且剩餘局數足夠
                qs.append({  # 新增一題
                    "question_id": f"f{fold}_{rank}_{player_id[:8]}_n{n}_{counts[n]}",  # 題號含折、等級、玩家與大小
                    "rank": rank, "player_id": player_id, "n_games": n,  # 標籤與題目大小
                    "game_keys": ";".join(keys[pos:pos + n]),  # 依序取下 n 局，以分號串接
                })  # 結束這一題
                pos += n  # 往後移，棋局不重複使用
                counts[n] += 1  # 記錄該大小的題數
                progressed = True  # 本輪有切出題目
    return qs  # 回傳這位玩家的題目


def main():  # 命令列進入點
    ap = argparse.ArgumentParser()  # 建立參數解析器
    ap.add_argument("--train-dir", default=str(ROOT / "data" / "training"))  # 官方訓練集資料夾
    ap.add_argument("--out-dir", default=str(ROOT / "splits"))  # 切分結果的輸出資料夾
    ap.add_argument("--folds", type=int, default=5)  # 折數
    a = ap.parse_args()  # 解析參數
    out_dir = Path(a.out_dir)  # 輸出資料夾
    out_dir.mkdir(exist_ok=True)  # 確保資料夾存在

    df = pd.concat([pd.read_csv(p, usecols=["player_id", "game_id", "rank"])  # 讀入每個等級檔的玩家與局號
                    for p in sorted(glob.glob(f"{a.train_dir}/train_*.csv"))], ignore_index=True)  # 依檔名順序合併所有等級
    df["game_key"] = df["rank"] + ":" + df["game_id"]  # 局的唯一鍵「等級:局號」
    players = df.groupby(["rank", "player_id"]).size().rename("n_games").reset_index()  # 每位玩家的局數
    folds = assign_folds(players, a.folds)  # 依玩家分折
    folds.to_csv(out_dir / "player_folds.csv", index=False)  # 寫出 player_folds.csv
    assert len(folds) == players["player_id"].nunique() == folds["player_id"].nunique()  # 每位玩家只有一列、只屬一折
    per = folds.groupby(["rank", "fold"]).size().unstack("fold")  # 每級每折的玩家數
    assert ((per.max(axis=1) - per.min(axis=1)) <= 1).all()  # 每級各折玩家數相差不超過 1
    print("players per rank x fold:\n", per.reindex(LEVELS).to_string())  # 印出分折統計

    games_by_player = df.groupby("player_id")["game_key"].apply(lambda s: sorted(s))  # 每位玩家排序後的棋局鍵
    rng = np.random.RandomState(SEED)  # 打亂棋局用的亂數（全程共用一個）
    summary = []  # 各折的題數統計
    for f in range(a.folds):  # 逐折出題
        rows = []  # 這一折的題目
        for r in folds[folds["fold"] == f].itertuples():  # 這一折的每位玩家
            keys = list(rng.permutation(games_by_player[r.player_id]))  # 把這位玩家的棋局打亂一次
            rows.extend(make_questions(keys, f, r.rank, r.player_id))  # 切成題目
        q = pd.DataFrame(rows, columns=["question_id", "rank", "player_id", "n_games", "game_keys"])  # 這一折的題目表
        used = [k for s in q["game_keys"] for k in s.split(";")]  # 所有題目用到的棋局
        assert len(used) == len(set(used)), f"fold {f}: a game appears in two questions"  # 同一局不可出現在兩題
        q.to_csv(out_dir / f"questions_fold{f}.csv", index=False)  # 寫出這一折的假題檔
        tab = q.groupby(["rank", "n_games"]).size().unstack("n_games").reindex(LEVELS)  # 每級每種大小的題數
        tab["fold"] = f  # 標上折號
        summary.append(tab)  # 收集統計
    print("questions per fold x rank x n_games:\n", pd.concat(summary).reset_index().to_string(index=False))  # 印出出題統計


if __name__ == "__main__":  # 直接執行時
    main()  # 執行切分
