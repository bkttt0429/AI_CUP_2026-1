"""Section 4M M8: the games at positions 21-40 of every player's seeded permutation (scoring.fold_players with max_games 40 minus the
20-game selection every score directory holds), folds 0-4, as a --game-keys CSV; plus the per-player game-count distribution.

  python -I scripts/m_series/x40_keys.py --out results/data/x40_keys.csv --report results/data/x40_keys.json
"""
import argparse  # 解析命令列參數
import json  # 寫出報告 JSON
import sys  # 設定匯入路徑
from pathlib import Path  # 路徑處理

import numpy as np  # 數值套件（本程式沒有直接用到）
import pandas as pd  # 讀取分折表、寫出鍵清單

ROOT = Path.cwd()  # 專案根目錄（須在根目錄執行）
sys.path[:0] = [str(ROOT / "src")]  # 加入 src 以匯入 aicup 套件
from aicup.ladder import FOLDS_CSV  # noqa: E402  玩家分折表的路徑
from aicup.scoring import fold_players, player_games  # noqa: E402  與評分程式相同的對局清單與抽局排列


def main():  # 產生第 21–40 局的鍵清單與報告
    ap = argparse.ArgumentParser()  # 命令列參數
    ap.add_argument("--out", required=True)  # 鍵清單 CSV 的輸出路徑
    ap.add_argument("--report", required=True)  # 報告 JSON 的輸出路徑
    ap.add_argument("--max-games", type=int, default=40)  # 每位玩家取到排列的第幾局（預設 40）
    ap.add_argument("--base-games", type=int, default=20, help="the games already scored (positions 1..base of the permutation); the extra keys are base+1..max")  # 已評分的局數（預設為排列的前 20 局）
    a = ap.parse_args()  # 解析參數
    games, folds = player_games(), pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 全部訓練對局與玩家分折表
    counts = games.groupby("player_id").size()  # 每位玩家的局數
    rep = {"players": int(len(counts)), "games": int(len(games)), "games_per_player_quantiles": {str(q): float(counts.quantile(q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},  # 報告：玩家數、局數、每人局數的分位數
           "players_with_le_base_games": int((counts <= a.base_games).sum()), "players_with_ge_max_games": int((counts >= a.max_games).sum()), "base_games": a.base_games, "max_games": a.max_games, "folds": {}}  # 局數不超過 20／至少 40 的玩家數與設定
    rows = []  # 鍵清單的各列
    for f in (0, 1, 2, 3, 4):  # 折 0–4
        base = fold_players(games, folds, f, max_games=a.base_games)  # 已評分的 20 局抽樣
        wide = fold_players(games, folds, f, max_games=a.max_games)  # 同一個排列取到 40 局
        b = {r.player_id: list(r.game_keys) for r in base.itertuples()}  # 玩家→已評分的局鍵
        n_new, n_players = 0, 0  # 新增的局數與有新增的玩家數
        for r in wide.itertuples():  # 逐一玩家
            old = b.get(r.player_id, [])  # 該玩家已評分的局
            assert list(r.game_keys)[:len(old)] == old, r.player_id  # 同一個排列：前 20 局就是已評分的抽樣
            extra = list(r.game_keys)[len(old):]  # 第 21 局之後的新增局
            if extra:  # 有新增局
                n_players += 1  # 計入有新增的玩家
            for k in extra:  # 逐一新增局
                rows.append({"fold": f, "player_id": r.player_id, "rank": r.rank, "game_key": k})  # 記下折、玩家、等級與局鍵
            n_new += len(extra)  # 累計新增局數
        rep["folds"][str(f)] = {"players": int(len(wide)), "players_with_extra": n_players, "extra_games": n_new}  # 該折的玩家數與新增量
    pd.DataFrame(rows).to_csv(a.out, index=False)  # 寫出給 --game-keys 用的 CSV
    rep["extra_games_total"] = len(rows)  # 新增局的總數
    Path(a.report).write_text(json.dumps(rep, indent=1))  # 寫出報告
    print(json.dumps(rep, indent=1))  # 同時印出報告
    print("wrote", a.out)  # 回報輸出路徑


if __name__ == "__main__":  # 直接執行時
    main()  # 產生鍵清單
