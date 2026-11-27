#!/usr/bin/env python
"""Write the HD table of the training set (issue #42): game_key, player_id, rank, color, hd for every game of data/training
(dataset v1.1; hd empty for a v1.0 record). Read once by scripts/hd_check.py (the HD features) and, filtered to hd >= 2, by the
--game-keys re-scoring of scripts/score_folds.py / scripts/l1b_score.py. CPU, a few minutes.

  python -I scripts/hd_table.py --out results/data/hd_table.csv --keys-out results/data/hd2_keys.csv
"""
import argparse  # 解析命令列參數
import sys  # 設定匯入路徑與結束碼
import time  # 計算耗時
from pathlib import Path  # 處理輸出路徑

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（scripts/ 的上一層）
if str(ROOT / "src") not in sys.path:  # src/ 還不在匯入路徑時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src/ 以匯入 aicup 套件
from aicup.handicap import hd_summary, hd_table  # noqa: E402  HD 表的建立與統計
from aicup.scoring import TRAIN_DIR  # noqa: E402  預設的訓練資料資料夾


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以說明的第一行當描述
    ap.add_argument("--train-dir", default=str(TRAIN_DIR))  # 訓練集 CSV 所在資料夾
    ap.add_argument("--out", required=True, help="the HD table (CSV)")  # HD 表的輸出路徑
    ap.add_argument("--keys-out", default=None, help="also write the game keys with HD >= --min-hd (a CSV with game_key, color, hd) for --game-keys")  # 另寫重新評分用的局清單
    ap.add_argument("--min-hd", type=int, default=2)  # 列入局清單的最小 HD 值
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 開始計時
    try:  # 讀不到訓練資料時改為回報錯誤
        t = hd_table(a.train_dir)  # 從每局 SGF 讀出 HD 值
    except FileNotFoundError as e:  # 資料夾裡沒有 train_*.csv
        print(e, file=sys.stderr)  # 印出錯誤訊息
        return 2  # 結束碼 2 表示缺少輸入
    s = hd_summary(t)  # 各 HD 值的局數，以及 HD>=1、HD>=2 的比例
    print(f"{s['games']} games; HD {s['hd']}; share HD>=1 {s['share_hd1']:.4f}, HD>=2 {s['share_hd2']:.4f} ({time.time() - t0:.0f} s)")  # 印出覆蓋率摘要
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    t.to_csv(a.out, index=False)  # 寫出 HD 表
    print(f"wrote {a.out}")  # 回報輸出位置
    if a.keys_out:  # 需要局清單時
        sel = t[t["hd"] >= a.min_hd][["game_key", "color", "hd"]]  # 選出 HD >= --min-hd 的局
        Path(a.keys_out).parent.mkdir(parents=True, exist_ok=True)  # 建立清單的資料夾
        sel.to_csv(a.keys_out, index=False)  # 寫出局清單
        print(f"wrote {a.keys_out}: {len(sel)} games with HD >= {a.min_hd}")  # 回報清單局數
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值當結束碼
