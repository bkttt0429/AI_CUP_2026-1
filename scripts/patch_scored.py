#!/usr/bin/env python
"""Merge a partial re-scoring over a scored directory, fold by fold (issue #42, the handicap board fix).

The HD >= 2 games (3.6%) were scored on boards missing their handicap stones and, per Q13, with the recorded (reversed)
colours. After the fix in aicup.sgf.setup_and_moves they are scored again, only them, with `--game-keys` of
scripts/score_folds.py / scripts/l1b_score.py into a PATCH directory; this script writes OUT = BASE with those games' table
rows (n_moves, sum_*) and per-move arrays replaced (aicup.scoring.patch_fold), every other game and file unchanged, so
every downstream script runs on OUT exactly as on BASE. Each fold's out/fold{F}/patch_meta.json and --summary record what
changed (moves before/after, mean |Δ| and mean Δ of every sum column, the patched keys).

  python -I scripts/patch_scored.py --base results/scores --patch results/scores_hdpatch --out results/scores_hd --folds 0 1 2 3 4 \\
      --summary results/data/patch_scores_hd.json
"""
import argparse  # 解析命令列參數
import sys  # 設定匯入路徑、輸出錯誤與結束碼
import time  # 計算經過的秒數
from pathlib import Path  # 處理檔案路徑

ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（本檔在 scripts/）
if str(ROOT / "src") not in sys.path:  # src 還不在匯入路徑上時
    sys.path.insert(0, str(ROOT / "src"))  # 加入 src，才能匯入 aicup 套件
from aicup.ladder import environment, to_json  # noqa: E402  執行環境紀錄與嚴格 JSON 輸出
from aicup.scoring import patch_fold  # noqa: E402  逐折併回的實作


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明的第一行作為 --help 描述
    ap.add_argument("--base", required=True, help="the scored directory (fold{F} subdirectories)")  # 原本的評分目錄
    ap.add_argument("--patch", required=True, help="the re-scoring of some of its games, same layout")  # 只含部分局重新評分結果的 PATCH 目錄（結構與 BASE 相同）
    ap.add_argument("--out", required=True, help="written: base with the patched games replaced")  # 輸出目錄：BASE 的內容，重評過的局換成 PATCH 的版本
    ap.add_argument("--folds", type=int, nargs="+", required=True)  # 要併回的折
    ap.add_argument("--summary", default=None, help="JSON with every fold's summary")  # 選用：各折摘要的 JSON 檔
    a = ap.parse_args(argv)  # 解析參數
    t0, out = time.time(), {}  # 起始時間與各折摘要
    for f in a.folds:  # 逐折處理
        try:  # 捕捉輸入或輸出目錄的問題
            s = patch_fold(Path(a.base) / f"fold{f}", Path(a.patch) / f"fold{f}", Path(a.out) / f"fold{f}")  # 併回這一折並取得摘要
        except (FileNotFoundError, FileExistsError, ValueError) as e:  # 沒有完成的分段檔、輸出目錄已有分段檔，或 PATCH 與 BASE 對不上（局不在 BASE、玩家／等級／顏色或 sum 欄不同）
            print(f"fold {f}: {e}", file=sys.stderr)  # 印出是哪一折出錯
            return 2  # 以結束碼 2 表示輸入有問題
        out[str(f)] = s  # 記下這一折的摘要
        top = sorted(s["sum_mean_abs_change"].items(), key=lambda kv: -kv[1])[:3]  # 平均絕對變化最大的 3 個 sum 欄
        print(f"fold {f}: {s['patched']} of {s['games']} games patched, target moves {s['n_moves_before']} -> {s['n_moves_after']}; "  # 回報換掉的局數，以及這些局修正前後的本人手數總和
              f"largest mean |change| " + " ".join(f"{c} {v:.3f}" for c, v in top) + f"  ({time.time() - t0:.0f} s)", flush=True)  # 以及變化最大的欄與累計耗時
    if a.summary:  # 有指定摘要檔時
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)  # 建立摘要檔的資料夾
        Path(a.summary).write_text(to_json({"folds": out, "base": a.base, "patch": a.patch, "out": a.out, **environment("cpu")}, indent=1) + "\n")  # 寫入各折摘要、三個目錄與執行環境
        print(f"wrote {a.summary}")  # 回報摘要檔位置
    return 0  # 正常結束


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值作為結束碼
