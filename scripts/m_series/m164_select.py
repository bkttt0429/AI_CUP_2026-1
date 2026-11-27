#!/usr/bin/env python
"""Section 4M M164 (2026-11-15): the label-free choice among identifier compositions (design doc M161q, registered rule).

For the tags given (in order of simplicity, simplest first; each built by m164_ident.py): the share of TEST opponents recognised
with a clear margin (largest excess of the opponent side's m15 over the target side's, from m84_<tag>_eval.json). Chosen = the
first tag in the given order whose value is within --tol (0.003) of the best one. No level label is involved.
The table goes to stderr and results/r_variants/m164_selected.json; stdout carries the chosen tag only.

  best=$(python -I scripts/m_series/m164_select.py SQ1 SQ2 SQ3)
"""
import argparse  # 解析命令列參數
import json  # 讀寫評估報告
import sys  # 標準錯誤輸出與中止程式
from pathlib import Path  # 處理檔案路徑

RV = Path.cwd() / "results/r_variants"  # 識別器評估報告所在的資料夾


def main():  # 依登錄的規則選出一個識別器
    ap = argparse.ArgumentParser(); ap.add_argument("tags", nargs="+"); ap.add_argument("--tol", type=float, default=0.003); a_ = ap.parse_args(); rows = []  # 參數：由簡到繁的識別器名稱與容許差
    for t in a_.tags:  # 每個識別器
        f = RV / f"m84_{t}_eval.json"  # m164_ident.py 寫出的評估報告
        if not f.exists():  # 沒有報告
            print(f"{t}: no evaluation file, skipped", file=sys.stderr); continue  # 略過這個識別器
        d = json.loads(f.read_text()); rows.append((t, d["test opponent side"]["largest_excess"], d["validation opponent side"]["largest_excess"], d["heldout"]["fused with the prior"]["top1_rows"], d["beta"], d["sources"]))  # 取出測試端與驗證端的最大超額、留出局 top-1、beta 與來源
    if not rows:  # 沒有可選的識別器
        sys.exit("no identifier to choose from")  # 中止
    best = max(r[1] for r in rows); chosen = next(r for r in rows if r[1] >= best - a_.tol)  # 測試端最大超額最高者；容許差內取最簡單的
    for t, te, va, h, beta, src in rows:  # 每個識別器印一列
        print(f"{t:6s} test-side largest excess {te:.4f} | validation side {va:.4f} | held-out top-1 by rows (with the prior) {h:.3f} | beta {beta:.2f} | block + {'+'.join(src)}" + ("   <- chosen" if t == chosen[0] else ""), file=sys.stderr)  # 表格印到標準錯誤並標出選中者
    (RV / "m164_selected.json").write_text(json.dumps({"chosen": chosen[0], "tol": a_.tol, "rows": [{"tag": r[0], "test": r[1], "validation": r[2], "heldout_rows": r[3], "beta": r[4], "sources": r[5]} for r in rows]}, indent=1))  # 寫出選擇紀錄
    print(chosen[0])  # 標準輸出只印選中的名稱，供 shell 接收


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
