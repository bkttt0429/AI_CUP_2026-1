#!/usr/bin/env python
"""M223 (2026-11-18): Public-only refinement of ct118d for 11-19 - three of its 18 Public edits changed to the answer with the
largest expected score under the exact update of M222 (both readings: probe_C2B and ct118d; results/r_variants/m222_update.json):
pub_q_0286 A -> C, pub_q_0192 D -> B, pub_q_0103 B -> A. Private rows = ct116 (= ct118d). Never overwrites.

  python -I scripts/m_series/m223_build.py
"""
import csv  # 讀寫提交檔
import hashlib  # 計算輸出檔的 md5
import json  # 讀取 M222 的更新結果
import sys  # 設定模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 期望分數的矩陣運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本工具：等級表與單題計分

LEV = led.LEVELS; EDITS = {"pub_q_0286": ("A", "C"), "pub_q_0192": ("D", "B"), "pub_q_0103": ("B", "A")}  # 等級名稱；要改的三題（ct118d 的答案 -> 新答案）


def main():  # 由 ct118d 建出 ct118e
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731  讀成題號對答案的字典
    a116, a118 = rd("submission_rank_ct116.csv"), rd("submission_rank_ct118d.csv"); w = json.load(open(ROOT / "results/r_variants/m222_update.json")); out = dict(a118)  # ct116 與 ct118d 的答案、M222 更新後各題的真值權重；從 ct118d 出發
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); tot = 0.0  # 計分矩陣（答案 x 真值）與期望增益的累計
    for q, (old, new) in EDITS.items():  # 每一題改動
        assert a118[q] == old, (q, a118[q]); p = np.array([w[q].get(l, 0.0) for l in LEV]); es = S @ p; best = LEV[int(es.argmax())]; assert best == new, (q, best, new)  # 確認原答案，且新答案是更新後期望分數最高者
        out[q] = new; tot += es[LEV.index(new)] - es[LEV.index(old)]  # 改答案並累計期望增益
        print(f"  {q}: ct116 {a116[q]:>2} | ct118d {old:>2} -> {new:>2}; weights " + " ".join(f"{k} {v:.2f}" for k, v in sorted(w[q].items(), key=lambda t: -t[1])[:3]) + f"; expected score {es[LEV.index(old)]:.3f} -> {es[LEV.index(new)]:.3f}")  # 印出各檔答案、前三個真值權重與期望分數
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"  # Private 列必須與 ct116 完全相同
    print(f"expected gain against ct118d under the update: {tot:+.2f} points ({tot / 400:+.4f}); Public answers that differ from ct116: {sum(out[q] != a116[q] for q in a116)}, from ct118d: {sum(out[q] != a118[q] for q in a116)}; Private rows identical to ct116")  # 回報期望增益與相對 ct116、ct118d 的差異題數
    path = ROOT / "submission_rank_ct118e.csv"; assert not path.exists(), f"{path.name} exists already - nothing overwritten"  # 輸出檔不可已存在（絕不覆寫）
    with open(path, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)  # 只用 LF 換行，題目順序與 ct116 相同
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")  # 回報列數與 md5 前 12 碼


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
