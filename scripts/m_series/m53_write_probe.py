"""Section 4M M53 probe files (2026-11-07): one bias-only push i -> j with odds ratio r on top of a base submission, for a Public probe.
The push is computed on ct9's test posteriors (submission_rank_ct9_proba.csv): on a question whose ct9 decision is i or j, m = P(i) + P(j) is
kept and P'(j) = m r P(j) / (r P(j) + P(i)); a question changes only if the BASE file says i and the adjusted expected-score decision says j
(so probes compose: each file = the previous accepted file + exactly one push, and the Public score difference reads that push alone).
  python -I scripts/m_series/m53_write_probe.py --base submission_rank_ct13.csv --push B A --r 2 --out submission_rank_ct14.csv
"""
import argparse  # 解析命令列參數
import hashlib  # 計算輸出檔的 md5
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀寫提交檔與機率檔

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序
from aicup.submission import validate_submission, write_submission  # noqa: E402  寫出並檢查提交檔
from m43_gate import decide  # noqa: E402  期望分決策

RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引


def main():  # 命令列進入點
    ap = argparse.ArgumentParser()  # 建立參數解析器
    ap.add_argument("--base", required=True); ap.add_argument("--push", nargs=2, required=True, metavar=("FROM", "TO")); ap.add_argument("--r", type=float, required=True)  # 底本檔、推移方向與勝算比
    ap.add_argument("--out", required=True); ap.add_argument("--proba", default="submission_rank_ct9_proba.csv")  # 輸出檔與計算推移用的機率檔
    a = ap.parse_args()  # 讀入參數
    i, j = RANK[a.push[0]], RANK[a.push[1]]  # 推移的來源與目標等級
    assert abs(i - j) == 1, "the push must be between adjacent classes"  # 只允許相鄰等級之間的推移
    pr = pd.read_csv(ROOT / a.proba); cols = [f"p_{lv}" for lv in LEVELS]  # 讀入測試題的十級機率
    P = pr[cols].to_numpy(float); qid = pr["question_id"].to_numpy()  # 機率矩陣與題號
    base = pd.read_csv(ROOT / a.base, dtype=str); order = base["question_id"]  # 底本答案與其題目順序
    d_base = base.set_index("question_id")["pred_rank"].reindex(qid).map(RANK).to_numpy()  # 底本答案對齊機率檔的題目順序
    d9 = decide(P)  # ct9 機率的期望分決策
    Q = P.copy(); m = P[:, i] + P[:, j]; s = a.r * P[:, j] / (a.r * P[:, j] + P[:, i] + 1e-12)  # 兩級合計 m，與乘上勝算比後 j 的占比
    route = (d9 == i) | (d9 == j); Q[route, j] = m[route] * s[route]; Q[route, i] = m[route] * (1 - s[route])  # 只調整 ct9 答 i 或 j 的題
    d_adj = decide(Q)  # 調整後的期望分決策
    mv = (d_base == i) & (d_adj == j)  # 底本答 i 且調整後答 j 才翻動
    d_new = d_base.copy(); d_new[mv] = j  # 在底本上套用這一個推移
    pub = pd.Series(qid).str.startswith("pub").to_numpy()  # 標出 Public 題
    rows = pd.DataFrame({"question_id": qid, "pred_rank": [LEVELS[k] for k in d_new]}).set_index("question_id").loc[order].reset_index()  # 依底本的題目順序排列
    out = ROOT / a.out  # 輸出檔路徑
    write_submission(list(rows.itertuples(index=False, name=None)), out)  # 寫出提交檔
    problems = validate_submission(out, order.tolist(), 800, {"pub_": 400, "priv_": 400})  # 檢查表頭、題號與順序，並核對共 800 題、Public／Private 各 400
    print(f"push {a.push[0]}->{a.push[1]} r={a.r:g} on {a.base}: {int(mv.sum())} flips (public {int(mv[pub].sum())}, private {int(mv[~pub].sum())}); "  # 印出翻動題數
          f"base questions decided {a.push[0]}: {int((d_base == i).sum())}; validation problems: {problems or 'none'}")  # 再印底本答 i 的題數與檢查結果
    print(f"flipped public questions: {', '.join(qid[mv & pub])}")  # 列出翻動的 Public 題
    print(f"flipped private questions: {', '.join(qid[mv & ~pub])}")  # 列出翻動的 Private 題
    for name in ("submission_rank_ct9.csv", "submission_rank_ct13.csv"):  # 與兩個參考檔比較
        other = pd.read_csv(ROOT / name, dtype=str).set_index("question_id")["pred_rank"].reindex(qid).to_numpy(); mine = np.array([LEVELS[k] for k in d_new])  # 參考檔與新檔的答案
        ch = mine != other  # 答案不同的題
        print(f"{out.name} vs {name}: {int(ch.sum())} differ (public {int(ch[pub].sum())}, private {int(ch[~pub].sum())})")  # 印出差異題數
    print("predicted counts: " + " ".join(f"{lv} {int((d_new == k).sum())}" for k, lv in enumerate(LEVELS)))  # 各等級的預測題數
    print("md5 " + hashlib.md5(out.read_bytes()).hexdigest()[:12] + " " + out.name)  # 印出 md5 前 12 碼供上傳核對


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
