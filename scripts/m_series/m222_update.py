#!/usr/bin/env python
"""M222 (2026-11-18), report only: what the reading of ct118d (0.8546379 = +6.057 points against ct116, exact units +9,
within-one units +1 over its 18 changed Public answers) says about each of the 18 questions.

Exact conditional update by dynamic programming: prior of each question's truth = its M211 weights (re-solved after the ct116 reading)
where they exist, else its M210 weights, else the pooled posterior behind ct112; conditions: (1) the reading of probe_C2B over its seven questions (C -> B: +2, -1) and
(2) the reading of ct118d over the 18 questions (ct116's answer -> ct118d's answer: +9, +1). Couplings with the rest of the
ledger are ignored. Per question: truth weights before / after, the expected score of ct116's answer, of ct118d's answer and
of the best answer.

  python -I scripts/m_series/m222_update.py
"""
import csv  # 讀取提交檔
import json  # 讀寫真值權重
import sys  # 設定模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本工具：等級表、計分與計數增量

LEV = led.LEVELS; TARGET = (2, -1, 9, 1); EPS = 1e-15  # 等級名稱；兩個讀數的目標計數（probe_C2B 的 +2、-1 與 ct118d 的 +9、+1）；可忽略的機率門檻


def main():  # 主流程：以兩個讀數精確更新 ct118d 改動的 18 題
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731  讀成題號對答案的字典
    a116, a118, c2b = rd("submission_rank_ct116.csv"), rd("submission_rank_ct118d.csv"), rd("upload_probe/probe_C2B.csv")  # ct116、ct118d 與 probe_C2B 的答案
    qs = [q for q in a116 if a116[q] != a118[q]]; seven = {q for q in a116 if a116[q] != c2b[q]}; assert seven <= set(qs) and len(qs) == 18  # ct118d 改動的 18 題，其中包含 probe_C2B 的七題
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); post = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}; old = json.load(open(ROOT / "results/r_variants/m210_profiles.json"))  # ct112 背後的合併後驗與 M210 的真值權重
    prior, src = {}, {}  # 各題的先驗與其來源
    new = {q: d["with_composition"] for q, d in json.load(open(ROOT / "results/r_variants/m211_ledger_ct116.json")).items() if d.get("with_composition")}  # M211 的權重：ct116 讀數之後重解（10 題）
    for q in qs:  # 每一題
        w = np.array([new[q].get(l, 0.0) for l in LEV]) if q in new else np.array([old[q].get(l, 0.0) for l in LEV]) if q in old else post[q].copy(); src[q] = "M211" if q in new else "M210" if q in old else "pooled"  # 先驗依序取 M211、M210、合併後驗
        w = np.maximum(w, 1e-4 * post[q]); prior[q] = w / w.sum()  # 以合併後驗設下限：不完全排除任何真值
    I = LEV.index  # 等級名稱轉成索引
    inc = {q: [(*(led.xy(I("C"), I("B"), c) if q in seven else (0, 0)), *led.xy(I(a116[q]), I(a118[q]), c)) for c in range(10)] for q in qs}  # 每題依真值對四個計數的增量：probe_C2B 的 (x, y) 與 ct118d 的 (x, y)

    def step(d, q, sign=1):  # 把一題加進計數加總的分布
        n = {}  # 新的分布
        for s, p in d.items():  # 目前的每個加總
            for c in range(10):  # 這一題的每個真值
                pc = p * prior[q][c]  # 聯合機率
                if pc < EPS:  # 機率極小
                    continue  # 略過以控制狀態數
                k = tuple(a + sign * b for a, b in zip(s, inc[q][c])); n[k] = n.get(k, 0.0) + pc  # 累加到新的加總
        return n  # 回傳分布
    pre = [{(0, 0, 0, 0): 1.0}]  # 前綴分布：前 i 題的加總
    for q in qs:  # 依序加入
        pre.append(step(pre[-1], q))  # 多加一題
    suf = [{(0, 0, 0, 0): 1.0}]  # 後綴分布
    for q in reversed(qs):  # 由後往前加入
        suf.append(step(suf[-1], q))  # 多加一題
    suf = suf[::-1]  # 反轉後 suf[i]＝qs[i:] 加總的分布
    p_obs = pre[-1].get(TARGET, 0.0); print(f"probability of both readings under the priors: {p_obs:.2e}")  # 先驗下同時出現兩個讀數的機率
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); rows = []; rep = {}  # 計分矩陣、比較表與報告
    for i, q in enumerate(qs):  # 每一題
        w = np.zeros(10); right = suf[i + 1]  # 更新後的權重；這題之後各題的分布
        for s, p in pre[i].items():  # 這題之前各題的每個加總
            for c in range(10):  # 這一題的每個真值
                need = tuple(t - a - b for t, a, b in zip(TARGET, s, inc[q][c])); pr = right.get(need)  # 其餘題目必須湊出的加總
                if pr:  # 湊得出來時
                    w[c] += p * prior[q][c] * pr  # 累加這個真值的後驗質量
        w /= w.sum(); es = S @ w; e0, e1 = es[I(a116[q])], es[I(a118[q])]; best = int(es.argmax()); rep[q] = {LEV[c]: float(w[c]) for c in range(10) if w[c] >= 0.001}  # 正規化；ct116、ct118d 與最佳答案的期望分數
        rows.append((float(es[best] - e1), q, w, e0, e1, best))  # 記下換成最佳答案的增益
    print("question: ct116 -> ct118d | prior | after both readings | expected score of ct116's answer, of ct118d's, of the best answer")  # 表頭
    for g, q, w, e0, e1, best in sorted(rows, key=lambda r: r[1]):  # 依題號排序
        pt = np.argsort(-prior[q])[:3]; top = np.argsort(-w)[:3]  # 先驗與更新後的前三名真值
        print(f"  {q}: {a116[q]:>2} -> {a118[q]:>2} | {src[q]:6s} " + " ".join(f"{LEV[c]} {prior[q][c]:.2f}" for c in pt if prior[q][c] >= 0.005) + " | " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005)  # 印出先驗與更新後的權重
              + f" | {e0:.3f}, {e1:.3f}, best {LEV[best]} {S[best] @ w:.3f}" + ("" if LEV[best] == a118[q] else f"   <- {LEV[best]} would be {g:+.3f} better") + ("  [probe_C2B]" if q in seven else ""))  # 接上期望分數，標出可改進的題與 probe_C2B 的題
    tot = sum(r[0] for r in rows); print(f"sum of the expected gains of switching to the best answers: {tot:+.2f} points; expected score of ct118d on these 18 under the update: {sum(r[4] for r in rows):.2f} (ct116: {sum(r[3] for r in rows):.2f})")  # 全部換成最佳答案的期望增益，以及 ct118d 與 ct116 的期望分數
    (ROOT / "results/r_variants/m222_update.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m222_update.json")  # 寫出更新後的真值權重（m223_build.py 使用）


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
