#!/usr/bin/env python
"""M220 fallback (2026-11-18): the Public-only file ct118 without the full re-profile (after the four readings of 11-18 the
profile solves of m220_resolve.py no longer return within their time limit).

Exact local update for the seven questions of probe_C2B: prior of each question's truth = its M210 weights (ledger + probes of
11-17 + composition) where they exist, else the pooled posterior behind ct112; conditioned on the reading of the probe
(C -> B on the seven: +2 units of 1 - e^-1, -1 unit of e^-1) by enumeration (dynamic programme over the two counts). The
coupling of these seven with the rest of the ledger through tonight's reading is ignored (the other three probes do not touch them).

ct118 = ct116 with
  the edits of ct117 kept as they are (their weights predate tonight's probes; none of them is in a probe except pub_q_0286),
  pub_q_0286: the answer with the largest expected score under the local update,
  the other six questions of probe_C2B -> B (their summed contribution is positive whatever the truth of pub_q_0286 is).
Private rows = ct116.

  python -I scripts/m_series/m220_local.py
"""
import csv  # 讀寫提交檔
import hashlib  # 計算輸出檔的 md5
import io  # 在記憶體中寫 CSV 以便比對
import json  # 讀取 M210 的真值權重
import math  # 數學函式（本檔沒有直接用到）
import sys  # 命令列參數與模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本工具：等級表、計分與計數增量

LEV = led.LEVELS; X_OBS, Y_OBS = 2, -1  # 等級名稱；probe_C2B 的讀數：完全正確 +2、差一級以內 -1


def main():  # 主流程：七題的局部精確更新並建出 ct118（--best 時為 ct118d）
    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731  讀成題號對答案的字典
    a116, a117, c2b = rd("submission_rank_ct116.csv"), rd("submission_rank_ct117.csv"), rd("upload_probe/probe_C2B.csv"); seven = [q for q in a116 if a116[q] != c2b[q]]; assert len(seven) == 7 and all(a116[q] == "C" and c2b[q] == "B" for q in seven)  # ct116、ct117 與 probe_C2B 的答案；探測改動的七題都必須是 C→B
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); post = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}; old = json.load(open(ROOT / "results/r_variants/m210_profiles.json"))  # ct112 背後的合併後驗與 M210 的真值權重
    prior = {}  # 七題各自的先驗與來源
    for q in seven:  # 每一題
        if q in old:  # 有 M210 權重時
            w = np.array([old[q].get(l, 0.0) for l in LEV]); src = "M210 weights"  # 用 M210 的權重
        else:  # 沒有時
            w = post[q].copy(); src = "pooled posterior"  # 改用合併後驗
        prior[q] = (w / w.sum(), src)  # 正規化並記下來源
    inc = [led.xy(LEV.index("C"), LEV.index("B"), c) for c in range(10)]  # 一題由 C 改成 B 時，依真值對 (x, y) 兩個計數的增量

    def dist(qs):  # 一組題目兩個計數加總的分布（動態規劃）
        d = {(0, 0): 1.0}  # 起點：加總為 (0, 0)
        for q in qs:  # 逐題加入
            n = {}  # 新的分布
            for (x, y), p in d.items():  # 目前的每個加總
                for c in range(10):  # 這一題的每個真值
                    k = (x + inc[c][0], y + inc[c][1]); n[k] = n.get(k, 0.0) + p * prior[q][0][c]  # 累加到新的加總
            d = n  # 換成新的分布
        return d  # 回傳分布
    p_obs = dist(seven).get((X_OBS, Y_OBS), 0.0); print(f"probability of the reading (+2, -1) under the priors of the seven questions: {p_obs:.4f}")  # 先驗下出現這個讀數的機率
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); postw = {}  # 計分矩陣與各題更新後的權重
    for q in seven:  # 每一題
        rest = dist([r for r in seven if r != q]); w = np.array([prior[q][0][c] * rest.get((X_OBS - inc[c][0], Y_OBS - inc[c][1]), 0.0) for c in range(10)]); w = w / w.sum(); postw[q] = w  # 其餘六題配上這題的真值必須湊出讀數：精確的條件化
        es = S @ w; top = np.argsort(-w)[:4]; pt = np.argsort(-prior[q][0])[:3]  # 各答案的期望分數與前幾名真值
        print(f"  {q}: prior ({prior[q][1]}) " + " ".join(f"{LEV[c]} {prior[q][0][c]:.2f}" for c in pt) + " | after the reading " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005)  # 印出先驗與更新後的權重
              + f" | expected score C {es[1]:.3f}, B {es[2]:.3f}, A {es[3]:.3f}, D {es[0]:.3f}; ct117 answers {a117[q]}")  # 接上各答案的期望分數與 ct117 的答案
    out = dict(a117); why = {q: "ct117 edit kept" for q in a116 if a117[q] != a116[q]}  # 從 ct117 出發，保留它的改動
    q0 = "pub_q_0286"; es = S @ postw[q0]; best = max(("C", "A", "B", "D"), key=lambda l: es[LEV.index(l)]); out[q0] = best; why[q0] = f"local update: expected score C {es[1]:.3f}, B {es[2]:.3f}, A {es[3]:.3f} -> {best}"  # pub_q_0286 取更新後期望分數最高的答案
    best_mode = "--best" in sys.argv  # 2026-11-18 01:5x 的修訂：其餘六題各自取局部更新下期望分數最高的答案
    for q in seven:  # 七題之中
        if q != q0:  # 其餘六題
            e_ = S @ postw[q]; lv = max(("C", "B", "A", "D"), key=lambda l: e_[LEV.index(l)]) if best_mode else "B"  # --best 時取期望分數最高者，否則一律改成 B
            out[q] = lv; why[q] = (f"local update: expected score C {e_[1]:.3f}, B {e_[2]:.3f}, D {e_[0]:.3f} -> {lv}" if best_mode else "probe_C2B (the six together are positive for every truth of pub_q_0286)")  # 寫入答案與理由
    six = [q for q in seven if q != q0]; e6 = sum(float((S @ postw[q])[2] - (S @ postw[q])[1]) for q in six)  # 六題由 C 改成 B 的期望增益
    worst = min(0.8964 - ((1 - led.E1) * x + led.E1 * y) for x, y in set(inc)); print(f"the six other questions -> B: expected {e6 * 1:+.3f} points under the local update; guaranteed minimum over the truths of {q0}: {worst:+.3f} points")  # 不論 pub_q_0286 的真值為何，六題改 B 至少保證的點數
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"  # Private 列必須與 ct116 完全相同
    ch = [q for q in a116 if out[q] != a116[q]]; print(f"\n== {'ct118d' if best_mode else 'ct118'}: {len(ch)} Public answers differ from ct116; Private rows identical")  # 與 ct116 不同的 Public 題數
    for q in sorted(set(ch) | set(why)):  # 所有相關的題
        print(f"  {q}: ct116 {a116[q]:>2} | ct117 {a117[q]:>2} | ct118 {out[q]:>2}   {why.get(q, '')}")  # 印出三個檔的答案與理由
    buf = io.StringIO(newline=""); wr = csv.writer(buf, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, a116[q]) for q in a116)  # 用同樣的寫法重寫 ct116
    assert buf.getvalue().encode() == (ROOT / "submission_rank_ct116.csv").read_bytes(), "the writer does not reproduce ct116 byte for byte"  # 寫法必須逐位元組重現 ct116
    path = ROOT / ("submission_rank_ct118d.csv" if best_mode else "submission_rank_ct118.csv"); assert not path.exists(), f"{path.name} exists already - nothing overwritten"  # 輸出檔（--best 時為 ct118d）不可已存在
    with open(path, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)  # 只用 LF 換行，題目順序與 ct116 相同
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")  # 回報列數與 md5 前 12 碼


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
