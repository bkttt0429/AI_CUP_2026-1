#!/usr/bin/env python
"""M210 (2026-11-17), Public only: truth weights of every Public question that some upload has constrained, under the ledger
+ the two probes of M206 + the exact Public composition (the joint problem of m208_ledger_probe.py), in parallel.

For each such question and each candidate truth: the cost of the cheapest joint assignment with that truth fixed; weights =
exp(-(cost - min)), normalised. Output: results/r_variants/m210_profiles.json {question: {level: weight}} and a table of the
questions where an answer other than the reference file's has a larger expected score.

  python -I scripts/m_series/m210_profile_all.py [--ref submission_rank_ct116.csv] [--workers 44]
"""
import json  # 讀寫帳本與結果
import math  # 計算指數
import multiprocessing as mp  # 平行求解
import sys  # 命令列參數與模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本工具
import m208_ledger_probe as m8  # noqa: E402  M208 的聯立求解與兩個探測檔的讀數

LEV = led.LEVELS; ST = {}  # 等級名稱；分叉前放好、子程序共用的問題資料


def one(task):  # 子程序：固定一題的一個真值後求解
    q, c = task; a_, cost = m8.solve(ST["pub"], ST["cand"], ST["obs"], ST["prior"], ST["counts"], fix=(q, c), time_limit=150)  # 固定（題目, 真值）求最便宜的聯立解
    return q, c, (cost if isinstance(a_, dict) else None), (a_ == "timeout")  # 回傳成本（無解為 None）與是否逾時


def main():  # 主流程：平行剖析每一個受約束的 Public 題
    opt = dict(zip(sys.argv[1:], sys.argv[2:])); ref_file = opt.get("--ref", "submission_rank_ct116.csv"); workers = int(opt.get("--workers", "44"))  # 參數：參考檔與平行數
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}  # 先驗＝ct112 背後的合併後驗
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **m8.PROBES}; qs, cand, obs = led.build(up, prior); comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}  # 帳本加上兩個探測檔；Public 的等級組成
    ref = led.decisions(ref_file); pub = [q for q in ref.index if q.startswith("pub")]; cand2 = {q: (cand[q] if q in cand else list(range(10))) for q in pub}; ST.update(pub=pub, cand=cand2, obs=obs, prior=prior, counts=counts)  # 參考檔答案、全部 Public 題與候選真值，放進共用資料
    tasks = [(q, c) for q in qs for c in cand2[q]]; print(f"{len(qs)} constrained Public questions, {len(tasks)} solves, {workers} workers", flush=True)  # 每個（題目, 候選真值）求解一次
    with mp.get_context("fork").Pool(workers) as pool:  # 以 fork 建立工作程序池（共用 ST）
        res = pool.map(one, tasks, chunksize=1)  # 平行求解
    cost = {}; late = 0  # 各題各真值的成本與逾時次數
    for q, c, v, to in res:  # 每個結果
        late += bool(to)  # 計入逾時
        if v is not None:  # 有解時
            cost.setdefault(q, {})[c] = v  # 記下成本
    prof = {}  # 各題的真值權重
    for q, d in cost.items():  # 每一題
        m = min(d.values()); w = {c: math.exp(-(v - m)) for c, v in d.items()}; t = sum(w.values()); prof[q] = {c: v / t for c, v in w.items()}  # 權重＝exp(-(成本 - 最小值))，再正規化
    (ROOT / "results/r_variants/m210_profiles.json").write_text(json.dumps({q: {LEV[c]: v for c, v in w.items()} for q, w in prof.items()}, indent=1)); print(f"profiles of {len(prof)} questions written to results/r_variants/m210_profiles.json; solves that hit the time limit: {late}", flush=True)  # 寫出各題的真值權重
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); rows = []  # 計分矩陣與比較表
    for q, w in prof.items():  # 每一題
        wv = np.zeros(10)  # 十個等級的權重向量
        for c, v in w.items():  # 每個有權重的真值
            wv[c] = v  # 填入權重
        es = S @ wv; cur = int(ref[q]); best = int(es.argmax()); rows.append((float(es[best] - es[cur]), q, cur, best, float(wv[best]), float(wv[cur])))  # 各答案的期望分數；目前答案與期望分數最高的答案
    rows.sort(reverse=True); print(f"questions where another answer has a larger expected score than {ref_file} (gain, question, answer -> better answer, weight of the better answer as the truth, weight of the current answer):")  # 依期望增益排序並印出表頭
    tot = 0.0  # 列出項目的期望增益合計
    for g, q, cur, best, wb, wc in rows:  # 每一題
        if best != cur and g > 0.02:  # 換答案的期望增益超過 0.02 才列出
            tot += g; print(f"  {g:+.3f}  {q}: {LEV[cur]} -> {LEV[best]}   truth {LEV[best]} {wb:.2f}, {LEV[cur]} {wc:.2f}; pooled posterior " + " ".join(f"{LEV[c]} {prior[q][c]:.2f}" for c in np.argsort(-prior[q])[:3]))  # 印出增益、答案變化、真值權重與合併後驗前三名
    print(f"sum of the expected gains listed: {tot:+.2f} points")  # 印出合計


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
