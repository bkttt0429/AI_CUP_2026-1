#!/usr/bin/env python
"""M220 (2026-11-18), Public only: the joint ledger problem of M210 / M211 with the four boundary probes of 2026-11-18 added,
and the Public-only file ct118 (Private rows = ct116).

Readings (all against ct116 = 0.8394955; decoded with the formula of m206_decode.py, each has one solution):
  probe_2D3D 0.8347546 (dE, dW) = (-3, 0);  probe_3D4D 0.8369955 (-1, -1);  probe_C2B 0.8417364 (+2, -1);  probe_5D4D 0.8363349 (-2, 0)
The problem: every recorded upload + the two probes of 11-17 + the reading of ct116 + these four, with the exact Public
composition (m208_ledger_probe.solve). For each constrained Public question and candidate truth: the cost of the cheapest
joint assignment with that truth fixed -> weights (results/r_variants/m220_profiles.json).

ct118 (rule written in the design doc before this ran):
  (b) the tightened rule of ct117 on the NEW weights: a truth other than ct116's answer with weight >= 0.95 on a question
      changed by >= 2 recorded uploads -> that answer;
  (c) ct117's two special edits (pub_q_0103 -> B, pub_q_0157 -> 4D) stay unless the new weights give them a lower
      expected score than ct116's answer;
  (d) the questions of probe_C2B that (b) does not edit -> B (the reading itself: +2 exact, -1 within one over the seven).

  python -I scripts/m_series/m220_resolve.py [--workers 88]
"""
import csv  # 讀寫提交檔
import hashlib  # 計算輸出檔的 md5
import json  # 讀寫帳本與結果
import math  # 計算指數
import multiprocessing as mp  # 平行求解
import sys  # 命令列參數、模組搜尋路徑與中止程式
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本工具
import m208_ledger_probe as m8  # noqa: E402  M208 的聯立求解

LEV = led.LEVELS; ST = {}; REF_FILE, REF_SCORE = "submission_rank_ct116.csv", 0.8394955  # 等級名稱、子程序共用資料；參考檔 ct116 與其 Public 分數
OLD = {"upload_probe/used_20261117/probe_4D5D_locked.csv": 0.8347546, "upload_probe/used_20261117/probe_A2B.csv": 0.8363349, REF_FILE: REF_SCORE}  # 已有的讀數：11-17 的兩個探測檔與 ct116
NEW = {"upload_probe/probe_2D3D.csv": 0.8347546, "upload_probe/probe_3D4D.csv": 0.8369955, "upload_probe/probe_C2B.csv": 0.8417364, "upload_probe/probe_5D4D.csv": 0.8363349}  # 11-18 四個邊界探測檔的讀數
SPECIAL = {"pub_q_0103": "B", "pub_q_0157": "4D"}  # ct117 的兩題特別改動


def one(task):  # 子程序：固定一題的一個真值後求解
    q, c = task; a_, cost = m8.solve(ST["pub"], ST["cand"], ST["obs"], ST["prior"], ST["counts"], fix=(q, c), time_limit=150)  # 固定（題目, 真值）求最便宜的聯立解
    return q, c, (cost if isinstance(a_, dict) else None), (a_ == "timeout")  # 回傳成本（無解為 None）與是否逾時


def main():  # 主流程：重解帳本並依登錄的規則建出 ct118
    opt = dict(zip(sys.argv[1:], sys.argv[2:])); workers = int(opt.get("--workers", "88"))  # 參數：平行數
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}  # 先驗＝ct112 背後的合併後驗
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER)), **OLD, **NEW}; qs, cand, obs = led.build(up, prior)  # 全部讀數建成聯立問題
    for o in obs[-len(NEW):]:  # 四個新的探測檔
        print(f"  {o['name']}: against ct9 {len(o['moves'])} changed questions, exact {o['X']:+d}, within one {o['Y']:+d}")  # 印出改動題數與解出的兩個計數
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}  # Public 的等級組成
    ref = led.decisions(REF_FILE); pub = [q for q in ref.index if q.startswith("pub")]; cand2 = {q: (cand[q] if q in cand else list(range(10))) for q in pub}; ST.update(pub=pub, cand=cand2, obs=obs, prior=prior, counts=counts)  # ct116 的答案、全部 Public 題與候選真值，放進共用資料
    m0, c0 = m8.solve(pub, cand2, obs, prior, counts); print(f"{len(obs)} uploads, {len(qs)} constrained Public questions; joint problem with the composition: " + (f"cost {c0:.2f}" if isinstance(m0, dict) else f"NOT solved ({m0})"), flush=True)  # 先確認含組成的聯立問題有解
    if not isinstance(m0, dict):  # 無解時
        sys.exit("the joint problem with the new readings has no solution within the candidate sets - nothing written")  # 不寫任何檔就中止
    tasks = [(q, c) for q in qs for c in cand2[q]]; print(f"{len(tasks)} solves, {workers} workers", flush=True)  # 每個（題目, 候選真值）求解一次
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
    (ROOT / "results/r_variants/m220_profiles.json").write_text(json.dumps({q: {LEV[c]: v for c, v in w.items()} for q, w in prof.items()}, indent=1)); print(f"profiles of {len(prof)} questions written to results/r_variants/m220_profiles.json; solves that hit the time limit: {late}", flush=True)  # 寫出各題的真值權重
    S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)]); n_obs = {q: sum(q in o["moves"] for o in obs[:-len(NEW)]) for q in qs}  # 計分矩陣；支持數＝今晚探測之前已記錄、改過該題的上傳數
    old = json.load(open(ROOT / "results/r_variants/m210_profiles.json")) if (ROOT / "results/r_variants/m210_profiles.json").exists() else {}  # M210 的舊權重（有的話，供對照）

    def wv(q):  # 一題的十級真值權重向量
        v = np.zeros(10)  # 預設為 0
        for c, x in prof.get(q, {}).items():  # 有權重的真值
            v[c] = x  # 填入權重
        return v  # 回傳

    def show(q, note=""):  # 產生一題的說明行
        w = wv(q); es = S @ w; cur = int(ref[q]); top = np.argsort(-w)[:3]  # 權重、各答案的期望分數、ct116 的答案與前三個真值
        return f"  {q}: ct116 {LEV[cur]:>2}; weights " + " ".join(f"{LEV[c]} {w[c]:.2f}" for c in top if w[c] >= 0.005) + f"; best answer {LEV[int(es.argmax())]} ({es.max() - es[cur]:+.3f} against ct116's); recorded uploads that changed it {n_obs.get(q, 0)}" + (f"; M210 weights " + " ".join(f"{k} {v:.2f}" for k, v in sorted(old[q].items(), key=lambda t_: -t_[1])[:2]) if q in old else "") + note  # 權重、最佳答案與增益、支持數與 M210 的舊權重

    rd = lambda f: {r["question_id"]: r["pred_rank"] for r in csv.DictReader(open(ROOT / f))}  # noqa: E731  讀成題號對答案的字典
    a116, a117 = rd(REF_FILE), rd("submission_rank_ct117.csv"); e117 = [q for q in a116 if a116[q] != a117[q]]; out = dict(a116); why = {}  # ct116 與 ct117 的答案、ct117 改過的題；從 ct116 出發並記錄改動理由
    print("\n== the twelve edits of ct117 under the new weights")  # 小標：ct117 的十二題改動
    for q in e117:  # ct117 改過的每一題
        w = wv(q); es = S @ w; print(show(q, f"; ct117 answers {a117[q]} ({es[LEV.index(a117[q])] - es[int(ref[q])]:+.3f})"))  # 印出新權重下 ct117 答案的期望增益
    print("\n== the questions of tonight's probes under the new weights")  # 小標：今晚探測的題目
    for f in NEW:  # 每個新的探測檔
        d = rd(f); print(f" {f}:")  # 讀入探測檔的答案並印出檔名
        for q in [q for q in a116 if a116[q] != d[q]]:  # 探測檔改動的題
            print(show(q))  # 印出說明行
    for q in qs:  # 規則 (b)：逐一檢查受約束的 Public 題
        w = wv(q); t = int(w.argmax())  # 權重最高的真值
        if w[t] >= 0.95 and t != int(ref[q]) and n_obs.get(q, 0) >= 2:  # 權重 >= 0.95、不同於 ct116 的答案且至少兩筆上傳改過
            out[q] = LEV[t]; why[q] = f"(b) weight {w[t]:.2f}, {n_obs[q]} uploads"  # 改成該答案並記下理由
    for q, lv in SPECIAL.items():  # 規則 (c)：ct117 的兩題特別改動
        if q not in why:  # (b) 沒有處理過
            es = S @ wv(q)  # 新權重下各答案的期望分數
            if es[LEV.index(lv)] >= es[int(ref[q])]:  # 期望分數不低於 ct116 的答案
                out[q] = lv; why[q] = f"(c) special edit of ct117 kept ({es[LEV.index(lv)] - es[int(ref[q])]:+.3f})"  # 保留特別改動
            else:  # 期望分數較低
                why[q] = f"(c) special edit of ct117 DROPPED ({es[LEV.index(lv)] - es[int(ref[q])]:+.3f})"  # 放棄特別改動，維持 ct116
    c2b = rd("upload_probe/probe_C2B.csv")  # probe_C2B 的答案
    for q in [q for q in a116 if a116[q] != c2b[q]]:  # 規則 (d)：probe_C2B 改動的題
        if q not in why:  # (b) 沒有處理過
            out[q] = "B"; why[q] = "(d) probe_C2B"  # 照探測的讀數改成 B
    assert all(out[q] == a116[q] for q in a116 if not q.startswith("pub")), "a Private row changed"  # Private 列必須與 ct116 完全相同
    ch = [q for q in a116 if out[q] != a116[q]]; print(f"\n== ct118: {len(ch)} Public answers differ from ct116 ({len([q for q in ch if q in e117 and out[q] == a117[q]])} of them as in ct117); Private rows identical")  # 與 ct116 不同的 Public 題數
    for q in sorted(set(ch) | set(why) | set(e117)):  # 所有相關的題
        print(f"  {q}: ct116 {a116[q]:>2} | ct117 {a117[q]:>2} | ct118 {out[q]:>2}   {why.get(q, 'not edited (rule (b) no longer holds)' if q in e117 else '')}")  # 印出三個檔的答案與理由
    es_gain = sum(float((S @ wv(q))[LEV.index(out[q])] - (S @ wv(q))[int(ref[q])]) for q in ch if q in prof); print(f"expected gain of ct118 over ct116 under the new weights: {es_gain:+.2f} points (nominal; attribution to single questions is soft)")  # 新權重下 ct118 對 ct116 的期望增益（名目值）
    import io  # 在記憶體中寫 CSV 以便比對
    buf = io.StringIO(newline=""); wr = csv.writer(buf, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, a116[q]) for q in a116)  # 用同樣的寫法重寫 ct116
    assert buf.getvalue().encode() == (ROOT / REF_FILE).read_bytes(), "the writer does not reproduce ct116 byte for byte"  # 寫法必須逐位元組重現 ct116
    path = ROOT / "submission_rank_ct118.csv"; assert not path.exists(), "submission_rank_ct118.csv exists already - nothing overwritten"  # 輸出檔不可已存在（絕不覆寫）
    with open(path, "w", newline="") as fh:  # 寫出 ct118
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows((q, out[q]) for q in a116)  # 只用 LF 換行，題目順序與 ct116 相同
    print(f"written {path.name}: {len(out)} rows, md5 {hashlib.md5(path.read_bytes()).hexdigest()[:12]}")  # 回報列數與 md5 前 12 碼


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
