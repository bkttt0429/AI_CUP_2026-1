#!/usr/bin/env python
"""M211 (2026-11-17), REPORT ONLY (m208_ledger_probe.py with the reading of ct116, 0.8394955, as one more upload): the probe readings of M202 / M206 solved jointly with every recorded upload.

The ledger (m58_decode.py): every upload against ct9 gives two exact counts over its changed Public questions; the truths
are the cheapest joint assignment under a prior. Added here:
  the two small probes of 2026-11-17 (three 4D -> 5D locked questions back to 4D; seven questions A -> B) as two more uploads;
  the exact level composition of the Public half (M202 / M206) as ten count constraints over ALL 400 Public questions
  (the questions no upload ever changed become variables too, with every level as a candidate).
Prior: the pooled posterior behind ct112. For the questions of the two probes: the weight of each truth
(exp(-cost) of the best joint assignment with that truth fixed), with and without the composition constraints.

  python -I scripts/m_series/m208_ledger_probe.py
"""
import json  # 讀寫帳本與結果
import math  # 對數與指數
import sys  # 設定模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
from scipy.optimize import Bounds, LinearConstraint, milp  # 0/1 整數規劃求解器
from scipy.sparse import lil_matrix  # 稀疏的約束矩陣

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m58_decode as led  # noqa: E402  Public 帳本：上傳紀錄、分數分解與單題計分

LEV = led.LEVELS; PROBES = {"upload_probe/probe_4D5D_locked.csv": 0.8347546, "upload_probe/probe_A2B.csv": 0.8363349, "submission_rank_ct116.csv": 0.8394955}  # 等級名稱；兩個探測檔與 ct116 的 Public 分數


def solve(qs, cand, obs, prior, counts=None, fix=None, time_limit=120):  # 求最便宜的聯立真值配置（可加組成等式、固定某題真值）
    """(assignment, cost) of the cheapest joint truth assignment; counts: {level index: exact number of questions} over qs."""
    var = [(q, c) for q in qs for c in cand[q]]; idx = {v: i for i, v in enumerate(var)}; cost = np.array([-math.log(max(prior[q][c], 1e-6)) for q, c in var])  # 變數＝(題目, 候選真值)；成本＝負對數先驗
    n_rows = len(qs) + 2 * len(obs) + (10 if counts else 0); A = lil_matrix((n_rows, len(var))); lo = np.zeros(n_rows); hi = np.zeros(n_rows)  # 約束列：每題一列、每筆上傳兩列、組成十列
    for r, q in enumerate(qs):  # 每題恰有一個真值
        for c in cand[q]:  # 該題的每個候選真值
            A[r, idx[(q, c)]] = 1  # 係數為 1
        lo[r] = hi[r] = 1  # 加總等於 1
    for k, o in enumerate(obs):  # 每筆上傳
        rx, ry = len(qs) + 2 * k, len(qs) + 2 * k + 1  # 這筆上傳的兩列：完全正確數與差一級以內數
        for q, (d0, d1) in o["moves"].items():  # 這筆上傳相對於 ct9 改動的每一題
            for c in cand[q]:  # 每個候選真值
                x, y = led.xy(d0, d1, c)  # 在該真值下這個改動對兩個計數的增量
                if x:  # 完全正確數有變
                    A[rx, idx[(q, c)]] = x  # 填入係數
                if y:  # 差一級以內數有變
                    A[ry, idx[(q, c)]] = y  # 填入係數
        lo[rx] = hi[rx] = o["X"]; lo[ry] = hi[ry] = o["Y"]  # 等於由分數解出的精確計數
    if counts:  # 加上 Public 的等級組成
        base = len(qs) + 2 * len(obs)  # 組成約束的起始列
        for (q, c), i in idx.items():  # 每個變數
            A[base + c, i] = 1  # 計入等級 c 的題數
        for c in range(10):  # 每個等級
            lo[base + c] = hi[base + c] = counts[c]  # 等於量到的題數
    lb = np.zeros(len(var)); ub = np.ones(len(var))  # 0/1 變數的上下界
    if fix:  # 固定某題的真值
        if fix not in idx:  # 該真值不在候選內
            return None, None  # 視為無解
        lb[idx[fix]] = 1  # 強制選這個真值
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(len(var)), bounds=Bounds(lb, ub), options={"time_limit": time_limit})  # 求解整數規劃（有時限）
    if res.status != 0 or res.x is None:  # 沒有解或逾時
        return ("timeout" if res.status == 1 else None), None  # 逾時與無解分開回報
    return {q: c for (q, c), v in zip(var, res.x) if v > 0.5}, float(cost @ np.round(res.x))  # 回傳各題真值與總成本


def main():  # 主流程：把探測與 ct116 的讀數、組成併進帳本聯立求解
    z = np.load(ROOT / "results/r_variants/m163_test_posterior_ct112.npz", allow_pickle=True); prior = {str(q): np.maximum(p / p.sum(), 1e-6) for q, p in zip(z["qid"], z["learned"].astype(float))}  # 先驗＝ct112 背後的合併後驗
    up = {**led.KNOWN_SCORES, **json.load(open(led.LEDGER))}; qs0, cand0, obs0 = led.build(up, prior); m0, c0 = solve(qs0, cand0, obs0, prior); assert isinstance(m0, dict), "the ledger without the probes is not feasible"  # 既有的上傳帳本先求一次（必須可行）
    qs1, cand1, obs1 = led.build({**up, **PROBES}, prior); m1, c1 = solve(qs1, cand1, obs1, prior); print(f"ledger: {len(obs0)} uploads, {len(qs0)} constrained Public questions, cost {c0:.2f}; with the two probes: {len(obs1)} uploads, {len(qs1)} questions, " + ("INFEASIBLE" if not isinstance(m1, dict) else f"cost {c1:.2f}"), flush=True)  # 加入兩個探測檔與 ct116 後再求一次
    for o in obs1[-3:]:  # 兩個探測檔與 ct116
        print(f"  {o['name']}: against ct9 {len(o['moves'])} changed questions, exact {o['X']:+d}, within one {o['Y']:+d}")  # 印出改動題數與解出的兩個計數
    comp = json.load(open(ROOT / "results/trunk_seq/es/public_composition.json"))["counts"]; counts = {i: int(comp[l]) for i, l in enumerate(LEV)}; assert sum(counts.values()) == 400  # 量到的 Public 各等級題數（合計 400）
    c112 = led.decisions("submission_rank_ct112.csv"); pub = [q for q in c112.index if q.startswith("pub")]; cand2 = {q: (cand1[q] if q in cand1 else list(range(10))) for q in pub}  # ct112 的答案與全部 Public 題；沒被改過的題以全部等級當候選
    m2, c2 = solve(pub, cand2, obs1, prior, counts); print("with the composition as well (all 400 Public questions): " + ("INFEASIBLE or timeout: " + str(m2) if not isinstance(m2, dict) else f"cost {c2:.2f}"), flush=True)  # 加上組成等式，對全部 400 題求解
    if not isinstance(m2, dict):  # 受約束題目的候選集合（答過的答案上下兩級內）可能太窄，湊不出精確的組成
        cand2 = {q: list(range(10)) for q in pub}; m2, c2 = solve(pub, cand2, obs1, prior, counts); print("  retried with every level as a candidate everywhere: " + (str(m2) if not isinstance(m2, dict) else f"cost {c2:.2f}"), flush=True)  # 所有題目都以全部等級當候選重試
    pa = led.decisions("upload_probe/probe_A2B.csv"); pl = led.decisions("upload_probe/probe_4D5D_locked.csv"); c9, c28 = led.decisions("submission_rank_ct9.csv"), led.decisions("submission_rank_ct28.csv")  # 兩個探測檔與 ct9、ct28 的答案
    groups = {"A -> B probe (7)": [q for q in pub if pa[q] != c112[q]], "4D -> 5D push, pool disagrees (3)": [q for q in pub if pl[q] != c112[q]]}  # 要剖析的題組：A→B 探測與合併層不同意的 4D→5D 推移

    def profile(q, qs, cand, obs, cnt):  # 一題各候選真值的權重
        w = {}  # 各真值的最低成本
        for c in cand[q]:  # 每個候選真值
            a_, cost = solve(qs, cand, obs, prior, cnt, fix=(q, c), time_limit=90)  # 固定這個真值求最便宜的聯立解
            if isinstance(a_, dict):  # 可行時
                w[c] = cost  # 記下成本
        if not w:  # 全部不可行
            return {}  # 沒有權重
        m = min(w.values()); w = {c: math.exp(-(v - m)) for c, v in w.items()}; t = sum(w.values()); return {c: v / t for c, v in w.items()}  # 權重＝exp(-成本差)，正規化後回傳
    rep = {}  # 報告
    for gname, qg in groups.items():  # 每個題組
        print(f"\n== {gname}: truth weights (ledger + probes | + composition); the answer of ct112; the pooled posterior alone")  # 題組標題
        for q in qg:  # 每一題
            w1 = profile(q, qs1, cand1, obs1, None) if q in cand1 else {}; w2 = profile(q, pub, cand2, obs1, counts) if isinstance(m2, dict) else {}  # 只用帳本與探測、以及再加上組成的兩種真值權重
            fmt = lambda w: " ".join(f"{LEV[c]} {v:.2f}" for c, v in sorted(w.items(), key=lambda t: -t[1])[:3] if v >= 0.005) or "-"  # noqa: E731  列出權重最高的三個真值
            pr = prior[q]; top = np.argsort(-pr)[:3]; print(f"  {q}: {fmt(w1):28s} | {fmt(w2):28s} | ct112 {LEV[int(c112[q])]:>2} | posterior " + " ".join(f"{LEV[c]} {pr[c]:.2f}" for c in top), flush=True); rep[q] = {"ledger": {LEV[c]: v for c, v in w1.items()}, "with_composition": {LEV[c]: v for c, v in w2.items()}}  # 印出兩種權重、ct112 的答案與後驗前三名並存入報告
    if isinstance(m2, dict):  # 全部約束下最便宜的真值配置與 ct112 答案不同的地方
        truth = np.array([m2[q] for q in pub]); ans = np.array([int(c112[q]) for q in pub]); S = np.array([[led.s(a, c) for c in range(10)] for a in range(10)])  # 真值、ct112 的答案與計分矩陣
        print(f"\ncheapest joint assignment with everything: ct112 would score {float(S[ans, truth].mean()):.4f} on it (obtained 0.8332); it puts the truth away from ct112's answer on {int((truth != ans).sum())} questions ({int((np.abs(truth - ans) == 1).sum())} adjacent)")  # ct112 在這組真值下的分數與不同的題數
    (ROOT / "results/r_variants/m211_ledger_ct116.json").write_text(json.dumps(rep, indent=1)); print("written results/r_variants/m211_ledger_ct116.json")  # 寫出報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
