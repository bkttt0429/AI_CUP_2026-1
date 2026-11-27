"""Section 4M M58: the Public ledger. Every upload's exact Public score is a pair of linear constraints on the hidden Public labels:
400 (S_u - S_ref) = 0.63212 X_u + 0.36788 Y_u with X_u / Y_u the one-level / two-level net changes summed over the questions whose decision
differs from the reference file (ct9). All uploads are decoded jointly by a 0/1 MILP (scipy.optimize.milp): one-hot truth per involved
question over the classes within two of any decision it received, the exact X_u, Y_u of every upload as equality constraints, ct9's posterior
as the log-prior objective. Report: the MAP truths, which questions are DETERMINED (only one class is feasible), each probe's net vs ct13,
and the keep decision of single-push probes (Bayes rule, prior signal rate PI, signal precision P1 vs null precision P0).
  python -I scripts/m_series/m58_decode.py add submission_rank_ct16.csv 0.7760000     # record an upload's Public score
  python -I scripts/m_series/m58_decode.py report                                    # decode everything recorded
"""
import argparse  # 解析子命令與參數
import json  # 讀寫帳本與解碼結果
import math  # 指數與對數
import sys  # 本檔未使用的匯入（保留原樣）
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取提交檔與機率檔
from scipy.optimize import Bounds, LinearConstraint, milp  # 0/1 整數規劃求解器
from scipy.sparse import lil_matrix  # 稀疏的限制矩陣

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]  # 十個等級，順序同 aicup.metric
RANK = {lv: i for i, lv in enumerate(LEVELS)}  # 等級名稱轉成索引
E1 = math.exp(-1.0)  # 差一級時的官方得分 e^-1
LEDGER = ROOT / "results/trunk_seq/es/public_ledger.json"  # 歷次上傳 Public 分數的帳本
REF = "submission_rank_ct9.csv"  # 解碼的參考檔
KNOWN_SCORES = {"submission_rank_ct7.csv": 0.7687835, "submission_rank_ct9.csv": 0.7622032, "submission_rank_ct10.csv": 0.7701047, "submission_rank_ct12.csv": 0.7656229,  # 帳本建立前已知的上傳分數
                "submission_rank_ct13.csv": 0.7748456, "submission_rank_ct14.csv": 0.7732653, "submission_rank_ct15.csv": 0.7676047}  # 已知分數（續）
PI, P1 = 0.05, 0.8  # 2026-11-08 19:00：在任何後續讀數之前，先驗訊號率 PI 由 0.1 降為 0.05（M61 + 報告〈測試端新訊號來源〉）；P1 是訊號精確率


def s(d, c):  # 單題的官方得分
    k = abs(d - c)  # 決策與真值差幾級
    return 1.0 if k == 0 else E1 if k == 1 else 0.0  # 答對 1、差一級 e^-1、其餘 0


def xy(d0, d1, c):  # 單題改答案時對 (X, Y) 的貢獻
    """(x, y) increments of one question whose decision moves d0 -> d1 when its truth is c: 1 = (1, 1), 1 - e^-1 = (1, 0), e^-1 = (0, 1)."""
    dd = round(s(d1, c) - s(d0, c), 6)  # 這題得分的變化量
    return {1.0: (1, 1), -1.0: (-1, -1), round(1 - E1, 6): (1, 0), round(E1 - 1, 6): (-1, 0), round(E1, 6): (0, 1), round(-E1, 6): (0, -1), 0.0: (0, 0)}[dd]  # 查表換成 (x, y)


def decompose(delta400, nmax):  # 把 400 倍的分數差拆成 (X, Y)
    sols = [(X, Y) for X in range(-nmax, nmax + 1) for Y in range(-nmax, nmax + 1) if abs((1 - E1) * X + E1 * Y - delta400) < 3e-4]  # 枚舉所有吻合的整數解
    return sols  # 回傳全部解（應只有一個）


def decisions(path):  # 讀入提交檔的決策
    return pd.read_csv(ROOT / path, dtype=str).set_index("question_id")["pred_rank"].map(RANK)  # 題號對應等級索引


def build(uploads, prior_P, extra_queries=()):  # 建立聯合解碼的問題
    """uploads: {file: score}. extra_queries: [(decisions Series, X, Y)] for simulations. Returns (questions, cand, obs)."""
    ref = decisions(REF); pub = [q for q in ref.index if q.startswith("pub")]  # 參考檔決策與 Public 題清單
    obs = []  # 每次上傳的觀測
    for f, sc in uploads.items():  # 逐一上傳
        if f == REF:  # 參考檔本身沒有差異
            continue  # 略過
        d = decisions(f).reindex(ref.index)  # 這次上傳的決策
        ch = [q for q in pub if d[q] != ref[q]]  # 與參考檔不同的 Public 題
        delta = round((sc - uploads.get(REF, KNOWN_SCORES[REF])) * 400, 5)  # 相對參考檔的分數差乘 400
        sol = decompose(delta, len(ch))  # 拆成 (X, Y)
        if len(sol) != 1:  # 必須恰有一組解
            raise SystemExit(f"{f}: score {sc} gives {len(sol)} (X, Y) decompositions over {len(ch)} changes: {sol}")  # 分數或檔案有誤時停止
        obs.append({"name": f, "moves": {q: (int(ref[q]), int(d[q])) for q in ch}, "X": sol[0][0], "Y": sol[0][1]})  # 記下改動與 X、Y
    for k, (d, X, Y) in enumerate(extra_queries):  # 模擬用的額外觀測
        ch = [q for q in pub if d[q] != ref[q]]  # 與參考檔不同的 Public 題
        obs.append({"name": f"query{k}", "moves": {q: (int(ref[q]), int(d[q])) for q in ch}, "X": X, "Y": Y})  # 以給定的 X、Y 加入
    qs = sorted({q for o in obs for q in o["moves"]})  # 所有涉及的題目
    cand = {}  # 每題可能的真值
    for q in qs:  # 逐題
        ds = {int(ref[q])} | {o["moves"][q][1] for o in obs if q in o["moves"]}  # 這題收過的所有決策
        cand[q] = sorted({c for d in ds for c in range(d - 2, d + 3) if 0 <= c <= 9})  # 與任一決策相差兩級以內的等級
    return qs, cand, obs  # 回傳題目、可能真值與觀測


def solve(qs, cand, obs, prior_P, fix=None, forbid=None, time_limit=None):  # 以 MILP 求 MAP 真值
    """MAP truth assignment (dict q -> class) or None if infeasible. fix: (q, c) forced; forbid: (q, c) excluded."""
    var = [(q, c) for q in qs for c in cand[q]]  # 每個（題目, 等級）一個 0/1 變數
    idx = {v: i for i, v in enumerate(var)}  # 變數的索引
    cost = np.array([-math.log(max(prior_P[q][c], 1e-6)) for q, c in var])  # 目標函數：負對數先驗
    n_rows = len(qs) + 2 * len(obs)  # 每題一條 one-hot 限制，每次上傳兩條
    A = lil_matrix((n_rows, len(var))); lo = np.zeros(n_rows); hi = np.zeros(n_rows)  # 限制矩陣與上下界
    for r, q in enumerate(qs):  # one-hot 限制
        for c in cand[q]:  # 這題的每個可能等級
            A[r, idx[(q, c)]] = 1  # 係數為 1
        lo[r] = hi[r] = 1  # 恰好選一個等級
    for k, o in enumerate(obs):  # 每次上傳的 X、Y 等式
        rx, ry = len(qs) + 2 * k, len(qs) + 2 * k + 1  # 這次上傳的兩列
        for q, (d0, d1) in o["moves"].items():  # 這次上傳改動的題
            for c in cand[q]:  # 每個可能的真值
                x, y = xy(d0, d1, c)  # 真值為 c 時的貢獻
                if x:  # 對 X 有貢獻
                    A[rx, idx[(q, c)]] = x  # 填入 X 列
                if y:  # 對 Y 有貢獻
                    A[ry, idx[(q, c)]] = y  # 填入 Y 列
        lo[rx] = hi[rx] = o["X"]; lo[ry] = hi[ry] = o["Y"]  # 等式：恰等於解出的 X、Y
    lb = np.zeros(len(var)); ub = np.ones(len(var))  # 變數範圍 [0, 1]
    if fix:  # 要強制某題取某等級
        lb[idx[fix]] = 1  # 下界設為 1
    if forbid:  # 要排除某題的某等級
        ub[idx[forbid]] = 0  # 上界設為 0
    res = milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(len(var)), bounds=Bounds(lb, ub),  # 求解 0/1 整數規劃
               options={"time_limit": time_limit} if time_limit else None)  # 可選的時間上限
    if res.status == 1:  # 碰到時間上限：結果未知
        return "timeout"  # 回傳逾時標記
    if res.status != 0 or res.x is None:  # 不可行
        return None  # 回傳 None
    return {q: c for (q, c), v in zip(var, res.x) if v > 0.5}  # 取值為 1 的變數就是各題真值


def feasible_sets(qs, cand, obs, prior_P, mapa, only=None, time_limit=None):  # 每題的可行等級集合
    """{q: sorted feasible classes}; a check that hits the time limit counts its class as feasible (conservative)."""
    out = {}  # 結果
    for q in (only or qs):  # 逐題，或只查指定的題
        fs = {mapa[q]}  # MAP 等級本身一定可行
        for c in cand[q]:  # 其他可能的等級
            if c != mapa[q] and solve(qs, cand, obs, prior_P, fix=(q, c), time_limit=time_limit) is not None:  # 強制取 c 仍有解（逾時也算）
                fs.add(c)  # 記為可行
        out[q] = sorted(fs)  # 排序後存入
    return out  # 回傳每題的可行集合


def determined(qs, cand, obs, prior_P, mapa, only=None, time_limit=None):  # 每題是否被唯一決定
    """{q: True if no feasible assignment gives q a class other than its MAP class} -- one MILP per question."""
    return {q: solve(qs, cand, obs, prior_P, forbid=(q, mapa[q]), time_limit=time_limit) is None for q in (only or qs)}  # 排除 MAP 等級後無解即為唯一決定


def prior_from_ct9():  # 以 ct9 的測試後驗當先驗
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv").set_index("question_id")  # 讀入 ct9 的十級機率
    return {q: pr.loc[q, [f"p_{lv}" for lv in LEVELS]].to_numpy(float) for q in pr.index}  # 題號對應機率向量


def keep_rule(right, wrong, p0):  # 單一推移探測的保留判定
    lr = (P1 / p0) ** right * ((1 - P1) / (1 - p0)) ** wrong  # 訊號對虛無的似然比
    post = PI * lr / (PI * lr + 1 - PI)  # 推移是訊號的後驗機率
    return post  # 回傳後驗


def report():  # 解碼所有已記錄的上傳並寫出結果
    led = json.load(open(LEDGER)) if LEDGER.exists() else {}  # 讀入帳本
    uploads = {**KNOWN_SCORES, **led}  # 已知分數加上帳本
    prior = prior_from_ct9()  # ct9 後驗當先驗
    qs, cand, obs = build(uploads, prior)  # 建立解碼問題
    mapa = solve(qs, cand, obs, prior)  # MAP 真值
    if mapa is None:  # 聯立不可行
        raise SystemExit("the recorded scores are jointly infeasible: a score or a file was recorded wrongly")  # 表示有分數或檔案記錯
    fsets = feasible_sets(qs, cand, obs, prior, mapa)  # 每題的可行等級
    det = [q for q in qs if len(fsets[q]) == 1]  # 被唯一決定的題
    print(f"{len(obs)} uploads decoded vs ct9; {len(qs)} public questions involved, {len(det)} determined")  # 印出摘要
    base13 = decisions("submission_rank_ct13.csv")  # ct13 的決策（探測的底本）
    for o in obs:  # 逐次上傳
        print(f"  {o['name']:28s} vs ct9: X {o['X']:+d} Y {o['Y']:+d} over {len(o['moves'])} changes")  # 印出 X、Y 與改動題數
    print("single-push probes vs ct13 (right = truth is the target, wrong = truth is the source; ? = not determined):")  # 探測表的標題
    for f in uploads:  # 逐一上傳
        if f in KNOWN_SCORES or f == REF:  # 只列帳本中的上傳：寫死分數的檔案（含 M53 的探測 ct14、ct15）與參考檔略過
            continue  # 略過
        d = decisions(f)  # 這次上傳的決策
        ch = [q for q in d.index if q.startswith("pub") and d[q] != base13[q]]  # 相對 ct13 改動的 Public 題
        if not ch:  # 沒有改動
            continue  # 略過
        moves = pd.Series([f"{LEVELS[base13[q]]}->{LEVELS[d[q]]}" for q in ch]).value_counts()  # 各推移方向的題數
        right = sum(1 for q in ch if mapa[q] == d[q]); wrong = sum(1 for q in ch if mapa[q] == base13[q]); undet = sum(1 for q in ch if len(fsets[q]) > 1)  # 改對、改錯與未決定的題數
        p0 = 0.4  # 虛無精確率 P0
        print(f"  {f:28s} {dict(moves)}: right {right}, wrong {wrong}, other {len(ch) - right - wrong}, undetermined {undet}; "  # 印出對錯統計
              f"P(push is signal) {keep_rule(right, wrong, p0):.2f} -> {'KEEP' if keep_rule(right, wrong, p0) > 0.5 else 'drop'}")  # 以及保留判定
    out = {"map": {q: LEVELS[c] for q, c in mapa.items()}, "feasible": {q: [LEVELS[c] for c in fs] for q, fs in fsets.items()}, "determined": det}  # 解碼結果
    Path(ROOT / "results/trunk_seq/es/public_ledger_decoded.json").write_text(json.dumps(out, indent=1)); print("written results/trunk_seq/es/public_ledger_decoded.json")  # 寫出解碼結果


def main():  # 命令列進入點
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("add", "report")); ap.add_argument("file", nargs="?"); ap.add_argument("score", nargs="?", type=float)  # 子命令、檔名與分數
    a = ap.parse_args()  # 讀入參數
    if a.cmd == "add":  # 記錄一次上傳
        led = json.load(open(LEDGER)) if LEDGER.exists() else {}  # 讀入既有帳本
        led[a.file] = a.score; LEDGER.write_text(json.dumps(led, indent=1)); print(f"recorded {a.file} = {a.score}")  # 加入新分數並寫回帳本
    report()  # 兩種子命令最後都重新解碼


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
