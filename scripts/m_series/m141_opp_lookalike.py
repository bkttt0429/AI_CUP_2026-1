#!/usr/bin/env python
"""Section 4M M141 (2026-11-15): the opponent-side look-alike reading as a new input of the learned pool (design doc M141).

Reading: every even game's opponent-side identity vector (var_S4e) against the prototypes of the labelled training players
(m123's reading: balanced level weights, sharpness 20) -> a log distribution over the ten levels; a question takes the mean over
its even games, centred per level over the questions. The pool of M137 (L0: alpha, six betas, gamma) gets one more coefficient
per question size. Composition: the four guards relative to the current final's rule (ct106 = L0 + guards relative to the
ct104b rule). Registered: the pool with the new input against the ct106 rule, M37 on seeds 0-2 and 6-8. On a pass the test-side
file is built (ct28's locked answers untouched, guards relative to ct106's answers; never overwrites).

  python -I scripts/m_series/m141_opp_lookalike.py [--name ct107] [--no-build]
"""
import argparse  # 解析命令列參數
import csv  # 寫出提交檔
import hashlib  # 計算輸出檔的 md5
import json  # 讀寫結果 JSON
import sys  # 設定模組搜尋路徑與中止程式
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取 CSV 與依題目分組加總
import torch  # 以 LBFGS 擬合合併係數

ROOT = Path.cwd()  # 以目前目錄為 repo 根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄的實驗模組可匯入
import m137b_build as mb7  # noqa: E402  匯入 M137b 的合併擬合、防護與共用設定
import m137f_build2 as mf7  # noqa: E402  匯入 M137e 修訂後的四條防護（guard2）
from aicup.metric import LEVELS  # noqa: E402  十個等級的名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  題目大小、類別平衡分數與成對差檢定

m3, mf, la, b, s2, mb, ES, FOLDS, REC, SM = mb7.m3, mb7.mf, mb7.la, mb7.b, mb7.s2, mb7.mb, mb7.ES, mb7.FOLDS, mb7.REC, mb7.SM  # 沿用 M137b 匯入的模組與設定
NAMES, FIVE, STAT, W_FIN, CACHE, RIDGE = mb7.NAMES, mb7.FIVE, mb7.STAT, mb7.W_FIN, mb7.CACHE, mb7.RIDGE  # 沿用 M137b 的來源名稱、權重、快取位置與收縮強度
EXTRA = ("Ro",)  # 新的輸入，每種題目大小各一個係數（初始 0）


def fit(X, y, n, w, extra, steps=300):  # 擬合含額外輸入的 L0 合併係數
    """{size: theta = alpha, six betas, gamma, one per extra input} (no level offset: the L0 pool)."""
    out = {}  # 各大小的係數
    for s in SIZES:  # 每種題目大小各擬合一組
        m = n == s  # 這個大小的題目
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0); E = [torch.tensor(X[k][m]) for k in extra]  # 基礎對數機率、本人側讀數、六個來源證據與額外輸入
        yt, wt = torch.tensor(y[m]), torch.tensor(w[m])  # 真實等級與類別平衡權重
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]] + [0.0] * len(extra), dtype=torch.float64)  # 初始值＝手工規則，額外輸入的係數從 0 開始
        dl = torch.zeros(len(th0), dtype=torch.float64, requires_grad=True)  # 要學的係數增量（沒有偏移）
        opt = torch.optim.LBFGS([dl], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")  # 以 LBFGS（強 Wolfe 線搜尋）最佳化係數增量

        def closure():  # LBFGS 每次評估損失時呼叫
            opt.zero_grad(); th = th0 + dl  # 清除梯度並算出目前的係數
            z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R  # 原有輸入的合併 logit
            for i, e in enumerate(E):  # 逐一加入額外輸入
                z = z + th[8 + i] * e  # 乘上各自的係數
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (dl ** 2).sum()  # 類別平衡交叉熵加上 L2 收縮
            loss.backward()  # 反向傳播求梯度
            return loss  # 交給 LBFGS
        opt.step(closure)  # 執行最佳化
        out[s] = (th0 + dl).detach().numpy().copy()  # 存下這個大小的係數
    return out  # 回傳各大小的係數


def apply(coef, X, n, extra):  # 以擬合的係數算出合併後驗（含額外輸入）
    P = np.zeros_like(X["logp"])  # 輸出的後驗機率
    for s in sorted(set(int(x) for x in n)):  # 逐一處理出現的題目大小
        m = n == s; th = coef[s]  # 這個大小的題目與係數
        z = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + sum(th[8 + i] * X[k][m] for i, k in enumerate(extra))  # 合併的 logit（含額外輸入）
        z = z - z.max(1, keepdims=True); e = np.exp(z); P[m] = e / e.sum(1, keepdims=True)  # 數值穩定的 softmax
    return P  # 回傳後驗


def opp_reading(vecs, protos, labels, beta, owner, n_q, use):  # 對手側讀數：每題取可用各局讀數的平均
    """[n_q, 10]: per question the mean over its used games of the per-game log reading of the opponent-side vectors (zeros when
    a question has no used game). vecs [members, G, 128] unit vectors, owner [G] question index, use [G] bool."""
    lr, _, _ = la.reading(vecs[:, use], protos, labels, beta)  # 只對可用的局逐局讀數
    out = np.zeros((n_q, 10)); cnt = np.zeros(n_q); np.add.at(out, owner[use], lr); np.add.at(cnt, owner[use], 1.0)  # 依題目加總讀數與局數
    return out / np.maximum(cnt, 1.0)[:, None], cnt  # 回傳每題的平均與可用局數


def main():  # 主流程：交叉驗證、登錄判定、測試端建檔
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct107"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default="m141_opp_lookalike.json"); a_ = ap.parse_args()  # 參數：輸出檔名、只檢定不建檔、報告檔名
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]  # 讀入 M129 算好的各 m15 來源密度比
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])  # 載入 5e 與五個配方的識別器（各含 m15 版）的輸出
    rec0, rec2 = dict(REC), dict(REC)  # 兩套配方表：原設定與五個配方改用 m15
    for k in FIVE:  # 五個投票配方
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))  # 改用 m15 來源並換上它的密度比
    LF = {v: m3.load_fold(v) for v in FOLDS}  # 各驗證折的題目
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"]); hd_val = np.asarray(z["hd"])  # 局鍵索引、查詢表與每局的讓子級別（HD：0 互先、1 讓先、2 以上讓子）
    assert np.array_equal(q["qkey"][:Q].astype(str), z["key"]) and (q["side"][:Q] == 1).all() and (q["side"][Q:2 * Q] == 0).all(), "queries: the first Q entries must be the opponent side, the next Q the target side"  # 確認查詢表的排列：先對手側、再本人側
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()  # 訓練玩家、等級與所屬折
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]  # 身分向量（S4e）：查詢向量、原型與網路數
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=2, keepdims=True), 1e-9)  # noqa: E731  正規化成單位向量
    Fd = {}  # 各折的輸入
    for v in FOLDS:  # 逐一準備驗證折
        y, n, pid, qd = LF[v]; y, n, pid = np.asarray(y), np.asarray(n), np.asarray(pid); gk = list(qd["game_keys"]); Tn = np.array([b.TN[int(m)] for m in n])[:, None]; Ls = {}; cache = {}  # 標籤、局數、玩家、各題局鍵與降溫係數
        for A, rec in ((0, rec0), (2, rec2)):  # 兩套配方表
            Ls[A] = {}  # 這一套的各來源證據
            for name, spec in rec.items():  # 逐一配方
                if spec not in cache:  # 相同設定只算一次
                    idn, enc, cc, temp, pi0, bmin, sc = spec; ev = m3.evidence(V[idn], v, cc, pi0, bmin, sc); ee = m3.evidence(V["enc"], v, 0, pi0, bmin, m3.R_ENC) if enc else {}  # 折外校準的錨點證據與編碼器證據
                    L = np.stack([sum((ev.get(k, m3.Z10) + b.U_ENC * ee.get(k, m3.Z10) for k in g_), m3.Z10) for g_ in gk]); cache[spec] = L * Tn if temp else L  # 每題加總各局證據，需要時乘上降溫係數
                Ls[A][name] = cache[spec]  # 存入這個配方
        qi = np.concatenate([np.full(sum(k in row for k in g_), j) for j, g_ in enumerate(gk)]); gi = np.concatenate([[row[k] for k in g_ if k in row] for g_ in gk]).astype(int); cand = np.flatnonzero(pfold != v); tq = np.zeros((NM, len(y), 128))  # 每局的題號與查詢列；候選人排除驗證折玩家
        for m in range(NM):  # 每個網路
            np.add.at(tq[m], qi, QV[m, Q + gi].astype(np.float64))  # 把本人側的局向量加總到題目
        tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)  # 正規化成單位向量
        protos = [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)]  # 各成員中留出第 v 折的網路對其他折訓練玩家的原型
        lr_, _, _ = la.reading(tq, protos, plab[cand], la.BE)  # 本人側的長得像的人讀數
        ov = unit(QV[:, gi].astype(np.float64))  # 題目各局的對手側向量（被多題使用的局每題各出現一次）
        Ro, cnt = opp_reading(ov, protos, plab[cand], la.BE, qi, len(y), hd_val[gi] == 0)  # 對手側讀數：只用互先局
        Ro50, _ = opp_reading(ov, protos, plab[cand], 50.0, qi, len(y), hd_val[gi] == 0)  # 只報告的版本：銳度 50
        Ro1, _ = opp_reading(ov, protos, plab[cand], la.BE, qi, len(y), hd_val[gi] <= 1)  # 只報告的版本：納入讓先局
        cw = np.zeros(len(y))  # 類別平衡權重
        for s in SIZES:  # 每種題目大小
            for c in range(10):  # 每個等級
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)  # 權重＝該大小該等級題數的倒數
        cen = lambda x: x - x.mean(0, keepdims=True)  # noqa: E731  各等級在本折的題目上置中
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=cen(lr_), Ro=cen(Ro), Ro50=cen(Ro50), Ro1=cen(Ro1), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)  # 存下這一折的輸入（讀數都先置中）
        print(f"fold {v}: inputs ready ({len(y)} questions; share of questions without an even game {float((cnt == 0).mean()):.3f}; opponent reading alone: level of its maximum equals the truth {float((Ro.argmax(1) == y)[cnt > 0].mean()):.3f}, within one {float((np.abs(Ro.argmax(1) - y) <= 1)[cnt > 0].mean()):.3f})", flush=True)  # 回報沒有互先局的比例與讀數單獨的準度
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; ok = []; X012 = None  # 合併四折的標籤，準備報告與通過紀錄
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731  四折類別平衡分數的平均
    VARI = {"L0 (ct106 rule)": (), "+ opponent reading": ("Ro",), "+ opponent reading, beta 50": ("Ro50",), "+ opponent reading, with HD1 games": ("Ro1",)}  # 比較的版本：基準（ct106 規則）與三種加入對手側讀數的版本
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子各檢定一次
        X, S = {}, {}  # 各折的合併輸入與逐題得分
        for v in FOLDS:  # 每個驗證折
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 基礎集成機率：C、T 兩種結構乘三個種子的平均
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], "Ro": f["Ro"], "Ro50": f["Ro50"], "Ro1": f["Ro1"], **{k: f["Ls"][2][k] for k in NAMES}}  # 合併的輸入（含三種對手側讀數）
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["dc"] = mb7.keep_old(db, du)  # ct104b 規則的答案
            S.setdefault("ct104b rule", []).append(SM[f["dc"], f["y"]])  # 記錄 ct104b 規則的得分
        if seeds == (0, 1, 2):  # 提交用的種子組
            X012 = X  # 留給測試端擬合
        coefs = {}  # 各版本各折的係數
        for name, extra in VARI.items():  # 每個版本
            for v in FOLDS:  # 交叉擬合：輪流留出一折評估
                tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}  # 其餘三折當訓練資料
                coef = fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), extra); coefs[(name, v)] = coef  # 擬合並保存係數
                ds = np.asarray(b.decide(apply(coef, X[v], Fd[v]["n"], extra)), int)  # 在留出折上以期望分數決策
                if not extra:  # 基準版本
                    Fd[v]["d106"] = mf7.guard2(ds, Fd[v]["dc"]); S.setdefault(name, []).append(SM[Fd[v]["d106"], Fd[v]["y"]])  # 加四條防護＝ct106 規則的答案
                else:  # 加了新輸入的版本
                    S.setdefault(name, []).append(SM[mf7.guard2(ds, Fd[v]["d106"]), Fd[v]["y"]]); S.setdefault(name + " (plain)", []).append(SM[ds, Fd[v]["y"]])  # 相對於 ct106 規則加防護與不加防護的得分
        print(f"seeds {seeds}: ct104b rule {bal(S['ct104b rule']):.4f}, ct106 rule {bal(S['L0 (ct106 rule)']):.4f}")  # 印出兩個參考規則的分數
        out = {}  # 這組種子的比較結果
        for name, extra in VARI.items():  # 每個版本
            if not extra:  # 基準本身不比較
                continue  # 略過
            A_, B_ = np.concatenate(S[name]), np.concatenate(S["L0 (ct106 rule)"]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)  # 對 ct106 規則的成對差與信賴區間
            pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[name], S["L0 (ct106 rule)"], ys, ns)]  # 各折的差
            good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)  # M37：CI 下界 > 0、差 >= 0.0025、至少三折為正
            print(f"  {name:36s}: {bal(S[name]):.4f}; against the ct106 rule {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items())  # 印出比較結果與各題目大小的差
                  + f"  changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}; without guards {bal(S[name + ' (plain)']):.4f}; coefficient of the new input " + " ".join(f"n={s}: {np.mean([coefs[(name, v)][s][8] for v in FOLDS]):+.2f}" for s in SIZES))  # 接上改動比例、不加防護的分數與新輸入的係數
            if name == "+ opponent reading":  # 登錄的主要版本
                ok.append(bool(good)); print("        by true level: " + "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10)))  # 記錄是否通過並印出依真實等級的得失
                print("        other coefficients (mean of the four fits): " + " | ".join(f"n={s}: alpha {np.mean([coefs[(name, v)][s][0] for v in FOLDS]):.2f}, v5 {np.mean([coefs[(name, v)][s][2] for v in FOLDS]):.2f}, own look-alike {np.mean([coefs[(name, v)][s][7] for v in FOLDS]):.2f}" for s in SIZES), flush=True)  # 印出其他係數（四次擬合的平均）
            out[name] = {**dd, "per_fold": pf, "meets_m37": bool(good)}  # 存入報告
        rep[f"seeds_{seeds[0]}"] = out  # 存入報告
    passed = all(ok); print(f"REGISTERED (M37 on both seed sets, the pool with the opponent reading against the ct106 rule): {'PASS' if passed else 'NO'}"); rep["passed"] = passed  # 登錄的判定：兩組種子都要通過
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告
    if not passed or a_.no_build:  # 沒通過或只檢定
        return  # 到此結束
    # ---------------------------------------------------------------- 測試端
    extra = ("Ro",); Xall = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}  # 只加對手側讀數；四折合起來的輸入（種子 0-2）
    coef = fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), extra); coef[15] = (coef[10] + coef[20]) / 2  # 以四折全部擬合；15 局題用 10 與 20 局係數的平均
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 讀入 ct9 的基礎機率與題號
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  讀取既有提交檔的答案
    c9, c28, c104b, c106 = (sub(k) for k in ("ct9", "ct28", "ct104b", "ct106")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 既有檔答案、ct28 鎖定題、各題局數與降溫係數、Public 題遮罩
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ = {}  # 測試端編碼器證據（每題加總）
    for name in NAMES:  # 六個來源
        idn, enc, cc, temp, pi0, bmin, sc = rec2[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)  # 測試端各來源的每題證據（配方表 2：五個配方用 m15）
    ti = np.flatnonzero(q["qtest"] >= 0); qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); protos = [PR[m, 4].astype(np.float64) for m in range(NM)]  # 測試查詢列、各局所屬題目與全體玩家網路（-1）的原型
    tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; opp = ti[q["side"][ti] == 1]; opp = opp[np.argsort(q["qtest"][opp])]; assert len(tgt) == len(opp) == len(tqid) == 10000  # 測試局本人側與對手側的查詢列，依局排序
    tq = np.zeros((NM, 800, 128))  # 每題本人側的平均向量
    for m in range(NM):  # 每個網路
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))  # 把各局向量加總到題目
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)  # 正規化成單位向量
    lr, _, _ = la.reading(tq, protos, plab, la.BE); lr = lr - lr.mean(0, keepdims=True)  # 本人側讀數並置中
    hd_t = np.asarray(T["5e"]["hd"]); Ro_t, cnt_t = opp_reading(unit(QV[:, opp].astype(np.float64)), protos, plab, la.BE, gq, 800, hd_t == 0); Ro_t = Ro_t - Ro_t.mean(0, keepdims=True)  # 測試端對手側讀數（只用互先局）並置中
    Xt = {"logp": np.log(np.maximum(P9, 1e-12)), "R": lr, "Ro": Ro_t, **{k: LQ[k] for k in NAMES}}  # 測試端的合併輸入
    X0 = {k: v for k, v in Xt.items() if k != "Ro"}; c0 = mb7.fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), "L0"); c0[15] = ((c0[10][0] + c0[20][0]) / 2, (c0[10][1] + c0[20][1]) / 2)  # 以與 ct106 建檔程式完全相同的方式擬合 L0 係數（不含新輸入）
    d0 = mf7.guard2(np.where(locked, l28, np.asarray(b.decide(mb7.apply(c0, X0, nq_t.astype(int))), int)), rk(c104b).astype(int)); assert (lev[d0] == c106).all(), int((lev[d0] != c106).sum()); print("checks: without the new input the builder reproduces ct106")  # 檢查：不加新輸入時要逐題重現 ct106
    Pl = apply(coef, Xt, nq_t.astype(int), extra); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin = lev[mf7.guard2(d_plain, rk(c106).astype(int))]; assert (fin[locked] == c28[locked]).all()  # 新合併的決策套上相對於 ct106 答案的防護；鎖定題不變
    np.savez(ROOT / "results/r_variants/m141_test_posterior.npz", qid=qids, learned=Pl, base=P9)  # 存下測試端的後驗
    print(f"test questions without an even game: {int((cnt_t == 0).sum())}; coefficients of the opponent reading " + " ".join(f"n={s}: {coef[s][8]:+.2f}" for s in SIZES))  # 回報沒有互先局的測試題數與對手側讀數的係數
    for nm, o in (("ct106", c106), ("ct104b", c104b)):  # 與既有各檔比較
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")  # 不同的題數與差幾級
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c106[fin != c106], fin[fin != c106])]).value_counts(); print("kinds of changes against ct106: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))  # 對 ct106 的改動類別
    for i in np.flatnonzero(fin != c106):  # 逐題列出改動
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct104b {c104b[i]}, ct106 {c106[i]} -> {fin[i]}")  # 題號、局數與各檔答案
    f = ROOT / f"submission_rank_{a_.name}.csv"  # 輸出檔路徑
    if f.exists():  # 不覆寫既有檔
        sys.exit(f"{f.name} exists already: nothing overwritten")  # 已存在就中止
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))  # 只用 LF 換行，表頭加每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]  # 計算 md5 前 12 碼
    print(f"{f.name} written (not copied to upload_next), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}")  # 回報檔案與格式檢查
    rep["file"] = {"name": f.name, "md5": md5, "differs_from_ct106": qids[fin != c106].tolist(), "public": int(((fin != c106) & pub).sum()), "private": int(((fin != c106) & ~pub).sum())}  # 檔案資訊寫入報告
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))  # 重寫報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
