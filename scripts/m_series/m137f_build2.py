#!/usr/bin/env python
"""Section 4M M137e amendment (2026-11-14): the L0 pool of M137b with the guards taken relative to the CURRENT FINAL answers
(the ct104b rule): no change out of A, no 5D -> 6D, no change out of D, no change of two or more levels. Derived from
m137b_build.py; only the composition differs. Builds ct106 when the amended rule meets M37 on both seed sets.

Original header of m137b_build.py follows.
Section 4M M137b (2026-11-14): the learned log-linear pool of M137 - ablations, the cross-validation under ct104b's two
composition guards, and (only when the registered build rule holds) the test-side file.

Variants: L = the registered model (alpha, six betas, gamma, a level offset; one set per question size); L0 = no level offset;
L1 = no offset and alpha fixed at 1; Lb = only the offset (the rest at the hand-set values). Guarded: a variant's decisions with
the changes "out of A" and "5D -> 6D" undone relative to the ct75u rule's answers, compared with the ct104b rule (the M129 rule
under the same guards). Build rule (design doc M137b): L0 under the guards must meet M37 on both seed sets; L replaces it only if
it is better by >= 0.002 on both. Test side: coefficients fitted on all four folds (seeds 0-2), 15-game questions use the mean
of the 10- and 20-game coefficients; ct28's 38 locked answers untouched; the guards relative to ct75u; never overwrites a file.

  python -I scripts/m_series/m137b_build.py [--name ct105] [--no-build]
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
import m103_stability as m3  # noqa: E402  M103 的錨點證據計算與共用設定
import m110_fusion as mf  # noqa: E402  提供 ct75u 規則的決策
import m123_lookalike_sym as la  # noqa: E402  「長得像的人」讀數（M123）
from aicup.metric import LEVELS  # noqa: E402  十個等級的名稱
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  題目大小、類別平衡分數與成對差檢定

b, s2, mb, ES, FOLDS, REC, SM = m3.b, m3.s2, m3.mb, m3.ES, m3.FOLDS, m3.REC, m3.SM  # 沿用 M103 的模組、路徑、折、配方表與計分矩陣
CACHE = ROOT / "results/m84_cache"; FIVE = ("v3", "v5", "s2", "s3c", "s3"); NAMES = ("v44",) + FIVE; STAT = "m15"  # 快取位置、五個投票配方、六個錨點來源與分帶統計量 m15
W_FIN = {5: 0.5, 10: 0.5, 20: 0.1}; RIDGE = 1e-3  # 長得像的人的手工權重（依局數）與 L2 收縮強度
DROPS = [(LEVELS.index("A"), None), (LEVELS.index("5D"), LEVELS.index("6D"))]  # 原本兩條防護：不移出 A、不做 5D→6D
VARIANTS = {"L0": (False, True, True)}  # 只做 L0；要學的項目：（偏移、alpha、來源權重）


def keep_old(d_new, d_old):  # 套用原本兩條防護：違反的改動還原成舊答案
    d = d_new.copy()  # 從新決策出發
    for o, w_ in DROPS:  # 逐條防護
        m = (d_old == o) & (d_new != d_old) if w_ is None else (d_old == o) & (d_new == w_)  # 找出違反這條防護的題目
        d[m] = d_old[m]  # 改回舊答案
    return d  # 回傳加防護後的決策


def guard2(d_new, ref):  # 修訂後的四條防護
    """The amended composition: relative to the reference answers, undo changes out of A, out of D, 5D -> 6D, and of two or more levels."""
    A_, D_, D5, D6 = LEVELS.index("A"), LEVELS.index("D"), LEVELS.index("5D"), LEVELS.index("6D")  # 四個相關等級的索引
    d = d_new.copy()  # 從新決策出發
    m = (((ref == A_) | (ref == D_)) & (d_new != ref)) | ((ref == D5) & (d_new == D6)) | (np.abs(d_new - ref) >= 2)  # 違反任一條防護的題目
    d[m] = ref[m]  # 改回參考答案
    return d  # 回傳加防護後的決策


def fit(X, y, n, w, variant, steps=300):  # 依題目大小擬合一個版本的合併係數
    """{size: (theta [8] = alpha, six betas, gamma; offset [10])} of one variant."""
    use_b, use_a, use_s = VARIANTS[variant]; out = {}  # 取出這個版本要學哪些項
    free = torch.tensor([1.0 if use_a else 0.0] + [1.0 if use_s else 0.0] * 7, dtype=torch.float64)  # 遮罩：哪些係數可以離開初始值
    for s in SIZES:  # 每種題目大小各擬合一組
        m = n == s  # 這個大小的題目
        lp, R = torch.tensor(X["logp"][m]), torch.tensor(X["R"][m]); L = torch.stack([torch.tensor(X[k][m]) for k in NAMES], 0)  # 基礎對數機率、長得像的人讀數、六個來源的證據
        yt, wt = torch.tensor(y[m]), torch.tensor(w[m])  # 真實等級與類別平衡權重
        th0 = torch.tensor([1.0] + [1.0 if k == "v5" else 0.0 for k in NAMES] + [W_FIN[s]], dtype=torch.float64)  # 初始值＝手工規則（alpha 1、只有 v5 為 1、gamma 用手工權重）
        dl = torch.zeros(8, dtype=torch.float64, requires_grad=True); bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)  # 要學的係數增量與 10 維等級偏移
        opt = torch.optim.LBFGS([dl, bias], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")  # 以 LBFGS（強 Wolfe 線搜尋）同時最佳化係數增量與偏移

        def closure():  # LBFGS 每次評估損失時呼叫
            opt.zero_grad()  # 清除上一次的梯度
            th = th0 + free * dl; bb = bias if use_b else bias * 0.0  # 目前的係數；不學偏移時偏移為 0
            z = th[0] * lp + (th[1:7, None, None] * L).sum(0) + th[7] * R + bb  # 對數線性合併的 logit
            loss = (torch.nn.functional.cross_entropy(z, yt, reduction="none") * wt).sum() / wt.sum() + RIDGE * (((free * dl) ** 2).sum() + (bb ** 2).sum())  # 類別平衡交叉熵加上對初始值的 L2 收縮
            loss.backward()  # 反向傳播求梯度
            return loss  # 交給 LBFGS
        opt.step(closure)  # 執行最佳化
        out[s] = ((th0 + free * dl).detach().numpy().copy(), (bias.detach().numpy().copy() if use_b else np.zeros(10)))  # 存下這個大小的係數與偏移
    return out  # 回傳各大小的係數


def apply(coef, X, n):  # 以擬合的係數算出合併後的後驗
    P = np.zeros_like(X["logp"])  # 輸出的後驗機率
    for s in sorted(set(int(x) for x in n)):  # 逐一處理出現的題目大小
        m = n == s; th, bias = coef[s]  # 這個大小的題目與係數
        z = th[0] * X["logp"][m] + sum(th[1 + i] * X[k][m] for i, k in enumerate(NAMES)) + th[7] * X["R"][m] + bias  # 合併的 logit
        z = z - z.max(1, keepdims=True); e = np.exp(z); P[m] = e / e.sum(1, keepdims=True)  # 數值穩定的 softmax
    return P  # 回傳後驗


def main():  # 主流程：交叉驗證、建檔規則、測試端建檔
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--name", default="ct106"); ap.add_argument("--no-build", action="store_true"); ap.add_argument("--json", default="m137f_build2.json"); a_ = ap.parse_args()  # 參數：輸出檔名、只檢定不建檔、報告檔名
    pj = json.load(open(ROOT / "results/r_variants/m129_bundle2.json")); scale = pj["density_ratio"]  # 讀入 M129 算好的各 m15 來源密度比
    ids = [REC[k][0] for k in FIVE]; V, T, key_pid, used = b.sources(["5e", *ids, *[f"{x}~{STAT}" for x in ids]])  # 載入 5e 與五個配方的識別器（各含 m15 版）的驗證端與測試端輸出
    rec0, rec2 = dict(REC), dict(REC)  # 兩套配方表：0＝原設定，2＝五個配方都用 m15（M129 的 A＝2）
    for k in FIVE:  # 五個投票配方
        idn, enc, cc, temp, pi0, bmin, _ = REC[k]; rec2[k] = (f"{idn}~{STAT}", enc, cc, temp, pi0, bmin, float(scale[idn]))  # 改用 m15 來源並換上它的密度比
    LF = {v: m3.load_fold(v) for v in FOLDS}  # 各驗證折的題目
    z = V["5e"]; row = {k: i for i, k in enumerate(z["key"])}; q = np.load(CACHE / "queries.npz"); Q = int(q["Q"])  # 局鍵到列號的索引與查詢表
    players = q["players"].astype(str); plab = q["plab"]; pfold = pd.read_csv(ROOT / "splits/player_folds.csv", dtype=str).set_index("player_id").fold.astype(int).reindex(players).to_numpy()  # 訓練玩家、等級與所屬折
    v_ = np.load(CACHE / f"var_{la.VAR}.npz"); QV, PR = v_["qvec"], v_["protos"]; NM = QV.shape[0]  # 身分向量（S4e）：查詢向量、原型與網路數
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
        lr_, _, _ = la.reading(tq, [PR[m, v - 1][cand].astype(np.float64) for m in range(NM)], plab[cand], la.BE)  # 長得像的人讀數：對其他折訓練玩家的原型
        cw = np.zeros(len(y))  # 類別平衡權重
        for s in SIZES:  # 每種題目大小
            for c in range(10):  # 每個等級
                m = (n == s) & (y == c); cw[m] = 1.0 / max(int(m.sum()), 1)  # 權重＝該大小該等級題數的倒數
        Fd[v] = dict(y=y, n=n, pid=pid, Ls=Ls, R=lr_ - lr_.mean(0, keepdims=True), wq=np.array([W_FIN[int(m)] for m in n]), cw=cw)  # 存下這一折的輸入（讀數先置中）
        print(f"fold {v}: inputs ready ({len(y)} questions)", flush=True)  # 回報進度
    ys = [Fd[v]["y"] for v in FOLDS]; ns = [Fd[v]["n"] for v in FOLDS]; y, n, pid = (np.concatenate([Fd[v][k] for v in FOLDS]) for k in ("y", "n", "pid")); rep = {}; passes = {k: [] for k in VARIANTS}; gdiff = {k: [] for k in VARIANTS}; X012 = None  # 合併四折的標籤，準備報告與通過紀錄
    bal = lambda S_: float(np.mean([balanced_mean_over_n(x, yy, nn, sizes=SIZES) for x, yy, nn in zip(S_, ys, ns)]))  # noqa: E731  四折類別平衡分數的平均
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子各檢定一次
        X, S = {}, {}  # 各折的合併輸入與各規則的逐題得分
        for v in FOLDS:  # 每個驗證折
            f = Fd[v]; P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 基礎集成機率：C、T 兩種結構乘三個種子的平均
            X[v] = {"logp": np.log(np.maximum(P, 1e-12)), "R": f["R"], **{k: f["Ls"][2][k] for k in NAMES}}  # 合併的輸入：基礎對數機率、讀數、配方表 2（五個配方用 m15）的六個來源證據
            d0, du = mf.rule75u(P, f["Ls"][0]); _, db = mf.rule75u(la.pooled(P, f["R"], f["wq"]), f["Ls"][2]); f["du"] = du; f["dc"] = keep_old(db, du)  # ct75u 與 M129 規則的決策；ct104b 規則的答案當防護的參考
            for nm, d_ in (("m129_rule", db), ("ct104b_rule", keep_old(db, du)), ("ct75u_rule", du)):  # 三個參考規則
                S.setdefault(nm, []).append(SM[d_, f["y"]])  # 記錄逐題得分
        if seeds == (0, 1, 2):  # 提交用的種子組
            X012 = X  # 留給測試端擬合
        for var in VARIANTS:  # 每個版本（只有 L0）
            for v in FOLDS:  # 交叉擬合：輪流留出一折評估
                tr = [u for u in FOLDS if u != v]; Xt = {k: np.concatenate([X[u][k] for u in tr]) for k in X[v]}  # 其餘三折當訓練資料
                coef = fit(Xt, np.concatenate([Fd[u]["y"] for u in tr]), np.concatenate([Fd[u]["n"] for u in tr]), np.concatenate([Fd[u]["cw"] for u in tr]), var)  # 擬合係數
                ds = np.asarray(b.decide(apply(coef, X[v], Fd[v]["n"])), int)  # 在留出折上以期望分數決策
                S.setdefault(var, []).append(SM[ds, Fd[v]["y"]]); S.setdefault(var + "+guards", []).append(SM[guard2(ds, Fd[v]["dc"]), Fd[v]["y"]])  # 記錄不加防護與加四條防護的得分
        print(f"seeds {seeds}: ct75u rule {bal(S['ct75u_rule']):.4f}, M129 rule {bal(S['m129_rule']):.4f}, ct104b rule (guards) {bal(S['ct104b_rule']):.4f}")  # 印出參考規則的分數
        out = {}  # 這組種子的比較結果
        for var in VARIANTS:  # 每個版本
            line = f"  {var:3s}"  # 報表行開頭
            for new, old in ((var, "m129_rule"), (var + "+guards", "ct104b_rule")):  # 不加防護對 M129 規則、加防護對 ct104b 規則
                A_, B_ = np.concatenate(S[new]), np.concatenate(S[old]); dd = paired_diff(A_, B_, y, pid, n, sizes=SIZES, n=5000)  # 成對差與信賴區間
                pf = [float(balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(c, yy, nn, sizes=SIZES)) for x, c, yy, nn in zip(S[new], S[old], ys, ns)]  # 各折的差
                good = (dd["ci95"][0] > 0) and (dd["diff"] >= 0.0025) and (sum(e > 0 for e in pf) >= 3)  # M37：CI 下界 > 0、差 >= 0.0025、至少三折為正
                line += f" | {'four guards vs ct104b rule' if '+' in new else 'plain vs M129 rule'}: {bal(S[new]):.4f} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}] folds " + " ".join(f"{e:+.4f}" for e in pf) + f" changed {float((A_ != B_).mean()):.3f} {'PASS' if good else 'no'}"  # 加入這個比較的報表
                out[f"{new} - {old}"] = {**dd, "per_fold": pf, "meets_m37": bool(good), "level": bal(S[new])}  # 存入報告
                if "+" in new:  # 加防護的比較
                    passes[var].append(bool(good)); gdiff[var].append(float(dd["diff"]))  # 記下是否通過與差值，供建檔規則使用
                    lv = "  ".join(f"{LEVELS[c]} {float((A_ - B_)[y == c].mean()):+.4f}" for c in range(10))  # 依真實等級的得失
            print(line); print(f"        guarded, by true level: {lv}", flush=True)  # 印出這個版本的結果
        rep[f"seeds_{seeds[0]}"] = out  # 存入報告
    choice = "L0" if all(passes["L0"]) else None  # L0 加四條防護在兩組種子都達 M37 才建檔
    print(f"REGISTERED BUILD RULE (amended composition): L0 under the four guards relative to the ct104b rule {'meets' if choice else 'does not meet'} M37 on both seed sets -> {'build' if choice else 'NO FILE'}")  # 印出登錄建檔規則的判定
    rep["choice"] = choice; (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))  # 寫出報告
    if choice is None or a_.no_build:  # 不建檔的情況
        return  # 到此結束
    # ---------------------------------------------------------------- 測試端（輸入與 m129_bundle2.py 建出的完全相同）
    Xall = {k: np.concatenate([X012[v][k] for v in FOLDS]) for k in X012[FOLDS[0]]}  # 四折合起來的輸入（種子 0-2）
    coef = fit(Xall, y, n, np.concatenate([Fd[v]["cw"] for v in FOLDS]), choice); coef[15] = ((coef[10][0] + coef[20][0]) / 2, (coef[10][1] + coef[20][1]) / 2)  # 以四折全部擬合；15 局題用 10 與 20 局係數的平均
    print("coefficients fitted on all four folds (seeds 0-2): " + " | ".join(f"n={s}: alpha {coef[s][0][0]:.2f}, beta " + " ".join(f"{k} {coef[s][0][1 + i]:+.2f}" for i, k in enumerate(NAMES)) + f", gamma {coef[s][0][7]:.2f}" for s in SIZES))  # 印出測試端用的係數
    pr = pd.read_csv(ROOT / "submission_rank_ct9_proba.csv", dtype={"question_id": str}); qids = pr.question_id.to_numpy(); P9 = pr[[f"p_{lv}" for lv in LEVELS]].to_numpy(float); lev = np.array(LEVELS); rk = np.vectorize(b.RANK.get)  # 讀入 ct9 的基礎機率與題號
    sub = lambda k: pd.read_csv(ROOT / f"submission_rank_{k}.csv", dtype=str).set_index("question_id").pred_rank.reindex(qids).to_numpy()  # noqa: E731  讀取既有提交檔的答案
    c9, c28, c44, c75u, c104b = (sub(k) for k in ("ct9", "ct28", "ct44", "ct75u", "ct104b")); locked = c28 != c9; l28 = rk(c28); tqid = T["5e"]["qid"].astype(str); nq_t = pd.Series(tqid).value_counts().reindex(qids).to_numpy(); Tq = np.array([b.TN[int(m)] for m in nq_t])[:, None]; pub = np.char.startswith(qids.astype(str), "pub_")  # 既有檔答案、ct28 鎖定題、各題局數與降溫係數、Public 題遮罩
    Leq = pd.DataFrame(m3.test_evidence(V, T, "enc", 0, 1, m3.R_ENC, 1)).groupby(T["enc"]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy()  # 測試端編碼器證據（每題加總）

    def evidence_of(rec):  # 依配方表算測試端各來源的每題證據
        LQ = {}  # 各來源的每題證據
        for name in NAMES:  # 六個來源
            idn, enc, cc, temp, pi0, bmin, sc = rec[name]; Lq = pd.DataFrame(m3.test_evidence(V, T, idn, cc, pi0, sc, bmin)).groupby(T[idn]["qid"]).sum().reindex(qids).fillna(0.0).to_numpy(); LQ[name] = ((Lq + b.U_ENC * Leq) if enc else Lq) * (Tq if temp else 1.0)  # 每題加總，加上編碼器證據與降溫
        return LQ  # 回傳各來源證據

    def compose(P, LQ):  # 重現 ct44 與 ct75u 的手工組成規則
        dR, dA = {}, {}  # 各來源的邊界區決策與全域決策
        for name in LQ:  # 逐一來源
            d0, da, dr, _, region = mb.decisions(P, LQ[name], 0.5, 3.0, 0.6); dR[name] = np.where(locked, l28, dr); dA[name] = np.where(locked, l28, da)  # 邊界區決策；鎖定題固定為 ct28 的答案
        base = np.where(locked, l28, d0); d44 = dR["v44"]; keep = d44 != base; fin = np.where(keep, d44, np.where(dR["v3"] == dR["v5"], dR["v5"], d44)); vR = np.stack([dR[k] for k in FIVE]); vA = np.stack([dA[k] for k in FIVE])  # 起點＝ct44 的答案，ct44 沒改的題在 v3、v5 一致時採 v5；五個投票來源的兩種決策疊成陣列
        rev = keep & (vR == vR[0]).all(0) & (vR[0] != d44); out = ~region & ~keep & ~locked & (vA == vA[0]).all(0) & (vA[0] != base); fin[rev] = vR[0][rev]; fin[out] = vA[0][out]  # ct44 改過的題五票一致反對時改用該票；未改過的題在邊界區外五票一致時改答案
        return d44, fin  # 回傳 ct44 與 ct75u 規則的答案
    d44_0, fin_0 = compose(P9, evidence_of(rec0)); assert (lev[d44_0] == c44).all() and (lev[fin_0] == c75u).all(); print("checks: with the learned pool off the builder reproduces the files ct44 and ct75u")  # 檢查：不用學出來的合併時要逐題重現 ct44 與 ct75u
    ti = np.flatnonzero(q["qtest"] >= 0); tgt = ti[q["side"][ti] == 0]; tgt = tgt[np.argsort(q["qtest"][tgt])]; assert len(tgt) == len(tqid) == 10000; qpos = {x: i for i, x in enumerate(qids)}; gq = np.array([qpos[x] for x in tqid]); tq = np.zeros((NM, 800, 128))  # 測試局本人側的查詢列，對應到 800 題
    for m in range(NM):  # 每個網路
        np.add.at(tq[m], gq, QV[m, tgt].astype(np.float64))  # 把各局向量加總到題目
    tq /= np.maximum(np.linalg.norm(tq, axis=2, keepdims=True), 1e-9)  # 正規化成單位向量
    lr, _, _ = la.reading(tq, [PR[m, 4].astype(np.float64) for m in range(NM)], plab, la.BE); lr = lr - lr.mean(0, keepdims=True)  # 以全體玩家網路（-1）的原型讀數並置中
    LQ2 = evidence_of(rec2); Xt = {"logp": np.log(np.maximum(P9, 1e-12)), "R": lr, **{k: LQ2[k] for k in NAMES}}  # 測試端的合併輸入（m15 版證據）
    Pl = apply(coef, Xt, nq_t.astype(int)); d_plain = np.where(locked, l28, np.asarray(b.decide(Pl), int)); fin_i = guard2(d_plain, rk(c104b).astype(int)); fin = lev[fin_i]; assert (fin[locked] == c28[locked]).all()  # 合併後的決策套上相對於 ct104b 檔答案的四條防護；鎖定題不變
    c105 = sub("ct105"); print(f"guards: {int((lev[d_plain] != c104b).sum())} -> {int((fin != c104b).sum())} questions differ from ct104b; against ct105: {int((fin != c105).sum())} differ (Public {int(((fin != c105) & pub).sum())})")  # 防護前後與 ct104b 不同的題數，以及與 ct105 的差異
    for nm, o in (("ct104b", c104b), ("ct75u", c75u), ("ct44", c44), ("ct28", c28)):  # 與既有各檔比較
        ch = fin != o; jj = np.abs(rk(fin) - rk(o)); print(f"{a_.name} vs {nm}: {int(ch.sum())} differ (Public {int((ch & pub).sum())}, Private {int((ch & ~pub).sum())}); one level {int((jj == 1).sum())}, two or more {int((jj >= 2).sum())}")  # 不同的題數與差幾級
    print("predicted levels, ct104b -> " + a_.name + ": " + "  ".join(f"{lv}{int((c104b == lv).sum())}->{int((fin == lv).sum())}" for lv in LEVELS))  # 各等級預測題數的變化
    kinds = pd.Series([f"{o}>{x}" for o, x in zip(c104b[fin != c104b], fin[fin != c104b])]).value_counts(); print("kinds of changes against ct104b: " + ", ".join(f"{k} {v}" for k, v in kinds.items()))  # 對 ct104b 的改動類別
    for i in np.flatnonzero(fin != c104b):  # 逐題列出改動
        print(f"    {qids[i]} ({int(nq_t[i])} games): ct9 {c9[i]}, ct75u {c75u[i]}, ct104b {c104b[i]} -> {fin[i]}")  # 題號、局數與各檔答案
    f = ROOT / f"submission_rank_{a_.name}.csv"  # 輸出檔路徑
    if f.exists():  # 不覆寫既有檔
        sys.exit(f"{f.name} exists already: nothing overwritten")  # 已存在就中止
    with open(f, "w", newline="") as fh:  # 寫出提交檔
        wr = csv.writer(fh, lineterminator="\n"); wr.writerow(["question_id", "pred_rank"]); wr.writerows(zip(qids, fin))  # 只用 LF 換行，表頭加每題答案
    raw = f.read_bytes(); md5 = hashlib.md5(raw).hexdigest()[:12]  # 計算 md5 前 12 碼
    print(f"{f.name} written (not copied to upload_next: the ledger back-test comes first), md5 {md5}; rows {len(fin)}, LF only {b'\\r' not in raw}, levels valid {set(fin) <= set(LEVELS)}; variant {choice}")  # 回報檔案與格式檢查
    rep["file"] = {"name": f.name, "md5": md5, "variant": choice, "differs_from_ct104b": qids[fin != c104b].tolist(), "public": int(((fin != c104b) & pub).sum()), "private": int(((fin != c104b) & ~pub).sum())}  # 檔案資訊寫入報告
    (ROOT / "results/r_variants" / a_.json).write_text(json.dumps(rep, indent=1, default=float))  # 重寫報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主流程
