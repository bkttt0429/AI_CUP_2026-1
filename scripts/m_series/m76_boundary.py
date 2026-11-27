"""Section 4M M76 (2026-11-10; user: Jev 只想用在分不開的邊界上，參考它的機率設計方法). Report only.
A boundary-only decision module in Jev's shape, on the opponent-anchor evidence (bands >= q90, as ct41):
  region  = the posterior's two most likely levels are adjacent, hold >= MASS of the probability, and their ratio is <= RMAX
  Noul    = pairwise posterior of the second level given the pair, from the model's two probabilities x the anchors' evidence for those two levels
  act     = switch to the second level only when its pairwise posterior >= TAU (0.5 = Bayes within the pair; > 0.5 = act only when confident);
            outside the region the base decision stands
against A = the anchors on all ten levels of every question (the current composition), and "A inside the region only".
Scenarios: CV density with matched weights; test-like density 0.74 with weights x0.74; weights too high (density 0.74, x1.0);
density lower than assumed (0.5, x0.74). Seeds 0-2 and 6-8. Gain vs the base ensemble, share of changed decisions, share of changes of >= 2 levels.
  python -I scripts/m_series/m76_boundary.py
"""
import json  # 寫出 JSON 報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  題目局數清單與配對 bootstrap
import m71_cv_density as cd  # noqa: E402  錨點稀疏化，模擬測試密度
import m71_step2 as s2  # noqa: E402  錨點證據的校準
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折

K = np.arange(10); S = np.where(K[:, None] == K[None, :], 1.0, np.where(np.abs(K[:, None] - K[None, :]) == 1, np.exp(-1), 0.0))  # 官方得分矩陣 S[決策, 真值]
KMIN = 3  # 只用帶索引 >= 3（由 0 起算，即 >= q90）的錨點，同 ct41


def fold_data(z_cal, z_eval, scale, seeds):  # 每折的真值、後驗與錨點對數證據
    out = []  # 每折一筆
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(v)  # 本折的真值、局數、玩家與題目表
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 標準成員的折外機率平均
        edges, cal, _, _ = s2.fit(z_cal, z_cal["fold"] != v)  # 只用其餘三折校準
        te = np.flatnonzero(z_eval["fold"] == v); tb = z_eval["target_black"][te]  # 本折的查詢與本人執色
        band = np.where(tb, np.searchsorted(edges[False], z_eval["opp_m"][te]), np.searchsorted(edges[True], z_eval["opp_m"][te]))  # 對手查詢的帶（對手執另一色）
        g = s2.group_of(z_eval["hd"][te], tb); lab = z_eval["opp_lab"][te]; logE = np.zeros((len(te), 10))  # 分組、配對標籤與對數證據
        for k in (0, 1, 2):  # 三個分組
            s = g == k; a = scale * cal[k]["a"] * (np.arange(len(cal[k]["a"])) >= KMIN); dd = lab[s][:, None] - K[None, :]  # 權重乘倍數、低於 q90 的帶歸 0；標籤差 d
            logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])  # 每局的對數證據 log e(c)
        key_row = {k: i for i, k in enumerate(z_eval["key"][te])}; L = np.zeros_like(P)  # 棋局鍵對應查詢列；題目的對數證據
        for i, gk in enumerate(q["game_keys"]):  # 逐題
            rows = [key_row[k] for k in gk if k in key_row]  # 這題中有查詢的局
            if rows:  # 有錨點證據
                L[i] = logE[rows].sum(axis=0)  # 各局證據相乘（對數相加）
        out.append((y, n, pid, P, L))  # 存下本折
    return out  # 回傳四折的資料


def decisions(P, L, mass, rmax, tau):  # 各種決策規則的結果
    d0 = (P @ S.T).argmax(1); Pa = P * np.exp(L - L.max(1, keepdims=True)); dA = (Pa @ S.T).argmax(1)  # 基礎集成的期望分決策，與錨點作用於全部十級後的決策
    top = np.argsort(-P, axis=1)[:, :2]; c1, c2 = top[:, 0], top[:, 1]; r = np.arange(len(P))  # 後驗的第一、二名
    region = (np.abs(c1 - c2) == 1) & (P[r, c1] + P[r, c2] >= mass) & (P[r, c1] / np.maximum(P[r, c2], 1e-12) <= rmax)  # 邊界區域：相鄰、合占夠大、比值不懸殊
    w1, w2 = P[r, c1] * np.exp(L[r, c1]), P[r, c2] * np.exp(L[r, c2]); p2 = w2 / (w1 + w2)  # 兩級的成對後驗（Noul）
    dB = np.where(region & (p2 >= tau), c2, np.where(region & (1 - p2 >= tau), c1, d0))  # 區域內取成對後驗 >= tau 的那一級；兩級都未達或在區域外，維持基礎集成的決策
    return d0, dA, np.where(region, dA, d0), dB, region  # 基礎集成、全部十級、只在區域內、成對規則的決策與區域遮罩


def main():  # 命令列進入點
    z = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); rep = {}  # 測試制度的驗證查詢表
    variants = [("A all levels, all questions", None), ("A inside the region only", None)] + [(f"pair-only, act at >= {t:.2f}", t) for t in (0.5, 0.6, 0.7)]  # 要比較的決策變體
    for sc_name, rho, scale in (("CV density, matched weights", 1.0, 1.0), ("test-like density 0.74, weights x0.74", 0.74, 0.74), ("weights too high (0.74, x1.0)", 0.74, 1.0), ("density lower than assumed (0.5, x0.74)", 0.5, 0.74)):  # 四種密度與權重情境
        print(f"== {sc_name}"); rep[sc_name] = {}  # 情境標題
        for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
            acc = {nm: [] for nm, _ in variants}; reg_share = []  # 各變體的結果與區域占比
            for draw in ((1, 2, 3) if rho < 1 else (1,)):  # 稀疏時抽三次
                data = fold_data(z, cd.thin(z, rho, draw) if rho < 1 else z, scale, seeds)  # 以完整表校準、稀疏表評估
                y, n, pid = (np.concatenate([d[k] for d in data]) for k in (0, 1, 2))  # 合併四折的真值、局數與玩家
                per = {nm: [] for nm, _ in variants}; d0s, regs = [], []  # 各變體每折的決策
                for yy, nn, pp, P, L in data:  # 逐折
                    for (nm, tau) in variants:  # 逐一變體
                        d0, dA, dAr, dB, region = decisions(P, L, 0.5, 3.0, tau if tau else 0.5)  # 區域：合占 >= 0.5、比值 <= 3
                        per[nm].append({"A all levels, all questions": dA, "A inside the region only": dAr}.get(nm, dB))  # 取該變體的決策
                    d0s.append(d0); regs.append(region)  # 基礎集成的決策與區域
                d0 = np.concatenate(d0s); reg_share.append(float(np.concatenate(regs).mean()))  # 合併並記錄區域占比
                for nm, _ in variants:  # 逐變體比較
                    d = np.concatenate(per[nm]); dd = paired_diff(S[d, y], S[d0, y], y, pid, n, sizes=SIZES, n=2000); ch = d != d0  # 對基礎集成決策的配對差；改動的題
                    acc[nm].append((dd["diff"], dd["ci95"][0], float(ch.mean()), float((np.abs(d - d0)[ch] >= 2).mean()) if ch.any() else 0.0))  # 差、CI 下界、改動比例、改動 >= 2 級的比例
            for nm, _ in variants:  # 逐變體彙整
                a = np.mean(acc[nm], axis=0); rep[sc_name].setdefault(nm, []).append(a.tolist())  # 各次抽樣取平均後記錄
            print(f"  seeds {seeds} (region = {np.mean(reg_share):.3f} of questions): " + " | ".join(f"{nm}: {np.mean(acc[nm], axis=0)[0]:+.4f} (CI lower {np.mean(acc[nm], axis=0)[1]:+.4f}; changed {np.mean(acc[nm], axis=0)[2]:.3f}, >=2 levels {np.mean(acc[nm], axis=0)[3]:.2f})" for nm, _ in variants), flush=True)  # 印出本組種子的結果
    Path(ROOT / "results/r_variants/m76_boundary.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
