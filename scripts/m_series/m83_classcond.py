"""Section 4M M83 (2026-11-11): class-conditional anchor evidence. Today e(c) = (1 - a_b) + a_b min(K(l - c) / P0(l - c), 30) with K, P0
functions of the difference only. Here one row per target class c: P0_c(l) from the low bands (<= q80) of the queries whose truth is c,
K_c(l) from the top bands (>= q98) minus (1 - a_top) P0_c, both shrunk with KAPPA pseudo-counts towards the pooled tables restricted to
the 10 legal labels and renormalised. Bands, a_b, groups and the cross-fit (fold v calibrated on the other folds) are unchanged.
Identifier: five components + matcher (m82_val.npz, f5e_). Primary: class-conditional vs pooled evidence, M37, seeds 0-2 and 6-8.
Report only: truncation-only (kappa = infinity); gain by true class.
  python -I scripts/m_series/m83_classcond.py
"""
import json  # 寫出 JSON 報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.metric import LEVELS  # noqa: E402  十個等級的順序
from aicup.qeval import SIZES, balanced_mean_over_n, paired_diff  # noqa: E402  官方分數與配對比較
import m71_step2 as s2  # noqa: E402  錨點證據的校準
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折

KAPPA = 50.0  # 向共用表收縮的虛擬計數
SM = np.zeros((10, 10)); _i = np.arange(10); SM[_i, _i] = 1.0; SM[_i[:-1], _i[:-1] + 1] = SM[_i[:-1] + 1, _i[:-1]] = np.exp(-1.0)  # 官方得分矩陣：同級 1、差一級 e^-1


def tables(z, tr, cal, band, g, kappa):  # 依目標等級條件化的證據比值表
    """{group: R [10 target classes, 10 labels]} = min(K_c(l) / P0_c(l), 30); kappa None = the pooled tables, truncated and renormalised."""
    y, lab, out = z["y"], z["opp_lab"], {}  # 真值、配對標籤與輸出
    for k in (0, 1, 2):  # 三個分組
        s = tr & (g == k); low, top = s & (band <= 1), s & (band >= 5); a_top, K, P0 = cal[k]["a_top"], cal[k]["K"], cal[k]["P0"]; R = np.zeros((10, 10))  # 低帶（<= q80）與高帶（>= q98）的查詢；共用的 K、P0
        for c in range(10):  # 逐一目標等級
            pk, p0 = K[np.arange(10) - c + 9], P0[np.arange(10) - c + 9]; pk, p0 = pk / pk.sum(), p0 / p0.sum()  # 共用表限制在 10 個合法標籤後重新正規化
            if kappa is None:  # 只截斷、不條件化
                Kc, P0c = pk, p0  # 直接用截斷後的共用表
            else:  # 條件化並向共用表收縮
                n0 = np.bincount(lab[low & (y == c)], minlength=10).astype(float); P0c = (n0 + kappa * p0) / (n0.sum() + kappa)  # 真值為 c 時低帶的標籤分布
                nt = np.bincount(lab[top & (y == c)], minlength=10).astype(float); Ht = (nt + kappa * ((1 - a_top) * p0 + a_top * pk)) / (nt.sum() + kappa)  # 真值為 c 時高帶的標籤分布
                Kc = np.clip(Ht - (1 - a_top) * P0c, 0, None); Kc = Kc / Kc.sum()  # 扣除背景、截 0 後正規化
            R[c] = np.minimum(Kc / P0c, 30.0)  # 比值上限 30
        out[k] = R  # 存下本組的表
    return out  # 回傳各分組的表


def evidence(z, v, kappa="pooled"):  # 第 v 折每局的對數證據
    """{game_key: log e [10]} of fold v, calibrated on the other folds. kappa "pooled" = today's evidence."""
    tr = z["fold"] != v; edges, cal, band_all, g_all = s2.fit(z, tr); te = np.flatnonzero(z["fold"] == v); tb = z["target_black"][te]  # 以其餘三折校準；本折查詢與本人執色
    band = np.where(tb, np.searchsorted(edges[False], z["opp_m"][te]), np.searchsorted(edges[True], z["opp_m"][te])); g = s2.group_of(z["hd"][te], tb); lab = z["opp_lab"][te]; out = np.zeros((len(te), 10))  # 對手查詢的帶、分組、配對標籤與輸出
    R = None if kappa == "pooled" else tables(z, tr, cal, band_all, g_all, kappa)  # 共用版不需要條件化表
    for k in (0, 1, 2):  # 三個分組
        s = g == k; a = cal[k]["a"][band[s]][:, None]  # 本組查詢與各帶權重
        ratio = cal[k]["ratio"][lab[s][:, None] - np.arange(10)[None, :] + 9] if R is None else R[k][:, lab[s]].T  # 共用版依標籤差查表；條件化版依（目標等級, 標籤）查表
        out[s] = np.log((1 - a) + a * ratio)  # 每局的對數證據
    return dict(zip(z["key"][te], out)), (R, cal)  # 棋局鍵對應證據；條件化表與校準


def main():  # 命令列進入點
    d = dict(np.load(ROOT / "results/r_variants/m82_val.npz")); z = {**{k: d[k] for k in ("key", "pid", "fold", "y", "hd", "target_black")}, "opp_lab": d["f5e_opp_lab"], "opp_m": d["f5e_opp_m"], "imp_lab": d["f5e_imp_lab"], "imp_m": d["f5e_imp_m"]}  # 識別器＝五成分加配對器（f5e_）的驗證查詢表
    VAR = {"pooled": "pooled", "truncated": None, "class": KAPPA}; EV = {name: {v: evidence(z, v, kap) for v in FOLDS} for name, kap in VAR.items()}  # 三種證據：共用、只截斷、條件化，每折各算一次
    R, cal = EV["class"][FOLDS[0]][1]; Rp = EV["truncated"][FOLDS[0]][1][0]  # 給第 1 折用（以其餘三折校準）的條件化表與截斷表
    print("HD0 evidence ratio K_c(l) / P0_c(l), rows = target class, columns = matched label (fold 1's calibration); class-conditional | pooled, truncated:")  # 表格標題
    for c in range(10):  # 逐一目標等級
        print(f"  {LEVELS[c]:>2s} " + " ".join(f"{x:5.2f}" for x in R[0][c]) + "   | " + " ".join(f"{x:5.2f}" for x in Rp[0][c]))  # 印出條件化與截斷兩版的這一列
    rep = {}  # 報告
    for seeds in ((0, 1, 2), (6, 7, 8)):  # 兩組種子
        S = {name: [] for name in (*VAR, "base")}; ys, ns, pids = [], [], []  # 各證據版本與基礎集成的逐題得分
        for v in FOLDS:  # 逐一驗證折
            y, n, pid, q = load_fold(v); P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 本折資料與標準成員的折外機率平均
            S["base"].append(SM[(P @ SM).argmax(1), y]); ys.append(np.asarray(y)); ns.append(np.asarray(n)); pids.append(np.asarray(pid))  # 基礎集成的期望分決策得分
            for name in VAR:  # 逐一證據版本
                ev = EV[name][v][0]; L = np.zeros_like(P)  # 本折的證據與題目的對數證據
                for i, gk in enumerate(q["game_keys"]):  # 逐題
                    for k in gk:  # 這題的每一局
                        if k in ev:  # 有錨點證據
                            L[i] += ev[k]  # 累加對數證據
                Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); S[name].append(SM[(Pf @ SM).argmax(1), y])  # 乘上證據後的期望分決策得分
        y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids); res = {}  # 合併四折；本組種子的結果

        def cmp(a, b, label):  # 配對比較兩個版本
            dd = paired_diff(np.concatenate(S[a]), np.concatenate(S[b]), y, pid, n, sizes=SIZES, n=5000); pf = [balanced_mean_over_n(x, yy, nn, sizes=SIZES) - balanced_mean_over_n(w, yy, nn, sizes=SIZES) for x, w, yy, nn in zip(S[a], S[b], ys, ns)]  # 玩家層配對 bootstrap 與各折的差
            ok = (dd["ci95"][0] > 0) and (sum(e > 0 for e in pf) >= 3) and (dd["diff"] >= 0.0025)  # M37 規則
            print(f"  {label:44s} {dd['diff']:+.4f} [{dd['ci95'][0]:+.4f}, {dd['ci95'][1]:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {x:+.4f}" for m, x in dd["by_n"].items()) + f"  -> {'PASS' if ok else 'no'}")  # 印出比較結果
            return {**dd, "per_fold": pf, "pass": bool(ok)}  # 回傳結果與判定
        print(f"seeds {seeds}:")  # 種子組標題
        for name in VAR:  # 各版本對基礎集成
            res[f"{name}_vs_base"] = cmp(name, "base", f"{name} evidence vs base")  # 比較並記錄
        res["primary"] = cmp("class", "pooled", "CLASS-CONDITIONAL vs POOLED (primary)"); res["truncated_vs_pooled"] = cmp("truncated", "pooled", "truncated vs pooled (report)"); res["class_vs_truncated"] = cmp("class", "truncated", "class-conditional vs truncated (report)")  # 主檢定（條件化對共用）與兩個只報告的比較
        for name in VAR:  # 各版本依真實等級的增益
            gain = np.concatenate(S[name]) - np.concatenate(S["base"]); print(f"  gain by true class, {name:9s}: " + " ".join(f"{LEVELS[c]} {np.mean([gain[(y == c) & (n == m)].mean() for m in SIZES]):+.4f}" for c in range(10)))  # 印出各等級在三種局數下的平均增益
        rep[f"seeds_{seeds[0]}"] = res  # 記錄本組種子
    ok = all(r["primary"]["pass"] for r in rep.values()); rep["adopt"] = bool(ok); print(f"PRE-REGISTERED: class-conditional evidence replaces the pooled one: {'YES' if ok else 'NO'}")  # 兩組主檢定都通過才採用
    Path(ROOT / "results/r_variants/m83_classcond.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
