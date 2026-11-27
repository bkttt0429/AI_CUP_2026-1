"""Section 4M M71 (2026-11-10): the CV gain at the TEST's anchor density. In the test-regime validation table (m71_val_regime.npz) a random
share 1 - rho of the opponent queries is replaced by an impostor draw (the target-side label and margin of a random other query scored
on the same model colour, same fold), so real anchors are thinned to rho of their validation density; the evidence uses
a_b = scale x the cross-fitted validation a_b. Report only (no selection): scale = rho is the deployed setting.
  python -I scripts/m_series/m71_cv_density.py
"""
import json  # 寫出 JSON 報告
import sys  # 調整模組搜尋路徑
from pathlib import Path  # 處理檔案路徑

import numpy as np  # 陣列運算

ROOT = Path.cwd()  # 以目前目錄當專案根目錄
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/m_series")]  # 讓 aicup 套件與同目錄模組可匯入
from aicup.qeval import SIZES, balanced_mean_over_n, evaluate, paired_diff  # noqa: E402  官方分數與配對比較
import m71_step2 as s2  # noqa: E402  錨點證據的校準
from ens_compare import ES, FOLDS, load_fold  # noqa: E402  折外機率與驗證折


def thin(z, rho, seed):  # 把錨點稀疏到 rho 倍的密度
    rng = np.random.default_rng(seed); out = {k: v.copy() for k, v in z.items()}  # 複製查詢表，不動原資料
    drop = rng.random(len(z["y"])) > rho  # 以 1 - rho 的機率選出要換掉的查詢
    for f in FOLDS:  # 逐折
        for model_black in (True, False):  # 逐一模型顏色
            src = np.flatnonzero((z["fold"] == f) & (z["target_black"] == model_black))  # 在此模型顏色上評分的冒充者
            dst = np.flatnonzero(drop & (z["fold"] == f) & (z["target_black"] != model_black))  # 在此模型顏色上評分的對手查詢
            pick = rng.choice(src, len(dst))  # 隨機抽出冒充者
            out["opp_lab"][dst], out["opp_m"][dst] = z["imp_lab"][pick], z["imp_m"][pick]  # 以冒充者的標籤與 margin 取代
    return out  # 回傳稀疏後的查詢表


def gain(z_cal, z_eval, scale, seeds):  # 給定密度與權重倍數下的 CV 增益
    S_new, S_old, pf, ys, ns, pids = [], [], [], [], [], []  # 收集各折結果
    for v in FOLDS:  # 逐一驗證折
        y, n, pid, q = load_fold(v)  # 本折的真值、局數、玩家與題目表
        P = np.mean([np.load(ES / f"val{v}/{arm}_s{s}.npz")["proba"].astype(np.float64) for arm in s2.STD for s in seeds], axis=0)  # 標準成員的折外機率平均
        edges, cal, _, _ = s2.fit(z_cal, z_cal["fold"] != v)  # 只用其餘三折校準
        te = np.flatnonzero(z_eval["fold"] == v); tb = z_eval["target_black"][te]  # 本折的查詢與本人執色
        band = np.where(tb, np.searchsorted(edges[False], z_eval["opp_m"][te]), np.searchsorted(edges[True], z_eval["opp_m"][te]))  # 對手查詢的帶（對手執另一色）
        g = s2.group_of(z_eval["hd"][te], tb); lab = z_eval["opp_lab"][te]; logE = np.zeros((len(te), 10))  # 分組、配對標籤與對數證據
        for k in (0, 1, 2):  # 三個分組
            s = g == k; a = scale * cal[k]["a"]; dd = lab[s][:, None] - np.arange(10)[None, :]  # 權重乘上倍數；標籤差 d
            logE[s] = np.log((1 - a[band[s]])[:, None] + a[band[s]][:, None] * cal[k]["ratio"][dd + 9])  # 每局證據 e(c) = (1 - a) + a·K/P0
        key_row = {k: i for i, k in enumerate(z_eval["key"][te])}; L = np.zeros_like(P)  # 棋局鍵對應查詢列；題目的對數證據
        for i, gk in enumerate(q["game_keys"]):  # 逐題
            rows = [key_row[k] for k in gk if k in key_row]  # 這題中有查詢的局
            if rows:  # 有錨點證據
                L[i] = logE[rows].sum(axis=0)  # 各局證據相乘（對數相加）
        Pf = P * np.exp(L - L.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)  # 後驗乘上證據後正規化
        so, sn = evaluate(P, y, q, sizes=SIZES)[1], evaluate(Pf, y, q, sizes=SIZES)[1]  # 加證據前後的逐題得分
        S_new.append(sn); S_old.append(so); ys.append(y); ns.append(n); pids.append(pid); pf.append(balanced_mean_over_n(sn, y, n, sizes=SIZES) - balanced_mean_over_n(so, y, n, sizes=SIZES))  # 收集結果與本折的差
    y, n, pid = np.concatenate(ys), np.concatenate(ns), np.concatenate(pids)  # 合併四折
    d = paired_diff(np.concatenate(S_new), np.concatenate(S_old), y, pid, n, sizes=SIZES, n=3000)  # 玩家層配對 bootstrap
    return d, pf  # 回傳整體差與各折差


def main():  # 命令列進入點
    z = dict(np.load(ROOT / "results/r_variants/m71_val_regime.npz")); rep = {}  # 測試制度的驗證查詢表
    for rho, scale, seeds, name in ((1.0, 1.0, (0, 1, 2), "full density, a x1.00 (CV as measured)"), (1.0, 0.74, (0, 1, 2), "full density, a x0.74 (under-weighted)"),  # 情境：滿密度（實測 CV）、權重偏低
                                    (0.74, 0.74, (0, 1, 2), "TEST-LIKE density 0.74, a x0.74, seeds 0-2"), (0.74, 0.74, (6, 7, 8), "TEST-LIKE density 0.74, a x0.74, seeds 6-8"),  # 像測試的密度 0.74，兩組種子
                                    (0.74, 1.0, (0, 1, 2), "density 0.74, a x1.00 (over-weighted)"), (0.5, 0.74, (0, 1, 2), "density 0.50, a x0.74 (if the density were lower still)")):  # 權重偏高、密度更低
        ds = []  # 各次抽樣的結果
        for seed in (1, 2, 3) if rho < 1 else (1,):  # 稀疏時抽三次取平均
            d, pf = gain(z, thin(z, rho, seed) if rho < 1 else z, scale, seeds); ds.append((d, pf))  # 以完整表校準、稀疏表評估
        diff = float(np.mean([d["diff"] for d, _ in ds])); lo = float(np.mean([d["ci95"][0] for d, _ in ds])); hi = float(np.mean([d["ci95"][1] for d, _ in ds])); pf = np.mean([p for _, p in ds], axis=0)  # 各次抽樣取平均
        by_n = {m: float(np.mean([d["by_n"][m] for d, _ in ds])) for m in ds[0][0]["by_n"]}  # 依題目局數的差
        rep[name] = {"diff": diff, "ci95": [lo, hi], "per_fold": pf.tolist(), "by_n": by_n}  # 記錄此情境
        print(f"{name:52s} {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]  folds " + " ".join(f"{e:+.4f}" for e in pf) + "  by n " + " ".join(f"{m}: {val:+.4f}" for m, val in by_n.items()))  # 印出此情境的結果
    Path(ROOT / "results/r_variants/m71_cv_density.json").write_text(json.dumps(rep, indent=1, default=float)); print("written")  # 寫出 JSON 報告


if __name__ == "__main__":  # 直接執行時
    main()  # 執行主程式
